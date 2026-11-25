"""Mixed-precision dual-stream DetectionTrainer for DeRoF-Det with atomic resumption."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from ultralytics import YOLO
from ultralytics.engine.trainer import LOGGER, RANK, TQDM, autocast, colorstr, unset_deterministic
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import callbacks

from data.paired_dataset import PairedRGBTDataset, make_loader
from engine.checkpoint import capture, json_write, restore_state_dict, save_atomic, utcnow
from engine.evaluator import Validator
from engine.protocols import APStopper, CLS_SHA256, EXPERIMENTS, spec_for
from models.baselines import P2AddModel, P2BaseV2Model, P2ConcatC2fModel
from models.detector import DeRoFDet, transfer_imagenet_backbone


def compute_source_hash():
    """Hash all repository Python source files for reproducible resume verification."""
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    files = [root / "train.py"]
    for pkg in ("models", "data", "engine"):
        files.extend(sorted((root / pkg).glob("*.py")))
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class DeRoFTrainer(DetectionTrainer):
    """Unified trainer for DeRoF-Det (<50M) and dual-stream P2 baselines."""

    def __init__(
        self,
        ident,
        data_root=None,
        weights_path=None,
        fixture=False,
        cap=None,
        p2_fusion_width=64,
        strict_tail=False,
        **kwargs,
    ):
        cfg = EXPERIMENTS[ident]
        self.ident = ident
        self.candidate = ident
        self.phase = "train"
        self.s_cfg = cfg
        self.spec = spec_for(ident)
        if cap is not None:
            self.spec["cap"] = cap
        self.data_root = data_root
        self.weights_path = weights_path
        self.fixture = fixture
        self.schedule_epochs = 2 if fixture else cfg["schedule_epochs"]
        self.stop_epoch = cap or (2 if fixture else cfg["stop_epoch"])
        self.patience_epochs = 0 if fixture else cfg["patience"]
        self.cap = self.stop_epoch
        self.winner = cfg["label"]
        self.p2_fusion_width = p2_fusion_width
        self.strict_tail = strict_tail
        self.source_sha = compute_source_hash()

        self.samples_since_update = 0
        self.optimizer_attempts = 0
        self.successful_updates = 0
        self.amp_skips = 0
        self.last_grad_norm = None
        self.last_update_samples = 0
        self.started = utcnow()
        self.saved = None
        self.full_state_restored = False
        self._validated_epoch = -1

        super().__init__(**kwargs)
        self.callbacks = callbacks.get_default_callbacks()
        self.add_callback("on_train_start", self.restore_state)
        self.add_callback("on_train_batch_end", lambda t: t.write_status())
        self.add_callback("on_train_epoch_end", self.on_epoch_cadence)
        self.add_callback("on_fit_epoch_end", self.record)
        self.add_callback("on_model_save", lambda t: t.write_status())
        self.add_callback("on_train_end", lambda t: t.write_status("completed"))

    def _resolve_weights(self):
        root = Path(__file__).resolve().parent.parent
        candidates = []
        if self.weights_path:
            candidates.append(Path(self.weights_path))
        env_dir = os.environ.get("RGBT_WEIGHTS_DIR")
        if env_dir:
            candidates.append(Path(env_dir) / "yolov8m-cls.pt")
        candidates.append(root / "weights/yolov8m-cls.pt")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Missing ImageNet-1K pretrained weights 'yolov8m-cls.pt'. "
            "Pass --weights <path> or place it under ./weights/yolov8m-cls.pt."
        )

    def get_dataset(self):
        if self.spec["nc"] == 3:
            return dict(train="paired", val="paired", nc=3, names={0: "person", 1: "rider", 2: "crowd"}, channels=6)
        return dict(train="paired", val="paired", nc=1, names={0: "person"}, channels=6)

    def setup_model(self):
        w = self._resolve_weights()
        if hashlib.sha256(w.read_bytes()).hexdigest() != CLS_SHA256:
            raise AssertionError(f"Weight checksum mismatch for {w}; expected SHA256={CLS_SHA256}")
        torch.random.default_generator.manual_seed(42)
        cls_model = YOLO(str(w)).model
        nc = int(self.spec["nc"])
        arch = self.s_cfg["arch"]
        if arch == "add":
            self.model = P2AddModel(pretrained=None, nc=nc)
        elif arch == "concat_c2f":
            self.model = P2ConcatC2fModel(pretrained=None, nc=nc)
        elif arch == "base_v2":
            self.model = P2BaseV2Model(pretrained=None, nc=nc, p2_fusion_width=self.p2_fusion_width)
        elif arch == "derof":
            self.model = DeRoFDet(
                pretrained=None,
                nc=nc,
                slim_variant=self.s_cfg["slim_variant"],
                mask=self.s_cfg["mask"],
                p2_fusion_width=self.p2_fusion_width,
                spmr_gamma_init=self.s_cfg.get("spmr_gamma_init", 0.0),
                aux_scale=self.s_cfg.get("aux_scale", 1.0),
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.model.degrade = self.s_cfg["degrade"]
        self.model.strict_tail = self.strict_tail
        transferred = transfer_imagenet_backbone(self.model, cls_model)
        assert transferred == 150, f"Expected 150 ImageNet tensors, got {transferred}"
        self.model.in1k_transferred = transferred

        count = sum(p.numel() for p in self.model.parameters())
        json_write(
            self.save_dir / "initialization.json",
            dict(
                id=self.ident,
                dataset=self.s_cfg["dataset"],
                subset=self.s_cfg.get("prefix"),
                arch=arch,
                slim_variant=self.s_cfg["slim_variant"],
                winner=self.winner,
                parameters=count,
                source="official_ImageNet1K_cls",
                pretrain_weights=str(w),
                weights_sha256=CLS_SHA256,
                nc=nc,
                mask=self.s_cfg["mask"],
                head="P2-P5",
                hw=list(self.spec["hw"]),
                p2_fusion_width=self.p2_fusion_width,
                degrade=self.s_cfg["degrade"],
                transferred_tensors_per_stream=transferred,
                batch=self.args.batch,
                nbs=self.args.nbs,
                lr0=self.args.lr0,
                lrf=self.args.lrf,
                cos_lr=self.args.cos_lr,
                epochs=self.schedule_epochs,
                stop_epoch=self.stop_epoch,
                patience=self.patience_epochs,
                close_mosaic=self.args.close_mosaic,
                val_batch=self.s_cfg.get("val_batch", 16),
                source_sha=self.source_sha,
            ),
        )
        ckpt = self.save_dir / "weights/full_last.pt"
        if ckpt.exists():
            self.saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        return None

    def get_dataloader(self, dataset_path, batch_size=8, rank=-1, mode="train"):
        train = mode == "train"
        ds = PairedRGBTDataset(
            self.spec,
            "train" if train else "val",
            self.args,
            data_root=self.data_root,
            degrade=self.s_cfg["degrade"],
            fixture=self.fixture,
        )
        bs = self.args.batch if train else (4 if self.fixture else self.s_cfg.get("val_batch", 16))
        return make_loader(ds, bs, self.args.workers, train)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return Validator(self.spec)

    def preprocess_batch(self, batch):
        expected = (6, *self.spec["hw"])
        if batch["img"].shape[1:] != expected:
            raise AssertionError(f"Expected batch tensor shape (B, {expected}), got {tuple(batch['img'].shape)}")
        self.samples_since_update += batch["img"].shape[0]
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        old_scale = self.scaler.get_scale()
        norms = [p.grad.detach().float().norm() / old_scale for p in self.model.parameters() if p.grad is not None]
        if norms:
            norm = torch.linalg.vector_norm(torch.stack(norms))
            self.last_grad_norm = float(norm) if torch.isfinite(norm) else None
            if self.successful_updates == 0 and torch.isfinite(norm):
                for branch in (self.model.model[:10], self.model.ir_backbone):
                    assert any(
                        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0
                        for p in branch.parameters()
                    ), "Stream backbone received zero or non-finite gradient"
        super().optimizer_step()
        self.optimizer_attempts += 1
        if self.scaler.get_scale() >= old_scale:
            self.successful_updates += 1
        else:
            self.amp_skips += 1
        self.last_update_samples = self.samples_since_update
        self.samples_since_update = 0

    def on_epoch_cadence(self, trainer):
        ep1 = trainer.epoch + 1
        trainer.args.val = True
        self._validated_epoch = ep1
        trainer.stop = ep1 >= self.stop_epoch

    def restore_state(self, trainer):
        self.stopper = APStopper(self.patience_epochs)
        if self.saved:
            restore_state_dict(self, self.saved)
            self.stopper.__dict__.update(self.saved["stopper"])
            self.stopper.patience = self.patience_epochs or float("inf")
            self.saved = None
        json_write(
            self.save_dir / "runtime.json",
            dict(
                **self.spec,
                winner=self.winner,
                arch=self.s_cfg["arch"],
                slim_variant=self.s_cfg["slim_variant"],
                schedule_epochs=self.schedule_epochs,
                actual_cap=self.stop_epoch,
                head="P2-P5",
                p2_fusion_width=self.p2_fusion_width,
                spmr_gamma_init=self.s_cfg.get("spmr_gamma_init", 0.0),
                aux_scale=self.s_cfg.get("aux_scale", 1.0),
                source_sha=self.source_sha,
                full_state_restored=self.full_state_restored,
                degrade=self.s_cfg["degrade"],
                batch=self.args.batch,
                nbs=self.args.nbs,
                lr0=self.args.lr0,
                lrf=self.args.lrf,
                cos_lr=self.args.cos_lr,
                close_mosaic=self.args.close_mosaic,
            ),
        )

    def save_model(self):
        state = capture(self)
        state["stopper"] = dict(self.stopper.__dict__)
        save_atomic(self.save_dir / "weights/full_last.pt", state)
        if self._validated_epoch != self.epoch + 1:
            return
        best_file = self.save_dir / "best_score.json"
        best = json.loads(best_file.read_text(encoding="utf-8"))["score"] if best_file.exists() else -1.0
        if self.fitness > best:
            save_atomic(self.save_dir / "weights/full_best.pt", state)
            json_write(best_file, dict(score=self.fitness, epoch=self.epoch + 1))

    def save_metrics(self, metrics):
        p = self.save_dir / "results.csv"
        rows = {}
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                rows = {int(row["epoch"]): row for row in csv.DictReader(f)}
        rows[self.epoch + 1] = dict(epoch=self.epoch + 1, **metrics)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", *metrics])
            writer.writeheader()
            for ep in sorted(rows):
                writer.writerow({k: rows[ep].get(k, "") for k in ["epoch", *metrics]})
        tmp.replace(p)

    def record(self, trainer):
        if self._validated_epoch != self.epoch + 1:
            return
        p = self.save_dir / "history.jsonl"
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
        rows = [r for r in rows if r["epoch"] != self.epoch + 1]
        rows.append(
            dict(
                epoch=self.epoch + 1,
                metrics=self.metrics,
                fitness=self.fitness,
                lr=self.lr,
                successful_steps=self.successful_updates,
                at=time.time(),
            )
        )
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def final_eval(self):
        if self.fixture:
            return
        best_ckpt = self.save_dir / "weights/full_best.pt"
        if not best_ckpt.exists():
            return
        state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        self.ema.ema.load_state_dict(state["ema"])
        self.evaluation_tag = "best_validation"
        result = self.validator(self)
        del self.evaluation_tag
        json_write(
            self.save_dir / "completion.json",
            dict(
                id=self.ident,
                epoch=self.epoch + 1,
                best_epoch=state["epoch"] + 1,
                best=result,
                successful_steps=self.successful_updates,
                source_sha=self.source_sha,
            ),
        )

    def write_status(self, status="running"):
        if self.ema is None or self.optimizer is None:
            return
        losses = self.loss_items.detach().float().cpu().tolist() if hasattr(self, "loss_items") else []
        if losses and not all(math.isfinite(x) for x in losses):
            raise FloatingPointError("Non-finite training loss encountered")
        record = dict(
            id=self.ident,
            winner=self.winner,
            status=status,
            pid=os.getpid(),
            started_at=self.started,
            updated_at=utcnow(),
            epoch=self.epoch + 1,
            planned_epochs=self.epochs,
            cap=self.cap,
            head="P2-P5",
            successful_steps=self.successful_updates,
            ema_updates=self.ema.updates,
            optimizer_attempts=self.optimizer_attempts,
            amp_skips=self.amp_skips,
            loss=losses,
            grad_norm=self.last_grad_norm,
            batch=self.batch_size,
            nbs=self.args.nbs,
            accumulation=self.accumulate,
            input_hw=list(self.spec["hw"]),
            lr=[g["lr"] for g in self.optimizer.param_groups],
            cuda_peak_allocated_mib=(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        )
        json_write(self.save_dir / "status.json", record)

    def _do_train(self, world_size=1):
        """Mixed-precision Ultralytics-compatible epoch loop honoring full-state continuation."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        self.last_opt_step = getattr(self, "last_opt_step", -1)
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self.epochs} epochs (stop_epoch={self.stop_epoch})..."
        )
        epoch = self.start_epoch
        if not getattr(self, "full_state_restored", False):
            self.optimizer.zero_grad()

        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            pbar = enumerate(self.train_loader)
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None

            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, group in enumerate(self.optimizer.param_groups):
                        group["lr"] = np.interp(
                            ni,
                            xi,
                            [self.args.warmup_bias_lr if j == 0 else 0.0, group["initial_lr"] * self.lf(epoch)],
                        )
                        if "momentum" in group:
                            group["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                self.scaler.scale(self.loss).backward()
                if ni - self.last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    self.last_opt_step = ni

                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            t_now = time.time()
            self.epoch_time = t_now - self.epoch_time_start
            self.epoch_time_start = t_now
            self.run_callbacks("on_fit_epoch_end")
            if self._get_memory(fraction=True) > 0.5:
                self._clear_memory()

            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1

        if RANK in {-1, 0}:
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")
