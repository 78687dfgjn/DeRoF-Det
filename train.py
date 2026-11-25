"""Unified CLI entrypoint for training and evaluating DeRoF-Det models."""
import argparse
import os
from pathlib import Path
import random

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
from ultralytics import settings
import ultralytics.engine.trainer as native_trainer

from engine.checkpoint import RunLock
from engine.protocols import EXPERIMENTS, P2_TRAIN, spec_for
from engine.trainer import DeRoFTrainer

settings.update({"sync": False, "wandb": False, "mlflow": False, "tensorboard": False})


def run(
    ident,
    data_root=None,
    weights=None,
    output_dir="runs",
    fixture=False,
    cap=None,
    batch=None,
    workers=None,
    p2_fusion_width=64,
    strict_tail=False,
):
    """Run or cleanly resume an experiment registered in `engine.protocols.EXPERIMENTS`."""
    if ident not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment '{ident}'. Choose from: {sorted(EXPERIMENTS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training.")

    cfg = EXPERIMENTS[ident]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    native_trainer.check_amp = lambda m: True

    root = Path(__file__).resolve().parent
    out = Path(output_dir) / ("fixture_runs" if fixture else "") / ident
    out.mkdir(parents=True, exist_ok=True)
    sched_epochs = 2 if fixture else cfg["schedule_epochs"]
    stop_ep = cap or (2 if fixture else cfg["stop_epoch"])
    bs = batch or cfg["batch"]
    pat = 0 if fixture else cfg["patience"]
    num_workers = 0 if fixture else (workers if workers is not None else P2_TRAIN["workers"])

    overrides = dict(
        P2_TRAIN,
        batch=bs,
        epochs=sched_epochs,
        model=str(root / "configs/yolov8m-p2.yaml"),
        data=str(root / "configs/dataset_rgbt.yaml"),
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        device="0" if torch.cuda.device_count() > 1 else "",
        deterministic=False,
        patience=pat,
        imgsz=max(spec_for(ident)["hw"]),
        workers=num_workers,
    )

    with RunLock(out / "training.lock"):
        trainer = DeRoFTrainer(
            ident=ident,
            data_root=data_root,
            weights_path=weights,
            fixture=fixture,
            cap=stop_ep,
            p2_fusion_width=p2_fusion_width,
            strict_tail=strict_tail,
            overrides=overrides,
        )
        ckpt = out / "weights/full_last.pt"
        if ckpt.exists():
            prev = torch.load(ckpt, map_location="cpu", weights_only=False)
            done = (prev["epoch"] + 1) >= stop_ep or (out / "completion.json").exists()
            del prev
            if done and (out / "completion.json").exists():
                trainer.write_status("completed")
                return
            if done:
                trainer._setup_train(world_size=1)
                trainer.restore_state(trainer)
                trainer.epoch = trainer.start_epoch - 1
                trainer.final_eval()
                trainer.write_status("completed")
                return
        trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual-stream DeRoF-Det models.")
    parser.add_argument("ident", choices=sorted(EXPERIMENTS), help="Experiment configuration identifier.")
    parser.add_argument("--data-root", type=str, default=None, help="Directory holding dataset JSON manifests.")
    parser.add_argument("--weights", type=str, default=None, help="Path to ImageNet-1K yolov8m-cls.pt weights.")
    parser.add_argument("--output-dir", type=str, default="runs", help="Root directory for training outputs.")
    parser.add_argument("--cap", type=int, default=None, help="Override stop epoch.")
    parser.add_argument("--batch", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--p2-width", type=int, default=64, help="MEAF fusion width on P2 level.")
    parser.add_argument("--strict-tail", action="store_true", help="Disable AMP on partial tail batches.")
    parser.add_argument("--fixture", action="store_true", help="Run 2-epoch micro-subset smoke validation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        ident=args.ident,
        data_root=args.data_root,
        weights=args.weights,
        output_dir=args.output_dir,
        fixture=args.fixture,
        cap=args.cap,
        batch=args.batch,
        workers=args.workers,
        p2_fusion_width=args.p2_width,
        strict_tail=args.strict_tail,
    )
