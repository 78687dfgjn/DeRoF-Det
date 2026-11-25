"""Atomic checkpoint capture/restoration, RNG snapshots, file locking, and JSON IO."""
import datetime
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

COUNTERS = (
    "last_opt_step",
    "accumulate",
    "samples_since_update",
    "last_update_samples",
    "successful_updates",
    "optimizer_attempts",
    "amp_skips",
)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


class RunLock:
    """Best-effort cross-process lock guarding a run directory."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            import fcntl

            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()


def capture(trainer):
    """Snapshot full optimizer, scheduler, EMA, AMP scaler, pending gradients, and RNG state."""
    return dict(
        format="rgbt-full-v1",
        epoch=trainer.epoch,
        model=_cpu_tree(trainer.model.state_dict()),
        ema=_cpu_tree(trainer.ema.ema.state_dict()),
        updates=trainer.ema.updates,
        optimizer=_cpu_tree(trainer.optimizer.state_dict()),
        scheduler=trainer.scheduler.state_dict(),
        scaler=trainer.scaler.state_dict(),
        gradients={k: _cpu_tree(p.grad) for k, p in trainer.model.named_parameters() if p.grad is not None},
        rng=rng_state(),
        counters={k: getattr(trainer, k, 0) for k in COUNTERS},
        best_fitness=trainer.best_fitness,
        fitness=trainer.fitness,
        candidate=trainer.candidate,
        phase=trainer.phase,
        train_args=vars(trainer.args),
        source_sha=getattr(trainer, "source_sha", None),
    )


def save_atomic(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore_state_dict(trainer, state):
    """Restore full training continuation state onto `trainer`."""
    assert state["candidate"] == trainer.candidate and state["phase"] == trainer.phase
    assert state["train_args"]["nbs"] == trainer.args.nbs and state["train_args"]["epochs"] == trainer.args.epochs
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.ema.ema.load_state_dict(state["ema"], strict=True)
    trainer.ema.updates = state["updates"]
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.scaler.load_state_dict(state["scaler"])
    for name, param in trainer.model.named_parameters():
        grad = state["gradients"].get(name)
        param.grad = None if grad is None else grad.to(param.device, param.dtype).clone()
    for key, val in state["counters"].items():
        setattr(trainer, key, val)
    trainer.start_epoch = state["epoch"] + 1
    trainer.best_fitness = state["best_fitness"]
    trainer.fitness = state["fitness"]
    trainer.full_state_restored = True
    restore_rng(state["rng"])
