"""Dataset protocols, training schedules, early-stopping criteria, and experiment registry."""

CLS_SHA256 = "864010e833b46be542ed849b2eb0fdacf51e4679dc677c18dcc4505244d08ec1"

DATA = {
    "R": dict(
        hw=[512, 640],
        nc=3,
        train_count=4900,
        val_count=1225,
        cap=60,
        patience=0,
        conf=0.001,
        iou=0.70,
        max_det=100,
        evaluator="tiny",
        selection_metric="mAP50_all",
    ),
    "U": dict(
        hw=[384, 640],
        nc=1,
        train_count=11208,
        val_count=5378,
        cap=250,
        patience=50,
        conf=0.05,
        iou=0.30,
        max_det=100,
        evaluator="coco",
        selection_metric="mAP50_all",
    ),
    "N": dict(
        hw=[384, 640],
        nc=1,
        train_count=4980,
        val_count=900,
        cap=60,
        patience=30,
        conf=0.05,
        iou=0.30,
        max_det=100,
        evaluator="coco",
        prefix="N_640x360",
        proto="N_protocol.json",
        selection_metric="mAP50_all",
    ),
}

P2_TRAIN = dict(
    task="detect",
    optimizer="SGD",
    epochs=60,
    batch=16,
    nbs=32,
    lr0=0.007,
    lrf=0.01,
    cos_lr=True,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.05,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    seed=42,
    amp=True,
    workers=6,
    cache=False,
    multi_scale=False,
    fraction=1.0,
    freeze=None,
    mosaic=1.0,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mixup=0.0,
    copy_paste=0.0,
    val=True,
    save=True,
    save_period=-1,
    plots=False,
    rect=False,
)

EXPERIMENTS = {
    # Multi-dataset primary comparisons (DeRoF-Det <50M variants vs P2 baselines)
    "R-ABC-slim1-in1k-fast": dict(dataset="R", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", schedule_epochs=70, stop_epoch=45, patience=0, batch=8, val_batch=16, degrade=True),
    "U-ABC-slim1-in1k-fast": dict(dataset="U", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", schedule_epochs=60, stop_epoch=40, patience=0, batch=16, val_batch=16, degrade=False),
    "R-ABC-slimB-in1k-fast": dict(dataset="R", arch="derof", slim_variant="slimB", mask="ABC", label="DeRoF-Det-slimB", schedule_epochs=70, stop_epoch=45, patience=0, batch=8, val_batch=16, degrade=True),
    "U-ABC-slimB-in1k-fast": dict(dataset="U", arch="derof", slim_variant="slimB", mask="ABC", label="DeRoF-Det-slimB", schedule_epochs=60, stop_epoch=40, patience=0, batch=16, val_batch=16, degrade=False),
    "N-P2-Add-in1k-fast": dict(dataset="N", arch="add", slim_variant=None, mask="ADD", label="BASE-add", schedule_epochs=60, stop_epoch=60, patience=30, batch=16, val_batch=16, degrade=False),
    "N-P2-ConcatC2f-in1k-fast": dict(dataset="N", arch="concat_c2f", slim_variant=None, mask="CONCAT_C2F", label="BASE-concat-c2f", schedule_epochs=60, stop_epoch=60, patience=30, batch=16, val_batch=16, degrade=False),
    "N-ABC-slim1-in1k-fast": dict(dataset="N", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", schedule_epochs=60, stop_epoch=60, patience=30, batch=16, val_batch=16, degrade=False),
    "N-ABC-slimB-in1k-fast": dict(dataset="N", arch="derof", slim_variant="slimB", mask="ABC", label="DeRoF-Det-slimB", schedule_epochs=60, stop_epoch=60, patience=30, batch=16, val_batch=16, degrade=False),
    # NII-CU high-resolution & 4ch comparison tiers (evaluated by COCO AP50:95)
    "N4ch-P2-Base-v2-in1k-fast": dict(dataset="N", arch="base_v2", slim_variant=None, mask="BASE", label="P2-Base-v2", prefix="N4ch_640x468", proto="N4ch_protocol.json", hw=[480, 640], train_count=2653, val_count=485, batch=16, val_batch=16, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
    "N4ch-ABC-slim1-in1k-fast": dict(dataset="N", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", prefix="N4ch_640x468", proto="N4ch_protocol.json", hw=[480, 640], train_count=2653, val_count=485, batch=16, val_batch=16, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
    "N960-P2-Base-v2-in1k-fast": dict(dataset="N", arch="base_v2", slim_variant=None, mask="BASE", label="P2-Base-v2", prefix="N_960x540", proto="N960_protocol.json", hw=[544, 960], train_count=4980, val_count=900, batch=8, val_batch=8, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
    "N960-ABC-slim1-in1k-fast": dict(dataset="N", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", prefix="N_960x540", proto="N960_protocol.json", hw=[544, 960], train_count=4980, val_count=900, batch=8, val_batch=8, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
    "N1280-P2-Base-v2-in1k-fast": dict(dataset="N", arch="base_v2", slim_variant=None, mask="BASE", label="P2-Base-v2", prefix="N_1280x720", proto="N1280_protocol.json", hw=[736, 1280], train_count=4980, val_count=900, batch=4, val_batch=8, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
    "N1280-ABC-slim1-in1k-fast": dict(dataset="N", arch="derof", slim_variant="slim1", mask="ABC", label="DeRoF-Det-slim1", prefix="N_1280x720", proto="N1280_protocol.json", hw=[736, 1280], train_count=4980, val_count=900, batch=4, val_batch=8, schedule_epochs=60, stop_epoch=60, patience=30, conf=0.001, iou=0.65, max_det=300, selection_metric="AP50_95", degrade=False),
}

for _sv in ("slim1", "slimB"):
    for _mask in ("A", "B", "C", "AB", "AC", "BC"):
        EXPERIMENTS[f"R-{_mask}-{_sv}-in1k-fast"] = dict(
            dataset="R",
            arch="derof",
            slim_variant=_sv,
            mask=_mask,
            label=f"DeRoF-Det-{_sv}-{_mask}",
            schedule_epochs=70,
            stop_epoch=45,
            patience=0,
            batch=8,
            val_batch=16,
            degrade=True,
        )


def spec_for(ident):
    """Resolve full evaluation & dataset specification dictionary for an experiment ID."""
    cfg = EXPERIMENTS[ident]
    base = dict(DATA[cfg["dataset"]])
    base.update(id=ident, mask=cfg["mask"], cap=cfg["stop_epoch"], patience=cfg["patience"], strict=False)
    for k in ("prefix", "proto", "hw", "train_count", "val_count", "conf", "iou", "max_det", "selection_metric", "val_batch"):
        if k in cfg:
            base[k] = list(cfg[k]) if k == "hw" else cfg[k]
    return base


class APStopper:
    """Strict validation-fitness early stopping tracker."""

    def __init__(self, patience):
        self.patience = patience or float("inf")
        self.best_fitness = -float("inf")
        self.best_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness is None:
            return False
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_epoch = epoch
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= self.patience - 1
        return delta >= self.patience
