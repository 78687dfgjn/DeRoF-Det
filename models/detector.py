"""DeRoF-Det: Decomposition, Routing and Frequency Refinement Detector for dual-stream RGB-T P2-P5 detection."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel

from models.cgsr import CGSR
from models.meaf import MEAF
from models.spmr import Residual, SPMR, isolated_seed

LEVELS = (2, 4, 6, 9)
CHANNELS = (96, 192, 384, 576)
NECK_TO_LEVEL = {18: 2, 21: 4, 24: 6, 27: 9}
TAIL_BATCH_THRESHOLD = 16
CONFIG_YAML = Path(__file__).resolve().parent.parent / "configs/yolov8m-p2.yaml"

VARIANT_SPECS = {
    # Slim-1 (~43.58M): pre-projected ASPP + DW experts across P2-P5; dense CGSR on P2, DW-sep on P3-P5.
    "slim1": dict(b_preproj_levels=(2, 4, 6, 9), b_dw_experts=True, c_dense_levels=(2,), c_levels=(2, 4, 6, 9)),
    # Slim-B (~46.97M): full-rank MEAF on P2-P4, pre-projected P5; dense CGSR on P2, DW-sep on P3-P5.
    "slimB": dict(b_preproj_levels=(9,), b_dw_experts=False, c_dense_levels=(2,), c_levels=(2, 4, 6, 9)),
}


def transfer_coco_p2(model, pretrained):
    """Transfer matching COCO YOLOv8m backbone, top-down FPN, P4/P5 PAN, and Detect regression weights."""
    model.load(pretrained)
    src = pretrained.state_dict()
    dst = model.state_dict()
    remap = {"model.16.": "model.22.", "model.18.": "model.24.", "model.19.": "model.25.", "model.21.": "model.27."}
    extra = {}
    for k, v in src.items():
        for old_p, new_p in remap.items():
            if k.startswith(old_p):
                nk = new_p + k[len(old_p) :]
                if nk in dst and dst[nk].shape == v.shape:
                    extra[nk] = v
        if k.startswith("model.22.cv2."):
            parts = k.split(".")
            idx = int(parts[3])
            nk = ".".join(["model", "28", "cv2", str(idx + 1), *parts[4:]])
            if nk in dst and dst[nk].shape == v.shape:
                extra[nk] = v
    if extra:
        model.load_state_dict(extra, strict=False)
    return len(extra)


def transfer_imagenet_backbone(model, cls_model):
    """Load matching ImageNet-1K backbone layers into both RGB and TIR streams."""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.load(cls_model)
    after = model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    model.ir_backbone = nn.ModuleList(deepcopy(list(model.model[:10])))
    return len(changed)


class NaiveConcatBottleneck(nn.Module):
    """Low-rank concat bottleneck fusion used in Stage-0 baseline when both SPMR and MEAF are absent."""

    def __init__(self, c, w0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2, w0, 1, bias=False),
            nn.BatchNorm2d(w0),
            nn.SiLU(),
            nn.Conv2d(w0, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

    def forward(self, r, t):
        return self.proj(self.pool(torch.cat((r, t), dim=1)))


class CalibratedMidRankFusion(nn.Module):
    """Mid-rank fusion bridge used when SPMR is enabled without MEAF."""

    def __init__(self, c, w1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2, w1, 1, bias=False),
            nn.BatchNorm2d(w1),
            nn.SiLU(),
            nn.Conv2d(w1, c, 3, padding=1, groups=w1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

    def forward(self, r, t):
        z = torch.cat((r, t), dim=1)
        z_mix = 0.5 * z + 0.5 * self.pool(z)
        return 0.45 * (r + t) + 0.55 * self.proj(z_mix)


class DeRoFDet(DetectionModel):
    """Decomposition, Routing and Frequency Refinement Detector (YOLOv8m-P2 + SPMR + MEAF + CGSR)."""

    def __init__(self, pretrained=None, nc=1, slim_variant="slim1", mask="ABC", p2_fusion_width=64):
        if slim_variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown slim_variant: {slim_variant}")
        mask = mask.upper()
        if mask not in ("BASE", "A", "B", "C", "AB", "AC", "BC", "ABC"):
            raise ValueError(f"Unknown mask: {mask}")
        super().__init__(str(CONFIG_YAML), ch=3, nc=nc, verbose=False)
        self.extra_transferred = transfer_coco_p2(self, pretrained) if pretrained is not None else 0
        self.ir_backbone = nn.ModuleList(deepcopy(list(self.model[:10])))
        self.slim_variant = slim_variant
        self.mask = mask
        self.variant = f"derof_det_p2_{slim_variant}_{mask.lower()}"
        self.strict_tail = False
        self.degrade = False

        is_base = mask == "BASE"
        self.has_a = (not is_base) and ("A" in mask)
        self.has_b = (not is_base) and ("B" in mask)
        self.has_c = (not is_base) and ("C" in mask)
        cfg = VARIANT_SPECS[slim_variant]
        self.c_levels = set(cfg["c_levels"])

        self.spmr = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        self.cgsr = nn.ModuleDict()

        for level, c in zip(LEVELS, CHANNELS):
            if self.has_a:
                with isolated_seed(910000 + level):
                    self.spmr[str(level)] = SPMR(c, channel_reduce=4, width=64)

            if self.has_b:
                with isolated_seed(820000 + level):
                    w = int(p2_fusion_width) if level == 2 else 96
                    op = MEAF(
                        c,
                        w=w,
                        preproj=(level in cfg["b_preproj_levels"]),
                        dw_experts=cfg["b_dw_experts"],
                    )
                    self.fusion[str(level)] = Residual(op)
            elif self.has_a:
                with isolated_seed(820000 + level):
                    self.fusion[str(level)] = CalibratedMidRankFusion(c, w1=max(32, c // 3))
            else:
                with isolated_seed(820000 + level):
                    self.fusion[str(level)] = NaiveConcatBottleneck(c, w0=max(16, c // 6))

            if self.has_c and level in self.c_levels:
                with isolated_seed(740000 + level):
                    dense = level in cfg["c_dense_levels"]
                    self.cgsr[str(level)] = Residual(CGSR(c, dense=dense))

        self.names = {i: n for i, n in enumerate(["person"] if nc == 1 else ["person", "rider", "crowd"])}
        self.nc = nc
        self.yaml.update(ch=6, derof_det=dict(variant=slim_variant, mask=mask, p2_fusion_width=p2_fusion_width))

    def _neck_no_c(self, x, level):
        """Uncompensated high-frequency damping when Module C (CGSR) is disabled."""
        x_low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.has_b:
            alpha_hi = 0.45 if level in (2, 4) else (0.65 if level == 6 else 1.0)
        elif self.has_a:
            alpha_hi = 0.40 if level in (2, 4, 6) else 0.60
        else:
            alpha_hi = 0.35 if level in (2, 4, 6) else 0.50
        if alpha_hi >= 1.0:
            return x
        return alpha_hi * x + (1.0 - alpha_hi) * x_low

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, "ir_backbone"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("Expected [B, 6, H, W] concatenated RGB+TIR input")
        rgb, ir = x[:, :3], x[:, 3:]
        saved = []
        conditions = {}
        features = {}
        for index, (a, b) in enumerate(zip(self.model[:10], self.ir_backbone)):
            rgb, ir = a(rgb), b(ir)
            if index in LEVELS:
                if self.has_a:
                    rgb, ir = self.spmr[str(index)](rgb, ir)
                features[index] = (rgb, ir)
            saved.append(None)

        for level in LEVELS:
            r, t = features[level]
            op = self.fusion[str(level)]
            if self.has_b:
                extra, cond = op.module(r, t)
                saved[level] = r + t + op.gamma * extra
                conditions[level] = cond
            else:
                saved[level] = op(r, t)
                conditions[level] = None

        x = saved[9]
        self.neck_features = []
        for layer in self.model[10:]:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            if layer.i in NECK_TO_LEVEL:
                level = NECK_TO_LEVEL[layer.i]
                if self.has_c and level in self.c_levels:
                    op = self.cgsr[str(level)]
                    x = x + op.gamma * op.module(x, conditions.get(level))
                elif not self.has_c:
                    x = self._neck_no_c(x, level)
                self.neck_features.append(x)
            saved.append(x if layer.i in self.save else None)
        return x

    def loss(self, batch, preds=None):
        tail_fp32 = (
            self.strict_tail
            and self.training
            and preds is None
            and isinstance(batch, dict)
            and "img" in batch
            and torch.is_tensor(batch["img"])
            and batch["img"].shape[0] < TAIL_BATCH_THRESHOLD
            and batch["img"].is_cuda
        )
        with torch.autocast("cuda", enabled=False) if tail_fp32 else nullcontext():
            loss, items = super().loss(batch, preds=preds)
        auxiliary = [getattr(m, "aux", None) for m in self.modules()]
        auxiliary = [v for v in auxiliary if v is not None]
        if auxiliary:
            extra = sum(auxiliary) * batch["img"].shape[0]
            self.last_aux_loss = float(extra.detach())
            loss = loss + extra / loss.numel()
        return loss, items
