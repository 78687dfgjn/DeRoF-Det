"""Standard P2-P5 dual-stream baselines (Add, Concat-C2f, Late Fusion, and Base-v2)."""
from copy import deepcopy
import torch
from torch import nn
from ultralytics.nn.modules import C2f, Conv
from ultralytics.nn.tasks import DetectionModel

from models.detector import CHANNELS, CONFIG_YAML, DeRoFDet, LEVELS, transfer_coco_p2


class _DualBackboneP2Base(DetectionModel):
    def __init__(self, pretrained=None, nc=1):
        super().__init__(str(CONFIG_YAML), ch=3, nc=nc, verbose=False)
        self.extra_transferred = transfer_coco_p2(self, pretrained) if pretrained is not None else 0
        self.ir_backbone = nn.ModuleList(deepcopy(list(self.model[:10])))
        self.strict_tail = False
        self.degrade = False
        self.names = {i: n for i, n in enumerate(["person"] if nc == 1 else ["person", "rider", "crowd"])}
        self.nc = nc

    def _run_head(self, saved):
        x = saved[9]
        for layer in self.model[10:]:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return x


class P2AddModel(_DualBackboneP2Base):
    """Element-wise P2-P5 dual-stream addition baseline."""

    def __init__(self, pretrained=None, nc=1):
        super().__init__(pretrained, nc=nc)
        self.variant = "v8m_p2_add"
        self.yaml.update(ch=6, baseline="P2-Add")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, "ir_backbone"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("Expected [B, 6, H, W] RGB+TIR")
        rgb, ir = x[:, :3], x[:, 3:]
        saved = []
        for index, (a, b) in enumerate(zip(self.model[:10], self.ir_backbone)):
            rgb, ir = a(rgb), b(ir)
            saved.append((rgb + ir) if index in LEVELS else None)
        return self._run_head(saved)


class P2ConcatC2fModel(_DualBackboneP2Base):
    """Concat(2C) + 1x1 Conv + C2f(n=3) fusion baseline across P2-P5."""

    def __init__(self, pretrained=None, nc=1, c2f_n=3):
        super().__init__(pretrained, nc=nc)
        self.fuse_c2f = nn.ModuleDict(
            {str(lvl): nn.Sequential(Conv(c * 2, c, 1), C2f(c, c, n=c2f_n, shortcut=True)) for lvl, c in zip(LEVELS, CHANNELS)}
        )
        self.variant = f"v8m_p2_concat_c2f_n{c2f_n}"
        self.yaml.update(ch=6, baseline=f"P2-Concat-C2f(n={c2f_n})")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, "ir_backbone"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("Expected [B, 6, H, W] RGB+TIR")
        rgb, ir = x[:, :3], x[:, 3:]
        saved = []
        for index, (a, b) in enumerate(zip(self.model[:10], self.ir_backbone)):
            rgb, ir = a(rgb), b(ir)
            saved.append(self.fuse_c2f[str(index)](torch.cat([rgb, ir], dim=1)) if index in LEVELS else None)
        return self._run_head(saved)


class P2LateModel(DetectionModel):
    """Dual P2-P5 backbones and PAN-FPN necks with late addition before Detect."""

    def __init__(self, pretrained=None, nc=1):
        super().__init__(str(CONFIG_YAML), ch=3, nc=nc, verbose=False)
        self.extra_transferred = transfer_coco_p2(self, pretrained) if pretrained is not None else 0
        self.ir_stream = nn.ModuleList(deepcopy(list(self.model[:-1])))
        self.strict_tail = False
        self.degrade = False
        self.names = {i: n for i, n in enumerate(["person"] if nc == 1 else ["person", "rider", "crowd"])}
        self.nc = nc
        self.variant = "v8m_p2_late"
        self.yaml.update(ch=6, baseline="P2-LateFusion")

    @property
    def ir_backbone(self):
        return self.ir_stream[:10]

    def _forward_stream(self, stream, x):
        saved = []
        for layer in stream:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [x if j == -1 else saved[j] for j in layer.f]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return saved

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if not hasattr(self, "ir_stream"):
            return super()._predict_once(x, profile, visualize, embed)
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError("Expected [B, 6, H, W] RGB+TIR")
        rgb, ir = x[:, :3], x[:, 3:]
        sr = self._forward_stream(self.model[:-1], rgb)
        st = self._forward_stream(self.ir_stream, ir)
        detect = self.model[-1]
        return detect([sr[j] + st[j] for j in detect.f])


class P2BaseV2Model(DeRoFDet):
    """Stage-0 dual-stream baseline (low-rank Concat-Bottleneck without SPMR/MEAF/CGSR)."""

    def __init__(self, pretrained=None, nc=1, p2_fusion_width=64):
        super().__init__(pretrained=pretrained, nc=nc, slim_variant="slim1", mask="BASE", p2_fusion_width=p2_fusion_width)
        self.variant = "v8m_p2_base_v2"
        self.yaml.update(ch=6, baseline="P2-Base-v2")
