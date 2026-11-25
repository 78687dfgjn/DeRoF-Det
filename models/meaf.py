"""Module B (MEAF): Modality-Expert Atrous Fusion with foreground/background separation."""
import torch
from torch import nn
from torch.nn import functional as F

from models.spmr import conv_bn_silu, modal_descriptors, spatial_norm


def _conv3x3(in_c, out_c, padding=1, dilation=1, bias=True, sep=False):
    if sep:
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=padding, dilation=dilation, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, 1, bias=bias),
        )
    return nn.Conv2d(in_c, out_c, 3, stride=1, padding=padding, dilation=dilation, bias=bias)


class PriorHead(nn.Module):
    """Foreground saliency predictor with optional 1x1 bottleneck projection."""

    def __init__(self, in_c, width, sep=False, bneck=False):
        super().__init__()
        if bneck:
            self.enlayer = nn.Sequential(
                nn.Conv2d(in_c, width, 1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                _conv3x3(width, width, padding=1, dilation=1, bias=True, sep=sep),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        else:
            self.enlayer = nn.Sequential(
                _conv3x3(in_c, width, padding=1, dilation=1, bias=True, sep=sep),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        self.outlayer = nn.Sequential(nn.Conv2d(width, 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        feat = self.enlayer(x)
        return feat, self.outlayer(feat)


class _BottleneckPrior(nn.Module):
    """Depthwise bottleneck spatial prior (1x1 c->w + DW 3x3 + 1x1 w->1)."""

    def __init__(self, c, w):
        super().__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(c, w, 1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, groups=w, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        feat = self.enlayer(x)
        return feat, self.outlayer(feat)


class ASPPBlock(nn.Module):
    """Four-branch multi-dilation atrous spatial pyramid pooling block (d = 1, 3, 5, 7)."""

    def __init__(self, in_c, width, sep=False):
        super().__init__()
        self.aspp1 = nn.Sequential(nn.Conv2d(in_c, width, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.aspp2 = self._dilated_branch(in_c, width, dilation=3, sep=sep)
        self.aspp3 = self._dilated_branch(in_c, width, dilation=5, sep=sep)
        self.aspp4 = self._dilated_branch(in_c, width, dilation=7, sep=sep)

    @staticmethod
    def _dilated_branch(in_c, width, dilation, sep):
        return nn.Sequential(
            _conv3x3(in_c, width, padding=dilation, dilation=dilation, bias=True, sep=sep),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)), dim=1)


class _MaskGuidedASPP(nn.Module):
    """Foreground/background dual-branch ASPP refinement with optional c->w pre-projection."""

    def __init__(self, c, w, preproj=True):
        super().__init__()
        if preproj:
            self.ff_pre = nn.Sequential(nn.Conv2d(c, w, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
            self.bf_pre = nn.Sequential(nn.Conv2d(c, w, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
            aspp_in = w
        else:
            self.ff_pre = nn.Identity()
            self.bf_pre = nn.Identity()
            aspp_in = c
        self.ff_conv = ASPPBlock(aspp_in, w, sep=False)
        self.bf_conv = ASPPBlock(aspp_in, w, sep=False)
        self.fg_bg_head = PriorHead(w * 8, w, sep=False, bneck=True)

    def forward(self, feat, pred):
        _, _, h, w = feat.size()
        prob = torch.sigmoid(F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True))
        ff_feat = self.ff_conv(self.ff_pre(feat * prob))
        bf_feat = self.bf_conv(self.bf_pre(feat * (1 - prob)))
        enhanced_feat, new_pred = self.fg_bg_head(torch.cat((ff_feat, bf_feat), dim=1))
        return enhanced_feat, prob, new_pred


class MEAF(nn.Module):
    """Modality-Expert Atrous Fusion (Module B): 3-expert routing + dual-branch dilated ASPP."""

    def __init__(self, c, w=96, preproj=True, dw_experts=True):
        super().__init__()
        self.width = int(w)
        self.preproj = bool(preproj)
        self.prior = _BottleneckPrior(c, w) if preproj else PriorHead(c, w, sep=False, bneck=False)
        self.atrous_core = _MaskGuidedASPP(c, w, preproj=preproj)
        self.project = nn.Conv2d(w, c, 1, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    conv_bn_silu(c, w),
                    conv_bn_silu(w, w, 3, groups=w if dw_experts else 1),
                    nn.Conv2d(w, c, 1),
                )
                for _ in range(3)
            ]
        )
        self.expert_route = nn.Conv2d(5, 3, 1)

    def forward(self, r, t):
        weights = self.expert_route(modal_descriptors(r, t)).softmax(dim=1)
        joint = sum(op(val) * weights[:, i : i + 1] for i, (op, val) in enumerate(zip(self.experts, (r, t, r + t))))
        _, p1 = self.prior(joint)
        enhanced, p1_prob, p2 = self.atrous_core(joint, p1)
        h, w = joint.shape[-2:]
        p2_prob = torch.sigmoid(F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=True))
        with torch.autocast(device_type=joint.device.type, enabled=False):
            diff = (spatial_norm(r) - spatial_norm(t)).abs().mean(1, keepdim=True)
            cond = torch.cat(
                [p1_prob.float(), p2_prob.float(), (p2_prob - p1_prob).abs().float(), diff],
                dim=1,
            )
        return self.project(enhanced), cond
