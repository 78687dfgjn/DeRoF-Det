"""Module A (SPMR): Shared-Private Multiscale Recalibration with auxiliary reconstruction loss."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def isolated_seed(seed):
    """Run deterministic module initialization without perturbing the caller RNG state."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        yield


def conv_bn_silu(cin, cout, k=1, groups=1, dilation=1):
    """Standard Conv-BatchNorm-SiLU building block."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=dilation * (k // 2), dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(),
    )


def spatial_norm(x):
    """Per-sample, per-channel spatial Z-score normalization in FP32."""
    x = x.float()
    return (x - x.mean((-2, -1), keepdim=True)) / x.var((-2, -1), unbiased=False, keepdim=True).add(1e-6).sqrt()


def modal_descriptors(r, t):
    """Compute a 5-channel normalized summary (mean_r, mean_t, |r-t|, std_r, std_t) for routing."""
    rn, tn = spatial_norm(r), spatial_norm(t)
    return torch.cat(
        [
            rn.mean(1, keepdim=True),
            tn.mean(1, keepdim=True),
            (rn - tn).abs().mean(1, keepdim=True),
            rn.var(1, unbiased=False, keepdim=True).add(1e-6).sqrt(),
            tn.var(1, unbiased=False, keepdim=True).add(1e-6).sqrt(),
        ],
        dim=1,
    )


class Residual(nn.Module):
    """Zero-initialized scalar residual wrapper."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.zeros(()))


class ChannelRecalibration(nn.Module):
    """Channel attention aggregating global average- and max-pooled statistics."""

    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def _pool_gate(self, pooled):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(pooled), inplace=True)))

    def forward(self, inputs):
        x1 = self._pool_gate(F.adaptive_avg_pool2d(inputs, output_size=(1, 1)))
        x2 = self._pool_gate(F.adaptive_max_pool2d(inputs, output_size=(1, 1)))
        attn = (x1 + x2).view(-1, self.input_channels, 1, 1)
        return inputs * attn


class MultiscaleStripAttention(nn.Module):
    """Channel prior recalibration followed by multi-scale orthogonal depthwise strip attention."""

    def __init__(self, channels, channel_reduce=4):
        super().__init__()
        self.ca = ChannelRecalibration(channels, channels // channel_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.ca(self.act(self.conv(inputs)))
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv7_1(self.dconv1_7(x_init))
        x_2 = self.dconv11_1(self.dconv1_11(x_init))
        x_3 = self.dconv21_1(self.dconv1_21(x_init))
        spatial_att = self.conv(x_1 + x_2 + x_3 + x_init)
        return self.conv(spatial_att * inputs)


class SPMR(nn.Module):
    """Shared-Private Multiscale Recalibration (Module A) for dual-stream RGB-T calibration."""

    def __init__(self, channels, channel_reduce=4, width=64, gamma_init=0.0):
        super().__init__()
        self.shared = conv_bn_silu(channels * 2, width)
        self.to_r = nn.Conv2d(width, channels, 1)
        self.to_t = nn.Conv2d(width, channels, 1)
        self.r_private = conv_bn_silu(channels, width)
        self.t_private = conv_bn_silu(channels, width)
        self.r_decode = nn.Conv2d(width, channels, 1)
        self.t_decode = nn.Conv2d(width, channels, 1)
        self.gates = nn.Conv2d(width, 3, 1)
        self.rgb_attn = Residual(MultiscaleStripAttention(channels, channel_reduce=channel_reduce))
        self.ir_attn = Residual(MultiscaleStripAttention(channels, channel_reduce=channel_reduce))
        if gamma_init:
            with torch.no_grad():
                self.rgb_attn.gamma.fill_(float(gamma_init))
                self.ir_attn.gamma.fill_(float(gamma_init))
        self.aux = None

    def forward(self, r, t):
        shared = self.shared(torch.cat((r, t), dim=1))
        sr, st = self.to_r(shared), self.to_t(shared)
        pr = self.r_decode(self.r_private(r - sr))
        pt = self.t_decode(self.t_private(t - st))
        w = self.gates(shared).sigmoid()
        dr = self.rgb_attn.module(sr * w[:, 0:1] + pr * w[:, 1:2])
        dt = self.ir_attn.module(st * w[:, 0:1] + pt * w[:, 2:3])
        self.aux = (
            F.mse_loss(spatial_norm(sr + pr), spatial_norm(r))
            + F.mse_loss(spatial_norm(st + pt), spatial_norm(t))
        ) * 0.005
        return r + self.rgb_attn.gamma * dr, t + self.ir_attn.gamma * dt
