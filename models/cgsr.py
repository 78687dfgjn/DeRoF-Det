"""Module C (CGSR): Condition-Guided Spectral Refinement with 2D-DCT frequency decomposition."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class _DCTSpectralBranch(nn.Module):
    """FP32 2D-DCT orthogonal high-pass decomposition and spatial-channel frequency modulation."""

    def __init__(self, channels, dense=False, ratio=(0.25, 0.25), patch=(8, 8)):
        super().__init__()
        if channels % 32 != 0:
            raise ValueError(f"Expected channels divisible by 32, got {channels}")
        self.ratio = ratio
        self.patch = patch
        self.channel1 = nn.Conv2d(channels, channels, 1, groups=32, bias=False)
        self.channel2 = nn.Conv2d(channels, channels, 1, groups=32, bias=False)
        if dense:
            self.out = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=True),
                nn.GroupNorm(32, channels),
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=True),
                nn.GroupNorm(32, channels),
            )
        self.register_buffer("_basis_h", torch.empty(0), persistent=False)
        self.register_buffer("_basis_w", torch.empty(0), persistent=False)

    @staticmethod
    def dct_basis(size, device):
        """Construct an orthonormal 1D DCT-II transform matrix."""
        k = torch.arange(size, device=device, dtype=torch.float32)[:, None]
        n = torch.arange(size, device=device, dtype=torch.float32)[None, :]
        matrix = torch.cos(math.pi / size * (n + 0.5) * k) * math.sqrt(2.0 / size)
        matrix[0] *= 1.0 / math.sqrt(2.0)
        return matrix

    def high_pass(self, x):
        """Suppress the low-frequency upper-left DCT quadrant and invert back to spatial coordinates."""
        h, w = x.shape[-2:]
        if self._basis_h.shape != (h, h) or self._basis_h.dtype != torch.float32 or self._basis_h.device != x.device:
            self._basis_h = self.dct_basis(h, x.device)
        if self._basis_w.shape != (w, w) or self._basis_w.dtype != torch.float32 or self._basis_w.device != x.device:
            self._basis_w = self.dct_basis(w, x.device)
        a, b = self._basis_h, self._basis_w
        coeff = torch.matmul(torch.matmul(a, x), b.T)
        mask = torch.ones((h, w), device=x.device, dtype=torch.float32)
        mask[: int(h * self.ratio[0]), : int(w * self.ratio[1])] = 0.0
        return torch.matmul(torch.matmul(a.T, coeff * mask), b)

    def refine(self, z, high):
        """Re-weight features using high-pass response and pooled frequency channel attention."""
        maximum = F.adaptive_max_pool2d(high, self.patch).relu().sum((2, 3), keepdim=True)
        average = F.adaptive_avg_pool2d(high, self.patch).relu().sum((2, 3), keepdim=True)
        attention = torch.sigmoid(self.channel2(self.channel1(maximum) + self.channel1(average)))
        return self.out(z * high + z * attention)


class CGSR(nn.Module):
    """Condition-Guided Spectral Refinement (Module C) conditioned on MEAF uncertainty maps."""

    def __init__(self, channels, dense=False, ratio=(0.25, 0.25), patch=(8, 8)):
        super().__init__()
        self.freq_branch = _DCTSpectralBranch(channels, dense=dense, ratio=ratio, patch=patch)
        self.gate = nn.Sequential(nn.Conv2d(8, 16, 1), nn.SiLU(), nn.Conv2d(16, 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.diagnostics = {}

    def forward(self, x, condition=None):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.float()
            high = self.freq_branch.high_pass(z)
            low = z - high
            base_refined = self.freq_branch.refine(z, high)
            content = torch.cat(
                [
                    z.mean(1, keepdim=True),
                    z.var(1, unbiased=False, keepdim=True).add(1e-6).sqrt(),
                    high.abs().mean(1, keepdim=True),
                    low.abs().mean(1, keepdim=True),
                ],
                dim=1,
            )
            if condition is None:
                condition = z.new_zeros(z.shape[0], 4, *z.shape[-2:])
            elif condition.shape[-2:] != z.shape[-2:]:
                condition = F.interpolate(condition.float(), size=z.shape[-2:], mode="bilinear", align_corners=False)
            alpha, beta = (0.5 * self.gate(torch.cat((content, condition.float()), dim=1)).tanh()).chunk(2, dim=1)
            out = base_refined + alpha * high + beta * low
        self.diagnostics = {
            "reconstruction_error": float((z - high - low).abs().max().detach()),
            "high_energy": float(high.square().mean().detach()),
            "low_energy": float(low.square().mean().detach()),
            "alpha": float(alpha.mean().detach()),
            "beta": float(beta.mean().detach()),
        }
        return out.to(dtype)
