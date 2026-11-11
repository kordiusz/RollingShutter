"""Shared network building blocks: U-Net, coordinate channels, backward warp, correlation layer.

The stage-1 models that used to live here (MotionNet: 9 camera-motion parameters + exact RS geometry,
RefinePipeline) were removed on 2026-11-11; description and results: docs/etap1_motionnet.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin, cout, ch=(32, 64, 128, 256)):
        super().__init__()
        self.inc = conv_block(cin, ch[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                          nn.LeakyReLU(0.1, inplace=True), conv_block(ch[i + 1], ch[i + 1]))
            for i in range(len(ch) - 1))
        self.ups = nn.ModuleList(
            conv_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1)))
        self.head = nn.Conv2d(ch[0], cout, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-3)  # ~identity at start (tiny flow / residual)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, s], 1))
        return self.head(x)


def coord_channels(n, h, w, device):
    """x, y in [-1, 1]: tells a CNN where in the image a pixel is (its readout time depends on it)."""
    ys = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(n, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(n, 1, h, w)
    return torch.cat([xs, ys], 1)


def warp(img, flow):
    """Backward warp: out[y,x] = img[y+dy, x+dx]."""
    n, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device, dtype=img.dtype),
                            torch.arange(w, device=img.device, dtype=img.dtype), indexing="ij")
    gx = 2 * (xs + flow[:, 0]) / (w - 1) - 1
    gy = 2 * (ys + flow[:, 1]) / (h - 1) - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear",
                         padding_mode="border", align_corners=True)


def correlation(fa, fb, r):
    """Cost volume: cosine similarity of A's feature with B's feature shifted by
    (dx, dy) in [-r, r]^2 -> (2r+1)^2 channels (as in FlowNet / PWC-Net)."""
    fa, fb = F.normalize(fa, dim=1), F.normalize(fb, dim=1)
    H, W = fa.shape[-2:]
    fb = F.pad(fb, (r, r, r, r))
    return torch.cat([(fa * fb[:, :, dy:dy + H, dx:dx + W]).sum(1, keepdim=True)
                      for dy in range(2 * r + 1) for dx in range(2 * r + 1)], 1)
