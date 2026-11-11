"""Shared training helpers: masked metrics and run bookkeeping (used by train_raft.py and train_unet.py)."""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


def masked_l1(a, b, m):
    return (m * (a - b).abs()).sum() / (m.sum() * a.shape[1] + 1e-8)


def psnr(a, b, m):
    """Per-image masked PSNR, returns tensor (N,)."""
    mse = (m * (a - b) ** 2).flatten(1).sum(1) / (m.flatten(1).sum(1) * a.shape[1] + 1e-8)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gauss(ws=11, sigma=1.5, device="cpu"):
    g = torch.exp(-(torch.arange(ws, device=device) - ws // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, m):
    """Per-image masked SSIM (standard 11x11 Gaussian window), returns (N,)."""
    c = a.shape[1]
    w = _gauss(device=a.device).expand(c, 1, 11, 11)
    f = lambda x: F.conv2d(x, w, padding=5, groups=c)
    mu_a, mu_b = f(a), f(b)
    s_aa, s_bb, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    smap = smap.mean(1, keepdim=True)
    return (smap * m).flatten(1).sum(1) / (m.flatten(1).sum(1) + 1e-8)


def run_context(dev):
    """Where/what this run is: git commit of the code, GPU, versions."""
    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], capture_output=True, text=True,
                                  cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def dataset_info(root):
    root = Path(root)
    info = {"path": str(root)}
    cfg = root / "config.json"
    if cfg.exists():
        info["generator_config"] = json.loads(cfg.read_text())
    return info
