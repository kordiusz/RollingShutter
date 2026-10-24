"""RAFT + rolling-shutter geometry: correction of camera AND object motion.

1) RAFT finds, for every pixel of RS image A, where the same scene point is in
   RS image B (rotated back to A's orientation):  d = p_B - p_A   (N,2,S,S).
2) Geometry (no learned weights) turns that into a per-pixel velocity:
       camera A sees the point at time t_A(row of p_A):    p_A = q + u * t_A
       camera B sees it at time t_B(column of p_B):         p_B = q - (b, 0) + u * t_B
   =>  d + (b, 0) = u * (t_B - t_A)   =>   u = (d + (b, 0)) / dt          [px/ms]
   q is where the point was at the target moment t = 0 (middle of A's readout),
   b is the stereo baseline in px (B sees the scene shifted left by b).
   Where dt ~ 0 (the anti-diagonal: both cameras read the point at the same
   moment) u cannot be measured; there it is filled in from the neighbourhood.
3) Output image: for every output pixel q find the source p in A with
   p = q + u(p) * t_A(p)   (fixed-point iteration), then warp A.
Everything is differentiable, so RAFT can be fine-tuned with an image loss.

Units: times in ms, displacements in px of the S x S image, velocity in px/ms.
Time convention = synth_rs.readout_times / synth_video.py (t = 0 at the middle
of A's readout; B's readout centre is shifted by `sync`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import warp


# ------------------------------------------------------------------ geometry
def base_grid(n, h, w, device, dtype=torch.float32):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return xs.expand(n, h, w), ys.expand(n, h, w)


def readout_time(px, py, S, T, sync, rot_k=1, cam="a"):
    """Capture time [ms] of a (sub)pixel given in A's image frame.
    px, py: (N,H,W); T, sync: (N,) in ms. Camera A reads rows; camera B reads
    its own sensor rows, which in A's frame are columns (rot_k = 1 or -1)."""
    T = T.view(-1, 1, 1)
    if cam == "a":
        r = py
        off = 0.0
    else:
        k = rot_k % 4
        r = {0: py, 1: S - 1 - px, 2: S - 1 - py, 3: px}[k]
        off = sync.view(-1, 1, 1)
    return off + (r / (S - 1) - 0.5) * T


def sample_field(f, px, py):
    """Bilinear sample of field f (N,C,H,W) at pixel coords (px, py) (N,H,W)."""
    h, w = f.shape[-2:]
    g = torch.stack([2 * px / (w - 1) - 1, 2 * py / (h - 1) - 1], -1)
    return F.grid_sample(f, g, mode="bilinear", padding_mode="border", align_corners=True)


def _base_vec(base, n):
    """Stereo offset as (N,2,1,1): a scalar per sample means (b, 0) (B sits to the
    right); a (N,2) tensor gives the full residual offset (bx, by) between the views."""
    base = torch.as_tensor(base, dtype=torch.float32)
    if base.dim() == 2:
        return base.view(n, 2, 1, 1)
    return torch.stack([base, torch.zeros_like(base)], 1).view(n, 2, 1, 1)


def velocity_from_flow(d, T, sync, base, rot_k=1, eps_frac=0.05, fill=61):
    """d: (N,2,S,S) flow A->B. Returns velocity u (N,2,S,S) [px/ms] and
    confidence c (N,1,S,S) in [0,1] (~0 where both cameras read at the same time)."""
    n, _, S, _ = d.shape
    xs, ys = base_grid(n, S, S, d.device, d.dtype)
    tA = readout_time(xs, ys, S, T, sync, rot_k, "a")
    tB = readout_time(xs + d[:, 0], ys + d[:, 1], S, T, sync, rot_k, "b")
    dt = (tB - tA)[:, None]                                  # (N,1,S,S)
    num = d + _base_vec(base, n)
    eps = (eps_frac * T).view(n, 1, 1, 1)
    # direct estimate: exact where |dt| >= eps; |dt| is clamped so it never divides by ~0
    dt_safe = torch.where(dt >= 0, 1.0, -1.0) * dt.abs().clamp_min(eps)
    u_dir = num / dt_safe
    c = (dt ** 2 / (dt ** 2 + eps ** 2)) ** 2               # confidence: ~1 far from the diagonal, ~0 on it
    # the unreliable band (~25 px wide) is filled from RELIABLE neighbours only
    num_b = F.avg_pool2d(u_dir * c, fill, 1, fill // 2, count_include_pad=False)
    den_b = F.avg_pool2d(c, fill, 1, fill // 2, count_include_pad=False).clamp_min(1e-4)
    u = c * u_dir + (1 - c) * num_b / den_b
    return u, c


def backward_flow(u, T, sync, rot_k=1, iters=6, cam="a", base=None):
    """For every output pixel q: where to take the colour in RS image A
    (cam="a": p = q + u(p) t_A(p)) or in RS image B (cam="b":
    p = q - (b,0) + u(p) t_B(p)). Returns flow p - q (N,2,S,S) and validity."""
    n, _, S, _ = u.shape
    qx, qy = base_grid(n, S, S, u.device, u.dtype)
    if cam == "b":
        bv = _base_vec(base, n)
        qx, qy = qx - bv[:, 0], qy - bv[:, 1]
    px, py = qx, qy
    for _ in range(iters):
        up = sample_field(u, px, py)
        t = readout_time(px, py, S, T, sync, rot_k, cam)
        px, py = qx + up[:, 0] * t, qy + up[:, 1] * t
    gx, gy = base_grid(n, S, S, u.device, u.dtype)
    valid = ((px >= 0) & (px <= S - 1) & (py >= 0) & (py <= S - 1)).to(u.dtype)[:, None]
    return torch.stack([px - gx, py - gy], 1), valid


def correct_from_flow(rs_a, d, T, sync, base, rot_k=1):
    """Pure geometry: RS image A + flow A->B -> corrected (GS) image."""
    u, w = velocity_from_flow(d, T, sync, base, rot_k)
    fl, valid = backward_flow(u, T, sync, rot_k)
    return warp(rs_a, fl), {"velocity": u, "conf": w, "flow": fl, "valid": valid}


# ------------------------------------------------------------------ model
class RaftGeom(nn.Module):
    """RAFT (pretrained, optionally fine-tuned) + fixed RS geometry."""

    def __init__(self, size="large", pretrained=True, iters=12, rot_k=1):
        super().__init__()
        from torchvision.models.optical_flow import (raft_large, raft_small,
                                                     Raft_Large_Weights, Raft_Small_Weights)
        if size == "large":
            self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT if pretrained else None)
        else:
            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT if pretrained else None)
        self.iters, self.rot_k = iters, rot_k

    def flow_ab(self, rs_a, rs_b):
        """RAFT expects images in [-1, 1] and sides divisible by 8."""
        return self.raft(rs_a * 2 - 1, rs_b * 2 - 1, num_flow_updates=self.iters)[-1]

    def forward(self, rs_a, rs_b, T, sync, base):
        d = self.flow_ab(rs_a, rs_b)
        pred, aux = correct_from_flow(rs_a, d, T, sync, base, self.rot_k)
        aux["flow_ab"] = d
        return pred, aux


# ------------------------------------------------------------------ data + checkpoints
class RSSampleDataset(torch.utils.data.Dataset):
    """Samples from synth_video.py (and synth_rs.py) folders: RS pair, target and
    the timing needed by the geometry (all read from meta.json)."""

    def __init__(self, root, limit=None):
        from pathlib import Path
        self.dirs = sorted(p for p in Path(root).iterdir() if p.is_dir())[:limit]
        if not self.dirs:
            raise RuntimeError(f"no samples in {root}")

    def __len__(self):
        return len(self.dirs)

    @staticmethod
    def _img(path, k=None):
        import cv2
        import numpy as np
        im = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if k is not None:
            im = np.rot90(im, -k)                          # B back to A's orientation
        return torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float() / 255

    def __getitem__(self, i):
        import json
        d = self.dirs[i]
        m = json.loads((d / "meta.json").read_text())
        k = m.get("b_rot_k", 1)
        f = lambda v: torch.tensor(float(v))
        return {"rs_a": self._img(d / "rs_a.png"), "rs_b": self._img(d / "rs_b.png", k),
                "gs_a": self._img(d / "gs_a.png"),
                "T": f(m["readout_a_s"] * 1000), "sync": f(m["sync_offset_s"] * 1000),
                "base": f(m["baseline_px"]), "rot_k": torch.tensor(k)}


def load_raft_geom(path=None, device="cpu", size="large", iters=12, rot_k=1):
    """Fine-tuned checkpoint (path) or, without a path, the pretrained RAFT (zero-shot)."""
    if path is None:
        return RaftGeom(size=size, pretrained=True, iters=iters, rot_k=rot_k).to(device).eval()
    ck = torch.load(path, map_location=device)
    m = RaftGeom(pretrained=False, **ck["model_args"]).to(device)
    m.raft.load_state_dict(ck["raft"])
    return m.eval()
