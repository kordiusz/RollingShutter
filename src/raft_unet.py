"""U-Net on the rolling-shutter video dataset (synth_video.py), two variants.

"refine" (main model): RAFT + RS geometry (raft_geom.py, frozen or slowly
   fine-tuned) gives the corrected image from camera A. The U-Net fixes what the
   geometry cannot express:
     * local errors of the backward flow (the anti-diagonal band where the
       velocity is only interpolated, motion that is not constant over the readout),
     * pixels that A does not show: areas that left A's frame and background
       uncovered by moving objects. Camera B is warped to the same target moment
       with the same velocity field, and a blend mask chooses the camera per pixel,
     * a small image residual (interpolation blur, gain difference, noise).
   output = blend * warp(A, flow_a + da) + (1 - blend) * warp(B, flow_b + db) + residual
   At the start (tiny head weights, blend ~0.88 for A) the output is close to the RAFT + geometry
   result, slightly averaged with B.

"plain" (baseline): the same U-Net with no RAFT and no geometry. Input: RS A, RS B
   (rotated back), coordinates and per-pixel readout times of both cameras.
   Output: flow for A + image residual. Answers "can a U-Net alone learn it?"
   (on the camera-motion set the analogous flow_dual model barely learned).
"""
import torch
import torch.nn as nn

from model import UNet, coord_channels, warp
from raft_geom import RaftGeom, backward_flow, base_grid, readout_time, velocity_from_flow

U_SCALE = 1.0      # velocity [px/ms] is fed to the U-Net as is (typically |u| < 1)
T_SCALE = 30.0     # readout times [ms] are divided by this (max readout in the datasets)


def time_maps(n, S, T, sync, rot_k, device):
    """Readout time of every pixel for camera A and camera B (A's frame), scaled to ~[-0.5, 0.5]."""
    xs, ys = base_grid(n, S, S, device)
    ta = readout_time(xs, ys, S, T, sync, rot_k, "a")
    tb = readout_time(xs, ys, S, T, sync, rot_k, "b")
    return torch.stack([ta, tb], 1) / T_SCALE


class RaftUNet(nn.Module):
    """RAFT + RS geometry (stage 1) + refinement U-Net (stage 2). One checkpoint."""

    # U-Net input: warped A (3), warped B (3), valid A, valid B, confidence (3),
    # velocity (2), coordinates (2)
    CIN, COUT = 13, 8

    def __init__(self, width=32, raft_size="large", raft_iters=12, rot_k=1, train_raft=False):
        super().__init__()
        self.geom = RaftGeom(size=raft_size, pretrained=False, iters=raft_iters, rot_k=rot_k)
        self.unet = UNet(self.CIN, self.COUT, tuple(width * m for m in (1, 2, 4, 8)))
        self.train_raft = train_raft
        if not train_raft:
            for p in self.geom.parameters():
                p.requires_grad_(False)

    @property
    def rot_k(self):
        return self.geom.rot_k

    def train(self, mode=True):
        super().train(mode)
        self.geom.eval()                # RAFT always with frozen BatchNorm statistics
        return self

    def stage1(self, rs_a, rs_b, T, sync, base):
        """RAFT flow A->B, velocity, backward flows for A and B and their warps."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_raft):
            d = self.geom.flow_ab(rs_a, rs_b)
            u, conf = velocity_from_flow(d, T, sync, base, self.rot_k)
            fa, va = backward_flow(u, T, sync, self.rot_k, cam="a")
            fb, vb = backward_flow(u, T, sync, self.rot_k, cam="b", base=base)
        return {"flow_ab": d, "velocity": u, "conf": conf, "flow_a": fa, "valid_a": va,
                "flow_b": fb, "valid_b": vb, "geom": warp(rs_a, fa)}

    def forward(self, rs_a, rs_b, T, sync, base):
        n, _, h, w = rs_a.shape
        s = self.stage1(rs_a, rs_b, T, sync, base)
        wb = warp(rs_b, s["flow_b"])
        x = torch.cat([s["geom"] - 0.5, wb - 0.5, s["valid_a"], s["valid_b"], s["conf"],
                       s["velocity"] / U_SCALE, coord_channels(n, h, w, rs_a.device)], 1)
        with torch.autocast(device_type=rs_a.device.type, dtype=torch.float16, enabled=rs_a.is_cuda):
            out = self.unet(x).float()
        fa = s["flow_a"] + 4.0 * out[:, 0:2]
        fb = s["flow_b"] + 4.0 * out[:, 2:4]
        blend = torch.sigmoid(out[:, 4:5] + 2.0)          # starts ~0.88 -> prefers camera A
        pred = (blend * warp(rs_a, fa) + (1 - blend) * warp(rs_b, fb) + out[:, 5:8]).clamp(0, 1)
        s.update(blend=blend, flow_a_ref=fa, flow_b_ref=fb)
        return pred, s


class PlainUNet(nn.Module):
    """Baseline without RAFT and geometry: U-Net on [A, B, coords, readout times] -> flow + residual."""

    def __init__(self, width=32, rot_k=1, flow_scale=10.0):
        super().__init__()
        self.unet = UNet(3 + 3 + 2 + 2, 5, tuple(width * m for m in (1, 2, 4, 8)))
        self.rot_k, self.flow_scale = rot_k, flow_scale

    def forward(self, rs_a, rs_b, T, sync, base):
        n, _, h, w = rs_a.shape
        x = torch.cat([rs_a - 0.5, rs_b - 0.5, coord_channels(n, h, w, rs_a.device),
                       time_maps(n, h, T, sync, self.rot_k, rs_a.device)], 1)
        with torch.autocast(device_type=rs_a.device.type, dtype=torch.float16, enabled=rs_a.is_cuda):
            out = self.unet(x).float()
        flow = out[:, 0:2] * self.flow_scale
        pred = (warp(rs_a, flow) + out[:, 2:5]).clamp(0, 1)
        # same evaluation region as RAFT + geometry: where the source pixel is inside A
        return pred, {"flow_a": flow, "valid_a": valid_from_flow(flow)}


def valid_from_flow(flow):
    """1 where a backward flow points inside the source image."""
    n, _, h, w = flow.shape
    xs, ys = base_grid(n, h, w, flow.device)
    px, py = xs + flow[:, 0], ys + flow[:, 1]
    return ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).float()[:, None]


class LegacyRSNet(nn.Module):
    """Stage-1 models of train.py (e.g. flow_dual, trained on the camera-motion set)
    with the interface of this file, to evaluate them on the video dataset."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, rs_a, rs_b, T, sync, base):
        pred, flow, _ = self.net(rs_a, rs_b, torch.full_like(T, 245.9))   # f: only MotionNet uses it
        pred = pred.float()
        if flow is None:
            return pred, {"valid_a": torch.ones_like(pred[:, :1])}
        return pred, {"flow_a": flow.float(), "valid_a": valid_from_flow(flow.float())}


def image_tensor(im, size=256, device="cpu"):
    """BGR uint8 image -> RGB tensor (1,3,size,size) in [0,1]."""
    import cv2
    import numpy as np
    im = cv2.resize(np.ascontiguousarray(im), (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float()[None].to(device) / 255


@torch.no_grad()
def predict_pair(model, img_a, img_b, T_ms, sync_ms=0.0, base_px=0.0, rot_k=1, size=256, device="cpu"):
    """One pair of BGR uint8 images: A as camera A outputs it, B as camera B outputs it
    (rotated by 90 deg). B is rotated back, both are resized to size x size (they must
    show the same square view). Returns tensors (1,3,S,S) rs_a, rs_b (aligned), pred."""
    import numpy as np
    rs_a, rs_b = image_tensor(img_a, size, device), image_tensor(np.rot90(img_b, -rot_k), size, device)
    t = lambda v: torch.tensor([float(v)], device=device)
    pred, _ = model.eval()(rs_a, rs_b, t(T_ms), t(sync_ms), t(base_px))
    return rs_a, rs_b, pred.float()


def build(mode, **kw):
    return RaftUNet(**kw) if mode == "refine" else PlainUNet(**kw)


def load_model(path, device="cpu"):
    """Checkpoint of train_unet.py, or of train.py (old models, wrapped in LegacyRSNet)."""
    ck = torch.load(path, map_location=device)
    if "mode" in ck:
        model = build(ck["mode"], **ck["model_args"])
        model.load_state_dict(ck["model"])
    else:
        from model import build_model
        net = build_model(**ck["model_args"])
        net.load_state_dict(ck["model"])
        model = LegacyRSNet(net)
        ck = dict(ck, mode="legacy_" + ck["model_args"].get("mode", "?"))
    return model.to(device).eval(), ck
