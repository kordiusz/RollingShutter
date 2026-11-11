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

Direct synthesis (no flow, no geometry; "bezpośrednia synteza ramek" from the topic card):
"ae"   classic autoencoder without skip connections, latent 16 x 16 x 512 (ConvAutoencoder).
"corr" siamese encoder + correlation layer (cost volume) + decoder with skips (CorrSynthNet).
Both get A, B, coordinates and readout times; their psnr is over the whole image (= psnr_full).

"corrgeom" the network of "corr" predicting the flow A->B + fixed RS geometry v1 (CorrGeomNet):
   RAFT + geometry with RAFT replaced by a small network trained from scratch.

single=True (plain / ae / corr): camera B is replaced by a copy of A, so the network gets no second
moment of readout. Ablation "does camera B help?" with the same architecture (notebook 09_ablacje).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNet, conv_block, coord_channels, correlation, warp
from raft_geom import RaftGeom, backward_flow, base_grid, correct_from_flow, readout_time, velocity_from_flow

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

    def __init__(self, width=32, rot_k=1, flow_scale=10.0, single=False):
        super().__init__()
        self.unet = UNet(3 + 3 + 2 + 2, 5, tuple(width * m for m in (1, 2, 4, 8)))
        self.rot_k, self.flow_scale, self.single = rot_k, flow_scale, single

    def forward(self, rs_a, rs_b, T, sync, base):
        rs_b = rs_a if self.single else rs_b
        n, _, h, w = rs_a.shape
        x = torch.cat([rs_a - 0.5, rs_b - 0.5, coord_channels(n, h, w, rs_a.device),
                       time_maps(n, h, T, sync, self.rot_k, rs_a.device)], 1)
        with torch.autocast(device_type=rs_a.device.type, dtype=torch.float16, enabled=rs_a.is_cuda):
            out = self.unet(x).float()
        flow = out[:, 0:2] * self.flow_scale
        pred = (warp(rs_a, flow) + out[:, 2:5]).clamp(0, 1)
        # same evaluation region as RAFT + geometry: where the source pixel is inside A
        return pred, {"flow_a": flow, "valid_a": valid_from_flow(flow)}


def synth_inputs(rs_a, rs_b, T, sync, rot_k):
    """[A, B, coordinates, readout times of A and B] -> 10 channels, shared by the synthesis models."""
    n, _, h, w = rs_a.shape
    return torch.cat([rs_a - 0.5, rs_b - 0.5, coord_channels(n, h, w, rs_a.device),
                      time_maps(n, h, T, sync, rot_k, rs_a.device)], 1)


def down(cin, cout):
    """Stride-2 conv + conv block (halves the resolution)."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         conv_block(cout, cout))


def up_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class ConvAutoencoder(nn.Module):
    """"ae": classic autoencoder, NO skip connections. Direct synthesis of the GS image.

    Encoder 256 -> 16 px (4 x stride 2), latent 16 x 16 x latent_ch (default 512 channels,
    131k numbers vs 196k in the RS pair), decoder back to 256 px, output = image (no flow,
    no copy of A). Everything the output shows has to pass through the latent, so this is
    the reference "what a bottleneck alone can do": fine texture is expected to blur."""

    def __init__(self, width=32, rot_k=1, latent_ch=512, single=False):
        super().__init__()
        self.rot_k, self.single = rot_k, single
        ch = [width, 2 * width, 4 * width, 8 * width, latent_ch]           # 256, 128, 64, 32, 16 px
        self.inc = conv_block(10, ch[0])
        self.enc = nn.ModuleList(down(ch[i], ch[i + 1]) for i in range(4))
        self.dec = nn.ModuleList(conv_block(ch[i + 1], ch[i]) for i in reversed(range(4)))
        self.head = nn.Conv2d(ch[0], 3, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, rs_a, rs_b, T, sync, base):
        rs_b = rs_a if self.single else rs_b
        x = synth_inputs(rs_a, rs_b, T, sync, self.rot_k)
        with torch.autocast(device_type=rs_a.device.type, dtype=torch.float16, enabled=rs_a.is_cuda):
            z = self.inc(x)
            for e in self.enc:
                z = e(z)                                                   # latent (N, latent_ch, 16, 16)
            for d in self.dec:
                z = d(F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
            out = self.head(z).float()
        pred = (out + 0.5).clamp(0, 1)                                     # starts as a grey image
        return pred, {"valid_a": torch.ones_like(pred[:, :1])}


class CorrSynthNet(nn.Module):
    """"corr": siamese encoder + correlation layer + decoder with skips. Direct synthesis.

    A and B go through the same encoder (features at 1/2, 1/4, 1/8). A cost volume compares
    A's features with B's shifted by up to +-r cells: at 1/8 (r=4) that is +-32 px, at 1/4
    (r=3) +-12 px, i.e. the 10-40 px matches the problem needs, which a plain U-Net has to
    discover by itself. With the readout times of both cameras the decoder can turn
    "where is this point in B" into "where was it at the reference moment". A 1/16 stage adds
    context. Output = rs_a + residual image: synthesis without any flow / warp / geometry
    (starts as the identity, i.e. the uncorrected input)."""

    COUT = 3                                                               # residual image

    def __init__(self, width=32, rot_k=1, r8=4, r4=3, single=False):
        super().__init__()
        self.rot_k, self.r8, self.r4, self.single = rot_k, r8, r4, single
        w = width
        self.f1 = conv_block(3, w)                                         # 256 (encoder, shared A/B)
        self.f2, self.f4, self.f8 = down(w, w), down(w, 2 * w), down(2 * w, 4 * w)   # 128, 64, 32
        c8 = (2 * r8 + 1) ** 2 + 2 * 4 * w + 2 + 2                         # corr + fa + fb + coords + times
        self.b8 = conv_block(c8, 8 * w)
        self.b16 = down(8 * w, 8 * w)                                      # 16 px: context
        self.u8 = conv_block(16 * w, 8 * w)
        c4 = 8 * w + (2 * r4 + 1) ** 2 + 2 * 2 * w
        self.u4 = conv_block(c4, 4 * w)
        self.u2 = conv_block(4 * w + 2 * w, 2 * w)
        self.u1 = conv_block(2 * w + 10, w)                                # + full-res inputs
        self.head = nn.Conv2d(w, self.COUT, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def features(self, x):
        f1 = self.f1(x - 0.5)
        f2 = self.f2(f1)
        f4 = self.f4(f2)
        return f2, f4, self.f8(f4)

    def decode(self, rs_a, rs_b, T, sync):
        """Encoder + correlation + decoder -> head output (N, COUT, S, S), float32."""
        rs_b = rs_a if self.single else rs_b
        x = synth_inputs(rs_a, rs_b, T, sync, self.rot_k)
        corr = correlation
        with torch.autocast(device_type=rs_a.device.type, dtype=torch.float16, enabled=rs_a.is_cuda):
            a2, a4, a8 = self.features(rs_a)
            b2, b4, b8 = self.features(rs_b)
            c8 = corr(a8.float(), b8.float(), self.r8).to(a8.dtype)
            aux8 = F.adaptive_avg_pool2d(x[:, 6:10], a8.shape[-2:]).to(a8.dtype)   # coords + times at 1/8
            y8 = self.b8(torch.cat([c8, a8, b8, aux8], 1))
            y16 = self.b16(y8)
            y8 = self.u8(torch.cat([up_to(y16, y8), y8], 1))
            c4 = corr(a4.float(), b4.float(), self.r4).to(a4.dtype)
            y4 = self.u4(torch.cat([up_to(y8, a4), c4, a4, b4], 1))
            y2 = self.u2(torch.cat([up_to(y4, a2), a2, b2], 1))
            y1 = self.u1(torch.cat([up_to(y2, x), x.to(y2.dtype)], 1))
            return self.head(y1).float()

    def forward(self, rs_a, rs_b, T, sync, base):
        pred = (rs_a + self.decode(rs_a, rs_b, T, sync)).clamp(0, 1)
        return pred, {"valid_a": torch.ones_like(pred[:, :1])}


class CorrGeomNet(CorrSynthNet):
    """"corrgeom": the network of "corr", but it predicts the flow A->B and the fixed RS geometry
    (v1, as in RAFT + geometry v1) turns it into the corrected image. No pretrained weights.

    The rung between "corr" (matching, no geometry) and RAFT + geometry (pretrained iterative
    matching + geometry): if it is much better than "corr", the geometry is the key; if it is
    close to "corr", it is RAFT's matching. Trained from the image loss only, like RAFT in 04.
    Starts as "static scene" (flow = -baseline, velocity 0, output = input A)."""

    COUT = 2                                                               # flow A->B
    FLOW_SCALE = 10.0                                                      # head output * 10 px

    def forward(self, rs_a, rs_b, T, sync, base):
        d = self.FLOW_SCALE * self.decode(rs_a, rs_b, T, sync)
        d = d - torch.stack([base, torch.zeros_like(base)], 1).view(-1, 2, 1, 1).to(d)
        pred, aux = correct_from_flow(rs_a, d, T, sync, base, self.rot_k)
        return pred, {"valid_a": aux["valid"], "loss_mask": aux["valid"], "flow_ab": d,
                      "velocity": aux["velocity"]}


def valid_from_flow(flow):
    """1 where a backward flow points inside the source image."""
    n, _, h, w = flow.shape
    xs, ys = base_grid(n, h, w, flow.device)
    px, py = xs + flow[:, 0], ys + flow[:, 1]
    return ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).float()[:, None]


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


MODELS = {"refine": RaftUNet, "plain": PlainUNet, "ae": ConvAutoencoder, "corr": CorrSynthNet,
          "corrgeom": CorrGeomNet}


def build(mode, **kw):
    return MODELS[mode](**kw)


def load_model(path, device="cpu"):
    """Checkpoint of train_unet.py."""
    ck = torch.load(path, map_location=device)
    model = build(ck["mode"], **ck["model_args"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
