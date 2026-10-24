"""Small U-Net for dual rolling-shutter correction.

Input : rs_a + rs_b (B already rotated back to A's orientation) + 2 coordinate
        channels. The coordinates matter: RS displacement depends on WHICH ROW
        or COLUMN a pixel is in (its readout time), and a plain CNN does not
        know where in the image it is (CoordConv trick).
Modes :
  "flow"   - predicts a 2-channel displacement field and warps rs_a with it
             (= "predykcja wektorów przesunięć"; geometric, sharp output)
  "direct" - predicts the corrected image directly (rs_a + residual)
single=True uses only camera A: the ablation "does camera B help?".
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


# --------------------------------------------------------------------------- #
# Parametric ("motion") model: predict 9 camera-motion numbers, get the flow
# from exact RS geometry. Same idea as Velocity-Net in Zhuang et al. 2019.
# Motion over A's readout, s = t / readout in [-0.5, 0.5]:
#   rotation vector  r(s) = theta * s + beta * s^2      (theta = w*Tr, beta = a*Tr^2/2)
#   camera centre    c(s) = tau * s                      (tau = v*Tr, plane depth = 1)
# --------------------------------------------------------------------------- #
def rodrigues_t(rv):
    th = torch.sqrt((rv ** 2).sum(-1, keepdim=True) + 1e-12)
    k = rv / th
    kx, ky, kz = k.unbind(-1)
    z = torch.zeros_like(kx)
    K = torch.stack([torch.stack([z, -kz, ky], -1), torch.stack([kz, z, -kx], -1),
                     torch.stack([-ky, kx, z], -1)], -2)
    th = th[..., None]
    eye = torch.eye(3, device=rv.device, dtype=rv.dtype).expand(K.shape)
    return eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _readout_row(px, py, S, cam, rot_k):
    """Sensor row index (= readout order) of a point given in A's image frame."""
    if cam == "a":
        return py
    k = rot_k % 4
    return {0: py, 1: S - 1 - px, 2: S - 1 - py, 3: px}[k]


def flow_from_motion(params, f, S, n=65, iters=6, cam="a", rot_k=1):
    """params (N,9) = [theta, beta, tau]; f (N,) focal length in px.
    Returns backward flow (N,2,S,S): gs_a[y,x] = rs[y+dy, x+dx], where rs is
    rs_a (cam="a") or rs_b rotated back to A's orientation (cam="b": the same
    rig motion, but time follows B's readout direction). For B this assumes
    zero baseline, no sync offset and equal readout time; the refinement
    network corrects the remaining small differences.
    Differentiable; solved on an n x n grid and upsampled (field is smooth)."""
    N = params.shape[0]
    theta, beta, tau = params[:, None, None, 0:3], params[:, None, None, 3:6], params[:, None, None, 6:9]
    dev, dt = params.device, params.dtype
    g = torch.linspace(0, S - 1, n, device=dev, dtype=dt)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    gx, gy = gx.expand(N, n, n), gy.expand(N, n, n)
    c0 = (S - 1) / 2
    fv = f.view(N, 1, 1)
    X = torch.stack([(gx - c0) / fv, (gy - c0) / fv, torch.ones_like(gx)], -1)
    px, py = gx, gy
    for _ in range(iters):
        s = (_readout_row(px, py, S, cam, rot_k) / (S - 1) - 0.5)[..., None]
        R = rodrigues_t(theta * s + beta * s ** 2)
        Xc = (R.transpose(-1, -2) @ (X - tau * s)[..., None])[..., 0]
        px = fv * Xc[..., 0] / Xc[..., 2] + c0
        py = fv * Xc[..., 1] / Xc[..., 2] + c0
    coarse = torch.stack([px - gx, py - gy], 1)
    return F.interpolate(coarse, size=(S, S), mode="bicubic", align_corners=True)


class MotionNet(nn.Module):
    """CNN encoder -> 9 motion parameters -> exact flow -> warp of rs_a."""
    # The network predicts motion in PIXELS (angle * f etc.), which is what it can
    # actually see in the image; dividing by the known focal length f gives radians.
    # Otherwise the target would depend on the FOV, which the image does not reveal.
    PARAM_SCALE_PX = (10.0,) * 3 + (5.0,) * 3 + (2.0,) * 3   # typical magnitudes in px

    def __init__(self, single=False, img_ch=3, width=32, radius=6):
        super().__init__()
        self.single, self.radius = single, radius
        w = width
        # 1) shared ("siamese") feature extractor, 1/4 resolution
        self.feat = nn.Sequential(
            nn.Conv2d(img_ch, w, 3, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(w, w, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1))
        # 2) regressor on [cost volume, features of A, coordinates]
        cin = (0 if single else (2 * radius + 1) ** 2) + 2 * w + 2
        chs = [2 * w, 4 * w, 8 * w, 8 * w]
        layers, c = [], cin
        for co in chs:                                   # 4 x stride 2 -> 1/64 overall
            layers += [nn.Conv2d(c, co, 3, stride=2, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                       nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True)]
            c = co
        self.enc = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(4)              # keep coarse spatial layout
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * 16, 256), nn.LeakyReLU(0.1, True),
                                nn.Linear(256, 9))   # LeakyReLU: plain ReLU here 'dies'
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.register_buffer("scale", torch.tensor(self.PARAM_SCALE_PX))

    @staticmethod
    def correlation(fa, fb, r):
        """Cost volume: similarity of A's feature with B's feature shifted by
        (dx, dy) in [-r, r]^2 -> (2r+1)^2 channels. At 1/4 res, r=6 covers +-24 px."""
        fa, fb = F.normalize(fa, dim=1), F.normalize(fb, dim=1)
        H, W = fa.shape[-2:]
        fb = F.pad(fb, (r, r, r, r))
        return torch.cat([(fa * fb[:, :, dy:dy + H, dx:dx + W]).sum(1, keepdim=True)
                          for dy in range(2 * r + 1) for dx in range(2 * r + 1)], 1)

    def forward(self, rs_a, rs_b=None, f=None):
        n, _, h, w = rs_a.shape
        fa = self.feat(rs_a - 0.5)
        parts = [fa]
        if not self.single:
            fb = self.feat(rs_b - 0.5)
            parts.insert(0, self.correlation(fa.float(), fb.float(), self.radius).to(fa.dtype))
        parts.append(coord_channels(n, *fa.shape[-2:], rs_a.device).to(fa.dtype))
        params_px = self.fc(self.pool(self.enc(torch.cat(parts, 1)))).float() * self.scale
        params = params_px / f.float().view(-1, 1)
        with torch.autocast(device_type=rs_a.device.type, enabled=False):  # geometry in fp32
            flow = flow_from_motion(params, f.float(), h)
        return warp(rs_a, flow.to(rs_a.dtype)), flow, params


class RSNet(nn.Module):
    def __init__(self, mode="flow", single=False, img_ch=3, ch=(32, 64, 128, 256),
                 flow_scale=10.0):
        super().__init__()
        assert mode in ("flow", "direct")
        self.mode, self.single, self.flow_scale = mode, single, flow_scale
        cin = img_ch * (1 if single else 2) + 2
        self.net = UNet(cin, 2 if mode == "flow" else img_ch, ch)

    def forward(self, rs_a, rs_b=None, f=None):
        n, _, h, w = rs_a.shape
        parts = [rs_a] if self.single else [rs_a, rs_b]
        x = torch.cat(parts + [coord_channels(n, h, w, rs_a.device)], 1)
        out = self.net(x)
        if self.mode == "flow":
            flow = out * self.flow_scale
            return warp(rs_a, flow), flow, None
        return (rs_a + out).clamp(0, 1), None, None


# --------------------------------------------------------------------------- #
# Stage 2: refinement. MotionNet (frozen) gives the global, camera-only
# correction; this network fixes what that model cannot express:
#   * B's stereo/sync offset (small OffsetHead, 2 numbers in px),
#   * local flow corrections for A and for B (U-Net),
#   * which camera each output pixel should come from (blend mask), so the
#     areas that left A's frame are filled from B,
#   * a small image residual (sharpening, brightness, leftovers).
# --------------------------------------------------------------------------- #
def out_of_frame(flow):
    """1 where a backward flow points outside the source image."""
    n, _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    px, py = xs + flow[:, 0], ys + flow[:, 1]
    return ((px < 0) | (px > w - 1) | (py < 0) | (py > h - 1)).to(flow.dtype)[:, None]


class OffsetHead(nn.Module):
    """Global (dx, dy) shift of B relative to A after the geometric correction
    (stereo baseline + sync offset). Correlation of features, then regression."""

    def __init__(self, feat_ch, radius=4, width=32):
        super().__init__()
        self.radius = radius
        c = (2 * radius + 1) ** 2
        self.net = nn.Sequential(
            nn.Conv2d(c, width, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fa, fb):
        corr = MotionNet.correlation(fa.float(), fb.float(), self.radius)
        return self.net(corr.to(fa.dtype)).float() * 4.0      # px, at full resolution


class RefinePipeline(nn.Module):
    """MotionNet (global geometry) + refinement U-Net. One module, one checkpoint."""

    def __init__(self, motion_args, width=32, freeze_motion=True):
        super().__init__()
        self.motion = build_model(**motion_args)
        assert isinstance(self.motion, MotionNet) and not self.motion.single, \
            "refinement needs a dual-camera MotionNet"
        self.freeze_motion = freeze_motion
        if freeze_motion:
            for p in self.motion.parameters():
                p.requires_grad_(False)
        fch = self.motion.feat[-1].out_channels
        self.offset = OffsetHead(fch, width=width)
        # inputs: warped A (3), warped B (3), out-of-frame A/B (2), global flow A (2), coords (2)
        self.unet = UNet(12, 8, tuple(width * m for m in (1, 2, 4, 8)))
        self.rot_k = 1

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_motion:
            self.motion.eval()          # keep BatchNorm statistics of the frozen model
        return self

    def forward(self, rs_a, rs_b, f):
        n, _, h, w = rs_a.shape
        # 1) global, camera-only correction (stage 1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_motion):
            _, flow_a, params = self.motion(rs_a, rs_b, f)
        flow_a = flow_a.detach() if self.freeze_motion else flow_a
        with torch.autocast(device_type=rs_a.device.type, enabled=False):
            flow_b = flow_from_motion(params.float().detach() if self.freeze_motion else params.float(),
                                      f.float(), h, cam="b", rot_k=self.rot_k)
        wa = warp(rs_a, flow_a.to(rs_a.dtype))
        wb = warp(rs_b, flow_b.to(rs_b.dtype))
        # 2) global offset of B (baseline + sync)
        off = self.offset(self.motion.feat(wa - 0.5), self.motion.feat(wb - 0.5))
        flow_b = flow_b + off[:, :, None, None]
        wb = warp(rs_b, flow_b.to(rs_b.dtype))
        # 3) local corrections + fusion
        x = torch.cat([wa - 0.5, wb - 0.5, out_of_frame(flow_a), out_of_frame(flow_b),
                       flow_a / 10.0, coord_channels(n, h, w, rs_a.device)], 1)
        out = self.unet(x.to(rs_a.dtype)).float()
        fa = flow_a + 4.0 * out[:, 0:2]
        fb = flow_b + 4.0 * out[:, 2:4]
        blend = torch.sigmoid(out[:, 4:5] + 2.0)        # starts ~0.88 -> prefers camera A
        ia, ib = warp(rs_a.float(), fa), warp(rs_b.float(), fb)
        pred = (blend * ia + (1 - blend) * ib + out[:, 5:8]).clamp(0, 1)
        return pred, fa, {"flow_b": fb, "params": params, "blend": blend, "offset": off}


def build_model(mode="motion", single=False, width=32, motion_args=None):
    if mode == "motion":
        return MotionNet(single=single, width=width)
    if mode == "refine":
        return RefinePipeline(motion_args, width=width)
    return RSNet(mode=mode, single=single, ch=tuple(width * m for m in (1, 2, 4, 8)))
