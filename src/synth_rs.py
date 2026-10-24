"""
Synthetic dual rolling-shutter (RS) dataset generator.

Setup being simulated
---------------------
Two RS cameras mounted side by side on one rigid rig:
  * Camera A  - normal orientation, reads rows top -> bottom.
  * Camera B  - physically rotated 90 deg about its optical axis, so in A's
                image frame it reads COLUMNS instead of rows. Its output image
                is therefore a rotated picture, exactly like the real camera.
Both cameras move together (same rotation / translation over time).

Idea (the simplest approach that still captures the physics)
------------------------------------------------------------
Start from an ordinary sharp still photo (= a global-shutter, GS, view of a
planar scene at depth Z=1). For every pixel of each RS image we know *when*
it was read out (its row for A, its column for B). We compute the camera pose
at that instant, cast the pixel's ray onto the scene plane, and look up the
colour in the still photo. That is a per-pixel backward warp (cv2.remap), so
there are no holes and the ground truth is exact.

Outputs per sample (folder <out>/<split>/<id>/):
  rs_a.png   RS image from camera A (model input 1)
  rs_b.png   RS image from camera B, rotated as the real camera sees it (input 2)
  gs_a.png   Global-shutter image of camera A at the reference time (target)
  gs_b.png   Global-shutter image of camera B (optional extra target)
  meta.json  All sampled parameters (motion, readout, baseline, ...)
  corr.npz   (with --save-corr) per-pixel (x, y) location in gs_a of every
             rs_a pixel (corr_a) and rs_b pixel (corr_b, in rs_b layout)

Usage
-----
  python src/synth_rs.py --src path/to/images --out dataset --per-image 4 --size 256
  python src/synth_rs.py --demo --out demo          # no images needed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict, field
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SynthConfig:
    size: int = 256                      # output S x S (square: rot90 keeps shape)
    fov_deg: tuple = (50.0, 80.0)        # horizontal FOV of the crop -> focal length
    readout_ms: tuple = (10.0, 33.0)     # A: time to read the S rows of the crop
    b_readout_scale: float = 1.0         # B readout = A readout * this (see README)
    b_rot_k: int = 1                     # np.rot90 k that turns A-frame into B image
    sync_offset_ms: tuple = (0.0, 0.0)   # B readout centre minus A's, e.g. (-2, 2)
    # Motion, expressed in PIXELS of image motion during A's readout (intuitive):
    max_pan_px: float = 24.0             # pan/tilt induced shift
    max_roll_px: float = 8.0             # roll induced shift at the crop corner
    max_trans_px: float = 3.0            # translation induced shift (plane Z=1)
    accel_frac: float = 0.5              # angular velocity may change by +-50%
    baseline_px: tuple = (0.0, 6.0)      # stereo disparity A->B (B to the right)
    exposure_ms: tuple = (0.0, 0.0)      # >0 adds motion blur (n sub-frames)
    exposure_sub: int = 5
    gain: tuple = (0.9, 1.1)             # per-camera brightness gain
    noise_std: tuple = (0.0, 0.01)       # per-camera Gaussian noise (0..1 scale)
    gray: bool = False
    margin_frac: float = 0.35            # extra source border around the crop


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def rodrigues(rv: np.ndarray) -> np.ndarray:
    """Vectorised axis-angle -> rotation matrix. rv (..., 3) -> (..., 3, 3)."""
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    Km = np.stack([np.stack([zero, -kz, ky], -1),
                   np.stack([kz, zero, -kx], -1),
                   np.stack([-ky, kx, zero], -1)], -2)
    th = th[..., None]
    eye = np.broadcast_to(np.eye(3), Km.shape)
    return eye + np.sin(th) * Km + (1 - np.cos(th)) * (Km @ Km)


@dataclass
class Motion:
    """Rig motion around reference time t=0 (= middle of A's readout).

    World frame = camera A at t=0, scene = textured plane at Z=1.
    R_cw(t) = exp(w t + 0.5 a t^2),  centre c(t) = v t.
    Units: seconds, radians, plane-depth units.
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))


def source_coords(T: np.ndarray, f: float, S: int, P: int, motion: Motion,
                  baseline: float) -> tuple[np.ndarray, np.ndarray]:
    """For an S x S image whose pixel (y, x) is captured at time T[y, x],
    return the (x, y) coordinates in the P x P source patch (= A's GS view
    with margin) that the pixel sees. Optical axes of A and B are parallel;
    B's centre is offset by `baseline` (plane units) along A's x axis."""
    c0 = (S - 1) / 2.0
    cP = (P - 1) / 2.0
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
    ray = np.stack([(xs - c0) / f, (ys - c0) / f, np.ones_like(xs)], -1)  # K^-1 p

    rv = motion.w * T[..., None] + 0.5 * motion.a * (T[..., None] ** 2)
    R = rodrigues(rv)                                   # camera -> world
    d = np.einsum("...ij,...j->...i", R, ray)           # ray dir in world
    c = motion.v * T[..., None]                         # rig centre
    if baseline:
        c = c + np.einsum("...ij,j->...i", R, np.array([baseline, 0.0, 0.0]))
    lam = (1.0 - c[..., 2]) / d[..., 2]                 # hit plane Z = 1
    X = c + lam[..., None] * d
    mx = (f * X[..., 0] + cP).astype(np.float32)
    my = (f * X[..., 1] + cP).astype(np.float32)
    return mx, my


def readout_times(S: int, readout_s: float, offset_s: float, rot_k: int,
                  is_b: bool) -> np.ndarray:
    """Capture time of each pixel, laid out in A's image frame.
    The sensor's own row index r in [0, S-1] is read at
    offset + (r/(S-1) - 0.5) * readout. For camera B the sensor rows are the
    rows of the rotated image, mapped back to A's frame."""
    row = np.repeat(np.arange(S, dtype=np.float64)[:, None], S, axis=1)
    if is_b:
        row = np.rot90(row, -rot_k)       # B's row index, seen in A's frame
    return offset_s + (row / (S - 1) - 0.5) * readout_s


def pixel_time(x, y, S, readout_s, offset_s, rot_k, is_b):
    """Continuous capture time of a (sub)pixel given in A's image frame.
    Same convention as readout_times()."""
    r = y
    if is_b:
        k = rot_k % 4
        r = {0: y, 1: S - 1 - x, 2: S - 1 - y, 3: x}[k]
    return offset_s + (r / (S - 1) - 0.5) * readout_s


def gs_to_rs(f, S, motion: Motion, readout_s, offset_s=0.0, rot_k=1, is_b=False,
             baseline=0.0, iters=10, xs=None, ys=None):
    """Inverse of source_coords: for every pixel of the GS image of camera A
    (t = 0), find where it lands in the RS image (A-frame layout).
    Fixed-point iteration: guess row -> time -> pose -> project -> new row.
    xs, ys: optional arrays of GS points (default: the full pixel grid)."""
    c0 = (S - 1) / 2.0
    if xs is None:
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
    X = np.stack([(xs - c0) / f, (ys - c0) / f, np.ones_like(xs)], -1)  # plane Z=1
    px, py = xs.copy(), ys.copy()
    for _ in range(iters):
        T = pixel_time(px, py, S, readout_s, offset_s, rot_k, is_b)
        R = rodrigues(motion.w * T[..., None] + 0.5 * motion.a * T[..., None] ** 2)
        c = motion.v * T[..., None]
        if baseline:
            c = c + np.einsum("...ij,j->...i", R, np.array([baseline, 0.0, 0.0]))
        Xc = np.einsum("...ji,...j->...i", R, X - c)          # R^T (X - c)
        px = f * Xc[..., 0] / Xc[..., 2] + c0
        py = f * Xc[..., 1] / Xc[..., 2] + c0
    return px.astype(np.float32), py.astype(np.float32)


def flow_from_meta(meta: dict, n: int = 33, cam: str = "a") -> np.ndarray:
    """Ground-truth backward flow: flow[y, x] = (dx, dy) such that
    gs_a[y, x] == rs[y + dy, x + dx], where rs is camera A's RS image
    (cam="a") or camera B's RS image rotated back to A's orientation (cam="b";
    includes B's column readout, stereo baseline and sync offset).
    Shape (S, S, 2), float32. Computed from meta.json, so nothing extra has to
    be stored on disk. The field is very smooth, so it is solved exactly on a
    coarse grid (n x n points) and interpolated with a cubic spline."""
    from scipy.interpolate import RectBivariateSpline
    S = meta["size"]
    mot = Motion(w=np.array(meta["w_rad_s"]), a=np.array(meta["a_rad_s2"]),
                 v=np.array(meta["v_planeunits_s"]))
    g = np.linspace(0, S - 1, n)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    if cam == "a":
        px, py = gs_to_rs(meta["f"], S, mot, meta["readout_a_s"], xs=gx, ys=gy)
    else:
        px, py = gs_to_rs(meta["f"], S, mot, meta["readout_b_s"], meta["sync_offset_s"],
                          meta.get("b_rot_k", 1), is_b=True,
                          baseline=meta["baseline_px"] / meta["f"], xs=gx, ys=gy)
    full = np.arange(S, dtype=np.float64)
    fx = RectBivariateSpline(g, g, px - gx)(full, full)
    fy = RectBivariateSpline(g, g, py - gy)(full, full)
    return np.stack([fx, fy], -1).astype(np.float32)


def render(src: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    return cv2.remap(src, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT101)


def render_cam(src, T, f, S, P, motion, baseline, exposure_s, n_sub):
    """Render one RS image (A-frame layout); averages sub-frames for blur."""
    taus = [0.0] if exposure_s <= 0 else np.linspace(0, exposure_s, n_sub)
    acc, mx0, my0 = None, None, None
    for tau in taus:
        mx, my = source_coords(T + tau, f, S, P, motion, baseline)
        img = render(src, mx, my)
        acc = img if acc is None else acc + img
        if mx0 is None:
            mx0, my0 = mx, my
    return acc / len(taus), mx0, my0


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def U(rng, r):
    lo, hi = r
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


def sample_params(rng: np.random.Generator, cfg: SynthConfig):
    S = cfg.size
    f = (S / 2) / math.tan(math.radians(U(rng, cfg.fov_deg)) / 2)
    Tr = U(rng, cfg.readout_ms) / 1000.0

    # angular velocity from desired pixel shifts during A's readout
    pan = rng.uniform(0, cfg.max_pan_px)
    th = rng.uniform(0, 2 * math.pi)
    wxy = pan / (f * Tr)
    roll = rng.uniform(-cfg.max_roll_px, cfg.max_roll_px)
    wz = roll / ((S / math.sqrt(2)) * Tr)
    w = np.array([wxy * math.sin(th), wxy * math.cos(th), wz])
    # angular acceleration: w changes by up to accel_frac*|w| over half readout
    adir = rng.normal(size=3)
    adir /= np.linalg.norm(adir) + 1e-12
    a = adir * np.linalg.norm(w) * rng.uniform(0, cfg.accel_frac) / (Tr / 2)
    # translation (plane at Z=1, so shift_px = f * |v| * Tr)
    vdir = rng.normal(size=3)
    vdir /= np.linalg.norm(vdir) + 1e-12
    v = vdir * rng.uniform(0, cfg.max_trans_px) / (f * Tr)

    return dict(
        f=f, readout_a_s=Tr, readout_b_s=Tr * cfg.b_readout_scale,
        sync_offset_s=U(rng, cfg.sync_offset_ms) / 1000.0,
        baseline_px=U(rng, cfg.baseline_px),
        exposure_s=U(rng, cfg.exposure_ms) / 1000.0,
        motion=Motion(w=w, a=a, v=v),
        gain_a=U(rng, cfg.gain), gain_b=U(rng, cfg.gain),
        noise_a=U(rng, cfg.noise_std), noise_b=U(rng, cfg.noise_std),
    )


def random_patch(img: np.ndarray, P: int, rng) -> np.ndarray:
    h, w = img.shape[:2]
    s_min = P / min(h, w)
    s = rng.uniform(s_min, max(s_min, 1.0)) if s_min < 1 else s_min
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, (max(P, round(w * s)), max(P, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y0 = rng.integers(0, h - P + 1)
    x0 = rng.integers(0, w - P + 1)
    patch = img[y0:y0 + P, x0:x0 + P]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch)


def make_sample(src_img: np.ndarray, rng: np.random.Generator, cfg: SynthConfig,
                max_tries: int = 10):
    S = cfg.size
    M = int(round(cfg.margin_frac * S))
    P = S + 2 * M
    patch = random_patch(src_img, P, rng).astype(np.float32) / 255.0

    for _ in range(max_tries):
        p = sample_params(rng, cfg)
        f, mot = p["f"], p["motion"]
        b = p["baseline_px"] / f                       # plane units (Z = 1)

        T_a = readout_times(S, p["readout_a_s"], 0.0, cfg.b_rot_k, is_b=False)
        T_b = readout_times(S, p["readout_b_s"], p["sync_offset_s"], cfg.b_rot_k, is_b=True)
        rs_a, ax, ay = render_cam(patch, T_a, f, S, P, mot, 0.0, p["exposure_s"], cfg.exposure_sub)
        rs_b, bx, by = render_cam(patch, T_b, f, S, P, mot, b, p["exposure_s"], cfg.exposure_sub)

        lo, hi = 0.0, P - 1.0
        if all(m.min() >= lo and m.max() <= hi for m in (ax, ay, bx, by)):
            break
    else:
        return None  # motion too large for the margin; caller skips

    zero = np.zeros((S, S))
    gs_a = render(patch, *source_coords(zero, f, S, P, Motion(), 0.0))
    gs_b = render(patch, *source_coords(zero, f, S, P, Motion(), b))

    def photometric(img, gain, sigma):
        img = img * gain
        if sigma > 0:
            img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)

    rs_a = photometric(rs_a, p["gain_a"], p["noise_a"])
    rs_b = photometric(rs_b, p["gain_b"], p["noise_b"])
    # target = what camera A itself would see with a global shutter (same gain, no noise)
    gs_a = photometric(gs_a, p["gain_a"], 0.0)
    gs_b = photometric(gs_b, p["gain_b"], 0.0)

    k = cfg.b_rot_k
    out = dict(
        rs_a=rs_a, rs_b=np.rot90(rs_b, k), gs_a=gs_a, gs_b=np.rot90(gs_b, k),
        corr_a=np.stack([ax - M, ay - M], -1),
        corr_b=np.rot90(np.stack([bx - M, by - M], -1), k),
    )
    mot = p.pop("motion")
    meta = {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv)
            for kk, vv in p.items()}
    meta.update(w_rad_s=mot.w.tolist(), a_rad_s2=mot.a.tolist(),
                v_planeunits_s=mot.v.tolist(), size=S, b_rot_k=k,
                ref_time="middle of camera A readout")
    return out, meta


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_sample(folder: Path, out: dict, meta: dict, save_corr: bool):
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("rs_a", "rs_b", "gs_a", "gs_b"):
        cv2.imwrite(str(folder / f"{name}.png"), to_u8(out[name]))
    with open(folder / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    if save_corr:
        np.savez_compressed(folder / "corr.npz",
                            corr_a=out["corr_a"].astype(np.float32),
                            corr_b=np.ascontiguousarray(out["corr_b"]).astype(np.float32))


def split_of(path: str, val_frac: float) -> str:
    """Deterministic split by source image, so no image leaks into both."""
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16)
    return "val" if (h % 10_000) / 10_000 < val_frac else "train"


def load_image(path: str, gray: bool):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return img


def _job(args):
    idx, path, n, cfg_dict, out_dir, seed, val_frac, save_corr = args
    cfg = SynthConfig(**cfg_dict)
    rng = np.random.default_rng([seed, idx])
    img = load_image(path, cfg.gray)
    if img is None:
        return 0
    split = split_of(path, val_frac)
    done = 0
    for j in range(n):
        res = make_sample(img, rng, cfg)
        if res is None:
            continue
        out, meta = res
        meta["source"] = os.path.basename(path)
        save_sample(Path(out_dir) / split / f"{idx:06d}_{j:02d}", out, meta, save_corr)
        done += 1
    return done


def demo_image(size=1024):
    """Checkerboard + grid + circles: makes RS distortion easy to see."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = (((yy // 64) + (xx // 64)) % 2 * 120 + 70).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, size, 64):
        cv2.line(img, (i, 0), (i, size), (40, 40, 220), 3)
        cv2.line(img, (0, i), (size, i), (220, 120, 40), 3)
    for i in range(6):
        cv2.circle(img, (int(size * (0.15 + 0.14 * i)), int(size * 0.5)), 60,
                   (40, 200, 40), 6)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder of sharp source images (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--save-corr", action="store_true")
    ap.add_argument("--demo", action="store_true", help="use a built-in test pattern")
    ap.add_argument("--config", help="JSON file overriding SynthConfig fields")
    a = ap.parse_args()

    cfg = SynthConfig(size=a.size)
    if a.config:
        with open(a.config) as fh:
            for k, v in json.load(fh).items():
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg_dict = asdict(cfg)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "config.json", "w") as fh:
        json.dump(cfg_dict, fh, indent=1)

    if a.demo:
        path = str(Path(a.out) / "_demo_source.png")
        cv2.imwrite(path, demo_image())
        paths = [path]
    else:
        if not a.src:
            ap.error("--src is required unless --demo")
        paths = sorted(str(p) for p in Path(a.src).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not paths:
            ap.error(f"no images found in {a.src}")

    jobs = [(i, p, a.per_image, cfg_dict, a.out, a.seed, a.val_frac, a.save_corr)
            for i, p in enumerate(paths)]
    total = 0
    if a.workers > 1 and len(jobs) > 1:
        with Pool(a.workers) as pool:
            for k, n in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
                total += n
                if (k + 1) % 50 == 0:
                    print(f"{k + 1}/{len(jobs)} images, {total} samples")
    else:
        for job in jobs:
            total += _job(job)
    print(f"done: {total} samples from {len(paths)} images -> {a.out}")


if __name__ == "__main__":
    main()
