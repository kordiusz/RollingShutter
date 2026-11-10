"""
Dual rolling-shutter dataset from a self-made "layered" scene (no video, no Blender).

What it does, in plain words
----------------------------
We build a small fake 3D scene ourselves:
  * a background picture far away (depth Z_bg),
  * a few cut-out objects (random shapes, textured with photos) floating at
    different depths in front of it; each object can slide and spin on its own
    (e.g. a fan, a passing car),
  * the camera rig shakes (rotation) and moves (translation).
Then we "film" it exactly like a rolling-shutter camera: every row of camera A
(every column of camera B) is taken at its own moment in time. It is the same
idea as synth_video.py (rows from successive moments of a fast video), except
the "video" is rendered on the fly for the exact moment of every row, so there
are no stripes / ghosting from a limited frame rate.

Because the objects really sit at different depths, camera B (shifted sideways
by the stereo baseline) sees near objects shifted more than the background,
and objects cover different parts of the background: real parallax and
occlusions, unlike the constant shift in synth_rs.py / synth_video.py.

Outputs: same folder format as synth_rs.py / synth_video.py
(rs_a.png, rs_b.png rotated as camera B outputs it, gs_a.png, gs_b.png,
meta.json with kind="layers"), so preview.py and train_raft.py work unchanged.
There is no dense ground-truth flow (objects move on their own), so it is
meant for image-supervised models (train_raft.py), like the video set.

  python src/synth_layers.py --src path/to/photos --out dataset_layers --n 4000
  python src/synth_layers.py --src photos --out rs_layers --n 6000 --procedural-frac 0.2   # 80% photos, 20% shapes
  # realistic: backgrounds from DIV2K, real objects cut out of COCO (src/make_cutouts.py)
  python src/synth_layers.py --bg-src DIV2K_train_HR --obj-src cutouts --out rs_layers --n 6000 --procedural-frac 0.2
  python src/synth_layers.py --out demo_layers --n 16        # procedural textures, no photos needed
  python src/preview.py demo_layers/train --n 8 --out preview_layers.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from synth_rs import (IMG_EXT, SynthConfig, random_patch, readout_times, rodrigues,
                      sample_params, save_sample, split_of)


@dataclass
class LayerConfig:
    size: int = 256
    supersample: int = 2                 # render at 2x and shrink (anti-aliasing)
    # camera / readout / shake: same meaning as in synth_rs.SynthConfig
    fov_deg: tuple = (45.0, 60.0)
    readout_ms: tuple = (15.0, 33.0)
    b_readout_scale: float = 1.0
    b_rot_k: int = 1
    sync_offset_ms: tuple = (-3.0, 3.0)
    max_pan_px: float = 30.0
    max_roll_px: float = 8.0
    max_trans_px: float = 3.0            # camera translation, px of BACKGROUND motion (near objects move more)
    accel_frac: float = 0.5
    baseline_px: tuple = (0.0, 3.0)      # stereo disparity A->B of the BACKGROUND
    exposure_ms: tuple = (0.0, 0.0)      # >0 adds motion blur
    exposure_sub: int = 5
    gain: tuple = (0.9, 1.1)
    noise_std: tuple = (0.0, 0.02)
    margin_frac: float = 0.35
    # objects
    n_objects: tuple = (1, 5)            # how many cut-outs in front of the background
    obj_radius_frac: tuple = (0.08, 0.3) # object size (radius) / image size
    obj_depth_frac: tuple = (0.25, 0.9)  # object depth / background depth (smaller = nearer)
    obj_speed_px: tuple = (0.0, 40.0)    # own motion of an object during A's readout, px
    obj_spin_rad: tuple = (0.0, 0.6)     # own rotation of an object during A's readout, rad
    p_static_obj: float = 0.3            # chance that an object does not move on its own
    min_colour_dist: float = 25.0        # object vs background behind it, mean-colour delta E (Lab)
    contrast_tries: int = 8              # textures tried per object to reach min_colour_dist
    cutout_max_upscale: float = 2.0      # real cut-outs are not enlarged more than this (blur)
    cutout_max_spin_rad: float = 0.15    # real cut-outs (people, cars) rotate at most this per readout


# --------------------------------------------------------------------------- #
# Textures and shapes
# --------------------------------------------------------------------------- #
def procedural_texture(rng, P):
    """Colourful random texture (used when no photos are given)."""
    kind = rng.integers(4)
    if kind == 0:                                            # multi-scale coloured noise
        img = np.zeros((P, P, 3), np.float32)
        for s in (4, 16, 64):
            n = rng.random((max(2, P // s), max(2, P // s), 3)).astype(np.float32)
            img += cv2.resize(n, (P, P), interpolation=cv2.INTER_CUBIC) / 3
    elif kind == 1:                                          # stripes at a random angle
        yy, xx = np.mgrid[0:P, 0:P].astype(np.float32)
        th, per = rng.uniform(0, np.pi), rng.uniform(8, 40)
        s = (np.sin((xx * np.cos(th) + yy * np.sin(th)) * 2 * np.pi / per) > 0)[..., None]
        img = np.where(s, rng.random(3), rng.random(3)).astype(np.float32)
    elif kind == 2:                                          # checkerboard
        c = int(rng.integers(12, 48))
        yy, xx = np.mgrid[0:P, 0:P]
        s = (((yy // c) + (xx // c)) % 2 == 0)[..., None]
        img = np.where(s, rng.random(3), rng.random(3)).astype(np.float32)
    else:                                                    # random shapes on a plain colour
        img = np.ones((P, P, 3), np.float32) * rng.random(3).astype(np.float32)
        for _ in range(int(rng.integers(20, 60))):
            col = tuple(float(v) for v in rng.random(3))
            p = tuple(int(v) for v in rng.integers(0, P, 2))
            if rng.random() < 0.5:
                cv2.circle(img, p, int(rng.integers(4, P // 6)), col, -1, cv2.LINE_AA)
            else:
                q = tuple(int(v) for v in rng.integers(0, P, 2))
                cv2.line(img, p, q, col, int(rng.integers(2, 10)), cv2.LINE_AA)
    return np.clip(img, 0, 1)


def get_texture(rng, pool, P):
    """P x P float texture in [0, 1]: a random crop of a random photo, or procedural."""
    if pool:
        for _ in range(5):
            img = cv2.imread(pool[int(rng.integers(len(pool)))], cv2.IMREAD_COLOR)
            if img is not None:
                return random_patch(img, P, rng).astype(np.float32) / 255.0
    return procedural_texture(rng, P)


def get_object(rng, pool, Q, r, cfg):
    """Foreground object on a Q x Q canvas, size ~r (radius): (texture, alpha, source).
    From the object pool: a PNG with transparency (real cut-out, e.g. from
    make_cutouts.py) is used as it is; a normal photo gets a random shape cut out
    of it. Without a pool: procedural texture + random shape."""
    for _ in range(5 if pool else 0):
        path = pool[int(rng.integers(len(pool)))]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:                                  # real cut-out with alpha
            if img.dtype != np.uint8:
                img = (img / 257).astype(np.uint8)
            h, w = img.shape[:2]
            s = min(2 * r / max(h, w), cfg.cutout_max_upscale)
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            if rng.random() < 0.5:
                img = img[:, ::-1]
            h, w = img.shape[:2]
            canvas = np.zeros((Q, Q, 4), np.float32)
            y0, x0 = (Q - h) // 2, (Q - w) // 2
            canvas[y0:y0 + h, x0:x0 + w] = img.astype(np.float32) / 255.0
            return canvas[..., :3].copy(), canvas[..., 3].copy(), "cutout:" + Path(path).parent.name
        tex = random_patch(img[..., :3], Q, rng).astype(np.float32) / 255.0
        return tex, random_shape(rng, Q, r), "photo"
    return procedural_texture(rng, Q), random_shape(rng, Q, r), "procedural"


def colour_distance(tex_a, mask_a, tex_b):
    """How different two textures look: distance of their mean colours in Lab
    (CIE76 delta E, ~2 = barely visible, >25 = clearly different colours)."""
    la = cv2.cvtColor(tex_a, cv2.COLOR_BGR2Lab)[mask_a > 0.5]
    lb = cv2.cvtColor(tex_b, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    if len(la) == 0 or len(lb) == 0:
        return 1e9
    return float(np.linalg.norm(la.mean(0) - lb.mean(0)))


def random_shape(rng, Q, r):
    """Alpha mask (Q x Q, float 0..1) of a random shape with radius ~r, centred."""
    m = np.zeros((Q, Q), np.float32)
    c = (Q - 1) / 2
    kind = rng.integers(4)
    if kind == 0:                                            # ellipse
        cv2.ellipse(m, (int(c), int(c)), (int(r), int(r * rng.uniform(0.4, 1))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
    elif kind == 1:                                          # rectangle
        w, h = r * rng.uniform(0.5, 1.4), r * rng.uniform(0.3, 1.0)
        box = cv2.boxPoints(((c, c), (2 * w, 2 * h), float(rng.uniform(0, 180))))
        cv2.fillPoly(m, [np.round(box).astype(np.int32)], 1.0, cv2.LINE_AA)
    elif kind == 2:                                          # irregular blob / polygon
        n = int(rng.integers(5, 12))
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = r * rng.uniform(0.5, 1.0, n)
        pts = np.stack([c + rad * np.cos(ang), c + rad * np.sin(ang)], 1)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0, cv2.LINE_AA)
    else:                                                    # fan / propeller (blades + hub)
        nb = int(rng.integers(2, 6))
        a0 = rng.uniform(0, 2 * np.pi)
        for i in range(nb):
            a = a0 + 2 * np.pi * i / nb
            cv2.ellipse(m, (int(c + 0.5 * r * np.cos(a)), int(c + 0.5 * r * np.sin(a))),
                        (int(0.5 * r), int(0.13 * r) + 1), math.degrees(a), 0, 360, 1.0, -1,
                        cv2.LINE_AA)
        cv2.circle(m, (int(c), int(c)), int(0.15 * r) + 1, 1.0, -1, cv2.LINE_AA)
    return m


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _ray_pose(xs, ys, T, f, S, motion, base_world):
    """Camera centre c and ray direction d (world frame) of pixels (xs, ys)
    read at times T. World = camera A at t=0. Camera centre c(t) = v t
    (+ R(t) [b,0,0] for B), orientation R(t) = exp(w t + a t^2 / 2)."""
    c0 = (S - 1) / 2.0
    ray = np.stack([(xs - c0) / f, (ys - c0) / f, np.ones_like(xs)], -1)
    Tx = T[..., None]
    R = rodrigues(motion["w"] * Tx + 0.5 * motion["a"] * Tx ** 2)
    d = np.einsum("...ij,...j->...i", R, ray)
    c = motion["v"] * Tx
    if base_world:
        c = c + np.einsum("...ij,j->...i", R, np.array([base_world, 0.0, 0.0]))
    return c, d


def render_layers(T, f, S, layers, motion, base_world, block=64):
    """Image (S x S, A-frame layout) whose pixel (y, x) is captured at time T[y, x].

    Each layer is a plane Z = const facing the camera; for each pixel we cast
    the ray, hit the plane, convert the hit point to the layer's own texture
    coordinates (objects can slide and spin), sample colour + alpha and paint
    the layers far -> near. Geometry is computed in blocks of rows to keep
    memory low (many worker processes run at once)."""
    c0 = (S - 1) / 2.0
    maps = [(np.empty((S, S), np.float32), np.empty((S, S), np.float32)) for _ in layers]
    for y0 in range(0, S, block):
        y1 = min(S, y0 + block)
        ys, xs = np.mgrid[y0:y1, 0:S].astype(np.float64)
        Tb = T[y0:y1]
        c, d = _ray_pose(xs, ys, Tb, f, S, motion, base_world)
        for L, (mx, my) in zip(layers, maps):
            Z = L["Z"]
            lam = (Z - c[..., 2]) / d[..., 2]
            px = (c[..., 0] + lam * d[..., 0]) * f / Z        # hit point, in "A at t=0" pixels
            py = (c[..., 1] + lam * d[..., 1]) * f / Z
            if L["kind"] == "bg":
                mx[y0:y1], my[y0:y1] = px + L["cP"], py + L["cP"]
                continue
            dx = px - (L["cx"] - c0 + L["ux"] * Tb)           # minus object centre at time T
            dy = py - (L["cy"] - c0 + L["uy"] * Tb)
            phi = L["phi0"] + L["spin"] * Tb
            cs, sn = np.cos(phi), np.sin(phi)
            mx[y0:y1] = cs * dx + sn * dy + L["qc"]            # rotate by -phi
            my[y0:y1] = -sn * dx + cs * dy + L["qc"]

    out = None
    for L, (mx, my) in zip(layers, maps):
        if L["kind"] == "bg":
            out = cv2.remap(L["tex"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            continue
        col = cv2.remap(L["tex"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        al = cv2.remap(L["alpha"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        al = al[..., None]
        out = out * (1 - al) + col * al
    return out


def render_cam(T, f, S, layers, motion, base_world, exposure_s, n_sub):
    taus = [0.0] if exposure_s <= 0 else np.linspace(-exposure_s / 2, exposure_s / 2, n_sub)
    acc = None
    for tau in taus:
        img = render_layers(T + tau, f, S, layers, motion, base_world)
        acc = img if acc is None else acc + img
    return acc / len(taus)


def bg_coords_ok(t, f, S, P, motion, base_world, Z):
    """Does the background texture (P x P) cover everything this camera sees
    at time t? At a fixed time the mapping is a homography, so checking the
    image border is enough."""
    e = np.arange(S, dtype=np.float64)
    lo, hi = np.zeros(S), np.full(S, S - 1.0)
    xs = np.concatenate([e, e, lo, hi])
    ys = np.concatenate([lo, hi, e, e])
    c, d = _ray_pose(xs, ys, np.full(xs.shape, t), f, S, motion, base_world)
    lam = (Z - c[..., 2]) / d[..., 2]
    cP = (P - 1) / 2.0
    for k in (0, 1):
        m = (c[..., k] + lam * d[..., k]) * f / Z + cP
        if m.min() < 0 or m.max() > P - 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# One sample
# --------------------------------------------------------------------------- #
def make_sample(rng, cfg: LayerConfig, bg_pool, obj_pool, max_tries=10):
    S, ss = cfg.size, cfg.supersample
    S2 = S * ss
    U = lambda r: float(rng.uniform(*r)) if r[0] != r[1] else float(r[0])

    # camera, readout, shake, baseline, photometry: reuse synth_rs sampling
    keys = set(SynthConfig.__dataclass_fields__)
    scfg = SynthConfig(**{k: v for k, v in asdict(cfg).items() if k in keys})
    Z_bg = 1.0
    M2 = int(round(cfg.margin_frac * S2))
    P2 = S2 + 2 * M2
    for _ in range(max_tries):
        p = sample_params(rng, scfg)
        f = p["f"]                                            # focal length at output size
        f2 = f * ss
        mot = p["motion"]
        motion = {"w": mot.w, "a": mot.a, "v": mot.v * Z_bg}  # translation: trans px at background
        b_world = p["baseline_px"] * Z_bg / f                 # background disparity = baseline_px
        T_a = readout_times(S2, p["readout_a_s"], 0.0, cfg.b_rot_k, is_b=False)
        T_b = readout_times(S2, p["readout_b_s"], p["sync_offset_s"], cfg.b_rot_k, is_b=True)
        lo_t = min(T_a.min(), T_b.min()) - p["exposure_s"] / 2
        hi_t = max(T_a.max(), T_b.max()) + p["exposure_s"] / 2
        if all(bg_coords_ok(t, f2, S2, P2, motion, bw, Z_bg)
               for t in (lo_t, hi_t) for bw in (0.0, b_world)):
            break
    else:
        return None

    layers = [dict(kind="bg", Z=Z_bg, tex=get_texture(rng, bg_pool, P2), cP=(P2 - 1) / 2.0)]
    objs = []
    Tr = p["readout_a_s"]
    for _ in range(int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1))):
        r = U(cfg.obj_radius_frac) * S2
        Q = int(2 * r + 8)
        Z = U(cfg.obj_depth_frac) * Z_bg
        moving = rng.random() >= cfg.p_static_obj
        sp = U(cfg.obj_speed_px) * ss / Tr if moving else 0.0           # px/s at supersampled size
        th = rng.uniform(0, 2 * np.pi)
        spin = U(cfg.obj_spin_rad) / Tr * rng.choice([-1, 1]) if moving else 0.0
        cx, cy = rng.uniform(0.1, 0.9) * S2, rng.uniform(0.1, 0.9) * S2
        # the object must stand out from the background behind it (else it is invisible)
        bx, by = int(cx - (S2 - 1) / 2 + layers[0]["cP"]), int(cy - (S2 - 1) / 2 + layers[0]["cP"])
        ri = int(r)
        under = layers[0]["tex"][max(0, by - ri):by + ri + 1, max(0, bx - ri):bx + ri + 1]
        best, best_d = None, -1.0
        for _ in range(cfg.contrast_tries):
            cand = get_object(rng, obj_pool, Q, r, cfg)
            dist = colour_distance(cand[0], cand[1], under)
            if dist > best_d:
                best, best_d = cand, dist
            if dist >= cfg.min_colour_dist:
                break
        tex, alpha, source = best
        if source.startswith("cutout"):                        # people / cars do not spin fast
            lim = cfg.cutout_max_spin_rad / Tr
            spin = float(np.clip(spin, -lim, lim))
        objs.append(dict(kind="obj", Z=Z, tex=tex, alpha=alpha, qc=(Q - 1) / 2.0,
                         cx=cx, cy=cy, ux=sp * math.cos(th), uy=sp * math.sin(th),
                         phi0=0.0, spin=spin, r=r, colour_dist=best_d, source=source))
    layers += sorted(objs, key=lambda L: -L["Z"])              # paint far -> near

    rs_a = render_cam(T_a, f2, S2, layers, motion, 0.0, p["exposure_s"], cfg.exposure_sub)
    rs_b = render_cam(T_b, f2, S2, layers, motion, b_world, p["exposure_s"], cfg.exposure_sub)
    zero = np.zeros((S2, S2))
    gs_a = render_layers(zero, f2, S2, layers, motion, 0.0)
    gs_b = render_layers(zero, f2, S2, layers, motion, b_world)

    shrink = lambda im: cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA) if ss > 1 else im
    rs_a, rs_b, gs_a, gs_b = map(shrink, (rs_a, rs_b, gs_a, gs_b))

    def photo(img, g, sigma):
        img = img * g
        if sigma > 0:
            img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)
    rs_a = photo(rs_a, p["gain_a"], p["noise_a"])
    rs_b = photo(rs_b, p["gain_b"], p["noise_b"])
    gs_a = photo(gs_a, p["gain_a"], 0)
    gs_b = photo(gs_b, p["gain_b"], 0)

    k = cfg.b_rot_k
    out = dict(rs_a=rs_a, rs_b=np.rot90(rs_b, k), gs_a=gs_a, gs_b=np.rot90(gs_b, k))
    meta = dict(
        kind="layers", size=S, b_rot_k=k, f=float(f), f_is_estimate=False,
        readout_a_s=p["readout_a_s"], readout_b_s=p["readout_b_s"],
        sync_offset_s=p["sync_offset_s"], exposure_s=p["exposure_s"],
        baseline_px=p["baseline_px"],                           # disparity of the background
        gain_a=p["gain_a"], gain_b=p["gain_b"], noise_a=p["noise_a"], noise_b=p["noise_b"],
        # camera motion (NOT named w_rad_s etc. on purpose: the scene is not a single plane,
        # so synth_rs.flow_from_meta must not be used on these samples)
        cam_w_rad_s=mot.w.tolist(), cam_a_rad_s2=mot.a.tolist(), cam_v_s=motion["v"].tolist(),
        depth_bg=Z_bg,
        objects=[dict(depth=L["Z"], disparity_px=p["baseline_px"] * Z_bg / L["Z"],
                      radius_px=L["r"] / ss, centre_px=[L["cx"] / ss, L["cy"] / ss],
                      speed_px_per_readout=math.hypot(L["ux"], L["uy"]) * Tr / ss,
                      spin_rad_per_readout=L["spin"] * Tr,
                      colour_dist=L["colour_dist"], source=L["source"]) for L in objs],
        ref_time="middle of camera A readout")
    return out, meta


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _job(args):
    idx, cfg_dict, pools, out_dir, seed, val_frac, proc_frac = args
    cfg = LayerConfig(**cfg_dict)
    rng = np.random.default_rng([seed, idx])
    split = "val" if rng.random() < val_frac else "train"
    if (Path(out_dir) / split / f"{idx:06d}" / "meta.json").exists():
        return 1                                              # already made (resumed run)
    procedural = rng.random() < proc_frac                     # this sample: shapes only, no photos
    bg, obj = ([], []) if procedural else (pools[split]["bg"], pools[split]["obj"])
    res = make_sample(rng, cfg, bg, obj)
    if res is None:
        return 0
    out, meta = res
    meta["textures"] = "photos" if bg else "procedural"
    save_sample(Path(out_dir) / split / f"{idx:06d}", out, meta, save_corr=False)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder of photos used for backgrounds AND objects (recursive); "
                                  "without it, procedural textures are used")
    ap.add_argument("--bg-src", help="folder of background photos (overrides --src for backgrounds)")
    ap.add_argument("--obj-src", help="folder of foreground objects (overrides --src for objects); "
                                      "PNGs with transparency (make_cutouts.py) are used as real "
                                      "cut-outs, normal photos get a random shape")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="number of samples")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)),
                    help="parallel processes (each needs ~0.3 GB RAM)")
    ap.add_argument("--config", help="JSON file overriding LayerConfig fields")
    ap.add_argument("--procedural-frac", type=float, default=0.0,
                    help="with --src: fraction of samples made from procedural shapes "
                         "instead of photos, e.g. 0.2")
    a = ap.parse_args()

    cfg = LayerConfig(size=a.size)
    if a.config:
        for k, v in json.loads(Path(a.config).read_text()).items():
            if k in LayerConfig.__dataclass_fields__:
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg_dict = asdict(cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        dict(cfg_dict, src=a.src, bg_src=a.bg_src, obj_src=a.obj_src,
             procedural_frac=a.procedural_frac, n=a.n, seed=a.seed), indent=1))

    def list_images(folder):
        if not folder:
            return []
        found = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not found:
            ap.error(f"no images found in {folder}")
        return found

    bg_paths = list_images(a.bg_src or a.src)
    obj_paths = list_images(a.obj_src or a.src)
    if obj_paths and not bg_paths:
        ap.error("--obj-src needs backgrounds too: give --bg-src (or --src)")
    # every photo / cut-out goes to train OR val, never both
    pools = {}
    for split in ("train", "val"):
        pools[split] = {k: [p for p in v if split_of(p, a.val_frac) == split]
                        for k, v in (("bg", bg_paths), ("obj", obj_paths))}
    for k in ("bg", "obj"):
        if pools["train"][k] and not pools["val"][k]:
            pools["val"][k] = pools["train"][k]
            print(f"WARNING: too few {k} images for a separate val pool; val uses train images")
    print(f"backgrounds: {len(bg_paths) or 'procedural'} | objects: {len(obj_paths) or 'procedural'}")

    jobs = [(i, cfg_dict, pools, a.out, a.seed, a.val_frac, a.procedural_frac) for i in range(a.n)]
    total = 0
    if a.workers > 1 and a.n > 1:
        with Pool(a.workers, maxtasksperchild=200) as pool:
            for k, n in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
                total += n
                if (k + 1) % 200 == 0:
                    print(f"{k + 1}/{a.n} samples", flush=True)
    else:
        for job in jobs:
            total += _job(job)
    print(f"done: {total} samples -> {out}")


if __name__ == "__main__":
    main()
