"""
Dual rolling-shutter dataset from a self-made "layered" scene (no video, no Blender).

What it does, in plain words
----------------------------
We build a small fake 3D scene ourselves:
  * a background photo far away (a street, a room, a park ...),
  * a few real objects cut out of other photos (people, cars, dogs, balls ...)
    standing or flying in front of it at different distances; each can move on
    its own (a passing car, a thrown ball),
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

Realism rules for real cut-outs (folder name = COCO category, see OBJECTS):
  * things that stand on the ground (people, cars, animals) are placed with
    perspective: the bigger on screen, the nearer, the lower in the frame,
    using their real height (a person is 1.7 m, a bus 3.2 m); they move
    sideways and do not spin; a soft shadow sits under them,
  * the horizon (where the ground is) comes from the people and cars already
    in the background photo when coco_backgrounds.py wrote scene_info.json
    next to it, so inserted things match the photo's own perspective,
  * indoor backgrounds (from the captions) only get indoor things (no cars),
  * flying / thrown things (birds, balls, kites) can be anywhere,
  * colours of a cut-out are pulled towards the scene's colours (lighting),
  * a cut-out must stand out from the background behind it (contrast check),
  * no object is used twice in one sample.

Outputs: same folder format as synth_rs.py / synth_video.py
(rs_a.png, rs_b.png rotated as camera B outputs it, gs_a.png, gs_b.png,
meta.json with kind="layers"), so preview.py and train_raft.py work unchanged.
There is no dense ground-truth flow (objects move on their own), so it is
meant for image-supervised models, like the video set.

  # realistic: scene backgrounds + real objects from COCO (coco_backgrounds.py, make_cutouts.py)
  python src/synth_layers.py --bg-src coco/backgrounds --obj-src coco/cutouts --out rs_layers_3 \
      --n 6000 --config configs/config_layers_real.json
  python src/synth_layers.py --src photos --out rs_layers --n 6000     # any photos, random shapes
  python src/synth_layers.py --out demo_layers --n 16                   # procedural, no photos needed
  python src/gallery.py rs_layers_3                                     # browse everything
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

# Real objects (COCO category = folder name of the cut-outs) and how they behave:
#   group  "ground": stands on the ground, moves sideways, does not spin
#          "free":   anywhere in the frame, any direction, may spin
#   height: typical real height in metres -> believable distance for its size on screen
#   speed:  own speed relative to a walking person (0 = never moves: a hydrant, a bench)
#   weight: how often the category is picked (0 = never: things that would look silly
#           floating in the air, such as fruit or cups, even if the folder exists)
#   indoor: may appear in an indoor scene (a car in a kitchen is not realistic)
OBJECTS = {
    # name            group     height speed weight indoor
    "person":        ("ground", 1.70, 1.0, 12, True),
    "car":           ("ground", 1.50, 1.3, 5, False),
    "bicycle":       ("ground", 1.00, 1.3, 2, True),
    "motorcycle":    ("ground", 1.10, 1.3, 2, False),
    "bus":           ("ground", 3.20, 1.2, 2, False),
    "truck":         ("ground", 2.80, 1.2, 2, False),
    "train":         ("ground", 4.00, 1.3, 0.5, False),
    "boat":          ("ground", 2.00, 0.8, 0.5, False),
    "dog":           ("ground", 0.60, 1.2, 3, True),
    "cat":           ("ground", 0.30, 1.0, 1, True),
    "horse":         ("ground", 1.60, 1.3, 1, False),
    "sheep":         ("ground", 0.90, 0.8, 0.5, False),
    "cow":           ("ground", 1.40, 0.7, 0.5, False),
    "elephant":      ("ground", 3.00, 0.7, 0.3, False),
    "bear":          ("ground", 1.50, 1.0, 0.3, False),
    "zebra":         ("ground", 1.40, 1.2, 0.3, False),
    "giraffe":       ("ground", 4.50, 1.0, 0.3, False),
    "fire hydrant":  ("ground", 0.80, 0.0, 0.5, False),
    "stop sign":     ("ground", 0.80, 0.0, 0.5, False),
    "parking meter": ("ground", 1.30, 0.0, 0.5, False),
    "bench":         ("ground", 0.90, 0.0, 0.5, False),
    "chair":         ("ground", 0.90, 0.0, 0.5, True),
    "potted plant":  ("ground", 0.70, 0.0, 0.5, True),
    "suitcase":      ("ground", 0.70, 0.5, 0.5, True),
    "skateboard":    ("ground", 0.15, 1.3, 0.5, True),
    "bird":          ("free",   0.30, 1.3, 2, False),
    "sports ball":   ("free",   0.25, 1.3, 2, True),
    "frisbee":       ("free",   0.25, 1.3, 1, True),
    "kite":          ("free",   1.00, 1.0, 1, False),
    "umbrella":      ("free",   1.00, 0.7, 0.5, False),
    "airplane":      ("free",   3.00, 1.3, 0.3, False),
    # never used: hand-held things and food look silly floating in a scene
    "backpack":      ("free",   0.50, 0.7, 0, True), "handbag":       ("free", 0.35, 0.7, 0, True),
    "bottle":        ("free",   0.25, 0.7, 0, True), "cup":           ("free", 0.10, 0.5, 0, True),
    "teddy bear":    ("free",   0.40, 0.7, 0, True), "apple":         ("free", 0.08, 1.3, 0, True),
    "orange":        ("free",   0.08, 1.3, 0, True), "banana":        ("free", 0.20, 1.0, 0, True),
    "tennis racket": ("free",   0.70, 1.3, 0, True), "baseball bat":  ("free", 0.80, 1.3, 0, True),
    "cell phone":    ("free",   0.15, 0.7, 0, True), "book":          ("free", 0.25, 0.7, 0, True),
    "surfboard":     ("free",   0.50, 1.0, 0, False),
}
UNKNOWN_OBJECT = ("free", 0.50, 1.0, 1, True)  # cut-outs from folders not listed above (your own PNGs)
SCENE_INFO = "scene_info.json"                 # per-background horizon / indoor flag (coco_backgrounds.py)
SCENE_CAM_HEIGHT_M = 1.5                       # camera height assumed when that horizon was estimated


@dataclass
class LayerConfig:
    size: int = 256
    supersample: int = 2                 # render at 2x and shrink (anti-aliasing); 1 for 640-px COCO photos
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
    n_objects: tuple = (1, 3)            # how many objects in front of the background
    obj_speed_px: tuple = (0.0, 40.0)    # own motion of an object during A's readout, px (x category speed)
    obj_spin_rad: tuple = (0.0, 0.6)     # own rotation during A's readout, rad ("free" objects / shapes)
    p_static_obj: float = 0.3            # chance that a movable object stands still
    obj_depth_frac: tuple = (0.25, 0.9)  # object distance / background distance (smaller = nearer)
    # real cut-outs
    obj_height_frac: tuple = (0.15, 0.7) # height on screen / image size ("ground" objects)
    free_size_mult: float = 0.6          # "free" objects are this much smaller than the range above
    bg_dist_m: tuple = (4.0, 25.0)       # how far the background is (metres), log-uniform
    cam_height_m: tuple = (0.8, 1.8)     # camera height above the ground (metres)
    horizon_frac: tuple = (0.4, 0.6)     # horizon row / image size (camera looks about level)
    ground_cut_frac: float = 0.4         # feet may be below the frame by this fraction of the height
    ground_dir_jitter_deg: float = 20.0  # "ground" objects move sideways +- this angle
    shadow: float = 0.35                 # darkness of the contact shadow under "ground" objects (0 = none)
    colour_match: float = 0.4            # pull a cut-out's colours this far towards the scene's (0..1)
    cutout_max_upscale: float = 1.3      # prefer cut-outs that need less enlarging than this (blur)
    cutout_max_spin_rad: float = 0.15    # "free" cut-outs rotate at most this per readout
    # random shapes (ordinary photos / procedural textures as objects)
    obj_radius_frac: tuple = (0.08, 0.3) # shape size (radius) / image size
    # all objects
    min_colour_dist: float = 20.0        # object vs background behind it, mean-colour delta E (Lab)
    contrast_tries: int = 8              # objects tried to reach min_colour_dist


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


_scene_info_cache = {}


def scene_info_of(path):
    """Horizon / indoor info written by coco_backgrounds.py next to the photo, or None."""
    folder = Path(path).parent
    if folder not in _scene_info_cache:
        f = folder / SCENE_INFO
        try:
            _scene_info_cache[folder] = json.loads(f.read_text()) if f.exists() else {}
        except ValueError:
            _scene_info_cache[folder] = {}
    return _scene_info_cache[folder].get(Path(path).name)


def random_patch_geom(img, P, rng):
    """random_patch() that also tells where the crop came from: (patch, scale_y, y0)."""
    h, w = img.shape[:2]
    s_min = P / min(h, w)
    s = rng.uniform(s_min, max(s_min, 1.0)) if s_min < 1 else s_min
    if abs(s - 1) > 1e-3:
        img = cv2.resize(img, (max(P, round(w * s)), max(P, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h2, w2 = img.shape[:2]
    y0 = int(rng.integers(0, h2 - P + 1))
    x0 = int(rng.integers(0, w2 - P + 1))
    patch = img[y0:y0 + P, x0:x0 + P]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    return np.ascontiguousarray(patch), h2 / h, y0


def get_background(rng, pool, P):
    """Background texture (P x P, float) and, when the photo came with scene info,
    the horizon row inside the crop and the indoor flag:
    (tex, horizon_crop_px, indoor, photo file name)."""
    if pool:
        for _ in range(5):
            path = pool[int(rng.integers(len(pool)))]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            patch, sy, y0 = random_patch_geom(img, P, rng)
            info = scene_info_of(path) or {}
            hz = info.get("horizon_frac")
            horizon = hz * img.shape[0] * sy - y0 if hz is not None else None
            return (patch.astype(np.float32) / 255.0, horizon, bool(info.get("indoor", False)),
                    Path(path).name)
    return procedural_texture(rng, P), None, False, None


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
# Real cut-outs
# --------------------------------------------------------------------------- #
def load_cutout(path):
    """PNG with transparency -> float32 RGBA in [0, 1], or None for an ordinary photo."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    return img.astype(np.float32) / 255.0


def place_cutout(rgba, height_px, flip, pad_frac=0.35):
    """Scale a cut-out to `height_px` on screen and put it on a square canvas with
    room around it for a shadow. Returns (colour, alpha, foot_row, width_px, upscale)."""
    rows = np.where(rgba[..., 3].max(1) > 0.5)[0]
    cols = np.where(rgba[..., 3].max(0) > 0.5)[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    rgba = rgba[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]           # tight box
    h, w = rgba.shape[:2]
    s = height_px / h
    nw, nh = max(2, round(w * s)), max(2, round(h * s))
    rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if flip:
        rgba = rgba[:, ::-1]
    Q = int(max(nw, nh) * (1 + pad_frac)) + 4
    canvas = np.zeros((Q, Q, 4), np.float32)
    y0, x0 = (Q - nh) // 2, (Q - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rgba
    return canvas[..., :3].copy(), np.clip(canvas[..., 3], 0, 1).copy(), y0 + nh - 1, nw, s


def contact_shadow(Q, foot, width, strength):
    """Soft dark ellipse on the ground under the feet (Q x Q map, values 0..strength)."""
    if strength <= 0:
        return None
    m = np.zeros((Q, Q), np.float32)
    ax, ay = max(2, int(0.45 * width)), max(1, int(0.07 * width))
    cv2.ellipse(m, (int(round((Q - 1) / 2)), int(foot)), (ax, ay), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    k = max(3, int(0.12 * width) | 1)
    return cv2.GaussianBlur(m, (k, k), 0) * strength


def match_colour(tex, alpha, scene_tex, amount):
    """Pull the cut-out's mean colour and brightness (Lab) `amount` of the way
    towards the scene's, so it looks lit like the scene (0 = unchanged)."""
    m = alpha > 0.5
    if amount <= 0 or m.sum() < 16:
        return tex
    lab = cv2.cvtColor(tex, cv2.COLOR_BGR2Lab)
    target = cv2.cvtColor(scene_tex, cv2.COLOR_BGR2Lab).reshape(-1, 3).mean(0)
    lab = lab + amount * (target - lab[m].mean(0))
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR), 0, 1)


def colour_distance(tex_a, mask_a, tex_b):
    """How different two textures look: distance of their mean colours in Lab
    (CIE76 delta E, ~2 = barely visible, >25 = clearly different colours)."""
    la = cv2.cvtColor(tex_a, cv2.COLOR_BGR2Lab)[mask_a > 0.5]
    lb = cv2.cvtColor(tex_b, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    if len(la) == 0 or len(lb) == 0:
        return 1e9
    return float(np.linalg.norm(la.mean(0) - lb.mean(0)))


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
    the ray, hit the plane, convert the hit point to the layer's own canvas
    coordinates (objects can slide and spin), sample colour + alpha and paint
    the layers far -> near. Object colour is stored premultiplied by alpha, so
    the edges do not pick up the colour outside the cut-out. Geometry is
    computed in blocks of rows to keep memory low (many workers run at once)."""
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
        if L.get("shade") is not None:                        # contact shadow on what is behind
            sh = cv2.remap(L["shade"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            out = out * (1 - sh[..., None])
        col = cv2.remap(L["tex"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        al = cv2.remap(L["alpha"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out = out * (1 - al[..., None]) + col                 # col is premultiplied
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
# Objects
# --------------------------------------------------------------------------- #
def _U(rng, r):
    return float(rng.uniform(*r)) if r[0] != r[1] else float(r[0])


def pick_category(rng, obj_pool, indoor=False):
    cats = [c for c, v in obj_pool.items() if v and OBJECTS.get(c, UNKNOWN_OBJECT)[3] > 0
            and (not indoor or OBJECTS.get(c, UNKNOWN_OBJECT)[4])]
    if not cats:
        cats = [c for c, v in obj_pool.items() if v and OBJECTS.get(c, UNKNOWN_OBJECT)[3] > 0]
    if not cats:
        return None
    w = np.array([OBJECTS.get(c, UNKNOWN_OBJECT)[3] for c in cats], float)
    return cats[int(rng.choice(len(cats), p=w / w.sum()))]


def ground_placement(rng, cfg, scene, height_m, S2):
    """Where a thing of real height `height_m` standing on the ground can be:
    (height on screen px, depth in background units, foot row) or None.
    Bigger on screen = nearer = feet lower in the frame (simple perspective)."""
    f2, D, h_cam, y_h = scene["f2"], scene["bg_dist_m"], scene["cam_height_m"], scene["horizon_px"]
    for _ in range(12):
        A = _U(rng, cfg.obj_height_frac) * S2
        Z = f2 * height_m / A                                     # distance in metres
        if not (cfg.obj_depth_frac[0] * D <= Z <= cfg.obj_depth_frac[1] * D):
            continue
        foot = y_h + f2 * h_cam / Z
        if foot > S2 - 1 + cfg.ground_cut_frac * A or foot - A < -0.3 * A:
            continue
        return A, Z / D, foot
    return None


def make_object(rng, cfg, scene, bg_tex, obj_pool, used, S2, ss, Tr):
    """One foreground layer (or None). Tries several objects and keeps the one
    that stands out best from the background behind it."""
    f2 = scene["f2"]
    cP = (bg_tex.shape[0] - 1) / 2.0
    best, best_d = None, -1.0
    for _ in range(cfg.contrast_tries):
        cat = pick_category(rng, obj_pool, scene["indoor"])
        group, height_m, speed_f = OBJECTS.get(cat, UNKNOWN_OBJECT)[:3]
        path, rgba, img = None, None, None
        if cat is not None:
            paths = [p for p in obj_pool[cat] if p not in used] or obj_pool[cat]
            path = paths[int(rng.integers(len(paths)))]
            rgba = load_cutout(path)
            if rgba is None:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

        if rgba is not None:                                     # real cut-out
            if group == "ground":
                geo = ground_placement(rng, cfg, scene, height_m, S2)
                if geo is None:
                    continue
                A, z, foot = geo
            else:
                A = _U(rng, cfg.obj_height_frac) * S2 * cfg.free_size_mult
                z, foot = _U(rng, cfg.obj_depth_frac), None
            placed = place_cutout(rgba, A, rng.random() < 0.5)
            if placed is None:
                continue
            tex, alpha, foot_c, width, up = placed
            if up > cfg.cutout_max_upscale and rng.random() < 0.8:
                continue                                         # would be blurry: try another
            Q = tex.shape[0]
            qc = (Q - 1) / 2.0
            cx = rng.uniform(0.1, 0.9) * S2
            cy = foot - (foot_c - qc) if foot is not None else rng.uniform(0.1, 0.9) * S2
            tex = match_colour(tex, alpha, bg_tex, cfg.colour_match)
            shade = contact_shadow(Q, foot_c, width, cfg.shadow) if group == "ground" else None
            source, size_px = "cutout:" + cat, A
        else:                                                    # random shape, photo or procedural
            group, speed_f = "free", 1.0
            r = _U(rng, cfg.obj_radius_frac) * S2
            Q = int(2 * r + 8)
            qc = (Q - 1) / 2.0
            tex = (random_patch(img, Q, rng).astype(np.float32) / 255.0 if img is not None
                   else procedural_texture(rng, Q))
            alpha = random_shape(rng, Q, r)
            z = _U(rng, cfg.obj_depth_frac)
            cx, cy = rng.uniform(0.1, 0.9) * S2, rng.uniform(0.1, 0.9) * S2
            shade, source, size_px = None, ("photo" if img is not None else "procedural"), 2 * r

        # must stand out from the background behind it, else it is invisible
        bx, by = int(cx - (S2 - 1) / 2 + cP), int(cy - (S2 - 1) / 2 + cP)
        h = int(size_px / 2) + 1
        under = bg_tex[max(0, by - h):by + h + 1, max(0, bx - h):bx + h + 1]
        dist = colour_distance(tex, alpha, under)
        if dist > best_d:
            best, best_d = dict(tex=tex, alpha=alpha, shade=shade, Q=Q, qc=qc, cx=cx, cy=cy, z=z,
                                group=group, speed_f=speed_f, source=source, cat=cat,
                                path=path, size_px=size_px), dist
        if dist >= cfg.min_colour_dist:
            break
    if best is None:
        return None
    o = best
    if o["path"]:
        used.add(o["path"])

    # own motion during the readout
    moving = o["speed_f"] > 0 and rng.random() >= cfg.p_static_obj
    sp = _U(rng, cfg.obj_speed_px) * o["speed_f"] * ss / Tr if moving else 0.0   # px/s at S2
    if o["group"] == "ground":
        th = (0.0 if rng.random() < 0.5 else math.pi) + math.radians(
            rng.uniform(-cfg.ground_dir_jitter_deg, cfg.ground_dir_jitter_deg))
        spin = 0.0
    else:
        th = rng.uniform(0, 2 * math.pi)
        spin = _U(rng, cfg.obj_spin_rad) / Tr * rng.choice([-1, 1]) if moving else 0.0
        if o["source"].startswith("cutout"):                     # a bird or a ball barely rotates
            spin = float(np.clip(spin, -cfg.cutout_max_spin_rad / Tr, cfg.cutout_max_spin_rad / Tr))
    return dict(kind="obj", Z=o["z"], tex=o["tex"] * o["alpha"][..., None], alpha=o["alpha"],
                shade=o["shade"], qc=o["qc"], cx=o["cx"], cy=o["cy"],
                ux=sp * math.cos(th), uy=sp * math.sin(th), phi0=0.0, spin=spin,
                size_px=o["size_px"], colour_dist=best_d, source=o["source"],
                category=o["cat"] or "shape", group=o["group"], static=not moving)


# --------------------------------------------------------------------------- #
# One sample
# --------------------------------------------------------------------------- #
def make_sample(rng, cfg: LayerConfig, bg_pool, obj_pool, max_tries=10):
    S, ss = cfg.size, cfg.supersample
    S2 = S * ss

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

    bg_tex, horizon_crop, indoor, bg_name = get_background(rng, bg_pool, P2)
    layers = [dict(kind="bg", Z=Z_bg, tex=bg_tex, cP=(P2 - 1) / 2.0)]
    # the ground: horizon estimated from the people / cars in the background photo when
    # coco_backgrounds.py provided it (then the camera height it assumed), else random
    if horizon_crop is not None:
        horizon_px, cam_height_m, horizon_src = horizon_crop - M2, SCENE_CAM_HEIGHT_M, "photo"
    else:
        horizon_px, cam_height_m, horizon_src = _U(rng, cfg.horizon_frac) * S2, _U(rng, cfg.cam_height_m), "random"
    scene = dict(f2=f2, bg_dist_m=float(np.exp(rng.uniform(*np.log(cfg.bg_dist_m)))),
                 cam_height_m=cam_height_m, horizon_px=horizon_px, horizon_src=horizon_src,
                 indoor=indoor, background=bg_name)
    Tr = p["readout_a_s"]
    objs, used = [], set()
    for _ in range(int(rng.integers(cfg.n_objects[0], cfg.n_objects[1] + 1))):
        L = make_object(rng, cfg, scene, bg_tex, obj_pool, used, S2, ss, Tr)
        if L is not None:
            objs.append(L)
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
        scene=dict(bg_dist_m=scene["bg_dist_m"], cam_height_m=scene["cam_height_m"],
                   horizon_px=scene["horizon_px"] / ss, horizon_src=scene["horizon_src"],
                   indoor=scene["indoor"], background=scene["background"]),
        objects=[dict(category=L["category"], group=L["group"], source=L["source"],
                      depth=L["Z"], depth_m=L["Z"] * scene["bg_dist_m"],
                      disparity_px=p["baseline_px"] * Z_bg / L["Z"],
                      size_px=L["size_px"] / ss, centre_px=[L["cx"] / ss, L["cy"] / ss],
                      static=L["static"],
                      speed_px_per_readout=math.hypot(L["ux"], L["uy"]) * Tr / ss,
                      spin_rad_per_readout=L["spin"] * Tr,
                      colour_dist=L["colour_dist"]) for L in objs],
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
    done = Path(out_dir) / split / f"{idx:06d}" / "meta.json"
    if done.exists():                                         # already made (resumed run)?
        try:
            json.loads(done.read_text())
            return 1
        except (OSError, ValueError):
            pass                                              # cut off while saving: make it again
    procedural = rng.random() < proc_frac                     # this sample: shapes only, no photos
    bg, obj = ([], {}) if procedural else (pools[split]["bg"], pools[split]["obj"])
    res = make_sample(rng, cfg, bg, obj)
    if res is None:
        return 0
    out, meta = res
    meta["textures"] = "photos" if bg else "procedural"
    save_sample(Path(out_dir) / split / f"{idx:06d}", out, meta, save_corr=False)
    return 1


def list_images(folders):
    found = []
    for folder in folders or []:
        here = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not here:
            raise SystemExit(f"no images found in {folder}")
        found += here
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="folder of photos used for backgrounds AND objects (recursive); "
                                  "without it, procedural textures are used")
    ap.add_argument("--bg-src", action="append",
                    help="folder of background photos (repeatable; overrides --src for backgrounds)")
    ap.add_argument("--obj-src", action="append",
                    help="folder of foreground objects (repeatable; overrides --src for objects): "
                         "PNGs with transparency in <category>/ subfolders (make_cutouts.py) are "
                         "real cut-outs, ordinary photos get a random shape")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="number of samples")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)),
                    help="parallel processes (each needs ~0.3 GB RAM)")
    ap.add_argument("--config", help="JSON file overriding LayerConfig fields")
    ap.add_argument("--procedural-frac", type=float, default=0.0,
                    help="fraction of samples made from procedural shapes instead of photos")
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

    bg_paths = list_images(a.bg_src or ([a.src] if a.src else []))
    obj_paths = list_images(a.obj_src or ([a.src] if a.src else []))
    if obj_paths and not bg_paths:
        ap.error("--obj-src needs backgrounds too: give --bg-src (or --src)")
    # every photo / cut-out goes to train OR val, never both; objects grouped by category folder
    pools = {}
    for split in ("train", "val"):
        obj = {}
        for p in obj_paths:
            if split_of(p, a.val_frac) == split:
                obj.setdefault(Path(p).parent.name, []).append(p)
        pools[split] = {"bg": [p for p in bg_paths if split_of(p, a.val_frac) == split], "obj": obj}
    for k in ("bg", "obj"):
        if pools["train"][k] and not pools["val"][k]:
            pools["val"][k] = pools["train"][k]
            print(f"WARNING: too few {k} images for a separate val pool; val uses train images")
    cats = sorted(pools["train"]["obj"])
    print(f"backgrounds: {len(bg_paths) or 'procedural'} | objects: {len(obj_paths) or 'procedural'}"
          + (f" in {len(cats)} categories" if cats else ""))
    unknown = [c for c in cats if c not in OBJECTS]
    if unknown:
        print(f"note: object folders not in OBJECTS (treated as free-floating): {', '.join(unknown)}")
    off = [c for c in cats if OBJECTS.get(c, UNKNOWN_OBJECT)[3] == 0]
    if off:
        print(f"note: object folders with weight 0 in OBJECTS, never used: {', '.join(off)}")

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
