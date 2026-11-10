"""Simple synthetic RS scenes with exact timing: a textured disc (or a fan) moving
over a textured background with a grid. Used for the demo and as a sanity test
of RAFT + geometry: the scene is rendered in continuous time, so there is no
frame interpolation, and the exact answer (GS image) is known.

Same conventions and output format as synth_video.py (a sample folder with
rs_a.png, rs_b.png rotated as camera B saves it, gs_a.png, gs_b.png, meta.json):
  camera A reads rows:     t_A(y) = (y/(S-1) - 0.5) * T
  camera B reads columns:  t_B(x) = sync + ((S-1-x)/(S-1) - 0.5) * T     (b_rot_k = 1)
  B sees the scene shifted left by the baseline b; GS = the scene at t = 0.

  python src/synth_shapes.py --out examples/kolo                      # disc moving right, 2 px/ms
  python src/synth_shapes.py --out examples/kolo_dol --vx 0 --vy 2
  python src/synth_shapes.py --out examples/wiatrak --fan --rpm 100
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from synth_rs import save_sample


def background_texture(rng, n=640):
    """Smooth colours + a darker grid every 32 px (RAFT needs texture; a bare grid is periodic)."""
    small = rng.uniform(0.25, 0.95, (n // 32, n // 32, 3)).astype(np.float32)
    tex = cv2.resize(small, (n, n), interpolation=cv2.INTER_CUBIC)
    tex[(np.arange(n) % 32) < 2, :] *= 0.35
    tex[:, (np.arange(n) % 32) < 2] *= 0.35
    return np.clip(cv2.GaussianBlur(tex, (0, 0), 0.8), 0, 1)


def checker_texture(n=320, cell=12):
    yy, xx = np.mgrid[0:n, 0:n]
    chk = ((yy // cell + xx // cell) % 2).astype(np.float32)[..., None]
    c1, c2 = np.array([0.95, 0.85, 0.2], np.float32), np.array([0.8, 0.15, 0.15], np.float32)   # BGR
    return np.clip(cv2.GaussianBlur(chk * c1 + (1 - chk) * c2, (0, 0), 0.8), 0, 1)


def disc_mask(radius):
    return lambda mx, my: np.hypot(mx, my) < radius


def fan_mask(n_blades=3, r_in=14, r_out=105, width_deg=28):
    def f(mx, my):
        r = np.hypot(mx, my)
        ang = np.degrees(np.arctan2(my, mx)) % (360 / n_blades)
        a = np.minimum(ang, 360 / n_blades - ang)
        half = width_deg / 2 * (1.2 - 0.6 * np.clip((r - r_in) / (r_out - r_in), 0, 1))
        return ((a < half) & (r < r_out)) | (r < r_in)
    return f


class Scene:
    """Background moving with v_bg [px/ms] (camera motion); object: centre c0 + v_obj * t,
    rotation omega * t [rad/ms] about its centre."""

    def __init__(self, bg, tex, mask, c0, v_obj=(0, 0), omega=0.0, v_bg=(0, 0)):
        self.bg, self.tex, self.mask = bg, tex, mask
        self.c0, self.v_obj = np.asarray(c0, float), np.asarray(v_obj, float)
        self.omega, self.v_bg = omega, np.asarray(v_bg, float)

    def render(self, t, shift_x=0.0):
        """Image whose pixel (y, x) shows the scene at time t[y, x]; shift_x = camera B's baseline."""
        S = t.shape[0]
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
        xs = xs + shift_x
        cx, cy = self.c0[0] + self.v_obj[0] * t, self.c0[1] + self.v_obj[1] * t
        a = -self.omega * t
        dx, dy = xs - cx, ys - cy
        mx, my = np.cos(a) * dx - np.sin(a) * dy, np.sin(a) * dx + np.cos(a) * dy
        inside = self.mask(mx, my)
        ob = (self.bg.shape[0] - S) / 2
        back = cv2.remap(self.bg, (xs - self.v_bg[0] * t + ob).astype(np.float32),
                         (ys - self.v_bg[1] * t + ob).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
        oo = self.tex.shape[0] / 2
        fg = cv2.remap(self.tex, (mx + oo).astype(np.float32), (my + oo).astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
        return np.where(inside[..., None], fg, back).astype(np.float32)


def make_sample(scene, S=256, T=20.0, sync=0.0, base=3.0, rot_k=1):
    """RS pair + GS targets of one scene, in the dataset format (rs_b / gs_b rotated as camera B saves them)."""
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float64)
    tA = (ys / (S - 1) - 0.5) * T
    tB = sync + ((S - 1 - xs) / (S - 1) - 0.5) * T
    zero = np.zeros((S, S))
    out = dict(rs_a=scene.render(tA), rs_b=np.rot90(scene.render(tB, base), rot_k),
               gs_a=scene.render(zero), gs_b=np.rot90(scene.render(zero, base), rot_k))
    meta = dict(kind="shapes", size=S, b_rot_k=rot_k, readout_a_s=T / 1000, readout_b_s=T / 1000,
                sync_offset_s=sync / 1000, baseline_px=base, ref_time="middle of camera A readout",
                v_obj_px_per_ms=scene.v_obj.tolist(), omega_rad_per_ms=scene.omega, v_bg_px_per_ms=scene.v_bg.tolist())
    return out, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="sample folder")
    ap.add_argument("--vx", type=float, default=2.0, help="object velocity [px/ms]")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=50)
    ap.add_argument("--fan", action="store_true", help="3-blade fan instead of a disc")
    ap.add_argument("--rpm", type=float, default=100, help="fan speed (with --fan)")
    ap.add_argument("--cam-vx", type=float, default=0.0, help="background (camera) velocity [px/ms]")
    ap.add_argument("--cam-vy", type=float, default=0.0)
    ap.add_argument("--readout-ms", type=float, default=20.0)
    ap.add_argument("--sync-ms", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bg = background_texture(rng)
    if a.fan:
        scene = Scene(bg, checker_texture(), fan_mask(), (128, 128), omega=a.rpm * 2 * np.pi / 60000,
                      v_bg=(a.cam_vx, a.cam_vy))
    else:
        scene = Scene(bg, checker_texture(), disc_mask(a.radius), (128, 128), v_obj=(a.vx, a.vy),
                      v_bg=(a.cam_vx, a.cam_vy))
    out, meta = make_sample(scene, T=a.readout_ms, sync=a.sync_ms, base=a.baseline)
    meta["command_args"] = vars(a)
    save_sample(Path(a.out), out, meta, save_corr=False)
    print("saved", a.out, json.dumps({k: meta[k] for k in ("readout_a_s", "sync_offset_s", "baseline_px")}))


if __name__ == "__main__":
    main()
