"""Dual rolling-shutter dataset from high-fps video (X4K1000FPS, slow-motion recordings).

Idea: a high-fps video is (almost) a global-shutter camera sampled very often.
Camera A reads ROWS one after another, so row y of the RS image is taken from
the video frame at the moment row y was read. Camera B (rotated 90 deg) reads
COLUMNS, so column x is taken from the frame at the moment column x was read.
The target (GS) is the whole frame at the middle of A's readout.
Camera B also looks from a point shifted sideways (stereo baseline): its crop
window is moved by `baseline_px` (converted to source pixels).

Same output format and time conventions as synth_rs.py (rs_a, rs_b rotated as
camera B outputs it, gs_a, gs_b, meta.json), so preview.py works unchanged.
There is no exact flow ground truth for real video (only the target image).

  python src/synth_video.py --src /content/x4k/train --out /content/dataset_video \
      --fps 1000 --per-clip 2 --size 256
  # X4K train subset: 4 clips from each of the first 60 scenes, val = 10% of SCENES
  python src/synth_video.py --src x4k/train --out rs_video --per-clip 3 --augment \
      --clips-per-scene 4 --max-scenes 60
  # only a few clips (e.g. X4K test set): choose the validation clips by name
  python src/synth_video.py --src x4k/test --out dataset_video --per-clip 40 --augment \
      --val-clips TEST03,TEST08,TEST13
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from synth_rs import readout_times, save_sample, split_of

DEFAULT_CFG = {
    "size": 256,
    "readout_ms": [10.0, 30.0],     # time to read all rows (A) / columns (B)
    "sync_offset_ms": [-2.0, 2.0],  # B's readout centre minus A's
    "baseline_px": [0.0, 8.0],      # B shifted sideways by this many px (in the output size)
    "crop_frac": [0.5, 1.0],        # crop side / short side of the frame (smaller = more zoom = larger motion in px)
    "rot_k": 1,                     # np.rot90 k that turns A-frame into B's picture
    "interp": "linear",             # "nearest": frame closest in time; "linear": blend of the 2 nearest
    "gain": [0.9, 1.1],
    "noise_std": [0.0, 0.01],
    "fov_deg": 55.0,                # only to store an estimated focal length in meta.json
}


# ------------------------------------------------------------------ reading
def read_clip(path, max_frames=None):
    """mp4 file or folder of PNG/JPG frames -> uint8 array (T, H, W, 3), BGR."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        frames = [cv2.imread(str(f)) for f in files[:max_frames]]
    else:
        cap, frames = cv2.VideoCapture(str(p)), []
        while max_frames is None or len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames)


def crop_stack(clip, x0, y0, side, size):
    """The same square crop of every frame, resized to size x size -> float32 (T,S,S,3) in [0,1]."""
    return np.stack([cv2.resize(fr[y0:y0 + side, x0:x0 + side], (size, size),
                                interpolation=cv2.INTER_AREA) for fr in clip]).astype(np.float32) / 255.0


# ------------------------------------------------------------------ the core
def assemble(stack, t_ms, fps, mode="linear"):
    """RS image: pixel (y, x) comes from the video frame at time t_ms[y, x]
    (ms from the clip start). For camera A t_ms is constant along each row,
    for camera B along each column."""
    n, S = stack.shape[0], stack.shape[1]
    fi = np.clip(t_ms * fps / 1000.0, 0, n - 1)          # fractional frame index per pixel
    yy, xx = np.mgrid[0:S, 0:S]
    if mode == "nearest":
        return stack[np.rint(fi).astype(int), yy, xx]
    i0 = np.floor(fi).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (fi - i0)[..., None]
    return (1 - w) * stack[i0, yy, xx] + w * stack[i1, yy, xx]


def augment(clip, rng):
    """Free variety for small sets: mirror the scene and/or play it backwards
    (both are physically valid videos; the RS simulation is done afterwards)."""
    if rng.random() < 0.5:
        clip = clip[:, :, ::-1]
    if rng.random() < 0.5:
        clip = clip[::-1]
    return clip


def make_sample(clip, fps, rng, cfg):
    n, H, W = clip.shape[:3]
    S = cfg["size"]
    U = lambda r: float(rng.uniform(*r))
    sync, base = U(cfg["sync_offset_ms"]), U(cfg["baseline_px"])
    # short clips (X4K test: 33 frames = 32 ms) cannot hold the longest readouts: cap the range
    lo, hi = cfg["readout_ms"]
    hi_fit = 2 * ((n - 1) * 1000.0 / fps / 2 - abs(sync) - 1000.0 / fps)
    if hi_fit < lo:
        return None
    readout = U((lo, min(hi, hi_fit)))

    # 1) the moment of the target image: an exact video frame, with room for the whole readout
    half = readout / 2 + abs(sync) + 1000.0 / fps
    k_lo, k_hi = int(np.ceil(half * fps / 1000)), int(np.floor((n - 1) - half * fps / 1000))
    if k_hi < k_lo:
        return None                                       # clip too short for this readout time
    k_c = int(rng.integers(k_lo, k_hi + 1))
    t_c = k_c * 1000.0 / fps

    # 2) crop window for A, and the same window moved sideways for B (baseline)
    side = int(U(cfg["crop_frac"]) * min(H, W))
    shift = int(round(base * side / S))                   # baseline in source pixels
    if side + shift > W:                                  # leave room for B's shifted window
        side = W - int(np.ceil(base * W / S))
        shift = int(round(base * side / S))
    x0, y0 = int(rng.integers(0, W - side - shift + 1)), int(rng.integers(0, H - side + 1))
    A = crop_stack(clip, x0, y0, side, S)
    B = crop_stack(clip, x0 + shift, y0, side, S)         # B sits to the right -> sees the scene shifted left

    # 3) readout time of every pixel (ms), same convention as synth_rs.readout_times
    tA = t_c + 1000 * readout_times(S, readout / 1000, 0.0, cfg["rot_k"], is_b=False)
    tB = t_c + 1000 * readout_times(S, readout / 1000, sync / 1000, cfg["rot_k"], is_b=True)
    rs_a = assemble(A, tA, fps, cfg["interp"])
    rs_b = assemble(B, tB, fps, cfg["interp"])
    gs_a, gs_b = A[k_c], B[k_c]                           # targets: whole frame at t_c

    # 4) photometry: per-camera gain (also on the target of that camera) and sensor noise
    def photo(img, g, sigma):
        img = img * g
        if sigma > 0:
            img = img + rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)
    ga, gb = U(cfg["gain"]), U(cfg["gain"])
    rs_a, rs_b = photo(rs_a, ga, U(cfg["noise_std"])), photo(rs_b, gb, U(cfg["noise_std"]))
    gs_a, gs_b = photo(gs_a, ga, 0), photo(gs_b, gb, 0)

    k = cfg["rot_k"]
    out = dict(rs_a=rs_a, rs_b=np.rot90(rs_b, k), gs_a=gs_a, gs_b=np.rot90(gs_b, k))
    meta = dict(kind="video", fps=fps, size=S, b_rot_k=k, interp=cfg["interp"],
                readout_a_s=readout / 1000, readout_b_s=readout / 1000, sync_offset_s=sync / 1000,
                baseline_px=base, baseline_src_px=shift, t_c_ms=t_c, frame_c=k_c,
                crop=[x0, y0, side], gain_a=ga, gain_b=gb,
                f=(S / 2) / np.tan(np.radians(cfg["fov_deg"]) / 2), f_is_estimate=True,
                ref_time="middle of camera A readout")
    return out, meta


# ------------------------------------------------------------------ driver
def list_clips(root):
    """Clips = mp4 files or folders that directly contain frames."""
    root = Path(root)
    clips = sorted(p for p in root.rglob("*.mp4"))
    clips += sorted({f.parent for f in root.rglob("*.png")})
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with clips (mp4 files or folders of frames)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1000.0, help="frame rate of the SOURCE video")
    ap.add_argument("--per-clip", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", help="JSON overriding DEFAULT_CFG")
    ap.add_argument("--val-clips", default=None,
                    help="comma-separated clip names (folder or file stem) that go to val; "
                         "default: random split by scene (parent folder)")
    ap.add_argument("--augment", action="store_true", help="random mirror / time reversal per sample")
    ap.add_argument("--clips-per-scene", type=int, default=None,
                    help="use only N clips from each scene (X4K train: ~40 similar clips per scene)")
    ap.add_argument("--max-scenes", type=int, default=None, help="use only the first N scenes")
    a = ap.parse_args()

    cfg = dict(DEFAULT_CFG, size=a.size)
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, fps=a.fps, src=str(a.src)), indent=1))

    clips = list_clips(a.src)
    if a.clips_per_scene or a.max_scenes:                # subset: evenly spread clips of each scene
        by_scene = {}
        for c in clips:
            by_scene.setdefault(c.parent, []).append(c)
        scenes = sorted(by_scene)[:a.max_scenes]
        k = a.clips_per_scene
        clips = [c for sc in scenes
                 for c in (by_scene[sc] if not k else by_scene[sc][::max(1, len(by_scene[sc]) // k)][:k])]
        print(f"subset: {len(scenes)} scenes, {len(clips)} clips")
    clips = clips[:a.max_clips]
    val_names = set(a.val_clips.split(",")) if a.val_clips else None
    rng = np.random.default_rng(a.seed)
    done, counts = 0, {"train": 0, "val": 0}
    for ci, clip_path in enumerate(clips):
        clip = read_clip(clip_path)
        name = clip_path.stem if clip_path.is_file() else clip_path.name
        if val_names is not None:
            split = "val" if name in val_names else "train"
        else:
            split = split_of(clip_path.parent.name, a.val_frac)   # X4K train/<scene>/<clip>: split by SCENE
        print(f"{name}: {clip.shape[0]} frames {clip.shape[2]}x{clip.shape[1]} -> {split}", flush=True)
        for j in range(a.per_clip):
            res = make_sample(augment(clip, rng) if a.augment else clip, a.fps, rng, cfg)
            if res is None:
                continue
            sample, meta = res
            meta["source"] = str(clip_path.relative_to(a.src))
            meta["augment"] = bool(a.augment)
            save_sample(out / split / f"{ci:05d}_{j:02d}", sample, meta, save_corr=False)
            done += 1
            counts[split] += 1
        if (ci + 1) % 50 == 0:
            print(f"{ci + 1}/{len(clips)} clips, {done} samples", flush=True)
    if val_names is not None:
        missing = val_names - {(c.stem if c.is_file() else c.name) for c in clips}
        if missing:
            print("WARNING: val clips not found:", ", ".join(sorted(missing)))
    print(f"done: {done} samples ({counts['train']} train / {counts['val']} val) from {len(clips)} clips -> {out}")


if __name__ == "__main__":
    main()
