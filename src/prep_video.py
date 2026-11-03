"""Long high-speed video (e.g. a 1000 fps slow-motion film from YouTube/Vimeo) -> short clips
in the same layout as X4K1000FPS, ready for synth_video.py:

    <out>/<video name>/<video name>_0000.mp4  + _0000.json   (one folder = one "scene")
    <out>/<video name>/<video name>_0001.mp4  + _0001.json
    ...

synth_video.py splits train/val by that folder, so no video leaks into both splits,
and reads each clip's capture frame rate from the .json next to it.

KEY POINT: a slow-motion video on YouTube plays at 24-60 fps, but every frame in it was
captured 1/capture_fps apart (e.g. 1 ms for 1000 fps). The script needs that CAPTURE rate
(from the description / on-screen text). The file's own fps only converts times to frames.

What it does with each video (one sequential decode pass):
  * keeps only the given segments (playback time in the file, e.g. "0:12-0:40;1:05-1:30@2500"),
    which is how you skip intros, real-time parts, speed ramps and cross-fades;
  * hard cuts inside a segment: big jump of the frame difference -> a new shot, no clip crosses it;
  * duplicated frames (25->30 fps pulldown, frame doubling) are dropped, so the time step
    stays exactly 1/capture_fps. A frame counts as duplicate when it changes far less than
    its neighbours and no part of it changes (a small moving object keeps the frame);
  * black bars / letterbox are cropped (per clip), or use a fixed --roi;
  * clips with almost no motion are rejected (cheap pre-filter; synth_video.py --min-disp
    does the exact check on every sample);
  * clip length is given in ms of capture time (default 64 ms = 65 frames at 1000 fps, as X4K).

Examples
  # one file you already downloaded, whole video is 1000 fps
  python src/prep_video.py --video data/yt/raw/abc123.webm --fps 1000 --out data/yt/clips

  # one YouTube video, only two segments (the second one was shot at 2500 fps)
  python src/prep_video.py --video "https://www.youtube.com/watch?v=abc123" --fps 1000 \\
      --segments "0:12-0:40;1:05-1:30@2500" --out data/yt/clips --download-dir data/yt/raw

  # many videos from a list (CSV: name,source,fps,segments,roi; lines starting with # are skipped)
  python src/prep_video.py --list configs/videos_yt.csv --out data/yt/clips --download-dir data/yt/raw

Needs: numpy, opencv-python; yt-dlp for URLs (pip install yt-dlp); ffmpeg for writing mp4
(system ffmpeg or `pip install imageio-ffmpeg`; without it, clips are written as PNG folders).
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

THUMB_W = 320          # width of the gray thumbnail used for all frame tests


# ------------------------------------------------------------------ helpers
def parse_time(s):
    """'75', '75.5', '1:15', '0:01:15.5' -> seconds."""
    s = s.strip()
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_segments(text, default_fps):
    """'0:12-0:40;1:05-1:30@2500' -> [(12.0, 40.0, 1000), (65.0, 90.0, 2500)]. Empty -> whole video."""
    if not text or not text.strip():
        return [(0.0, float("inf"), default_fps)]
    segs = []
    for item in re.split(r"[;\s]+", text.strip()):
        if not item:
            continue
        rng, _, fps = item.partition("@")
        a, _, b = rng.partition("-")
        segs.append((parse_time(a) if a else 0.0, parse_time(b) if b else float("inf"),
                     float(fps) if fps else default_fps))
    return sorted(segs)


def parse_roi(text):
    if not text or not text.strip():
        return None
    x, y, w, h = (int(v) for v in re.split(r"[,x ]+", text.strip()))
    return x, y, w, h


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download(url, folder):
    """YouTube/Vimeo URL -> local video file (best video-only stream, no AV1: OpenCV may not decode it)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fmt = "bv*[vcodec!^=av01][height<=2160]/bv*[height<=2160]/b"
    cmd = ["--no-playlist", "-f", fmt, "--write-info-json",
           "-o", str(folder / "%(id)s.%(ext)s"), "--print", "after_move:filepath", url]
    exe = shutil.which("yt-dlp")
    full = [exe, *cmd] if exe else [sys.executable, "-m", "yt_dlp", *cmd]
    res = subprocess.run(full, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed for {url}:\n{res.stderr[-2000:]}\n(pip install -U yt-dlp)")
    path = Path(res.stdout.strip().splitlines()[-1])
    info = {}
    for cand in (path.with_suffix(".info.json"), folder / (path.stem + ".info.json")):
        if cand.exists():
            j = json.loads(cand.read_text(encoding="utf-8"))
            info = {k: j.get(k) for k in ("id", "title", "uploader", "channel", "license",
                                          "webpage_url", "upload_date", "duration")}
            break
    return path, info


class ClipWriter:
    """Writes one clip as mp4 (x264, near-lossless) or, without ffmpeg, as a folder of PNGs."""

    def __init__(self, ffmpeg, crf):
        self.ffmpeg, self.crf = ffmpeg, crf

    def write(self, frames, path_noext):
        h, w = frames[0].shape[:2]
        if self.ffmpeg is None:
            d = Path(path_noext)
            d.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                cv2.imwrite(str(d / f"{i:04d}.png"), fr)
            return d
        out = Path(str(path_noext) + ".mp4")
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(self.crf), "-pix_fmt", "yuv420p", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr in frames:
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {out}")
        return out


def content_box(frames, thr=24, min_frac=0.5):
    """Bounding box of non-black content over the clip (removes letterbox / pillarbox).
    Returns (x, y, w, h) with even w, h, or None if the content is too small."""
    mx = np.max([fr.max(axis=2) for fr in frames[:: max(1, len(frames) // 8)]], axis=0)
    rows, cols = np.where(mx.max(axis=1) > thr)[0], np.where(mx.max(axis=0) > thr)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    # ignore thin bright lines (subtitles / logos in the bars): need >2% of the row/col to be content
    rows = np.where((mx > thr).mean(axis=1) > 0.02)[0]
    cols = np.where((mx > thr).mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    H, W = mx.shape
    if (y1 - y0) < min_frac * H or (x1 - x0) < min_frac * W:
        return None
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    return int(x0), int(y0), int(w), int(h)


def block_max(absdiff, b=8):
    """Largest mean |diff| over b x b blocks of the thumbnail. Averaging in blocks removes
    compression noise (true duplicates -> ~0), while even a small moving object changes its block a lot."""
    h, w = absdiff.shape
    return float(cv2.resize(absdiff, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA).max())


# ------------------------------------------------------------------ one video
def process_video(path, name, segments, out_root, a, writer, info=None, roi=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path} (AV1? try: yt-dlp -f 'bv*[vcodec^=vp09]' ...)")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    seg_frames = [(int(round(s * file_fps)), int(round(e * file_fps)) if np.isfinite(e) else 1 << 40, f)
                  for s, e, f in segments]
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = dict(frames_read=0, frames_used=0, dups=0, cuts=0, clips=0, static=0, bad_box=0)
    rows = []
    idx = -1
    seg_i = 0
    state = None                      # per-shot state, reset at cuts and segment borders

    def new_state(cap_fps):
        n_clip = int(round(a.clip_ms * cap_fps / 1000.0)) + 1
        return dict(buf=[], thumbs=[], prev=None, diffs=deque(maxlen=30), bdiffs=deque(maxlen=30), skip=0,
                    fps=cap_fps, n_clip=n_clip, shot=stats["cuts"], start=None)

    def flush_clip(st, seg_no):
        frames = st["buf"]
        th0, th1 = st["thumbs"][0], st["thumbs"][-1]
        changed = float((np.abs(th1 - th0) > 10).mean())        # part of the image that changed
        if changed < a.min_changed:
            stats["static"] += 1
            return
        box = roi or content_box(frames)
        if box is None:
            stats["bad_box"] += 1
            return
        x, y, w, h = box
        frames = [fr[y:y + h, x:x + w] for fr in frames]
        if a.max_side and max(w, h) > a.max_side:
            s = a.max_side / max(w, h)
            size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
            frames = [cv2.resize(fr, size, interpolation=cv2.INTER_AREA) for fr in frames]
        k = stats["clips"]
        stem = f"{name}_{k:04d}"
        written = writer.write(frames, out_dir / stem)
        meta = dict(capture_fps=st["fps"], n_frames=len(frames), size=list(frames[0].shape[1::-1]),
                    source=str(path), video=name, segment=seg_no, shot=st["shot"],
                    start_frame=st["start"], start_time_s=round(st["start"] / file_fps, 3),
                    file_fps=file_fps, roi=list(box), frac_changed=round(changed, 4),
                    dups_dropped_in_clip=st.get("dups_clip", 0))
        if info:
            meta["info"] = info
        meta_path = (written.with_suffix(".json") if written.suffix else written / "clip.json")
        meta_path.write_text(json.dumps(meta, indent=1))
        rows.append(dict(clip=written.name, **{k_: meta[k_] for k_ in
                                               ("capture_fps", "start_time_s", "shot", "frac_changed")}))
        stats["clips"] += 1

    while True:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        stats["frames_read"] += 1
        while seg_i < len(seg_frames) and idx >= seg_frames[seg_i][1]:
            seg_i += 1
            state = None
        if seg_i >= len(seg_frames):
            break
        s0, s1, cap_fps = seg_frames[seg_i]
        if idx < s0:
            continue
        if state is None:
            state = new_state(cap_fps)
        th = cv2.cvtColor(cv2.resize(fr, (THUMB_W, int(THUMB_W * fr.shape[0] / fr.shape[1])),
                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if state["prev"] is not None:
            ad = np.abs(th - state["prev"])
            d, bm = float(ad.mean()), block_max(ad)
            med = float(np.median(state["diffs"])) if len(state["diffs"]) >= 5 else None
            med_bm = float(np.median(state["bdiffs"])) if len(state["bdiffs"]) >= 5 else None
            if d > a.cut_abs and (med is None or d > a.cut_rel * med):   # hard cut -> new shot
                stats["cuts"] += 1
                state = new_state(cap_fps)
            elif (med_bm is not None and med_bm > a.noise_floor and bm < a.dup_rel * med_bm
                  and bm < a.dup_px):
                # duplicate = changes far less than its neighbours AND no part of the image changed
                # (a small moving object keeps the frame even when the rest is still)
                stats["dups"] += 1                                       # duplicated frame -> drop
                state["dups_clip"] = state.get("dups_clip", 0) + 1
                continue
            else:
                state["diffs"].append(d)
                state["bdiffs"].append(bm)
        state["prev"] = th
        if state["skip"] > 0:                                            # gap between clips
            state["skip"] -= 1
            continue
        if not state["buf"]:
            state["start"], state["dups_clip"] = idx, 0
        state["buf"].append(fr)
        state["thumbs"].append(th)
        stats["frames_used"] += 1
        if len(state["buf"]) == state["n_clip"]:
            flush_clip(state, seg_i)
            state["buf"], state["thumbs"] = [], []
            state["skip"] = int(round(a.gap_ms * state["fps"] / 1000.0))
        if a.max_clips and stats["clips"] >= a.max_clips:
            break
    cap.release()

    with open(out_dir / "clips.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    used = max(stats["frames_used"] + stats["dups"], 1)
    stats.update(name=name, file_fps=round(file_fps, 3), file_frames=n_total,
                 dup_ratio=round(stats["dups"] / used, 3))
    (out_dir / "video.json").write_text(json.dumps(dict(stats, segments=segments, info=info or {}),
                                                  indent=1, default=str))
    msg = (f"{name}: {stats['clips']} clips | read {stats['frames_read']} frames, cuts {stats['cuts']}, "
           f"duplicates dropped {stats['dups']} ({100 * stats['dup_ratio']:.0f}%), "
           f"static clips rejected {stats['static']}")
    print(msg, flush=True)
    if stats["dup_ratio"] > 0.1:
        print(f"  WARNING {name}: many duplicated frames. Check the capture fps - the video may be "
              f"slowed down further than the camera rate (frame doubling / interpolation).")
    if min(f for _, _, f in segments) < 500:
        print(f"  WARNING {name}: capture fps < 500 - frames are > 2 ms apart; consider frame "
              f"interpolation (RIFE) before using it for RS synthesis.")
    return stats


# ------------------------------------------------------------------ driver
def jobs_from_args(a):
    if a.list:
        jobs = []
        with open(a.list, newline="", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            jobs.append(dict(name=r.get("name") or None, source=r["source"],
                             fps=float(r.get("fps") or a.fps), segments=r.get("segments", ""),
                             roi=r.get("roi", "")))
        return jobs
    return [dict(name=a.name, source=a.video, fps=a.fps, segments=a.segments or "", roi=a.roi or "")]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:60] or "video"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file or URL (YouTube/Vimeo, needs yt-dlp)")
    src.add_argument("--list", help="CSV with columns name,source,fps,segments,roi")
    ap.add_argument("--out", required=True, help="output root (one sub-folder per video)")
    ap.add_argument("--fps", type=float, default=1000.0, help="CAPTURE frame rate of the camera (default 1000)")
    ap.add_argument("--segments", help='playback-time ranges to use, e.g. "0:12-0:40;1:05-1:30@2500"')
    ap.add_argument("--name", help="scene name (default: file stem / video id)")
    ap.add_argument("--roi", help="fixed crop x,y,w,h in file pixels (default: auto-crop black bars)")
    ap.add_argument("--download-dir", default="data/yt/raw")
    ap.add_argument("--clip-ms", type=float, default=64.0, help="clip length in capture time (X4K: 64 ms)")
    ap.add_argument("--gap-ms", type=float, default=0.0,
                    help="skip this much capture time between clips (more variety, fewer clips)")
    ap.add_argument("--max-side", type=int, default=None, help="downscale frames so the long side <= this")
    ap.add_argument("--min-changed", type=float, default=0.02,
                    help="reject clip if < this part of the image changes between first and last frame")
    ap.add_argument("--cut-abs", type=float, default=12.0, help="cut: mean |diff| of thumbnails above this (0-255)")
    ap.add_argument("--cut-rel", type=float, default=6.0, help="... and above this x the recent median diff")
    ap.add_argument("--dup-rel", type=float, default=0.5,
                    help="duplicate: block change (see --dup-px) below this x its recent median")
    ap.add_argument("--dup-px", type=float, default=3.0,
                    help="duplicate only if no 8x8 block of the thumbnail changes by more than this on average (0-255)")
    ap.add_argument("--noise-floor", type=float, default=2.0,
                    help="duplicate test only when the recent median block change is above this (0-255), "
                         "i.e. the shot really moves; raise it for noisy / heavily compressed videos")
    ap.add_argument("--max-clips", type=int, default=None, help="per video (for quick tests)")
    ap.add_argument("--crf", type=int, default=12, help="x264 quality of written clips (lower = better)")
    a = ap.parse_args()

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        print("no ffmpeg found (pip install imageio-ffmpeg) - writing clips as PNG folders")
    writer = ClipWriter(ffmpeg, a.crf)
    total = 0
    for job in jobs_from_args(a):
        info = None
        src = job["source"]
        if re.match(r"https?://", src):
            path, info = download(src, a.download_dir)
            name = job["name"] or safe_name(info.get("id") or path.stem)
            lic = (info or {}).get("license")
            print(f"downloaded {path.name}: '{(info or {}).get('title')}' licence: {lic or 'standard YouTube'}")
        else:
            path = Path(src)
            name = job["name"] or safe_name(path.stem)
        segs = parse_segments(job["segments"], job["fps"])
        st = process_video(path, name, segs, a.out, a, writer, info, parse_roi(job["roi"]))
        total += st["clips"]
    print(f"done: {total} clips -> {a.out}")


if __name__ == "__main__":
    main()
