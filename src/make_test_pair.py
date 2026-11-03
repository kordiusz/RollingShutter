"""One test pair of RS images (camera A: rows, camera B: columns) from a 1000 fps clip,
for a quick visual test of a trained model (06_unet.ipynb, last section).

Same simulation as synth_video.py, but with fixed, strong settings (long readout,
small crop) and the strongest of several random draws. Use a clip from a
VALIDATION scene, so the model has never seen it.

  python src/make_test_pair.py --clip data/x4k/encoded_train/152/occ010.683_f4177.mp4 --out examples/test_pair

Writes:
  rs_A_poziomy.png   camera A, rows read top to bottom (horizontal lines)
  rs_B_pionowy.png   camera B, rotated by 90 deg exactly as the camera outputs it
                     (in A's frame it reads columns: vertical lines)
  gs_A_wzorzec.png   global-shutter target for A (only for comparison, not a model input)
  meta.json          readout time, sync, baseline, ... (the model needs T and sync)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_dataset import rs_displacement
from synth_rs import to_u8
from synth_video import DEFAULT_CFG, make_sample, read_clip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1000.0)
    ap.add_argument("--readout-ms", type=float, default=30.0)
    ap.add_argument("--crop-frac", type=float, nargs=2, default=[0.4, 0.5])
    ap.add_argument("--tries", type=int, default=20, help="random draws; the strongest RS is kept")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = dict(DEFAULT_CFG, readout_ms=[a.readout_ms, a.readout_ms], crop_frac=a.crop_frac,
               noise_std=[0.0, 0.0], gain=[1.0, 1.0])
    clip = read_clip(a.clip)
    rng = np.random.default_rng(a.seed)
    best = None
    for _ in range(a.tries):
        s = make_sample(clip, a.fps, rng, cfg)
        if s is None:
            continue
        to8 = lambda x: (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
        p95 = float(np.percentile(rs_displacement(to8(s[0]["rs_a"]), to8(s[0]["gs_a"])), 95))
        if best is None or p95 > best[0]:
            best = (p95, s)
    if best is None:
        raise SystemExit("clip too short for this readout time")
    p95, (imgs, meta) = best
    meta.update(source=str(a.clip), rs_disp_p95_px=round(p95, 2))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fname in (("rs_a", "rs_A_poziomy.png"), ("rs_b", "rs_B_pionowy.png"), ("gs_a", "gs_A_wzorzec.png")):
        cv2.imwrite(str(out / fname), to_u8(imgs[name]))     # BGR uint8, as in the dataset
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out} | RS displacement p95 {p95:.1f} px | readout {a.readout_ms} ms, "
          f"sync {meta['sync_offset_s'] * 1000:+.2f} ms, baseline {meta['baseline_px']:.1f} px")


if __name__ == "__main__":
    main()
