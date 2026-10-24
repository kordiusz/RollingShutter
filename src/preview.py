"""Contact sheet for a generated dataset: one row per sample.
Columns: RS A | RS B (as camera sees it) | RS B rotated back | GS A (target)

  python src/preview.py dataset/train --n 6 --out preview.png
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="preview.png")
    a = ap.parse_args()

    rows = []
    for d in sorted(p for p in Path(a.folder).iterdir() if p.is_dir())[: a.n]:
        meta = json.loads((d / "meta.json").read_text())
        k = meta.get("b_rot_k", 1)
        ra, rb, ga = (cv2.imread(str(d / f"{n}.png"), cv2.IMREAD_COLOR) for n in ("rs_a", "rs_b", "gs_a"))
        rb_back = np.ascontiguousarray(np.rot90(rb, -k))
        info = f"pan {np.hypot(*meta['w_rad_s'][:2]):.1f}rad/s ro {meta['readout_a_s']*1e3:.0f}ms"
        sep = np.full((ra.shape[0], 4, 3), 255, np.uint8)
        rows.append(np.hstack([label(ra, "RS A"), sep, label(rb, "RS B (camera)"), sep,
                               label(rb_back, "RS B rotated back"), sep,
                               label(ga, "GS A target"), sep]))
        rows[-1] = np.vstack([rows[-1], np.full((22, rows[-1].shape[1], 3), 255, np.uint8)])
        cv2.putText(rows[-1], d.name + "  " + info, (6, rows[-1].shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, np.vstack(rows))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
