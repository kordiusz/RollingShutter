"""Cut real objects (people, cars, bikes, animals, balls, ...) out of COCO photos.

Each object is saved as a PNG with transparency (RGBA), cropped to its box:
  <out>/<category>/<annotation id>.png
These are the foreground objects for synth_layers.py (--obj-src), so the
scenes contain real things moving in front of a background instead of random
shapes cut out of random photos.

Only "clean" objects are kept: big enough, not cut by the photo border, not
heavily covered by other objects, not a crowd annotation.

Data (COCO 2017 validation set, ~1 GB in total):
  http://images.cocodataset.org/zips/val2017.zip                      (photos, ~780 MB)
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip  (outlines, ~250 MB)
Unpack both, then:
  python src/make_cutouts.py --images C:/coco/val2017 \
      --ann C:/coco/annotations/instances_val2017.json --out C:/coco/cutouts
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# things that plausibly stand / move in front of a background
DEFAULT_CATS = ("person,bicycle,car,motorcycle,bus,truck,bird,cat,dog,horse,sheep,cow,"
                "elephant,bear,zebra,giraffe,frisbee,sports ball,kite,skateboard,surfboard,"
                "umbrella,backpack,suitcase,teddy bear,chair,potted plant,fire hydrant,stop sign")


def overlap_frac(a, b):
    """Fraction of box a ([x, y, w, h]) covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1e-6, a[2] * a[3])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with COCO photos (e.g. val2017)")
    ap.add_argument("--ann", required=True, help="instances_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cats", default=DEFAULT_CATS, help="comma-separated COCO category names")
    ap.add_argument("--min-size", type=int, default=96, help="min width and height of the object, px")
    ap.add_argument("--min-fill", type=float, default=0.3, help="min object area / box area")
    ap.add_argument("--max-overlap", type=float, default=0.3,
                    help="skip objects whose box is covered more than this by another object")
    ap.add_argument("--max-per-cat", type=int, default=400, help="balance: at most N per category")
    a = ap.parse_args()

    coco = json.loads(Path(a.ann).read_text())
    want = {c.strip() for c in a.cats.split(",")}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    by_image = defaultdict(list)
    for an in coco["annotations"]:
        by_image[an["image_id"]].append(an)

    # 1) choose clean objects
    chosen = defaultdict(list)
    for img_id, anns in by_image.items():
        im = images[img_id]
        W, H = im["width"], im["height"]
        for an in anns:
            name = cat_name[an["category_id"]]
            x, y, w, h = an["bbox"]
            if (name not in want or an.get("iscrowd") or not isinstance(an["segmentation"], list)
                    or w < a.min_size or h < a.min_size
                    or an["area"] < a.min_fill * w * h
                    or x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2):
                continue
            if any(o is not an and overlap_frac(an["bbox"], o["bbox"]) > a.max_overlap for o in anns):
                continue
            chosen[name].append(an)

    # 2) cut them out
    out = Path(a.out)
    total, missing = 0, 0
    for name in sorted(chosen):
        anns = sorted(chosen[name], key=lambda t: -t["area"])[: a.max_per_cat]
        saved = 0
        for an in anns:
            im = images[an["image_id"]]
            img = cv2.imread(str(Path(a.images) / im["file_name"]), cv2.IMREAD_COLOR)
            if img is None:
                missing += 1
                continue
            (out / name).mkdir(parents=True, exist_ok=True)
            mask = np.zeros(img.shape[:2], np.uint8)
            for poly in an["segmentation"]:
                pts = np.array(poly, np.float32).reshape(-1, 2)
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (3, 3), 0.8)            # soft edge, no jaggies
            x, y, w, h = (int(round(v)) for v in an["bbox"])
            pad = 3
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
            rgba = np.dstack([img[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
            cv2.imwrite(str(out / name / f"{an['id']}.png"), rgba)
            saved += 1
        total += saved
        print(f"{name:15s} {saved:4d}")
    if missing:
        print(f"WARNING: {missing} objects skipped, their photos are not in {a.images}")
    print(f"done: {total} cut-outs -> {out}")


if __name__ == "__main__":
    main()
