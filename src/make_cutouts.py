"""Cut real objects (people, cars, bikes, animals, balls, ...) out of COCO photos.

Each object is saved as a PNG with transparency (RGBA), cropped to its box:
  <out>/<category>/<annotation id>.png
These are the foreground objects for synth_layers.py (--obj-src). The folder
name is the COCO category, which tells the generator how the thing behaves
(stands on the ground / flies, real height, typical speed; see synth_layers.OBJECTS).

Only "clean" objects are kept: big enough, not cut by the photo border, not
heavily covered by other objects, not a crowd annotation. Per category the
largest ones are preferred (sharper) and at most --max-per-image come from one
photo (variety). With --keypoints (person_keypoints_*.json, same zip) only
people who stand or walk are kept (upright body, feet visible), so nobody is
pasted lying or sitting in mid-air.

Data: http://images.cocodataset.org/annotations/annotations_trainval2017.zip (outlines)
      http://images.cocodataset.org/zips/val2017.zip (5000 photos, 780 MB)
  # A) photos on disk (val2017 -> ~1400 objects):
  python src/make_cutouts.py --images coco/val2017 --ann coco/annotations/instances_val2017.json \
      --keypoints coco/annotations/person_keypoints_val2017.json --out coco/cutouts
  # B) many more objects: choose them in the big train2017 set (118k photos) and download
  #    only the photos needed (~160 KB each) into --images:
  python src/make_cutouts.py --ann coco/annotations/instances_train2017.json \
      --keypoints coco/annotations/person_keypoints_train2017.json --images coco/train2017 \
      --download --max-per-cat 250 --out coco/cutouts
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from coco_utils import download_images, load_instances, overlap_frac, split_of_annotation_file
from synth_layers import OBJECTS

# COCO keypoint order
KP = {n: i for i, n in enumerate(("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                   "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                                   "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                                   "right_knee", "left_ankle", "right_ankle"))}


def load_upright_people(path):
    """Annotation ids of people who stand or walk: head above hips above ankles,
    body roughly vertical, at least one ankle visible."""
    ok = set()
    for an in json.loads(Path(path).read_text())["annotations"]:
        k = np.array(an["keypoints"], float).reshape(-1, 3)
        vis = k[:, 2] > 0

        def pt(*names):
            p = [k[KP[n], :2] for n in names if vis[KP[n]]]
            return np.mean(p, 0) if p else None
        head = pt("nose", "left_eye", "right_eye", "left_ear", "right_ear")
        sh, hip, ank = pt("left_shoulder", "right_shoulder"), pt("left_hip", "right_hip"), pt("left_ankle", "right_ankle")
        if head is None or sh is None or hip is None or ank is None:
            continue
        height = ank[1] - head[1]
        if height <= 0 or not (head[1] < sh[1] < hip[1] < ank[1]):
            continue
        if abs(sh[0] - hip[0]) > 0.25 * height or abs(hip[0] - ank[0]) > 0.35 * height:
            continue                                              # leaning / lying / sitting
        if hip[1] - sh[1] < 0.15 * height or ank[1] - hip[1] < 0.3 * height:
            continue                                              # folded body
        ok.add(an["id"])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True,
                    help="folder with the COCO photos (with --download: where to put them)")
    ap.add_argument("--ann", required=True, help="instances_val2017.json or instances_train2017.json")
    ap.add_argument("--keypoints", help="person_keypoints_*.json of the same split: keep only "
                                        "people who stand or walk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", action="store_true",
                    help="fetch the photos of the chosen objects from images.cocodataset.org")
    ap.add_argument("--cats", default=",".join(c for c, v in OBJECTS.items() if v[3] > 0),
                    help="comma-separated COCO category names (default: those used by synth_layers)")
    ap.add_argument("--min-size", type=int, default=96, help="min width and height of the object, px")
    ap.add_argument("--min-fill", type=float, default=0.3, help="min object area / box area")
    ap.add_argument("--max-overlap", type=float, default=0.3,
                    help="skip objects whose box is covered more than this by another object")
    ap.add_argument("--max-per-cat", type=int, default=400, help="balance: at most N per category")
    ap.add_argument("--max-person", type=int, default=None,
                    help="people are used most often, so they get their own limit (default 3x max-per-cat)")
    ap.add_argument("--max-per-image", type=int, default=2, help="variety: at most N objects per photo")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    images, by_image, cat_name = load_instances(a.ann)
    want = {c.strip() for c in a.cats.split(",")}
    upright = load_upright_people(a.keypoints) if a.keypoints else None

    # 1) clean objects, grouped by category
    clean = defaultdict(list)
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
            if name == "person" and upright is not None and an["id"] not in upright:
                continue
            if any(o is not an and overlap_frac(an["bbox"], o["bbox"]) > a.max_overlap for o in anns):
                continue
            clean[name].append(an)

    # 2) per category: among the 3*N largest pick N at random; few per photo
    rng = np.random.default_rng(a.seed)
    per_image = defaultdict(int)
    chosen = {}
    for name in sorted(clean):
        limit = (a.max_person or 3 * a.max_per_cat) if name == "person" else a.max_per_cat
        big = sorted(clean[name], key=lambda t: -t["area"])[: 3 * limit]
        rng.shuffle(big)
        keep = []
        for an in big:
            if per_image[an["image_id"]] >= a.max_per_image:
                continue
            per_image[an["image_id"]] += 1
            keep.append(an)
            if len(keep) >= limit:
                break
        chosen[name] = keep

    needed = {images[an["image_id"]]["file_name"] for anns in chosen.values() for an in anns}
    print(f"{sum(len(v) for v in chosen.values())} objects in {len(chosen)} categories, "
          f"from {len(needed)} photos")
    if a.download:
        download_images(needed, split_of_annotation_file(a.ann), a.images)

    # 3) cut them out
    out = Path(a.out)
    total = missing = 0
    for name, anns in chosen.items():
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
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))       # drop the 1-px fringe of background
            mask = cv2.GaussianBlur(mask, (3, 3), 0.8)              # soft edge, no jaggies
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
