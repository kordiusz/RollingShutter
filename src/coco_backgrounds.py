"""Pick COCO photos that look like everyday SCENES (streets, rooms, parks, shops ...)
to serve as backgrounds for synth_layers.py (--bg-src).

Why: a background should be a place a camera could be pointed at, not a close-up
of a leaf or a plate of food. COCO comes with object outlines and five captions
per photo, which is enough to tell the two apart without looking at the pixels:
  * short side >= --min-side px (the generator crops ~435 px for a 256 px output),
  * no single object covering more than --max-obj-frac of the photo (no close-ups),
  * all objects together covering at most --max-total-frac (no group photos),
  * at least --min-objects annotated things (a real scene, not a plain surface),
  * with --captions: a caption names a place (street, kitchen, park, ...) and none
    says "close up" / "selfie".

It also writes <out>/scene_info.json, which synth_layers.py reads to place its
objects consistently with the photo:
  * horizon_frac: where the ground's horizon is (row / height), estimated from the
    feet and heights of the people, cars, ... already in the photo (a camera 1.5 m
    above the ground is assumed); missing when the photo has no such reference,
  * indoor: the captions describe a room / kitchen / shop ..., or the photo holds
    two indoor-only things such as a laptop or a sink (no cars inside).

  # photos already on disk (val2017: 5000 photos -> ~1700 scenes):
  python src/coco_backgrounds.py --images coco/val2017 --ann coco/annotations/instances_val2017.json \
      --captions coco/annotations/captions_val2017.json --out coco/backgrounds
  # more variety: choose scenes in the big train2017 set and download only those photos (~160 KB each):
  python src/coco_backgrounds.py --ann coco/annotations/instances_train2017.json \
      --captions coco/annotations/captions_train2017.json --download 3000 --out coco/backgrounds
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from coco_utils import download_images, load_captions, load_instances, split_of_annotation_file
from synth_layers import OBJECTS, SCENE_CAM_HEIGHT_M, SCENE_INFO

INDOOR_WORDS = ("kitchen", "room", "bedroom", "bathroom", "office", "living", "restaurant",
                "store", "shop", "hallway", "lobby", "classroom", "library", "gym", "indoor",
                "inside", "hotel", "cafe", "bar ", "diner", "garage", "warehouse", "basement",
                "museum", "mall", "terminal", "studio", "desk", "sofa", "couch", "counter")
# things that are only found inside: two of them in the photo = an indoor scene
INDOOR_THINGS = {"tv", "laptop", "mouse", "remote", "keyboard", "microwave", "oven", "toaster",
                 "sink", "refrigerator", "bed", "couch", "toilet", "dining table", "hair drier",
                 "toothbrush", "bowl", "wine glass", "fork", "knife", "spoon"}
OUTDOOR_WORDS = ("street", "road", "city", "town", "park", "field", "beach", "highway", "sidewalk",
                 "bridge", "river", "lake", "mountain", "hill", "forest", "garden", "yard",
                 "parking", "stadium", "snow", "harbor", "dock", "traffic", "intersection",
                 "plaza", "farm", "zoo", "pier", "trail", "outdoor", "outside", "sky")
PLACE_WORDS = ("street", "road", "city", "town", "building", "park", "field", "beach", "kitchen",
               "room", "bedroom", "bathroom", "office", "living", "restaurant", "store", "shop",
               "market", "station", "airport", "highway", "sidewalk", "bridge", "river", "lake",
               "mountain", "hill", "forest", "garden", "yard", "parking", "stadium", "court",
               "snow", "harbor", "dock", "traffic", "intersection", "crowd", "plaza", "square",
               "hallway", "lobby", "classroom", "library", "gym", "zoo", "farm", "pier", "trail")
BAD_WORDS = ("close up", "close-up", "closeup", "selfie")


def is_scene(im, anns, caption, a):
    if min(im["width"], im["height"]) < a.min_side:
        return False
    if len(anns) < a.min_objects:
        return False
    area = im["width"] * im["height"]
    if max(an["area"] for an in anns) > a.max_obj_frac * area:
        return False
    if sum(an["area"] for an in anns) > a.max_total_frac * area:      # group photos, piles of things
        return False
    if caption is not None:
        if any(b in caption for b in BAD_WORDS):
            return False
        if not any(re.search(r"\b" + w, caption) for w in PLACE_WORDS):
            return False
    return True


def horizon_from_objects(im, anns, cat_name):
    """Horizon row / photo height, from things standing on the ground in the photo:
    a thing of real height H whose feet are at row y_f and which is h px tall has
    the horizon at y_f - h * (camera height / H). Median over the references."""
    H = im["height"]
    est = []
    for an in anns:
        info = OBJECTS.get(cat_name[an["category_id"]])
        if info is None or info[0] != "ground" or an.get("iscrowd"):
            continue
        x, y, w, h = an["bbox"]
        if h < max(40, 0.08 * H) or y + h >= H - 2 or y <= 2:        # feet / head must be visible
            continue
        est.append((y + h) - h * SCENE_CAM_HEIGHT_M / info[1])
    if not est:
        return None, 0
    hz = float(np.median(est)) / H
    return (hz if -0.5 <= hz <= 1.2 else None), len(est)


def is_indoor(caption, anns, cat_name):
    """Indoor when the captions and the annotated things point inside more than
    outside: indoor words + two indoor-only things (a laptop, a sink ...) versus
    outdoor words ("a living room overlooking the city" is still a room)."""
    caption = caption or ""
    inside = sum(len(re.findall(w, caption)) for w in INDOOR_WORDS)
    outside = sum(len(re.findall(r"\b" + w, caption)) for w in OUTDOOR_WORDS)
    things = sum(cat_name[an["category_id"]] in INDOOR_THINGS for an in anns)
    return inside + (2 if things >= 2 else 0) > outside


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="instances_val2017.json or instances_train2017.json")
    ap.add_argument("--captions", help="captions_*.json of the same split (recommended)")
    ap.add_argument("--images", help="folder with the photos (needed unless --download)")
    ap.add_argument("--out", required=True, help="folder that receives the chosen photos")
    ap.add_argument("--download", type=int, metavar="N",
                    help="do not read --images: choose N scenes and download them into --out")
    ap.add_argument("--max", type=int, help="copy at most this many (random choice)")
    ap.add_argument("--min-side", type=int, default=400)
    ap.add_argument("--max-obj-frac", type=float, default=0.25,
                    help="no single object may cover more of the photo than this")
    ap.add_argument("--max-total-frac", type=float, default=0.5,
                    help="all objects together may cover at most this much (no group photos)")
    ap.add_argument("--min-objects", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.download and not a.images:
        ap.error("give --images (photos on disk) or --download N")

    images, by_image, cat_name = load_instances(a.ann)
    caps = load_captions(a.captions)
    chosen = [iid for iid, im in images.items()
              if by_image[iid] and is_scene(im, by_image[iid], caps.get(iid) if caps else None, a)]
    chosen.sort()
    print(f"{len(chosen)} of {len(images)} photos look like scenes")
    n = a.download or a.max
    if n and n < len(chosen):
        rng = np.random.default_rng(a.seed)
        chosen = sorted(rng.choice(chosen, n, replace=False).tolist())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info_file = out / SCENE_INFO
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    with_horizon = indoor = 0
    for iid in chosen:
        hz, n_ref = horizon_from_objects(images[iid], by_image[iid], cat_name)
        ind = is_indoor(caps.get(iid), by_image[iid], cat_name)
        info[images[iid]["file_name"]] = dict(horizon_frac=hz, n_ref=n_ref, indoor=ind)
        with_horizon += hz is not None
        indoor += ind
    info_file.write_text(json.dumps(info, indent=0))
    print(f"scene info: {with_horizon} with a horizon from the photo's own objects, {indoor} indoor")

    names = [images[iid]["file_name"] for iid in chosen]
    if a.download:
        ok, _ = download_images(names, split_of_annotation_file(a.ann), out)
        print(f"done: {ok} new photos downloaded, {len(names)} chosen -> {out}")
        return
    copied = missing = 0
    for name in names:
        src = Path(a.images) / name
        if not src.exists():
            missing += 1
            continue
        if not (out / name).exists():
            shutil.copy2(src, out / name)
            copied += 1
    if missing:
        print(f"WARNING: {missing} chosen photos are not in {a.images}")
    print(f"done: {copied} photos copied -> {out}")


if __name__ == "__main__":
    main()
