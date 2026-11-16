"""Small helpers shared by make_cutouts.py and coco_backgrounds.py (COCO 2017 data).

COCO photos can be fetched one by one from http://images.cocodataset.org/<split>/<file>,
so a script can pick what it needs in the annotation file and download only those
photos instead of the whole 18 GB train2017 archive.
"""
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COCO_URL = "http://images.cocodataset.org/{split}/{name}"


def load_instances(path):
    """instances_*.json -> (images by id, annotations grouped by image id, category names by id).
    instances_train2017.json is 450 MB: loading takes ~30 s and ~3 GB of RAM."""
    coco = json.loads(Path(path).read_text())
    images = {im["id"]: im for im in coco["images"]}
    by_image = defaultdict(list)
    for an in coco["annotations"]:
        by_image[an["image_id"]].append(an)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    return images, by_image, cats


def load_captions(path):
    """captions_*.json -> {image id: "all captions of the photo, lower case"}."""
    if not path:
        return {}
    caps = defaultdict(list)
    for c in json.loads(Path(path).read_text())["annotations"]:
        caps[c["image_id"]].append(c["caption"].lower())
    return {k: " ".join(v) for k, v in caps.items()}


def split_of_annotation_file(path):
    """'instances_train2017.json' -> 'train2017' (the folder / URL part of the photos)."""
    stem = Path(path).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def overlap_frac(a, b):
    """Fraction of box a ([x, y, w, h]) covered by box b."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1e-6, a[2] * a[3])


def _fetch(url, dest, tries=3):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
                fh.write(r.read())
            tmp.replace(dest)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
    return False


def download_images(file_names, split, dest, workers=8):
    """Download the COCO photos `file_names` of `split` (train2017 / val2017) into `dest`,
    skipping those already there. Returns (downloaded, failed)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    todo = sorted({f for f in file_names if not (dest / f).exists()})
    if not todo:
        return 0, 0
    print(f"downloading {len(todo)} photos ({len(todo) * 0.16:.0f} MB) from images.cocodataset.org ...",
          flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, good in enumerate(ex.map(lambda f: _fetch(COCO_URL.format(split=split, name=f), dest / f), todo)):
            ok += good
            fail += not good
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    if fail:
        print(f"WARNING: {fail} photos could not be downloaded (skipped)")
    return ok, fail
