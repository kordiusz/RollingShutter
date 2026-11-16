"""Visual comparison of the approaches on the same samples: networks without geometry
(autoencoder, plain U-Net, encoder + correlation) vs RAFT + geometry vs the main model.

For every sample one block of three rows:
  1. full images: RS A | RS B | each model | GS target (PSNR of the whole image under each tile;
     the frame on RS A marks the zoomed region),
  2. zoom of the region with the strongest RS distortion (largest |GS - RS A|),
  3. error x4 against GS (black = perfect).
GIF per sample (modele_<sample>.gif, style of demo.py): RS A -> each network without geometry -> main model,
PSNR and SSIM of the whole image under each frame (same functions as the training / main table: train_common).
PSNR / SSIM of every sample and model also go to a CSV next to --out (porownanie_modeli.csv).

The checkpoints are train_unet.py runs. With --runs the newest run of each default name is used
(same rule as project_paths.find_run / main_table.py):
  ae               autoencoder without skips
  unet_plain_long  plain U-Net, 200 epochs
  corr             siamese encoder + correlation
  raft_unet_ft     main model; its RAFT + geometry stage (v1, before the U-Net) is shown as an extra column
Or give them explicitly: --model "label=path/best.pt" (repeatable, in column order).

  python src/demo_models.py examples/x4k_val_00276_02 examples/x4k_val_00273_02 examples/kolo \
      --runs /content/drive/MyDrive/rs_project/runs --out porownanie.jpg --gif-dir gify
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from demo import label, load_sample, to_t, to_u8
from train_common import psnr as psnr_t, ssim as ssim_t


def metrics(im, gs):
    """PSNR [dB] and SSIM of the whole image, uint8 BGR images (as the main table: whole-image mask)."""
    a, b = to_t(im), to_t(gs)
    one = torch.ones_like(a[:, :1])
    return psnr_t(a, b, one).item(), ssim_t(a, b, one).item()

DEFAULT_RUNS = [  # (run name suffix, column label; no Polish letters: cv2.putText cannot draw them)
    ("ae", "autoenkoder bez skrotow (ae)"),
    ("unet_plain_long", "zwykly U-Net, 200 ep."),
    ("corr", "enkoder + korelacja (corr)"),
    ("raft_unet_ft", "model glowny: RAFT + geom. + U-Net"),
]
GEOM_LABEL = "RAFT + geometria (bez U-Netu)"


def find_run(runs, name, file="best.pt"):
    c = sorted(Path(runs).glob(f"*_{name}/{file}"))
    return c[-1] if c else None


def load_models(a, dev):
    from raft_unet import load_model
    specs = []
    for m in a.model or []:
        lab, path = m.split("=", 1)
        specs.append((lab, Path(path)))
    if a.runs:
        for name, lab in DEFAULT_RUNS:
            ck = find_run(a.runs, name)
            if ck is None:
                print(f"brak runu *_{name} w {a.runs}, pomijam")
                continue
            specs.append((lab, ck))
    models = []
    for lab, path in specs:
        m, ck = load_model(path, dev)
        print(f"{lab}: {path} ({ck['mode']})")
        models.append((lab, ck["mode"], m))
    return models


@torch.no_grad()
def run_models(models, s, dev):
    """-> list of (label, uint8 image). A refine model also adds its geometry-only stage before itself."""
    f = lambda v: torch.tensor([float(v)], device=dev)
    args = (to_t(s["rs_a"]).to(dev), to_t(s["rs_b"]).to(dev), f(s["T"]), f(s["sync"]), f(s["base"]))
    out = []
    for lab, mode, m in models:
        pred, aux = m(*args)
        if mode == "refine":
            out.append((GEOM_LABEL, to_u8(aux["geom"])))
        out.append((lab, to_u8(pred)))
    return out


def worst_region(s, size):
    """Top-left corner of the size x size window with the largest mean |GS - RS A| (or |A - B| without GS)."""
    ref = s["gs_a"] if s["gs_a"] is not None else s["rs_b"]
    d = np.abs(s["rs_a"].astype(np.float32) - ref.astype(np.float32)).mean(2)
    m = cv2.boxFilter(d, -1, (size, size), normalize=True, borderType=cv2.BORDER_CONSTANT)
    h, w = d.shape
    r = size // 2
    m = m[r:h - size + r + 1, r:w - size + r + 1]          # windows fully inside the image
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return int(x), int(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folders (rs_a.png, rs_b.png, meta.json, gs_a.png)")
    ap.add_argument("--runs", default=None, help="folder with runs (MyDrive/rs_project/runs): default models")
    ap.add_argument("--model", action="append", help='"label=path/best.pt" (train_unet.py checkpoint), repeatable')
    ap.add_argument("--zoom", type=int, default=64, help="side of the zoomed region in px of the 256 image")
    ap.add_argument("--scale", type=float, default=1.5, help="tile size = image size x scale")
    ap.add_argument("--out", default="porownanie_modeli.jpg")
    ap.add_argument("--gif-dir", default=None, help="also one GIF per sample in this folder: modele_<sample>.gif, "
                    "same style as demo.py (RS A -> each network without geometry -> main model)")
    ap.add_argument("--gif-scale", type=float, default=2.0, help="GIF frame size = image size x this (as demo.py)")
    ap.add_argument("--gif-ms", type=int, default=2000, help="time of one GIF frame [ms]")
    ap.add_argument("--gif-zoom", action="store_true",
                    help="also porownanie_<sample>.gif: every frame = full image + zoom, incl. geometry and GS")
    a = ap.parse_args()
    if not a.runs and not a.model:
        ap.error("give --runs or --model")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    models = load_models(a, dev)
    assert models, "no checkpoints found"
    sc = a.scale / 2
    up = lambda im: cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
    blocks, rows = [], []
    for path in a.samples:
        s = load_sample(path)
        gs = s["gs_a"]
        size = s["rs_a"].shape[0]
        zx, zy = worst_region(s, a.zoom)
        tile = up(s["rs_a"]).shape[:2]
        zoom = lambda im: cv2.resize(im[zy:zy + a.zoom, zx:zx + a.zoom], tile[::-1], interpolation=cv2.INTER_CUBIC)
        err = lambda im: np.clip(np.abs(im.astype(np.float32) - gs.astype(np.float32)) * 4, 0, 255).astype(np.uint8)
        blank = np.zeros((*tile, 3), np.uint8)

        cols = [("RS A - wejscie", s["rs_a"]), ("RS B - wejscie", s["rs_b"])] + run_models(models, s, dev)
        if gs is not None:
            cols.append(("GS - jak powinno byc", gs))
        met = {lab: metrics(im, gs) for lab, im in [cols[0]] + cols[2:-1]} if gs is not None else {}
        ptxt = lambda lab: f"PSNR {met[lab][0]:.1f} dB | SSIM {met[lab][1]:.3f}" if lab in met else None
        print(f"{s['name']}: T {s['T']:.1f} ms, powiekszony obszar ({zx}, {zy})")
        for lab, (p, q) in met.items():
            print(f"  {lab}: PSNR {p:.2f} dB, SSIM {q:.4f}")
            rows.append({"sample": s["name"], "method": lab, "psnr_full": round(p, 3), "ssim_full": round(q, 4)})

        r1, r2, r3, gif, simple = [], [], [], [], []
        k = a.scale
        gup = lambda im: cv2.resize(im, None, fx=a.gif_scale, fy=a.gif_scale, interpolation=cv2.INTER_CUBIC)
        for i, (lab, im) in enumerate(cols):
            is_b, is_gs = i == 1, gs is not None and i == len(cols) - 1
            full = up(im)
            if i == 0:                                        # mark the zoomed region on the input
                cv2.rectangle(full, (int(zx * k), int(zy * k)), (int((zx + a.zoom) * k), int((zy + a.zoom) * k)),
                              (0, 255, 255), 2)
            pt = None if (is_b or is_gs) else ptxt(lab)
            r1.append(label(full, lab, pt, sc))
            r2.append(label(zoom(im), f"powiekszenie x{size // a.zoom * a.scale:g}" if i == 0 else None, None, sc))
            r3.append(blank if (is_b or is_gs or gs is None) else label(up(err(im)), "blad x4" if i == 0 else None, None, sc))
            if not is_b:
                gif.append(np.hstack([label(full, lab, pt or s["name"], sc), label(zoom(im), "powiekszenie", None, sc)]))
            if not (is_b or is_gs or lab == GEOM_LABEL):      # same frames as demo.py: before -> after
                top = "PRZED: kamera A (rolling shutter)" if i == 0 else f"PO: {lab}"
                simple.append(label(gup(im), top, pt or s["name"], a.gif_scale / 2))
        blocks.append(np.vstack([np.hstack(r1), np.hstack(r2), np.hstack(r3)]))

        if a.gif_dir:
            from PIL import Image
            Path(a.gif_dir).mkdir(parents=True, exist_ok=True)
            for prefix, fr, dur, on in (("modele", simple, a.gif_ms, True), ("porownanie", gif, a.gif_ms, a.gif_zoom)):
                if not on:
                    continue
                frames = [Image.fromarray(f[..., ::-1]) for f in fr]
                gp = Path(a.gif_dir) / f"{prefix}_{s['name']}.gif"
                frames[0].save(gp, save_all=True, append_images=frames[1:], duration=dur, loop=0)
                print("saved", gp)

    w = max(b.shape[1] for b in blocks)
    sep = np.full((8, w, 3), 255, np.uint8)
    out = []
    for b in blocks:
        out += [np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0))), sep]
    cv2.imwrite(a.out, np.vstack(out[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", a.out)
    if rows:
        import csv
        cp = Path(a.out).with_suffix(".csv")
        with open(cp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("saved", cp)


if __name__ == "__main__":
    main()
