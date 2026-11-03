"""Statistics of a dual-RS dataset (synth_video.py / synth_rs.py output).

For every sample it measures how strong the rolling shutter really is:
the optical flow (Farneback, OpenCV) between the target GS image and RS image A.
That flow = how far each pixel was moved by the rolling shutter, in px of the
256 x 256 image. Plus the generation parameters from meta.json.

  python src/analyze_dataset.py --data data/rs_video_x4k --out data/analiza_rs_video_x4k

Outputs (in --out):
  samples.csv     one row per sample (parameters + measured RS strength)
  scenes.csv      one row per source scene
  summary.txt     counts and quantiles, the numbers for the dataset description
  hist.png        histograms (needs matplotlib; skipped without it)
  weakest.jpg / strongest.jpg   example samples: RS A | GS | RS displacement
  strong_<split>.txt            sample folders with displacement p95 >= --strong (for filtered training)

Classes (thresholds on the 95th percentile of the RS displacement, px):
  slaby < 2 <= sredni < 6 <= silny
Motion type: "globalny" (most of the image moves, typically camera motion) when the
median displacement is at least half of the 95th percentile and >= 1 px; otherwise
"lokalny" (only some objects move) or "brak" for weak samples.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("slaby", "sredni", "silny")


def rs_displacement(rs, gs):
    """Per-pixel magnitude of the flow GS -> RS (px)."""
    g = cv2.cvtColor(gs, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(g, r, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    return np.linalg.norm(fl, axis=2)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-10)))


def classify(p50, p95, weak, strong):
    cls = CLASSES[0] if p95 < weak else CLASSES[1] if p95 < strong else CLASSES[2]
    if cls == CLASSES[0]:
        kind = "brak"
    elif p50 >= 1.0 and p50 >= 0.5 * p95:
        kind = "globalny"
    else:
        kind = "lokalny"
    return cls, kind


def analyze_one(args):
    d, split, weak, strong = args
    m = json.loads((d / "meta.json").read_text())
    rs = cv2.imread(str(d / "rs_a.png"))
    gs = cv2.imread(str(d / "gs_a.png"))
    mag = rs_displacement(rs, gs)
    p50, p95 = float(np.percentile(mag, 50)), float(np.percentile(mag, 95))
    cls, kind = classify(p50, p95, weak, strong)
    S = rs.shape[0]
    T = m.get("readout_a_s", 0) * 1000
    src = str(m.get("source", "")).replace("\\", "/")
    side = m["crop"][2] if "crop" in m else S
    return {
        "split": split, "sample": d.name,
        "scene": src.split("/")[0] if "/" in src else "", "source": src,
        "readout_ms": round(T, 2), "sync_ms": round(m.get("sync_offset_s", 0) * 1000, 3),
        "baseline_px": round(m.get("baseline_px", 0), 2), "crop_src_px": side,
        "scale": round(side / S, 3), "augment": m.get("augment", ""),
        "psnr_in": round(psnr(rs, gs), 2),
        "disp_p50": round(p50, 2), "disp_p95": round(p95, 2), "disp_max": round(float(mag.max()), 2),
        "frac_moving": round(float((mag > 1.0).mean()), 3),
        # displacement in px of the SOURCE video (before resizing the crop to S x S)
        "disp_p95_src": round(p95 * side / S, 2),
        # speed estimate: rows at the top/bottom are T/2 away from the target time
        "speed_px_per_ms": round(p95 / (T / 2), 3) if T > 0 else 0.0,
        "class": cls, "motion": kind,
    }


def q(v, ps=(5, 25, 50, 75, 95)):
    v = np.asarray(v, float)
    return " / ".join(f"{np.percentile(v, p):.2f}" for p in ps)


def summary(rows, weak, strong):
    out = []
    w = out.append
    w(f"Proby: {len(rows)}  (progi klas: slaby < {weak} px <= sredni < {strong} px <= silny; "
      f"miara = 95. percentyl przesuniecia RS w obrazie 256x256)")
    for split in ("train", "val"):
        r = [x for x in rows if x["split"] == split]
        if not r:
            continue
        scenes = {x["scene"] for x in r}
        clips = {x["source"] for x in r}
        w(f"\n== {split}: {len(r)} prob, {len(clips)} klipow, {len(scenes)} scen")
        for c in CLASSES:
            n = sum(x["class"] == c for x in r)
            w(f"  klasa {c:7s}: {n:5d} ({100 * n / len(r):4.1f}%)")
        for k in ("brak", "lokalny", "globalny"):
            n = sum(x["motion"] == k for x in r)
            w(f"  ruch {k:9s}: {n:5d} ({100 * n / len(r):4.1f}%)")
        w("  kwantyle 5/25/50/75/95%:")
        for key, label in (("disp_p95", "przesuniecie RS p95 [px]"), ("disp_p50", "przesuniecie RS mediana [px]"),
                           ("frac_moving", "czesc pikseli > 1 px"), ("psnr_in", "PSNR wejscia [dB]"),
                           ("readout_ms", "czas odczytu [ms]"), ("scale", "skala wycinka (zrodlo/256)"),
                           ("disp_p95_src", "przesuniecie p95 w px zrodla"),
                           ("speed_px_per_ms", "predkosc [px/ms] (256)")):
            w(f"    {label:32s} {q([x[key] for x in r])}")
        for c in CLASSES:
            rc = [x["psnr_in"] for x in r if x["class"] == c]
            if rc:
                w(f"  sredni PSNR wejscia, klasa {c:7s}: {np.mean(rc):.2f} dB")
    return "\n".join(out)


def grid(rows, root, n, title):
    tiles = []
    for x in rows[:n]:
        d = root / x["split"] / x["sample"]
        rs, gs = cv2.imread(str(d / "rs_a.png")), cv2.imread(str(d / "gs_a.png"))
        mag = rs_displacement(rs, gs)
        heat = cv2.applyColorMap(np.clip(mag * 255 / 15, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

        def lab(im, t):
            im = im.copy()
            cv2.rectangle(im, (0, 0), (im.shape[1], 16), (0, 0, 0), -1)
            cv2.putText(im, t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            return im
        tiles.append(np.hstack([lab(rs, f"RS A  {x['split']}/{x['sample']}"),
                                lab(gs, f"GS   PSNR wej. {x['psnr_in']:.1f} dB"),
                                lab(heat, f"przesuniecie RS p95 {x['disp_p95']:.1f} px (0-15)")]))
    if not tiles:
        return None
    img = np.vstack(tiles)
    head = np.zeros((24, img.shape[1], 3), np.uint8)
    cv2.putText(head, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, img])


def plots(rows, out, weak, strong):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("brak matplotlib - pomijam hist.png (pip install matplotlib)")
        return
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    g = lambda k, s=None: np.array([x[k] for x in rows if s is None or x["split"] == s], float)
    a = ax[0, 0]
    a.hist(np.clip(g("disp_p95"), 0, 30), bins=60, color="#4C72B0")
    a.axvline(weak, color="k", ls="--"); a.axvline(strong, color="k", ls="--")
    a.set_title("Przesuniecie RS, 95. percentyl [px, 256x256]"); a.set_xlabel("px (obciete do 30)")
    a = ax[0, 1]
    a.hist(g("psnr_in"), bins=60, color="#55A868"); a.set_title("PSNR wejscia RS A vs GS [dB]")
    a = ax[0, 2]
    a.scatter(g("disp_p95"), g("psnr_in"), s=4, alpha=0.4); a.set_xscale("symlog", linthresh=1)
    a.set_xlabel("przesuniecie RS p95 [px]"); a.set_ylabel("PSNR wejscia [dB]")
    a.set_title("Sila RS a PSNR wejscia")
    a = ax[1, 0]
    a.scatter(g("scale"), g("disp_p95"), s=4, alpha=0.4); a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("skala wycinka (px zrodla / 256)"); a.set_ylabel("przesuniecie RS p95 [px]")
    a.set_title("Wiekszy wycinek = mniejszy ruch w px")
    a = ax[1, 1]
    a.scatter(g("readout_ms"), g("disp_p95"), s=4, alpha=0.4); a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("czas odczytu [ms]"); a.set_ylabel("przesuniecie RS p95 [px]")
    a.set_title("Dluzszy odczyt = silniejszy RS")
    a = ax[1, 2]
    labels, vals = [], []
    for s in ("train", "val"):
        for c in CLASSES:
            labels.append(f"{s}\n{c}"); vals.append(sum(1 for x in rows if x["split"] == s and x["class"] == c))
    a.bar(range(len(vals)), vals, color=["#C44E52", "#DD8452", "#55A868"] * 2)
    a.set_xticks(range(len(vals))); a.set_xticklabels(labels, fontsize=8); a.set_title("Liczba prob w klasach")
    fig.tight_layout()
    fig.savefig(out / "hist.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset folder with train/ and val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weak", type=float, default=2.0, help="p95 displacement below this = weak RS [px]")
    ap.add_argument("--strong", type=float, default=6.0, help="p95 displacement from this = strong RS [px]")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples per split (test)")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split in ("train", "val"):
        if (root / split).is_dir():
            ds = sorted(p for p in (root / split).iterdir() if (p / "meta.json").exists())[:a.limit]
            jobs += [(d, split, a.weak, a.strong) for d in ds]
    print(f"{len(jobs)} prob, licze ...", flush=True)
    with Pool(a.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(analyze_one, jobs, chunksize=8)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    with open(out / "samples.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    scenes = {}
    for x in rows:
        scenes.setdefault((x["split"], x["scene"]), []).append(x)
    with open(out / "scenes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "scene", "n", "disp_p95_median", "frac_silny", "frac_slaby", "psnr_in_mean", "motion_globalny"])
        for (s, sc), r in sorted(scenes.items()):
            wr.writerow([s, sc, len(r), round(float(np.median([x["disp_p95"] for x in r])), 2),
                         round(np.mean([x["class"] == "silny" for x in r]), 2),
                         round(np.mean([x["class"] == "slaby" for x in r]), 2),
                         round(float(np.mean([x["psnr_in"] for x in r])), 2),
                         round(np.mean([x["motion"] == "globalny" for x in r]), 2)])
    for split in ("train", "val"):
        keep = [x["sample"] for x in rows if x["split"] == split and x["disp_p95"] >= a.strong]
        (out / f"strong_{split}.txt").write_text("\n".join(keep))
    txt = summary(rows, a.weak, a.strong)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)

    by = sorted(rows, key=lambda x: x["disp_p95"])
    for name, sel, title in (("weakest.jpg", by[:8], "Najslabszy RS (zostaja w zbiorze, ale niewiele ucza)"),
                             ("strongest.jpg", by[::-1][:8], "Najsilniejszy RS")):
        img = grid(sel, root, 8, title)
        if img is not None:
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    plots(rows, out, a.weak, a.strong)
    print("zapisano w", out)


if __name__ == "__main__":
    main()
