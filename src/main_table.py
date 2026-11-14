"""Main comparison table: every method on the same val set, with the same metrics.

All numbers are per image and averaged over images (like the training logs):
  psnr_full / ssim_full  whole image (the common measure; the main column),
  psnr_vis               region the method can see in A (its own validity mask; = psnr_full
                         for methods without one), only for comparison with the older tables.
Uncertainty: the val set has only 13 scenes (27 samples each), so confidence intervals are
bootstrapped over SCENES (resampling whole scenes), not over samples. Differences are paired
(same samples) against a reference method (default: the main model).
Runtime: one method for everything, median of `reps` runs on one real pair after warm-up
(network / flow + geometry only, no data loading), on the device given in the table.

Methods (key: what):
  input, only_b, mean_ab                    trivial (no correction, camera B alone, average A+B)
  farneback_v1, dis_v1, dis_v2              classical optical flow (OpenCV, no learning) + our geometry
  raft_zs_v1, raft_zs_v2                    pretrained RAFT (no training on our data) + geometry
  raft_ft_v1, raft_ft2_v2                   RAFT fine-tuned through the geometry (04 / 07)
  ae, unet_plain30, unet_plain200, corr80   networks without geometry (08)
  unet_plain_single                         camera-B ablation (B replaced by a copy of A, 09)
  refine, refine_ft                         RAFT + geometry + U-Net (05), frozen / jointly fine-tuned

  python src/main_table.py --data data/rs_video_x4k --out results/tabela_glowna --only classic
  python src/main_table.py --data /content/.../rs_video_x4k --runs /content/drive/MyDrive/rs_project/runs \\
      --out /content/drive/MyDrive/rs_project/results/tabela_glowna
Per-sample results are cached in <out>/per_sample/<key>.csv (rerun skips them, --redo recomputes).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from raft_geom import RSSampleDataset, correct_ab, correct_from_flow, load_raft_geom
from train_common import psnr, ssim

GROUPS = {
    "input": "trywialne", "only_b": "trywialne", "mean_ab": "trywialne",
    "farneback_v1": "przepływ klasyczny + geometria", "dis_v1": "przepływ klasyczny + geometria",
    "dis_v2": "przepływ klasyczny + geometria",
    "raft_zs_v1": "RAFT + geometria", "raft_zs_v2": "RAFT + geometria",
    "raft_ft_v1": "RAFT + geometria", "raft_ft2_v2": "RAFT + geometria",
    "ae": "sieci bez geometrii", "unet_plain30": "sieci bez geometrii", "unet_plain200": "sieci bez geometrii",
    "corr80": "sieci bez geometrii", "unet_plain_single": "ablacja: bez kamery B",
    "refine": "RAFT + geometria + U-Net", "refine_ft": "RAFT + geometria + U-Net",
}
LABELS = {
    "input": "bez korekcji (obraz A)", "only_b": "sam obraz B", "mean_ab": "średnia A i B",
    "farneback_v1": "Farnebäck + geometria v1", "dis_v1": "DIS + geometria v1", "dis_v2": "DIS + geometria v2",
    "raft_zs_v1": "RAFT gotowy + geometria v1", "raft_zs_v2": "RAFT gotowy + geometria v2",
    "raft_ft_v1": "RAFT dostrojony (04) + geometria v1", "raft_ft2_v2": "RAFT dostrojony z v2 (07) + geometria v2",
    "ae": "autoenkoder bez skipów (ae)", "unet_plain30": "zwykły U-Net, 30 ep.", "unet_plain200": "zwykły U-Net, 200 ep.",
    "corr80": "enkoder + korelacja (corr), 80 ep.", "unet_plain_single": "zwykły U-Net bez kamery B, 200 ep.",
    "refine": "RAFT (zamrożony) + geometria + U-Net", "refine_ft": "RAFT + geometria + U-Net, wspólnie dostrojone",
}
CLASSIC = ["input", "only_b", "mean_ab", "farneback_v1", "dis_v1", "dis_v2"]
RAFT_RUNS = {"raft_ft_v1": ("raft_geom", "v1"), "raft_ft2_v2": ("raft_geom_v2", "v2")}
UNET_RUNS = {"ae": "ae", "unet_plain30": "unet_plain", "unet_plain200": "unet_plain_long", "corr80": "corr",
             "unet_plain_single": "unet_plain_single", "refine": "raft_unet", "refine_ft": "raft_unet_ft"}
ORDER = CLASSIC + ["raft_zs_v1", "raft_zs_v2", "raft_ft_v1", "raft_ft2_v2"] + list(UNET_RUNS)


# ------------------------------------------------------------------ methods
def find_run(runs, name, file="best.pt"):
    """Newest run folder whose name ends with _<name> (same rule as project_paths.find_run)."""
    c = sorted(Path(runs).glob(f"*_{name}/{file}"))
    return c[-1] if c else None


def _gray(img):
    import cv2
    return cv2.cvtColor((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)


def classical_flow(kind):
    import cv2
    if kind == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f = lambda a, b: dis.calc(a, b, None)
    else:
        f = lambda a, b: cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)

    def flow(x, y):                              # (N,3,S,S) x2 -> (N,2,S,S) flow x -> y
        return torch.stack([torch.from_numpy(f(_gray(p), _gray(q))).permute(2, 0, 1) for p, q in zip(x, y)])
    return flow


class FlowGeom(torch.nn.Module):
    """Classical flow (OpenCV, CPU) + geometry v1 / v2; same interface as the networks."""

    def __init__(self, kind, geom, rot_k=1):
        super().__init__()
        self.flow, self.geom, self.rot_k = classical_flow(kind), geom, rot_k

    def forward(self, rs_a, rs_b, T, sync, base):
        dev = rs_a.device
        a, b = rs_a.cpu(), rs_b.cpu()
        d_ab = self.flow(a, b).float()
        if self.geom == "v1":
            pred, aux = correct_from_flow(a, d_ab, T.cpu(), sync.cpu(), base.cpu(), self.rot_k)
        else:
            d_ba = self.flow(b, a).float()
            pred, aux = correct_ab(a, b, d_ab, d_ba, T.cpu(), sync.cpu(), base.cpu(), self.rot_k)
        return pred.to(dev), {"valid": aux["valid"].to(dev)}


class Trivial(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, rs_a, rs_b, T, sync, base):
        return {"input": rs_a, "only_b": rs_b, "mean_ab": 0.5 * (rs_a + rs_b)}[self.kind], {}


def build_method(key, runs, dev, rot_k):
    """-> (model, device it runs on, info dict) or None when the checkpoint is missing."""
    cpu = torch.device("cpu")
    if key in ("input", "only_b", "mean_ab"):
        return Trivial(key), dev, {}
    if key in ("farneback_v1", "dis_v1", "dis_v2"):
        kind, geom = key.split("_")
        return FlowGeom("farneback" if kind == "farneback" else "dis", geom, rot_k), cpu, {"device_note": "CPU (OpenCV)"}
    if key in ("raft_zs_v1", "raft_zs_v2"):
        m = load_raft_geom(None, dev, rot_k=rot_k, geom=key[-2:])
        return m, dev, {"params_M": _params(m), "steps": "0 (RAFT pretrenowany poza projektem)"}
    if key in RAFT_RUNS:
        name, geom = RAFT_RUNS[key]
        ck = find_run(runs, name) if runs else None
        if ck is None:
            return None
        m = load_raft_geom(ck, dev, rot_k=rot_k, geom=geom)
        return m, dev, {"params_M": _params(m), "run": ck.parent.name, "steps": _steps(ck.parent)}
    if key in UNET_RUNS:
        from raft_unet import load_model
        ck = find_run(runs, UNET_RUNS[key]) if runs else None
        if ck is None:
            return None
        m, _ = load_model(ck, dev)
        return m, dev, {"params_M": _params(m), "run": ck.parent.name, "steps": _steps(ck.parent)}
    raise KeyError(key)


def _params(m):
    return round(sum(p.numel() for p in m.parameters()) / 1e6, 2)


def _steps(run_dir):
    try:
        info = json.loads((Path(run_dir) / "run_info.json").read_text())
        a = info["command_args"]
        return int(a["epochs"]) * (int(info["n_train"]) // int(a["bs"]))
    except (OSError, KeyError, ValueError):
        return ""


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, dataset, dev, bs=8, workers=0):
    """Per-sample rows: sample, psnr_full, ssim_full, psnr_vis."""
    model.eval()
    rows, i = [], 0
    for b in DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=workers):
        b = {k: v.to(dev) for k, v in b.items()}
        pred, aux = model(b["rs_a"], b["rs_b"], b["T"], b["sync"], b["base"])
        pred = pred.float().clamp(0, 1)
        one = torch.ones_like(pred[:, :1])
        vis = aux.get("valid_a", aux.get("valid", one)).float()
        pf, sf, pv = psnr(pred, b["gs_a"], one), ssim(pred, b["gs_a"], one), psnr(pred, b["gs_a"], vis)
        for j in range(pred.shape[0]):
            rows.append({"sample": dataset.dirs[i].name, "psnr_full": pf[j].item(), "ssim_full": sf[j].item(),
                         "psnr_vis": pv[j].item()})
            i += 1
    return rows


@torch.no_grad()
def time_ms(model, sample, dev, reps=20):
    b = {k: v[None].to(dev) for k, v in sample.items()}
    args = (b["rs_a"], b["rs_b"], b["T"], b["sync"], b["base"])
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(3):
        model(*args)
    ts = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter()
        model(*args)
        sync(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def scenes_of(dataset):
    return [json.loads((d / "meta.json").read_text())["source"].replace("\\", "/").split("/")[0] for d in dataset.dirs]


def scene_bootstrap(x, scenes, B=5000, seed=0):
    """95% percentile CI of the mean, resampling whole scenes."""
    x, scenes = np.asarray(x, float), np.asarray(scenes)
    us = np.unique(scenes)
    idx = [np.where(scenes == s)[0] for s in us]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([idx[k] for k in rng.integers(0, len(us), len(us))]) for _ in range(B)]
    means = [x[m].mean() for m in means]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(data, out, runs=None, keys=None, dev=None, redo=False, bs=8, workers=0, timing=True):
    """Evaluates the methods (cached per key), returns the summary table (pandas) and saves it."""
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RSSampleDataset(Path(data) / "val")
    rot_k = int(ds[0]["rot_k"])
    out = Path(out)
    (out / "per_sample").mkdir(parents=True, exist_ok=True)
    meta_path = out / "methods.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key in keys or ORDER:
        f = out / "per_sample" / f"{key}.csv"
        if f.exists() and not redo:
            continue
        built = build_method(key, runs, dev, rot_k)
        if built is None:
            print(f"{key}: brak checkpointu w {runs}, pomijam")
            continue
        model, mdev, info = built
        t0 = time.time()
        rows = evaluate(model, ds, mdev, bs, workers)
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        if timing:
            info["ms_per_pair"] = round(time_ms(model, ds[0], mdev), 1)
        info["device"] = info.pop("device_note", str(torch.cuda.get_device_name(0) if mdev.type == "cuda" else "CPU"))
        meta[key] = info
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{key}: psnr_full {np.mean([r['psnr_full'] for r in rows]):.2f} dB ({time.time() - t0:.0f} s)", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return summarize(out, ds)


def summarize(out, ds, ref="refine_ft"):
    import pandas as pd
    out = Path(out)
    scenes = dict(zip([d.name for d in ds.dirs], scenes_of(ds)))
    meta = json.loads((out / "methods.json").read_text()) if (out / "methods.json").exists() else {}
    per = {}
    for key in ORDER:
        f = out / "per_sample" / f"{key}.csv"
        if f.exists():
            per[key] = pd.read_csv(f, dtype={"sample": str}).set_index("sample")
    if not per:
        return None
    ref_df = per.get(ref)
    rows = []
    for key, df in per.items():
        sc = [scenes[s] for s in df.index]
        lo, hi = scene_bootstrap(df["psnr_full"], sc)
        r = {"grupa": GROUPS[key], "metoda": LABELS[key], "klucz": key,
             "psnr_full": df["psnr_full"].mean(), "CI95 (sceny)": f"[{lo:.2f}; {hi:.2f}]",
             "mediana": df["psnr_full"].median(), "ssim_full": df["ssim_full"].mean(), "psnr_vis": df["psnr_vis"].mean()}
        if ref_df is not None and key != ref:
            common = df.index.intersection(ref_df.index)
            d = df.loc[common, "psnr_full"] - ref_df.loc[common, "psnr_full"]
            dlo, dhi = scene_bootstrap(d, [scenes[s] for s in common])
            r[f"różnica vs {ref}"] = d.mean()
            r["CI95 różnicy"] = f"[{dlo:.2f}; {dhi:.2f}]"
            r[f"lepsze od {ref}"] = f"{int((d > 0).sum())}/{len(d)}"
        m = meta.get(key, {})
        r.update({"ms/parę": m.get("ms_per_pair", ""), "urządzenie": m.get("device", ""),
                  "param. [M]": m.get("params_M", ""), "kroki treningu": m.get("steps", ""), "run": m.get("run", "")})
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "tabela_glowna.csv", index=False)
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset folder with val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None, help="folder with training runs (for the learned methods)")
    ap.add_argument("--only", choices=["classic", "all"], default="all")
    ap.add_argument("--keys", nargs="*", default=None, help="subset of method keys")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    keys = a.keys or (CLASSIC if a.only == "classic" else ORDER)
    tab = run(a.data, a.out, a.runs, keys, redo=a.redo, bs=a.bs, workers=a.workers)
    if tab is not None:
        import pandas as pd
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(tab.drop(columns=["klucz", "run"]).round(3).to_string(index=False))
        print("zapisano", Path(a.out) / "tabela_glowna.csv")


if __name__ == "__main__":
    main()
