"""Where does camera B help? Per-pixel error as a function of the readout-time difference.

Camera A reads rows, camera B columns. At pixel p (A's frame) the two cameras read at
t_A(p) and t_B(p); on the anti-diagonal t_B = t_A (both read the point at the same moment,
so B carries no information about motion), away from it |t_B - t_A| grows up to ~T.
If the gain of a model that sees B over the same model without B grows with |t_B - t_A|,
B helps through its different readout time, not through extra pixels or a second viewpoint.

The error of the uncorrected input also depends on |t_A| (the middle row, t_A = 0, is
almost undistorted), so pixels are binned in 2D: |t_A| (how much there is to correct) x
|t_B - t_A| (how much timing information B adds). t_B is taken at the same pixel position
(the displacement between the cameras, up to ~40 px, is ignored).

  from analysis_b import analyze, gain_table, plot_gain
  res = analyze({"A+B": model_ab, "A": model_a}, RSSampleDataset(val_dir), device)
  gain_table(res, "A+B", "A")         # dB per (|t_A|, |t_B - t_A|) bin
  plot_gain(res, "A+B", "A", "fig.png")

Used by notebook 09 (camera-B ablation, no RAFT, no training).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from raft_geom import base_grid, readout_time

TA_EDGES = (0.0, 5.0, 10.0, 15.01)                                 # |t_A| [ms]: T <= 30 ms
DT_EDGES = (0.0, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 31.0)       # |t_B - t_A| [ms]


def time_grids(n, S, T, sync, rot_k, device):
    """|t_A| and |t_B - t_A| [ms] of every pixel, (N, S, S)."""
    xs, ys = base_grid(n, S, S, device)
    ta = readout_time(xs, ys, S, T, sync, rot_k, "a")
    tb = readout_time(xs, ys, S, T, sync, rot_k, "b")
    return ta.abs(), (tb - ta).abs()


@torch.no_grad()
def analyze(models, dataset, device="cpu", bs=8, workers=0, pool=8):
    """models: {name: model of train_unet.py (any mode)}. Returns a dict with, per name and for
    "input": squared-error sums per (|t_A| bin, |dt| bin) and an error map per image position
    (mean over the set, pooled pool x pool), plus pixel counts per bin."""
    names = list(models) + ["input"]
    nt, nd = len(TA_EDGES) - 1, len(DT_EDGES) - 1
    sse = {k: np.zeros((nt, nd)) for k in names}
    cnt = np.zeros((nt, nd))
    pos = {k: None for k in names}
    dt_map, n_img = None, 0
    ta_e = torch.tensor(TA_EDGES, device=device)
    dt_e = torch.tensor(DT_EDGES, device=device)
    for m in models.values():
        m.to(device).eval()
    for b in DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=workers):
        b = {k: v.to(device) for k, v in b.items()}
        n, _, S, _ = b["rs_a"].shape
        rot_k = int(b["rot_k"][0])
        ta, dt = time_grids(n, S, b["T"], b["sync"], rot_k, device)
        it = (torch.bucketize(ta, ta_e, right=True) - 1).clamp(0, nt - 1)
        idt = (torch.bucketize(dt, dt_e, right=True) - 1).clamp(0, nd - 1)
        cell = (it * nd + idt).flatten()
        cnt += torch.bincount(cell, minlength=nt * nd).double().view(nt, nd).cpu().numpy()
        preds = {k: m(b["rs_a"], b["rs_b"], b["T"], b["sync"], b["base"])[0].float() for k, m in models.items()}
        preds["input"] = b["rs_a"]
        for k, p in preds.items():
            e = ((p - b["gs_a"]) ** 2).mean(1)                                     # (N, S, S)
            sse[k] += torch.bincount(cell, weights=e.flatten().double(),
                                     minlength=nt * nd).view(nt, nd).cpu().numpy()
            ep = F.avg_pool2d(e[:, None], pool).sum(0)[0].double().cpu().numpy()
            pos[k] = ep if pos[k] is None else pos[k] + ep
        d = F.avg_pool2d(dt[:, None], pool).sum(0)[0].double().cpu().numpy()
        dt_map = d if dt_map is None else dt_map + d
        n_img += n
    return {"names": names, "sse": sse, "count": cnt, "pos": {k: v / n_img for k, v in pos.items()},
            "dt_map": dt_map / n_img, "n_images": n_img, "ta_edges": TA_EDGES, "dt_edges": DT_EDGES}


def _psnr(mse):
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def psnr_bins(res, name):
    """PSNR [dB] per (|t_A| bin, |dt| bin); NaN where a bin has no pixels."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return _psnr(res["sse"][name] / res["count"])


def gain_table(res, with_b, without_b):
    """pandas table: PSNR gain [dB] of `with_b` over `without_b` per bin, plus both over the input
    and the share of pixels; rows |t_A|, columns |t_B - t_A|."""
    import pandas as pd
    lab = lambda e: [f"{a:g}-{b:g}" for a, b in zip(e[:-1], e[1:])]
    rows, cols = lab(res["ta_edges"]), lab(res["dt_edges"])
    out = {}
    pa, pb, pi = psnr_bins(res, with_b), psnr_bins(res, without_b), psnr_bins(res, "input")
    for i, r in enumerate(rows):
        out[(f"|t_A| {r} ms", f"{with_b} - {without_b} [dB]")] = pa[i] - pb[i]
        out[(f"|t_A| {r} ms", f"{with_b} - input [dB]")] = pa[i] - pi[i]
        out[(f"|t_A| {r} ms", f"{without_b} - input [dB]")] = pb[i] - pi[i]
        out[(f"|t_A| {r} ms", "pixels [%]")] = 100 * res["count"][i] / res["count"].sum()
    t = pd.DataFrame(out, index=[f"{c} ms" for c in cols]).T
    t.columns.name = "|t_B - t_A|"
    return t


def gain_by_dt(res, with_b, without_b, min_ta_bin=1):
    """1D: gain per |dt| bin, pooled over |t_A| bins >= min_ta_bin (where there is RS to correct)."""
    sl = slice(min_ta_bin, None)
    c = res["count"][sl].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = _psnr(res["sse"][with_b][sl].sum(0) / c)
        pb = _psnr(res["sse"][without_b][sl].sum(0) / c)
        pi = _psnr(res["sse"]["input"][sl].sum(0) / c)
    centres = [(a + b) / 2 for a, b in zip(res["dt_edges"][:-1], res["dt_edges"][1:])]
    return np.array(centres), pa - pb, pa - pi, pb - pi, c


def plot_gain(res, with_b, without_b, path=None):
    """Figure: (left) gain vs |t_B - t_A| per |t_A| band, (middle) mean |t_B - t_A| per position,
    (right) gain of with_b over without_b per position."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    centres = [(a + b) / 2 for a, b in zip(res["dt_edges"][:-1], res["dt_edges"][1:])]
    pa, pb = psnr_bins(res, with_b), psnr_bins(res, without_b)
    te = res["ta_edges"]
    for i in range(len(te) - 1):
        ok = res["count"][i] > 1000
        ax[0].plot(np.array(centres)[ok], (pa[i] - pb[i])[ok], "o-", label=f"|t_A| {te[i]:g}-{te[i + 1]:g} ms")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("|t_B - t_A| [ms]  (0 = both cameras read the pixel at the same moment)")
    ax[0].set_ylabel(f"PSNR gain {with_b} vs {without_b} [dB]")
    ax[0].set_title("gain from camera B vs readout-time difference")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(res["dt_map"], cmap="viridis")
    ax[1].set_title("mean |t_B - t_A| [ms] per position")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    g = _psnr(res["pos"][with_b]) - _psnr(res["pos"][without_b])
    lim = np.nanmax(np.abs(g))
    im = ax[2].imshow(g, cmap="RdBu", vmin=-lim, vmax=lim)
    ax[2].set_title(f"gain {with_b} vs {without_b} [dB] per position")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax[1:]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def correlation_gain_dt(res, with_b, without_b):
    """Pearson correlation between the per-position gain and the per-position |t_B - t_A|."""
    g = (_psnr(res["pos"][with_b]) - _psnr(res["pos"][without_b])).ravel()
    d = res["dt_map"].ravel()
    ok = np.isfinite(g)
    return float(np.corrcoef(g[ok], d[ok])[0, 1]) if ok.sum() > 2 else math.nan
