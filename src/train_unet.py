"""Training / evaluation of the U-Net models from raft_unet.py on the video dataset.

  # main model: RAFT + geometry (fine-tuned RAFT from 04, frozen) + refinement U-Net
  python src/train_unet.py --data data/rs_video_x4k --mode refine --raft-ckpt runs/2026-10-24_raft_geom/best.pt \
      --epochs 30 --bs 8 --out runs --name 2026-11-03_raft_unet
  # same, but RAFT is also fine-tuned (small lr), after the U-Net has learned
  python src/train_unet.py ... --mode refine --init runs/2026-11-03_raft_unet/best.pt --raft-lr 1e-5 --epochs 10
  # baseline: plain U-Net, no RAFT, no geometry
  python src/train_unet.py --data data/rs_video_x4k --mode plain --epochs 30 --bs 16 --name 2026-11-03_unet_plain
  # direct synthesis: autoencoder without skips / siamese encoder + correlation layer
  python src/train_unet.py --data data/rs_video_x4k --mode ae --epochs 60 --bs 16 --name 2026-11-11_ae
  python src/train_unet.py --data data/rs_video_x4k --mode corr --epochs 60 --bs 16 --name 2026-11-11_corr
  # small network from scratch predicting the flow A->B + RS geometry (instead of RAFT)
  python src/train_unet.py --data data/rs_video_x4k --mode corrgeom --epochs 80 --bs 16 --name 2026-11-12_corrgeom
  # ablation: the same model with camera B replaced by a copy of A
  python src/train_unet.py --data data/rs_video_x4k --mode corr --single --epochs 80 --bs 16 --name 2026-11-12_corr_single
  # evaluation (+ per-sample metrics for the breakdown by RS strength)
  python src/train_unet.py --data data/rs_video_x4k --eval runs/<run>/best.pt --eval-json eval.json --per-sample val.csv

Loss: L1 between output and target where at least one camera saw the scene (refine)
or over the whole image (plain). Metrics:
  psnr/ssim        where the source pixel lies inside A (= the region of train_raft.py, comparable with 04)
  psnr_geom/...    the same for RAFT + geometry alone (refine only): what the U-Net added
  psnr_full        the whole image (comparable between all models; borders count too)
  *_in             no correction (RS A vs target)
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from raft_geom import RaftGeom, RSSampleDataset
from raft_unet import build, load_model
from train_common import masked_l1, psnr, ssim, run_context, dataset_info


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def forward(model, b):
    return model(b["rs_a"], b["rs_b"], b["T"], b["sync"], b["base"])


def loss_mask(aux, pred):
    if "valid_b" in aux:
        return torch.maximum(aux["valid_a"], aux["valid_b"])
    return torch.ones_like(pred[:, :1])


@torch.no_grad()
def evaluate(model, loader, dev, names=None):
    """Mean metrics; with `names` (sample folder names in loader order) also per-sample rows."""
    model.eval()
    acc, rows = {}, []
    for b in loader:
        b = to_dev(b, dev)
        pred, aux = forward(model, b)
        m, one = aux["valid_a"], torch.ones_like(aux["valid_a"])
        r = {"psnr": psnr(pred, b["gs_a"], m), "ssim": ssim(pred, b["gs_a"], m),
             "psnr_in": psnr(b["rs_a"], b["gs_a"], m), "ssim_in": ssim(b["rs_a"], b["gs_a"], m),
             "psnr_full": psnr(pred, b["gs_a"], one), "psnr_full_in": psnr(b["rs_a"], b["gs_a"], one),
             "valid": m.flatten(1).mean(1)}
        if "valid_geom" in aux:                  # corrgeom: share of pixels the geometry corrects
            r["valid_geom"] = aux["valid_geom"].flatten(1).mean(1)
        if "geom" in aux:
            g = aux["geom"]
            r.update(psnr_geom=psnr(g, b["gs_a"], m), ssim_geom=ssim(g, b["gs_a"], m),
                     psnr_full_geom=psnr(g, b["gs_a"], one), blend_b=(1 - aux["blend"]).flatten(1).mean(1))
        for k, v in r.items():
            acc.setdefault(k, []).append(v.float().cpu())
    out = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    if names is not None:
        cols = {k: torch.cat(v).numpy() for k, v in acc.items()}
        rows = [{"sample": n, **{k: round(float(cols[k][i]), 4) for k in cols}} for i, n in enumerate(names)]
    return out, rows


@torch.no_grad()
def save_preview(model, batch, dev, path, n=4):
    import cv2
    model.eval()
    b = to_dev({k: v[:n] for k, v in batch.items()}, dev)
    pred, aux = forward(model, b)
    to8 = lambda x: (x.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)
    gray = lambda x: cv2.cvtColor((x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat = lambda e: cv2.applyColorMap(np.clip(e.cpu().numpy() * 4 * 255, 0, 255).astype(np.uint8),
                                       cv2.COLORMAP_INFERNO)
    rows = []
    for i in range(pred.shape[0]):
        ims = [to8(b["rs_a"][i]), to8(b["rs_b"][i])]
        if "geom" in aux:
            ims += [to8(aux["geom"][i]), heat((aux["geom"][i] - b["gs_a"][i]).abs().mean(0))]
        ims += [to8(pred[i]), heat((pred[i] - b["gs_a"][i]).abs().mean(0)), to8(b["gs_a"][i])]
        if "blend" in aux:
            ims.append(gray(aux["blend"][i, 0]))
        rows.append(np.hstack(ims))
    # refine: RS A | RS B | RAFT+geometry | its error | U-Net | its error | target | blend (white = A, black = B)
    # plain:  RS A | RS B | U-Net | its error | target
    cv2.imwrite(str(path), np.vstack(rows))


@torch.no_grad()
def inference_ms(model, dev, size, reps=30):
    """Pure network time for one pair (no data loading)."""
    model.eval()
    x = torch.rand(1, 3, size, size, device=dev)
    t = lambda v: torch.full((1,), v, device=dev)
    args = (x, x, t(20.0), t(0.0), t(4.0))
    for _ in range(3):
        model(*args)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(*args)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="refine", choices=["refine", "plain", "ae", "corr", "corrgeom"],
                    help="ae / corr: direct synthesis (autoencoder / encoder + correlation); "
                         "corrgeom: corr network -> flow A->B -> RS geometry; see raft_unet.py")
    ap.add_argument("--single", action="store_true",
                    help="plain / ae / corr: camera B replaced by a copy of A (ablation: does B help?)")
    ap.add_argument("--raft-ckpt", default=None, help="refine: RAFT from train_raft.py (best.pt); none = pretrained")
    ap.add_argument("--init", default=None, help="start from this checkpoint of train_unet.py (weights only)")
    ap.add_argument("--raft-lr", type=float, default=0.0, help="refine: > 0 also fine-tunes RAFT with this lr")
    ap.add_argument("--width", type=int, default=32, help="base channels of the U-Net")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only N samples per split (debug)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None, help="last.pt of an interrupted run")
    ap.add_argument("--eval", default=None, help="checkpoint to evaluate instead of training")
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--per-sample", default=None, help="eval: CSV with metrics of every val sample")
    a = ap.parse_args()
    if a.single and a.mode in ("refine", "corrgeom"):
        ap.error("--single works with plain / ae / corr (the geometry needs camera B)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("device:", dev)
    val_ds = RSSampleDataset(Path(a.data) / "val", limit=a.limit)
    rot_k = int(val_ds[0]["rot_k"])
    val_dl = DataLoader(val_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)
    preview = next(iter(DataLoader(val_ds, batch_size=4)))
    size = val_ds[0]["rs_a"].shape[-1]

    if a.eval:
        model, ck = load_model(a.eval, dev)
        t0 = time.time()
        r, rows = evaluate(model, val_dl, dev, names=[d.name for d in val_ds.dirs])
        r["ms_per_pair_total"] = (time.time() - t0) / len(val_ds) * 1000
        r["ms_per_pair"] = inference_ms(model, dev, size)
        r["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        for k, v in r.items():
            print(f"{k:>18}: {v:.4f}")
        ej = Path(a.eval_json) if a.eval_json else Path(a.eval).parent / "eval.json"
        prev = ej.with_name(ej.stem + "_preview.png")      # next to the JSON: old runs are not overwritten
        save_preview(model, preview, dev, prev)
        out = {"checkpoint": a.eval, "mode": ck["mode"], "data": str(Path(a.data) / "val"), "metrics": r,
               **run_context(dev)}
        ej.write_text(json.dumps(out, indent=1))
        if a.per_sample:
            with open(a.per_sample, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader(); wr.writerows(rows)
        print("saved", ej, "and", prev)
        return

    name = a.name or {"refine": "raft_unet", "plain": "unet_plain"}.get(a.mode, a.mode) + ("_single" if a.single else "")
    run = Path(a.out) / name
    run.mkdir(parents=True, exist_ok=True)
    train_ds = RSSampleDataset(Path(a.data) / "train", limit=a.limit)
    train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                          pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    if a.mode == "refine":
        if a.raft_ckpt:
            rk = torch.load(a.raft_ckpt, map_location="cpu")
            ra, raft_sd = rk["model_args"], rk["raft"]
        else:
            ra, raft_sd = dict(size="large", iters=12), None
        model_args = dict(width=a.width, raft_size=ra["size"], raft_iters=ra["iters"], rot_k=rot_k,
                          train_raft=a.raft_lr > 0)
        model = build("refine", **model_args)
        if raft_sd is None:
            raft_sd = RaftGeom(ra["size"], pretrained=True).raft.state_dict()
        model.geom.raft.load_state_dict(raft_sd)
    else:
        model_args = dict(width=a.width, rot_k=rot_k, single=a.single)
        model = build(a.mode, **model_args)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"])
    model = model.to(dev)

    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("geom.")], "lr": a.lr}]
    if a.mode == "refine" and a.raft_lr > 0:
        groups.append({"params": list(model.geom.parameters()), "lr": a.raft_lr})
    trainable = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=a.epochs * len(train_dl), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    start, best = 0, -math.inf
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, best = ck["epoch"], ck["best"]

    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    n_tr = sum(p.numel() for p in trainable) / 1e6
    print(f"{name} ({a.mode}): {n_par:.2f} M params ({n_tr:.2f} M trainable), "
          f"{len(train_ds)} train / {len(val_ds)} val samples")
    info = {"name": name, "mode": a.mode, "command_args": vars(a), "model_args": model_args,
            "params_M": n_par, "trainable_M": n_tr, "n_train": len(train_ds), "n_val": len(val_ds),
            "dataset": dataset_info(a.data), **run_context(dev)}
    (run / "run_info.json").write_text(json.dumps(info, indent=1, default=str))

    log = open(run / "log.csv", "a", newline="")
    wr = csv.writer(log)
    if start == 0:
        r0, _ = evaluate(model, val_dl, dev)
        msg = f"before training: val PSNR {r0['psnr']:.2f} dB (input {r0['psnr_in']:.2f}"
        print(msg + (f", RAFT+geometry {r0['psnr_geom']:.2f})" if "psnr_geom" in r0 else ")"))
        save_preview(model, preview, dev, run / "preview_ep00.png")
        wr.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "val_psnr_full", "geom_psnr", "geom_ssim",
                     "input_psnr", "input_ssim", "blend_b", "sec"])
        wr.writerow([0, "", r0["psnr"], r0["ssim"], r0["psnr_full"], r0.get("psnr_geom", ""),
                     r0.get("ssim_geom", ""), r0["psnr_in"], r0["ssim_in"], r0.get("blend_b", ""), 0])

    frozen, last_full = 0, math.nan
    for ep in range(start, a.epochs):
        model.train()
        t0, tot, nb, skipped = time.time(), 0.0, 0, 0
        for b in train_dl:
            b = to_dev(b, dev)
            pred, aux = forward(model, b)
            loss = masked_l1(pred, b["gs_a"], loss_mask(aux, pred))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            sched.step()
            skipped += scaler.is_enabled() and scaler.get_scale() < scale   # inf/NaN gradients: step skipped
            tot, nb = tot + loss.item(), nb + 1
        r, _ = evaluate(model, val_dl, dev)
        dt = time.time() - t0
        print(f"ep {ep + 1:3d}/{a.epochs} loss {tot / nb:.4f} | val PSNR {r['psnr']:.2f} dB "
              + (f"(RAFT+geom {r['psnr_geom']:.2f}, " if "psnr_geom" in r else "(")
              + f"input {r['psnr_in']:.2f}) SSIM {r['ssim']:.4f} | full image {r['psnr_full']:.2f} dB"
              + (f" | from B {100 * r['blend_b']:.1f}%" if "blend_b" in r else "")
              + (f" | geometry on {100 * r['valid_geom']:.1f}%" if "valid_geom" in r else "")
              + (f" | SKIPPED {skipped}/{nb} steps" if skipped else "") + f" | {dt:.0f}s", flush=True)
        wr.writerow([ep + 1, tot / nb, r["psnr"], r["ssim"], r["psnr_full"], r.get("psnr_geom", ""),
                     r.get("ssim_geom", ""), r["psnr_in"], r["ssim_in"], r.get("blend_b", ""), dt])
        log.flush()
        ck = {"mode": a.mode, "model_args": model_args, "model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": ep + 1, "best": max(best, r["psnr"]), "val": r}
        torch.save(ck, run / "last.pt")
        if r["psnr"] > best:
            best = r["psnr"]
            torch.save({k: v for k, v in ck.items() if k not in ("opt", "sched")}, run / "best.pt")
            (run / "results.json").write_text(json.dumps(
                {"best_epoch": ep + 1, "val_metrics": r, "finished": datetime.now().isoformat(timespec="seconds")},
                indent=1))
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == a.epochs:
            save_preview(model, preview, dev, run / f"preview_ep{ep + 1:02d}.png")
        if skipped > nb // 2:   # the weights no longer change: stop instead of burning GPU time
            raise SystemExit(f"STOP: {skipped}/{nb} optimizer steps skipped in epoch {ep + 1} "
                             f"(non-finite gradients), training is stuck. Last checkpoint: {run / 'last.pt'}")
        # dead output: worse than the input and identical for 3 epochs (e.g. the output saturated in
        # clamp(0, 1), so no gradient reaches the network; 2026-11-14 corr --single: 7.25 dB from epoch 7)
        frozen = frozen + 1 if (abs(r["psnr_full"] - last_full) < 1e-3
                                and r["psnr_full"] < r["psnr_full_in"]) else 0
        last_full = r["psnr_full"]
        if frozen >= 2:
            raise SystemExit(f"STOP: val psnr_full stuck at {last_full:.2f} dB (input {r['psnr_full_in']:.2f}) "
                             f"for 3 epochs, the network output is dead. Last checkpoint: {run / 'last.pt'}")
    print(f"done. best val PSNR {best:.2f} dB -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
