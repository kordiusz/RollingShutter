"""Train / evaluate the dual-RS correction model.

Modes:
  motion - CNN predicts 9 camera-motion parameters; the displacement field is
           computed from exact rolling-shutter geometry (main model)
  flow   - U-Net predicts a dense displacement field directly
  direct - U-Net predicts the corrected image directly
  refine - stage 2: frozen MotionNet + refinement U-Net (local flow fixes for
           A and B, blend mask filling A's missing borders from B, image residual)
Train:
  python src/train.py --data dataset --mode motion --name motion_dual
  python src/train.py --data dataset --mode motion --single --name motion_single   # ablation: A only
  python src/train.py --data dataset --mode flow   --name flow_dual
  python src/train.py --data dataset --mode refine --motion-ckpt runs/motion_dual/best.pt --name refine
Evaluate (PSNR, SSIM, flow error, inference time) on dataset/val:
  python src/train.py --data dataset --eval runs/motion_dual/best.pt

Outputs go to runs/<name>/: best.pt, last.pt, log.csv, preview_epXX.png
Metrics: psnr/ssim where the GS pixel is visible in rs_a (comparable across
all models); psnr_b where it is visible ONLY in rs_b (the border that left A's
frame: only the refine model can recover it). "_in" = no correction (rs_a vs
gs_a), the baseline to beat. epe = mean flow error in px (epe_in: zero flow).
"""
import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import build_model
from rs_dataset import RSPairDataset


# ------------------------------- metrics ---------------------------------- #
def masked_l1(a, b, m):
    return (m * (a - b).abs()).sum() / (m.sum() * a.shape[1] + 1e-8)


def masked_mse(a, b, m):
    return (m * (a - b) ** 2).sum() / (m.sum() * a.shape[1] + 1e-8)


def psnr(a, b, m):
    """Per-image masked PSNR, returns tensor (N,)."""
    mse = (m * (a - b) ** 2).flatten(1).sum(1) / (m.flatten(1).sum(1) * a.shape[1] + 1e-8)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gauss(ws=11, sigma=1.5, device="cpu"):
    g = torch.exp(-(torch.arange(ws, device=device) - ws // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a, b, m):
    """Per-image masked SSIM (standard 11x11 Gaussian window), returns (N,)."""
    c = a.shape[1]
    w = _gauss(device=a.device).expand(c, 1, 11, 11)
    f = lambda x: F.conv2d(x, w, padding=5, groups=c)
    mu_a, mu_b = f(a), f(b)
    s_aa, s_bb, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    smap = smap.mean(1, keepdim=True)
    return (smap * m).flatten(1).sum(1) / (m.flatten(1).sum(1) + 1e-8)


# ------------------------------- helpers ---------------------------------- #
def to_dev(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None):
    model.eval()
    acc = {"psnr": [], "ssim": [], "psnr_in": [], "ssim_in": [], "epe": [], "epe_in": [], "epe_b": []}
    se_b, se_b_in, n_b = 0.0, 0.0, 0.0           # border visible only in B: pooled MSE
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        b = to_dev(b, dev)
        pred, flow, aux = model(b["rs_a"], b["rs_b"], b["f"])
        pred = pred.float()
        m = b["mask"]
        acc["psnr"].append(psnr(pred, b["gs_a"], m))
        acc["ssim"].append(ssim(pred, b["gs_a"], m))
        acc["psnr_in"].append(psnr(b["rs_a"], b["gs_a"], m))
        acc["ssim_in"].append(ssim(b["rs_a"], b["gs_a"], m))
        mag = lambda fl: (fl ** 2).sum(1, keepdim=True).sqrt()
        mmean = lambda x, mk: (x * mk).flatten(1).sum(1) / mk.flatten(1).sum(1).clamp_min(1)
        acc["epe_in"].append(mmean(mag(b["flow"]), m))
        if flow is not None:
            acc["epe"].append(mmean(mag(flow.float() - b["flow"]), m))
        if isinstance(aux, dict) and "flow_b" in aux:
            acc["epe_b"].append(mmean(mag(aux["flow_b"].float() - b["flow_b"]), b["mask_b"]))
        mb = b["mask_any"] * (1 - m)
        se_b += (mb * (pred - b["gs_a"]) ** 2).sum().item()
        se_b_in += (mb * (b["rs_a"] - b["gs_a"]) ** 2).sum().item()
        n_b += mb.sum().item() * 3
    r = {k: torch.cat(v).mean().item() for k, v in acc.items() if v}
    if n_b > 0:
        r["psnr_b"] = 10 * math.log10(n_b / max(se_b, 1e-10))
        r["psnr_b_in"] = 10 * math.log10(n_b / max(se_b_in, 1e-10))
    return r


def compute_loss(mode, pred, flow, aux, b, flow_weight):
    """Training loss for each mode (see module docstring)."""
    pred = pred.float()
    if mode == "refine":
        # image over everything visible in at least one camera (borders get filled from B)
        loss = masked_l1(pred, b["gs_a"], b["mask_any"])
        if flow_weight > 0:
            loss = loss + flow_weight * (masked_mse(flow.float(), b["flow"], b["mask"]) +
                                         masked_mse(aux["flow_b"].float(), b["flow_b"], b["mask_b"]))
        return loss
    loss = masked_l1(pred, b["gs_a"], b["mask"])
    if flow is not None and flow_weight > 0:
        # MSE, not L1: with L1 the network got stuck predicting zero motion
        # (the median of symmetric targets); MSE gives a gradient proportional to the error
        loss = loss + flow_weight * masked_mse(flow.float(), b["flow"], b["mask"])
    return loss


@torch.no_grad()
def save_preview(model, batch, dev, path, n=4):
    import cv2
    model.eval()
    b = to_dev(batch, dev)
    pred, _, _ = model(b["rs_a"][:n], b["rs_b"][:n], b["f"][:n])
    rows = []
    for i in range(min(n, pred.shape[0])):
        ims = [b["rs_a"][i], b["rs_b"][i], pred[i].float(), b["gs_a"][i]]
        ims = [(x.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8) for x in ims]
        # visibility map: green = seen by A, blue = only by B, red = by neither (BGR)
        ma, many = b["mask"][i, 0].cpu().numpy() > 0, b["mask_any"][i, 0].cpu().numpy() > 0
        vis = np.zeros_like(ims[0])
        vis[ma] = (60, 170, 60)
        vis[many & ~ma] = (200, 120, 40)
        vis[~many] = (40, 40, 200)
        rows.append(np.hstack(ims + [vis]))
    # columns: RS A | RS B (aligned) | output | GS target | visibility map
    cv2.imwrite(str(path), np.vstack(rows))


@torch.no_grad()
def inference_time(model, dev, size, reps=50):
    model.eval()
    x = torch.rand(1, 3, size, size, device=dev)
    f = torch.full((1,), size * 0.9, device=dev)
    for _ in range(5):
        model(x, x, f)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        model(x, x, f)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000


def run_context(dev):
    """Where/what this run is: git commit of the code, GPU, versions."""
    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], capture_output=True, text=True,
                                  cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def dataset_info(root):
    root = Path(root)
    info = {"path": str(root)}
    cfg = root / "config.json"
    if cfg.exists():
        info["generator_config"] = json.loads(cfg.read_text())
    return info


# --------------------------------- main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root with train/ and val/")
    ap.add_argument("--mode", default="motion", choices=["motion", "flow", "direct", "refine"])
    ap.add_argument("--motion-ckpt", default=None, help="refine mode: trained dual MotionNet (best.pt)")
    ap.add_argument("--single", action="store_true", help="use camera A only (ablation)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--flow-weight", type=float, default=None,
                    help="weight of the supervised flow loss (MSE in px^2); default 1.0, "
                         "0.02 in refine mode; 0 = photometric only")
    ap.add_argument("--width", type=int, default=32, help="base channels of the U-Net")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="use only N training samples (debug)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval", default=None, help="checkpoint to evaluate instead of training")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-json", default=None, help="eval mode: also save the metrics to this JSON file")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    val_ds = RSPairDataset(Path(a.data) / "val")
    val_dl = DataLoader(val_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)

    if a.eval:
        ck = torch.load(a.eval, map_location=dev)
        model = build_model(**ck["model_args"]).to(dev)
        model.load_state_dict(ck["model"])
        r = evaluate(model, val_dl, dev)
        size = val_ds[0]["rs_a"].shape[-1]
        r["ms_per_frame"] = inference_time(model, dev, size)
        r["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        for k, v in r.items():
            print(f"{k:>12}: {v:.4f}")
        if a.eval_json:
            out = {"checkpoint": a.eval, "data": str(Path(a.data) / "val"), "metrics": r, **run_context(dev)}
            Path(a.eval_json).write_text(json.dumps(out, indent=1))
            print("saved", a.eval_json)
        return

    name = a.name or f"{a.mode}_{'single' if a.single else 'dual'}"
    run = Path(a.out) / name
    run.mkdir(parents=True, exist_ok=True)
    train_ds = RSPairDataset(Path(a.data) / "train", limit=a.limit)
    train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                          drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    if a.flow_weight is None:
        a.flow_weight = 0.02 if a.mode == "refine" else 1.0
    if a.mode == "refine":
        if not a.motion_ckpt:
            raise SystemExit("--mode refine needs --motion-ckpt runs/motion_dual/best.pt")
        mck = torch.load(a.motion_ckpt, map_location="cpu")
        model_args = dict(mode="refine", width=a.width, motion_args=mck["model_args"])
        model = build_model(**model_args)
        model.motion.load_state_dict(mck["model"])        # stage-1 weights, kept frozen
        model = model.to(dev)
    else:
        model_args = dict(mode=a.mode, single=a.single, width=a.width)
        model = build_model(**model_args).to(dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params "
          f"({sum(p.numel() for p in trainable) / 1e6:.2f} M trainable), "
          f"{len(train_ds)} train / {len(val_ds)} val samples")
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best = 0, -math.inf
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, best = ck["epoch"] + 1, ck["best"]

    info = {"name": name, "command_args": vars(a), "model_args": model_args,
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "trainable_M": sum(p.numel() for p in trainable) / 1e6,
            "n_train": len(train_ds), "n_val": len(val_ds),
            "dataset": dataset_info(a.data), **run_context(dev)}
    (run / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    best_r, best_ep = None, None
    preview_batch = next(iter(DataLoader(val_ds, batch_size=4)))
    log = open(run / "log.csv", "a", newline="")
    wr = csv.writer(log)
    if start == 0:
        wr.writerow(["epoch", "train_loss", "val_psnr", "val_ssim", "input_psnr", "input_ssim", "val_epe",
                     "val_psnr_b", "input_psnr_b", "sec"])

    for ep in range(start, a.epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for b in train_dl:
            b = to_dev(b, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                pred, flow, aux = model(b["rs_a"], b["rs_b"], b["f"])
            loss = compute_loss(a.mode, pred, flow, aux, b, a.flow_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        r = evaluate(model, val_dl, dev)
        dt = time.time() - t0
        print(f"ep {ep + 1:3d}/{a.epochs} loss {tot / nb:.4f} | val PSNR {r['psnr']:.2f} dB "
              f"(input {r['psnr_in']:.2f}) SSIM {r['ssim']:.4f} (input {r['ssim_in']:.4f})"
              + (f" EPE {r['epe']:.2f}px" if "epe" in r else "")
              + (f" | B-only border PSNR {r['psnr_b']:.2f} (input {r['psnr_b_in']:.2f})" if "psnr_b" in r else "")
              + f" | {dt:.0f}s")
        wr.writerow([ep + 1, tot / nb, r["psnr"], r["ssim"], r["psnr_in"], r["ssim_in"], r.get("epe", ""),
                     r.get("psnr_b", ""), r.get("psnr_b_in", ""), dt])
        log.flush()
        ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  epoch=ep, best=max(best, r["psnr"]), model_args=model_args)
        torch.save(ck, run / "last.pt")
        if r["psnr"] > best:
            best = r["psnr"]
            best_r, best_ep = r, ep + 1
            torch.save(ck, run / "best.pt")
            (run / "results.json").write_text(json.dumps(
                {"best_epoch": best_ep, "epochs_done": ep + 1, "val_metrics": best_r,
                 "finished": datetime.now().isoformat(timespec="seconds")}, indent=1))
        if (ep + 1) % 5 == 0 or ep == 0 or ep + 1 == a.epochs:
            save_preview(model, preview_batch, dev, run / f"preview_ep{ep + 1:02d}.png")
    print(f"done. best val PSNR {best:.2f} dB -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
