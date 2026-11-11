"""Zero-shot evaluation and fine-tuning of RAFT + RS geometry (raft_geom.py).

  # pretrained RAFT, no training: how far does geometry alone get us?
  python src/train_raft.py --data dataset_video --zero-shot --out runs --name 2026-10-24_raft_zeroshot
  # fine-tuning (only RAFT learns; geometry is fixed)
  python src/train_raft.py --data dataset_video --epochs 10 --bs 8 --out runs --name 2026-10-24_raft_geom
  # evaluate a checkpoint
  python src/train_raft.py --data dataset_video --eval runs/<run>/best.pt --eval-json eval.json
  # continue from a fine-tuned checkpoint (e.g. trained with v1) with the default geometry v2
  python src/train_raft.py --data dataset_video --init runs/<run>/best.pt --epochs 5 --name 2026-11-03_raft_geom_v2
  # evaluate with the old geometry (v1: no object-edge handling, camera B not used for the output)
  python src/train_raft.py --data dataset_video --eval runs/<run>/best.pt --geom v1

Loss: L1 between the corrected image and the target, where the source pixel
lies inside image A. Metrics on the same region: psnr/ssim (and *_in = no correction).
Works on video datasets (synth_video.py) and on the camera-motion set (synth_rs.py).
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from raft_geom import RaftGeom, RSSampleDataset, load_raft_geom
from train_common import masked_l1, psnr, ssim, run_context, dataset_info


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def forward(model, b):
    return model(b["rs_a"], b["rs_b"], b["T"], b["sync"], b["base"])


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    acc = {k: [] for k in ("psnr", "ssim", "psnr_in", "ssim_in", "valid")}
    for b in loader:
        b = to_dev(b, dev)
        pred, aux = forward(model, b)
        m = aux["valid"]
        acc["psnr"].append(psnr(pred, b["gs_a"], m)); acc["ssim"].append(ssim(pred, b["gs_a"], m))
        acc["psnr_in"].append(psnr(b["rs_a"], b["gs_a"], m)); acc["ssim_in"].append(ssim(b["rs_a"], b["gs_a"], m))
        acc["valid"].append(m.flatten(1).mean(1))
    return {k: torch.cat(v).mean().item() for k, v in acc.items()}


@torch.no_grad()
def save_preview(model, batch, dev, path, n=4):
    import cv2
    from torchvision.utils import flow_to_image
    model.eval()
    b = to_dev({k: v[:n] for k, v in batch.items()}, dev)
    pred, aux = forward(model, b)
    to8 = lambda x: (x.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8)
    rows = []
    for i in range(pred.shape[0]):
        fl = flow_to_image(aux["flow_ab"][i].float().cpu()).numpy().transpose(1, 2, 0)[..., ::-1]
        conf = (aux["conf"][i, 0].cpu().numpy() * 255).astype(np.uint8)
        err = (pred[i] - b["gs_a"][i]).abs().mean(0).cpu().numpy()
        rows.append(np.hstack([to8(b["rs_a"][i]), to8(b["rs_b"][i]), to8(pred[i]), to8(b["gs_a"][i]), fl,
                               cv2.cvtColor(conf, cv2.COLOR_GRAY2BGR),
                               cv2.applyColorMap(np.clip(err * 4 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)]))
    # columns: RS A | RS B (aligned) | output | target | RAFT flow A->B | confidence | |output - target|
    cv2.imwrite(str(path), np.vstack(rows))


def freeze_bn(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--zero-shot", action="store_true", help="evaluate pretrained RAFT, no training")
    ap.add_argument("--eval", default=None, help="checkpoint to evaluate")
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--size", default="large", choices=["large", "small"])
    ap.add_argument("--iters", type=int, default=12, help="RAFT refinement iterations")
    ap.add_argument("--geom", default="v2", choices=["v1", "v2"],
                    help="v2 (default): object edges + pixels unseen by A taken from B (RAFT runs A->B and B->A)")
    ap.add_argument("--freeze-fnet", action="store_true", help="keep RAFT's feature encoder frozen")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="(debug) random RAFT weights")
    ap.add_argument("--init", default=None, help="start fine-tuning from this checkpoint's RAFT weights (best.pt)")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("device:", dev)
    val_ds = RSSampleDataset(Path(a.data) / "val", limit=a.limit)
    rot_k = int(val_ds[0]["rot_k"])
    val_dl = DataLoader(val_ds, batch_size=a.bs, shuffle=False, num_workers=a.workers)
    preview = next(iter(DataLoader(val_ds, batch_size=4)))

    if a.eval or a.zero_shot:
        if a.eval:
            model = load_raft_geom(a.eval, dev, geom=a.geom)
        else:
            model = RaftGeom(a.size, not a.no_pretrained, a.iters, rot_k, a.geom).to(dev)
        t0 = time.time()
        r = evaluate(model, val_dl, dev)
        r["ms_per_pair"] = (time.time() - t0) / len(val_ds) * 1000
        for k, v in r.items():
            print(f"{k:>12}: {v:.4f}")
        name = a.name or ("zeroshot" if a.zero_shot else Path(a.eval).parent.name)
        run = Path(a.out) / name
        run.mkdir(parents=True, exist_ok=True)
        save_preview(model, preview, dev, run / "preview_eval.png")
        out = {"checkpoint": a.eval or f"pretrained raft_{a.size}", "geom": a.geom, "data": str(Path(a.data) / "val"),
               "metrics": r, **run_context(dev)}
        Path(a.eval_json or run / "eval.json").write_text(json.dumps(out, indent=1))
        print("saved", a.eval_json or run / "eval.json", "and", run / "preview_eval.png")
        return

    name = a.name or "raft_geom"
    run = Path(a.out) / name
    run.mkdir(parents=True, exist_ok=True)
    train_ds = RSSampleDataset(Path(a.data) / "train", limit=a.limit)
    train_dl = DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                          pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    model_args = dict(size=a.size, iters=a.iters, rot_k=rot_k, geom=a.geom)
    model = RaftGeom(pretrained=not a.no_pretrained, **model_args).to(dev)
    if a.init:
        model.raft.load_state_dict(torch.load(a.init, map_location=dev)["raft"])
        print("RAFT weights from", a.init)
    if a.freeze_fnet:
        for p in model.raft.feature_encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl), pct_start=0.05)
    info = {"name": name, "command_args": vars(a), "model_args": model_args, "n_train": len(train_ds),
            "n_val": len(val_ds), "dataset": dataset_info(a.data), **run_context(dev)}
    (run / "run_info.json").write_text(json.dumps(info, indent=1, default=str))

    r0 = evaluate(model, val_dl, dev)
    print(f"before fine-tuning (pretrained): val PSNR {r0['psnr']:.2f} dB (input {r0['psnr_in']:.2f})")
    save_preview(model, preview, dev, run / "preview_ep00.png")
    log = open(run / "log.csv", "a")
    log.write("epoch,train_loss,val_psnr,val_ssim,input_psnr,input_ssim,sec\n")
    log.write(f"0,,{r0['psnr']},{r0['ssim']},{r0['psnr_in']},{r0['ssim_in']},0\n")
    best = -math.inf
    for ep in range(a.epochs):
        model.train()
        freeze_bn(model)
        t0, tot, nb = time.time(), 0.0, 0
        for b in train_dl:
            b = to_dev(b, dev)
            pred, aux = forward(model, b)
            loss = masked_l1(pred, b["gs_a"], aux["valid"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        r = evaluate(model, val_dl, dev)
        dt = time.time() - t0
        print(f"ep {ep + 1:3d}/{a.epochs} loss {tot / nb:.4f} | val PSNR {r['psnr']:.2f} dB (input {r['psnr_in']:.2f}) "
              f"SSIM {r['ssim']:.4f} (input {r['ssim_in']:.4f}) | {dt:.0f}s", flush=True)
        log.write(f"{ep + 1},{tot / nb},{r['psnr']},{r['ssim']},{r['psnr_in']},{r['ssim_in']},{dt}\n")
        log.flush()
        ck = {"raft": model.raft.state_dict(), "model_args": model_args, "epoch": ep + 1, "val": r}
        torch.save(ck, run / "last.pt")
        if r["psnr"] > best:
            best = r["psnr"]
            torch.save(ck, run / "best.pt")
            (run / "results.json").write_text(json.dumps(
                {"best_epoch": ep + 1, "val_metrics": r, "pretrained_val_metrics": r0,
                 "finished": datetime.now().isoformat(timespec="seconds")}, indent=1))
        if (ep + 1) % 2 == 0 or ep + 1 == a.epochs:
            save_preview(model, preview, dev, run / f"preview_ep{ep + 1:02d}.png")
    print(f"done. best val PSNR {best:.2f} dB -> {run / 'best.pt'}")


if __name__ == "__main__":
    main()
