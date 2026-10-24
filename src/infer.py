"""Run a trained model locally (CPU is fine) - no training code needed.

One checkpoint file holds everything: a refine checkpoint contains both the
MotionNet and the refinement U-Net weights.

1) On generated samples (with ground truth) - comparison images + metrics:
     python src/infer.py --ckpt refine_best.pt --data dataset/val --n 8 --out results

2) On a real camera pair (one frame from each camera):
     python src/infer.py --ckpt refine_best.pt --a camA.png --b camB.png --f 620 --out results
   or on two folders of frames (video), paired in sorted filename order:
     python src/infer.py --ckpt refine_best.pt --a framesA/ --b framesB/ --f 620 --out results

   --a : frame from camera A, as the camera outputs it (normal orientation)
   --b : frame from camera B, as the camera outputs it (content rotated 90 deg)
   --f : focal length in pixels of the ORIGINAL frames (from calibration;
         both cameras assumed equal). It is rescaled automatically.
   --rot-k : how B's picture is rotated relative to A (np.rot90 k), same as
         b_rot_k used when generating the training data (default 1).
   Preprocessing (same geometry as the training data): B is rotated back,
   both are centre-cropped to the common square and resized to --size.
   Frames should already be undistorted (lens distortion) with OpenCV.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from model import build_model

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    model = build_model(**ck["model_args"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["model_args"]


def to_t(img_bgr, dev):
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].to(dev)


def to_img(t):
    return (t[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 0.5).astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (255, 255, 255), -1)
    cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def blend_map(aux, size):
    """Where the output came from: green = camera A, blue = camera B."""
    if not isinstance(aux, dict) or "blend" not in aux:
        return None
    m = aux["blend"][0, 0].cpu().numpy()
    vis = np.zeros((size, size, 3), np.uint8)
    vis[..., 1] = (m * 200).astype(np.uint8)
    vis[..., 0] = ((1 - m) * 230).astype(np.uint8)
    return vis


@torch.no_grad()
def run(model, rs_a, rs_b, f):
    t = time.perf_counter()
    pred, flow, aux = model(rs_a, rs_b, f)
    return pred, flow, aux, (time.perf_counter() - t) * 1000


def prepare_pair(img_a, img_b, rot_k, size, f_orig):
    """Real frames -> model inputs in the training-data geometry."""
    b_back = np.ascontiguousarray(np.rot90(img_b, -rot_k))
    side = min(img_a.shape[0], img_a.shape[1], b_back.shape[0], b_back.shape[1])

    def centre(im):
        y0, x0 = (im.shape[0] - side) // 2, (im.shape[1] - side) // 2
        return im[y0:y0 + side, x0:x0 + side]

    a, b = centre(img_a), centre(b_back)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA)
    return a, b, f_orig * size / side


def on_dataset(model, a, dev):
    from rs_dataset import RSPairDataset
    from train import psnr
    ds = RSPairDataset(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, times = [], []
    for i in range(min(a.n, len(ds))):
        s = ds[i]
        rs_a, rs_b, gs = s["rs_a"][None].to(dev), s["rs_b"][None].to(dev), s["gs_a"][None].to(dev)
        pred, flow, aux, ms = run(model, rs_a, rs_b, s["f"][None].to(dev))
        times.append(ms)
        m, m_any = s["mask"][None].to(dev), s["mask_any"][None].to(dev)
        p_in, p_out = psnr(rs_a, gs, m).item(), psnr(pred.float(), gs, m).item()
        p_any = psnr(pred.float(), gs, m_any).item()
        res.append((p_in, p_out, p_any))
        S = gs.shape[-1]
        tiles = [label(to_img(rs_a), "RS A"), label(to_img(rs_b), "RS B (obrocona)"),
                 label(to_img(pred), f"wynik {p_out:.1f} dB"), label(to_img(gs), f"cel (wejscie {p_in:.1f} dB)")]
        bm = blend_map(aux, S)
        if bm is not None:
            tiles.append(label(bm, "skad piksel: A ziel., B nieb."))
        cv2.imwrite(str(out / f"{ds.dirs[i].name}.png"), np.hstack(tiles))
        print(f"{ds.dirs[i].name}: PSNR input {p_in:.2f} -> output {p_out:.2f} dB "
              f"(incl. borders seen by B: {p_any:.2f} dB) | {ms:.0f} ms")
    r = np.array(res)
    print(f"\nmean over {len(res)}: input {r[:, 0].mean():.2f} dB -> output {r[:, 1].mean():.2f} dB, "
          f"with borders {r[:, 2].mean():.2f} dB | {np.mean(times[1:] or times):.0f} ms/frame on {dev}")


def list_frames(p):
    p = Path(p)
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
    return [p]


def on_real(model, a, dev):
    fa, fb = list_frames(a.a), list_frames(a.b)
    if len(fa) != len(fb):
        print(f"warning: {len(fa)} frames from A vs {len(fb)} from B, using the first {min(len(fa), len(fb))}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for pa, pb in zip(fa, fb):
        img_a, img_b = cv2.imread(str(pa)), cv2.imread(str(pb))
        ia, ib, f = prepare_pair(img_a, img_b, a.rot_k, a.size, a.f)
        pred, flow, aux, ms = run(model, to_t(ia, dev), to_t(ib, dev),
                                  torch.tensor([f], dtype=torch.float32, device=dev))
        res = to_img(pred)
        cv2.imwrite(str(out / f"{pa.stem}_wynik.png"), res)
        tiles = [label(ia, "kamera A"), label(ib, "kamera B (obrocona)"), label(res, "wynik")]
        bm = blend_map(aux, a.size)
        if bm is not None:
            tiles.append(label(bm, "skad piksel: A ziel., B nieb."))
        cv2.imwrite(str(out / f"{pa.stem}_porownanie.png"), np.hstack(tiles))
        if isinstance(aux, dict) and "params" in aux:
            pr = aux["params"][0].cpu().numpy() * f
        elif torch.is_tensor(aux):
            pr = aux[0].cpu().numpy() * f
        else:
            pr = None
        extra = "" if pr is None else " | ruch [px]: obrot " + np.array2string(pr[:3], precision=1)
        print(f"{pa.name} + {pb.name}: {ms:.0f} ms{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", help="folder with generated samples (e.g. dataset/val)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--a", help="camera A frame or folder of frames")
    ap.add_argument("--b", help="camera B frame or folder of frames")
    ap.add_argument("--f", type=float, help="focal length [px] of the original frames")
    ap.add_argument("--rot-k", type=int, default=1)
    ap.add_argument("--size", type=int, default=256, help="must match the training size")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, args = load_model(a.ckpt, dev)
    print(f"model: {args.get('mode')} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params) on {dev}")
    if a.data:
        on_dataset(model, a, dev)
    elif a.a and a.b:
        if a.f is None:
            raise SystemExit("--f (focal length in px, from calibration) is required for real frames")
        on_real(model, a, dev)
    else:
        raise SystemExit("give either --data or --a/--b")


if __name__ == "__main__":
    main()
