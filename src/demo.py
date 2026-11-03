"""Demo: RS pair -> RAFT + RS geometry -> GS estimate, as a picture and an animation.

Input: sample folders in the dataset format (rs_a.png, rs_b.png as camera B saves it,
meta.json with readout_a_s, sync_offset_s, baseline_px, b_rot_k; gs_a.png optional),
e.g. examples/kolo (python src/synth_shapes.py) or a sample of data/rs_video_x4k.
Or two images directly with --a / --b and the timing on the command line.

  python src/demo.py examples/kolo examples/x4k_val_00839_01 --out demo.png --gif demo.gif
  python src/demo.py --a rs_a.png --b rs_b.png --readout-ms 20 --sync-ms 0 --baseline 3 --out wynik.png
  python src/demo.py examples/kolo --ckpt runs/<run>/best.pt --compare-v1 --out demo.png

Picture columns: RS A | RS B (rotated back to A) | result | GS target (if known) [| result of geometry v1]
GIF: RS A and the result alternate, so the straightening is easy to see.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from raft_geom import load_raft_geom


def load_sample(folder):
    d = Path(folder)
    m = json.loads((d / "meta.json").read_text())
    k = m.get("b_rot_k", 1)
    gs = cv2.imread(str(d / "gs_a.png")) if (d / "gs_a.png").exists() else None
    return dict(name=d.name, rs_a=cv2.imread(str(d / "rs_a.png")),
                rs_b=np.ascontiguousarray(np.rot90(cv2.imread(str(d / "rs_b.png")), -k)), gs_a=gs,
                T=m["readout_a_s"] * 1000, sync=m.get("sync_offset_s", 0) * 1000, base=m.get("baseline_px", 0), rot_k=k)


def to_t(im):
    return torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float()[None] / 255


def to_u8(x):
    return (x[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 0.5).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) / 255 ** 2
    return 10 * np.log10(1 / max(mse, 1e-10))


@torch.no_grad()
def correct(model, s, dev="cpu"):
    f = lambda v: torch.tensor([float(v)], device=dev)
    pred, aux = model(to_t(s["rs_a"]).to(dev), to_t(s["rs_b"]).to(dev), f(s["T"]), f(s["sync"]), f(s["base"]))
    return to_u8(pred), aux


def label(im, top, bottom=None, scale=1.0):
    im = im.copy()
    h = int(18 * scale)
    for txt, y0 in ((top, 0), (bottom, im.shape[0] - h)):
        if txt:
            cv2.rectangle(im, (0, y0), (im.shape[1], y0 + h), (0, 0, 0), -1)
            cv2.putText(im, txt, (4, y0 + int(13 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (255, 255, 255),
                        1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="*", help="sample folders")
    ap.add_argument("--a", help="RS image of camera A (instead of a sample folder)")
    ap.add_argument("--b", help="RS image of camera B, as the camera saves it (rotated)")
    ap.add_argument("--readout-ms", type=float, default=20.0)
    ap.add_argument("--sync-ms", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=0.0)
    ap.add_argument("--rot-k", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="fine-tuned RAFT (best.pt); default: pretrained RAFT")
    ap.add_argument("--geom", default="v2", choices=["v1", "v2"])
    ap.add_argument("--compare-v1", action="store_true", help="add a column with the old geometry (v1)")
    ap.add_argument("--scale", type=float, default=2.0, help="enlarge the picture")
    ap.add_argument("--out", default="demo.png")
    ap.add_argument("--gif", default=None, help="also save an animation RS A <-> result")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [load_sample(s) for s in a.samples]
    if a.a:
        A, B = cv2.imread(a.a), cv2.imread(a.b)
        samples.append(dict(name=Path(a.a).stem, rs_a=A, rs_b=np.ascontiguousarray(np.rot90(B, -a.rot_k)), gs_a=None,
                            T=a.readout_ms, sync=a.sync_ms, base=a.baseline, rot_k=a.rot_k))
    if not samples:
        ap.error("give sample folders or --a / --b")
    torch.set_grad_enabled(False)
    model = load_raft_geom(a.ckpt, dev, rot_k=samples[0]["rot_k"], geom=a.geom)
    model_v1 = load_raft_geom(a.ckpt, dev, rot_k=samples[0]["rot_k"], geom="v1") if a.compare_v1 else None

    up = lambda im: cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
    rows, frames = [], []
    for s in samples:
        out, aux = correct(model, s, dev)
        has_gs = s["gs_a"] is not None
        p_in = f"PSNR {psnr(s['rs_a'], s['gs_a']):.1f} dB" if has_gs else None
        p_out = f"PSNR {psnr(out, s['gs_a']):.1f} dB" if has_gs else None
        print(f"{s['name']}: T {s['T']:.1f} ms, sync {s['sync']:.1f} ms, baza {s['base']:.1f} px"
              + (f" | wejscie {p_in}, wynik {p_out}" if has_gs else ""))
        tiles = [label(up(s["rs_a"]), "RS A - wejscie (czyta wiersze)", p_in, a.scale / 2),
                 label(up(s["rs_b"]), "RS B - wejscie (czyta kolumny)", None, a.scale / 2),
                 label(up(out), "WYNIK: RAFT + geometria", p_out, a.scale / 2)]
        if has_gs:
            tiles.append(label(up(s["gs_a"]), "GS - jak powinno byc", None, a.scale / 2))
        if model_v1 is not None:
            o1, _ = correct(model_v1, s, dev)
            tiles.append(label(up(o1), "stara geometria (v1)",
                               f"PSNR {psnr(o1, s['gs_a']):.1f} dB" if has_gs else None, a.scale / 2))
        rows.append(np.hstack(tiles))
        frames.append((label(up(s["rs_a"]), "PRZED: obraz z kamery A (rolling shutter)", s["name"], a.scale / 2),
                       label(up(out), "PO: RAFT + geometria", s["name"], a.scale / 2)))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows))
    print("saved", a.out)
    if a.gif:
        from PIL import Image
        imgs = []
        for before, after in frames:
            imgs += [Image.fromarray(before[..., ::-1]), Image.fromarray(after[..., ::-1])]
        imgs[0].save(a.gif, save_all=True, append_images=imgs[1:], duration=900, loop=0)
        print("saved", a.gif)


if __name__ == "__main__":
    main()
