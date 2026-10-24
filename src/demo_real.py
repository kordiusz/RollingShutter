"""Demo on a real camera pair (e.g. the fan): RAFT + RS geometry -> GS estimate.

Input: one frame from camera A and one from camera B, exactly as the cameras
save them (B's picture is rotated 90 deg). Steps:
  1. rotate B back to A's orientation,
  2. find the global shift between the views (stereo baseline + mounting) by
     phase correlation, so the 256x256 crops show the same part of the scene,
  3. cut a size x size crop around CENTER_A (e.g. the fan) from A and the
     matching crop from B, resize to 256x256 if size != 256 (as in the dataset),
  4. readout time and sync of the CROP follow from the full-frame readout time
     and where the crop lies on each sensor,
  5. RAFT (A -> B) once, then the geometry for each tested camera sync offset
     (free-running USB webcams are not synchronised: the unknown offset is swept).

  python src/demo_real.py --a kameraA.jpg --b kameraB.jpg --out wynik.png \
      --center-a 320 240 --t-full 30 --sync-sweep -10 -5 0 5 10 [--ckpt runs/<run>/best.pt]
"""
import argparse

import cv2
import numpy as np
import torch

from raft_geom import correct_from_flow, load_raft_geom


def _gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _center_square(im, L):
    h, w = im.shape[:2]
    y0, x0 = (h - L) // 2, (w - L) // 2
    return im[y0:y0 + L, x0:x0 + L]


def _crop(im, cx, cy, side):
    h, w = im.shape[:2]
    x0 = int(round(np.clip(cx - side / 2, 0, w - side)))
    y0 = int(round(np.clip(cy - side / 2, 0, h - side)))
    return im[y0:y0 + side, x0:x0 + side], (x0 + side / 2, y0 + side / 2)


def prepare_pair(img_a, img_b, rot_k=1, center_a=None, side=256, out=256,
                 t_full_ms=30.0, cam_sync_ms=0.0, align=True):
    """Returns crops (float RGB tensors 1x3xHxW), crop readout time T, the sync
    of the crop centres, and diagnostics."""
    HA, WA = img_a.shape[:2]
    HB = img_b.shape[0]                                   # B's sensor rows = rows of B's own picture
    b_back = np.ascontiguousarray(np.rot90(img_b, -rot_k))
    HBb, WBb = b_back.shape[:2]
    if side > min(HA, WA, HBb, WBb):
        raise ValueError(f"crop {side} larger than the images")
    cxa, cya = center_a if center_a is not None else (WA / 2, HA / 2)

    # global shift between the views: B_back(c_B + v + delta) ~ A(c_A + v)
    dx = dy = 0.0
    if align:
        L = min(HA, WA, HBb, WBb)
        ga, gb = _center_square(_gray(img_a), L), _center_square(_gray(b_back), L)
        win = cv2.createHanningWindow((L, L), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    # B is rotated, so both cameras see only the central (roughly square) region:
    # keep the crop centre where BOTH crops fit without being pushed in from the edge
    h = side / 2
    lo_x = max(h, h - (WBb / 2 - WA / 2 + dx)); hi_x = min(WA - h, WBb - h - (WBb / 2 - WA / 2 + dx))
    lo_y = max(h, h - (HBb / 2 - HA / 2 + dy)); hi_y = min(HA - h, HBb - h - (HBb / 2 - HA / 2 + dy))
    req = (cxa, cya)
    cxa, cya = float(np.clip(cxa, lo_x, hi_x)), float(np.clip(cya, lo_y, hi_y))
    moved = abs(cxa - req[0]) > 0.5 or abs(cya - req[1]) > 0.5
    cxb = WBb / 2 + (cxa - WA / 2) + dx
    cyb = HBb / 2 + (cya - HA / 2) + dy

    ca, (cxa, cya) = _crop(img_a, cxa, cya, side)
    cb, (cxb, cyb) = _crop(b_back, cxb, cyb, side)
    if side != out:
        ca = cv2.resize(ca, (out, out), interpolation=cv2.INTER_AREA)
        cb = cv2.resize(cb, (out, out), interpolation=cv2.INTER_AREA)

    # timing of the crop: rows of A / sensor rows of B (= columns of B_back)
    T_crop = t_full_ms * side / HA
    tA_c = ((cya - 0.5) / (HA - 1) - 0.5) * t_full_ms
    rB = {1: WBb - 1 - (cxb - 0.5), 3: cxb - 0.5}[rot_k % 4]
    tB_c = cam_sync_ms + (rB / (HB - 1) - 0.5) * t_full_ms
    sync_crop = tB_c - tA_c

    to_t = lambda im: torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float()[None] / 255
    return {"rs_a": to_t(ca), "rs_b": to_t(cb), "T": T_crop, "sync": sync_crop, "base": 0.0,
            "crop_a": ca, "crop_b": cb, "center_a": (cxa, cya), "center_b": (cxb, cyb), "side_scale": side / out,
            "shift": (dx, dy), "b_back": b_back, "center_moved": moved,
            "allowed_center_x": (lo_x, hi_x), "allowed_center_y": (lo_y, hi_y)}


def estimate_static_offset(d):
    """Residual offset between the crops when most of the scene is STATIC (camera on
    a tripod, only e.g. the fan moves): static pixels have d = -(bx, by), so the
    median of the flow is the offset. Do not use when the camera itself moves."""
    med = d.flatten(2).median(dim=2).values                # (N,2)
    return -med


@torch.no_grad()
def run_pair(model, pair, sync_offsets=(0.0,), device="cpu", base="static"):
    """RAFT once, geometry for every tested camera sync offset [ms].
    base: "static" = estimate the residual offset from the (mostly static) scene,
    a number = horizontal offset in px, or a pair (bx, by)."""
    a, b = pair["rs_a"].to(device), pair["rs_b"].to(device)
    d = model.flow_ab(a, b)
    if isinstance(base, str):
        bv = estimate_static_offset(d)
    else:
        bv = torch.tensor([list(base) if np.ndim(base) else [float(base), 0.0]], device=device)
    pair["base_used"] = bv[0].tolist()
    f = lambda v: torch.tensor([float(v)], device=device)
    outs = []
    for s in sync_offsets:
        pred, aux = correct_from_flow(a, d, f(pair["T"]), f(pair["sync"] + s), bv, model.rot_k)
        outs.append((s, pred, aux))
    return d, outs


def to_bgr8(t):
    return (t[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255 + 0.5).astype(np.uint8)


def figure(pair, d, outs):
    """One image: inputs + RAFT flow, then the result for each sync offset."""
    from torchvision.utils import flow_to_image

    def lab(im, t):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 18), (255, 255, 255), -1)
        cv2.putText(im, t, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return im
    fl = flow_to_image(d[0].float().cpu()).numpy().transpose(1, 2, 0)[..., ::-1].copy()
    top = [lab(pair["crop_a"], "kamera A (RS)"), lab(pair["crop_b"], "kamera B obrocona (RS)"),
           lab(fl, "RAFT: przeplyw A->B")]
    res = [lab(to_bgr8(p), f"wynik, sync {s:+.0f} ms") for s, p, _ in outs]
    S = pair["crop_a"].shape[0]
    blank = np.full((S, S, 3), 255, np.uint8)
    cols = max(len(top), len(res))
    rows = [top + [blank] * (cols - len(top))]
    for i in range(0, len(res), cols):
        chunk = res[i:i + cols]
        rows.append(chunk + [blank] * (cols - len(chunk)))
    return np.vstack([np.hstack(r) for r in rows])


def overview(img_a, pair, max_w=520):
    """Full frames side by side with the crop boxes: check that B rotated back
    looks upright like A (otherwise change ROT_K) and that the crop covers the fan."""
    a = img_a.copy()
    b = pair["b_back"].copy()
    S = pair["crop_a"].shape[0]
    side = int(round(S * pair.get("side_scale", 1)))
    for im, (cx, cy), col in ((a, pair["center_a"], (0, 200, 0)), (b, pair["center_b"], (0, 120, 255))):
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        cv2.rectangle(im, (x0, y0), (x0 + side, y0 + side), col, 3)
    h = max(a.shape[0], b.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    both = np.hstack([pad(a), pad(b)])
    s = min(1.0, 2 * max_w / both.shape[1])
    return cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="demo_wynik.png")
    ap.add_argument("--ckpt", default=None, help="fine-tuned RAFT+geometry; default: pretrained RAFT")
    ap.add_argument("--rot-k", type=int, default=1)
    ap.add_argument("--center-a", type=float, nargs=2, default=None, help="crop centre in A [px]")
    ap.add_argument("--side", type=int, default=256, help="crop size in the original image")
    ap.add_argument("--t-full", type=float, default=30.0, help="full-frame readout time [ms]")
    ap.add_argument("--sync-sweep", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    ap.add_argument("--no-align", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_raft_geom(a.ckpt, dev, rot_k=a.rot_k)
    model.rot_k = a.rot_k
    pair = prepare_pair(cv2.imread(a.a), cv2.imread(a.b), a.rot_k, a.center_a, a.side,
                        t_full_ms=a.t_full, align=not a.no_align)
    print(f"crop A centre {pair['center_a']}, B centre {pair['center_b']}, shift {pair['shift']}, "
          f"T_crop {pair['T']:.1f} ms, sync of crop centres {pair['sync']:+.1f} ms")
    if pair["center_moved"]:
        print("note: crop centre moved into the region both cameras see:", pair["center_a"])
    d, outs = run_pair(model, pair, a.sync_sweep, dev)
    print("residual offset between the crops (static scene):", np.round(pair["base_used"], 2), "px")
    cv2.imwrite(a.out.replace(".png", "_podglad.png"), overview(cv2.imread(a.a), pair))
    cv2.imwrite(a.out, figure(pair, d, outs))
    print("saved", a.out)


if __name__ == "__main__":
    main()
