"""PyTorch Dataset for folders produced by synth_rs.py.

Each item (float tensors, images in [0, 1]):
  rs_a   (3,S,S)  RS image of camera A
  rs_b   (3,S,S)  RS image of camera B, ROTATED BACK into A's orientation,
                  so the two inputs are roughly pixel-aligned for the CNN
  gs_a   (3,S,S)  target: global-shutter image of camera A
  flow   (2,S,S)  ground-truth displacement: gs_a[y,x] = rs_a[y+dy, x+dx]
  mask   (1,S,S)  1 where that GS pixel is visible in rs_a (use in loss/metrics)
  flow_b (2,S,S)  the same for rs_b (aligned): gs_a[y,x] = rs_b[y+dy, x+dx]
  mask_b (1,S,S)  1 where the GS pixel is visible in rs_b
  mask_any        1 where visible in at least one camera (what can be recovered)
  motion (9,)     motion over A's readout [theta, beta, tau], f (scalar) focal px
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from synth_rs import flow_from_meta


def _to_tensor(img):
    if img.ndim == 2:
        img = img[..., None]
    else:
        img = img[..., ::-1]  # BGR -> RGB
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0


class RSPairDataset(Dataset):
    def __init__(self, root, limit=None):
        self.dirs = sorted(p for p in Path(root).iterdir() if p.is_dir())
        if limit:
            self.dirs = self.dirs[:limit]
        if not self.dirs:
            raise RuntimeError(f"no samples in {root}")

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        meta = json.loads((d / "meta.json").read_text())
        k = meta.get("b_rot_k", 1)
        rs_a = cv2.imread(str(d / "rs_a.png"), cv2.IMREAD_COLOR)
        rs_b = cv2.imread(str(d / "rs_b.png"), cv2.IMREAD_COLOR)
        gs_a = cv2.imread(str(d / "gs_a.png"), cv2.IMREAD_COLOR)
        rs_b = np.rot90(rs_b, -k)                      # back to A's orientation

        flow = flow_from_meta(meta)                    # (S,S,2), camera A
        flow_b = flow_from_meta(meta, cam="b")         # camera B (rotated back)
        S = flow.shape[0]
        ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)

        def visible(fl):
            px, py = xs + fl[..., 0], ys + fl[..., 1]
            return ((px >= 0) & (px <= S - 1) & (py >= 0) & (py <= S - 1)).astype(np.float32)

        mask, mask_b = visible(flow), visible(flow_b)

        Tr = meta["readout_a_s"]
        motion = np.concatenate([np.array(meta["w_rad_s"]) * Tr,             # theta
                                 np.array(meta["a_rad_s2"]) * Tr ** 2 / 2,   # beta
                                 np.array(meta["v_planeunits_s"]) * Tr])     # tau
        return {
            "rs_a": _to_tensor(rs_a), "rs_b": _to_tensor(rs_b), "gs_a": _to_tensor(gs_a),
            "flow": torch.from_numpy(flow.transpose(2, 0, 1).copy()),
            "mask": torch.from_numpy(mask[None]),
            "flow_b": torch.from_numpy(flow_b.transpose(2, 0, 1).copy()),
            "mask_b": torch.from_numpy(mask_b[None]),
            "mask_any": torch.from_numpy(np.maximum(mask, mask_b)[None]),
            "motion": torch.from_numpy(motion.astype(np.float32)),
            "f": torch.tensor(meta["f"], dtype=torch.float32),
        }
