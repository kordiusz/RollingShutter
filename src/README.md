# Synthetic dual rolling-shutter dataset

Training data for a model that takes **two rolling-shutter (RS) images**, from
camera A (normal) and camera B (mounted rotated 90°), and outputs the
**global-shutter (GS)** image of camera A.

## The idea in one paragraph

An RS sensor reads one row at a time, so every row is a photo taken at a
slightly different moment. If the camera moves during readout, the image
shears, wobbles and stretches. Camera B is rotated 90°, so in A's frame it
reads **columns** instead of rows. The same motion therefore distorts the two
images in complementary directions (see `preview.png`). That is the extra
information the model can use.

To synthesise data we take an ordinary sharp photo, treat it as the GS view of
a flat scene, and invent a camera motion. For every output pixel we know
*when* it is read (its row for A, its column for B), so we compute the camera
pose at that instant and look up the pixel the camera would see. That is one
`cv2.remap` per image: no holes, and the ground truth is exact.

```
sharp photo ──► random crop ──► sample motion ω(t), v(t), readout, baseline …
                                   │
               per-pixel time t(row)  ─► pose at t ─► ray ─► plane ─► source pixel
                                   │
         rs_a (row readout)   rs_b (column readout, then rot90)   gs_a (t = mid-readout)
```

## Generators (all write the same sample format)

| script | scene | dataset | notes |
|---|---|---|---|
| `synth_rs.py` | photo as a flat static scene, camera motion only | `rs_dataset_256` (stage 1, DIV2K) | exact flow (`--save-corr`); its timing / motion helpers are used by the other generators |
| `synth_video.py` | 1000 fps recordings (X4K1000FPS): camera **and object** motion | `rs_video_x4k` (main) | row y of A from the frame recorded when row y was read |
| `synth_video_v2.py` | same, more clips per scene, `--min-disp` drops near-static samples | extended X4K | |
| `synth_layers.py` | background + cut-out objects at different depths, own motion and spin | layered set (Filip) | real parallax and occlusions, no frame-rate ghosting |

Sample folder: `rs_a.png`, `rs_b.png` (rotated, exactly as camera B outputs it), `gs_a.png` (**target**,
global shutter at the middle of A's readout), `gs_b.png`, `meta.json` (readout time, sync offset, baseline,
`b_rot_k`, ...). Loaded by `raft_geom.RSSampleDataset`.

## Models (`raft_unet.py`, trained with `train_unet.py`)

| `--mode` | what it does |
|---|---|
| `refine` | **main model**: RAFT flow A->B, RS geometry (per-pixel velocity `u = (d + b) / Δt`, back to one moment), then a U-Net fixes local errors, blends A and B (border seen only by B) and adds an image residual |
| `plain` | U-Net on [A, B, coordinates, readout times] -> flow for A + residual; no RAFT, no geometry (baseline) |
| `ae` | autoencoder without skips, latent 16 x 16 x 512, outputs the image directly |
| `corr` | siamese encoder + correlation layer A<->B + decoder with skips, outputs A + residual image |
| `corrgeom` | the `corr` network predicting the flow A->B + fixed RS geometry v1: RAFT + geometry with RAFT replaced by a small network trained from scratch |

`--single` (plain / ae / corr) replaces camera B with a copy of A: the ablation "does camera B help?".
RAFT + geometry alone: `raft_geom.py`, fine-tuning `train_raft.py`. Shared blocks: `model.py`
(U-Net, warp, correlation), metrics: `train_common.py`. Stage 1 (MotionNet) was removed: `docs/etap1_motionnet.md`.

## Quick start (Windows / any OS)

```bash
pip install numpy opencv-python
python src/synth_rs.py --demo --out demo                  # checkerboard test, no data needed
python src/preview.py demo/train --out preview.png

# real run: point --src at a folder of sharp photos
python src/synth_rs.py --src D:/data/DIV2K_train_HR --out dataset --per-image 20 --size 256 --config configs/config_example.json
python src/preview.py dataset/train --n 8 --out preview.png
```

Each sample folder has `rs_a.png`, `rs_b.png` (rotated, exactly as camera B
outputs it), `gs_a.png` (**target**), `gs_b.png` and `meta.json`. With
`--save-corr` it also writes `corr.npz`: for every RS pixel, the (x, y) it
corresponds to in `gs_a`. That is free supervision if you later use a
flow-based model. Train/val are split **by source image**, so no photo leaks
into both.

**Source images:** any large, sharp, static photos work. DIV2K (800 images,
2K resolution) is a good start: 800 × 20 = 16k pairs at 256². Best of all,
add frames from **your own cameras shot on a tripod**. A static camera has no
RS distortion, and the sensor noise and colours then match your hardware.

## Matching the simulator to your 2-camera rig

Treat the model input as the **square overlap** of the two views. With
identical W×H sensors (landscape A, portrait B), the overlap is the central
H×H. Both cameras read all H rows of it, so:

| parameter | how to set it |
|---|---|
| `readout_ms` | full-frame readout time of the sensor (datasheet, or measure with a flickering LED / spinning fan). Typically 10–33 ms. |
| `b_readout_scale` | 1.0 for identical sensors and modes |
| `fov_deg` | the camera's **vertical** FOV (it becomes the overlap's horizontal FOV) |
| `b_rot_k` | 1 if B is rotated so that `np.rot90(imgA, 1)` matches B's view, otherwise -1. **Check this with a quick test:** pan the rig fast and compare with `preview.py` output. |
| `baseline_px` | disparity f·b/Z at your typical scene distance, e.g. f=600 px, b=4 cm, Z=5 m → ~5 px. Use a range. |
| `sync_offset_ms` | 0 with hardware trigger; otherwise the measured start-time jitter between the cameras |
| `max_pan_px` etc. | how far the image moves during one readout in your use case (hand-held: 10–40 px, vehicle: depends on speed and vibration) |

Motion is specified in **pixels of image motion during readout**, not rad/s,
so it is easy to reason about. Angular velocity, acceleration and translation
are derived from it and saved to `meta.json`.

## Assumptions (what this simple version does NOT model)

1. **Flat scene.** Rotation-induced RS (the dominant effect for hand-held and
   vibrating rigs) is exact for any scene. Translation and stereo baseline,
   however, produce one global parallax instead of depth-dependent parallax.
2. **Static scene.** No independently moving objects.
3. **Parallel optical axes.** B differs from A only by the 90° roll and the
   baseline. Real rigs: rectify the images with stereo calibration first.
4. No lens distortion. Undistort real images before feeding the model.

These are the usual simplifications in the "synthesise RS from still images"
line of work (e.g. Rengarajan et al., *Unrolling the Shutter*, CVPR 2017).
Upgrade only if real-data validation shows a gap:

* **depth-varying parallax**: replace the plane with a per-pixel depth map
  (RGB-D datasets or a monocular depth estimate);
* **moving objects / real dynamics**: build RS images row by row from
  high-frame-rate GS video (240 fps + frame interpolation);
* **full realism**: render both cameras in Blender/CARLA;
* always keep a small **real** test set from your rig to measure the gap.
