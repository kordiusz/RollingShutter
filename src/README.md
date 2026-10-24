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

## Training (Colab)

Open `train_colab.ipynb` in Google Colab (GPU runtime). It downloads DIV2K,
generates the dataset with `config_webcam.json`, and trains three models for
comparison:

| run | inputs | what the network predicts | purpose |
|---|---|---|---|
| `motion_dual` | A + B | 9 camera-motion parameters → exact RS geometry → displacement field → warp of A | **main model** |
| `motion_single` | A only | same, without camera B | ablation: does camera B help? |
| `flow_dual` | A + B | dense displacement field (U-Net) | architecture comparison |

`motion` model (`model.py`, `MotionNet`): a shared feature extractor runs on A
and on B (rotated back to A's orientation). A **correlation layer** (cost
volume, as in FlowNet/PWC-Net) measures how A's features match B's under
shifts of up to ±24 px. A small CNN then regresses the motion over the readout:
rotation, its change, and translation. The displacement field is computed from
them by a differentiable version of the generator's geometry. This is the
same idea as Velocity-Net in Zhuang et al. 2019, extended to two cameras.

Loss: MSE on the displacement field (px²) + masked L1 on the image.
`python src/train.py --eval runs/<name>/best.pt` reports PSNR, SSIM, flow error
(EPE) and inference time, always next to the "no correction" baseline.

Lessons from debugging (worth a paragraph in the thesis):
* plain ReLU in the regression head "died" (constant output), so it uses
  LeakyReLU, and training uses gradient clipping (1.0);
* an L1 loss on the displacement field kept the network stuck at "zero
  motion" (the median of symmetric targets); MSE fixed it;
* the network predicts motion in **pixels** and divides by the known focal
  length, because the image alone does not reveal the field of view;
* early fusion (just stacking A and B as channels) did not learn in short
  tests; the explicit correlation layer is what makes comparing the two
  cameras learnable.

## Stage 2: refinement U-Net (`--mode refine`)

`RefinePipeline` in `model.py` = the trained MotionNet (frozen) + a U-Net.
1. MotionNet → 9 motion parameters → global flow for A, and (same motion, B's
   column readout) for B.
2. `OffsetHead`: correlation of A and B after that correction → one global
   (dx, dy) for B (stereo baseline + sync offset, which MotionNet ignores).
3. U-Net input: warped A, warped B, "outside the frame" maps for A and B,
   A's global flow, coordinates. Output: local flow corrections for A and B,
   a blend mask (which camera each pixel comes from) and an image residual.
4. output = mask · warp(A) + (1 − mask) · warp(B) + residual.

Loss: L1 on the image wherever at least one camera saw the scene, plus a
small MSE on both flows (ground truth for B comes from `meta.json`, so the
existing datasets work unchanged). Only the U-Net and OffsetHead are trained.
The checkpoint contains both networks.

## Local inference (`infer.py`, CPU is fine)

```bash
pip install torch opencv-python scipy numpy
python src/infer.py --ckpt refine_best.pt --data dataset/val --n 8 --out results      # synthetic, with metrics
python src/infer.py --ckpt refine_best.pt --a camA.png --b camB.png --f 620 --out results   # real pair
python src/infer.py --ckpt refine_best.pt --a framesA/ --b framesB/ --f 620 --out results   # video frames
```
For real frames: B is rotated back, both are centre-cropped to the common
square and resized to 256; `--f` is the focal length in px of the original
frames (from calibration). Undistort lens distortion first.

## Files

| file | purpose |
|---|---|
| `synth_rs.py` | generator (numpy + OpenCV only) |
| `preview.py` | contact sheet to eyeball samples |
| `rs_dataset.py` | PyTorch `Dataset` (aligned inputs, GT flow, visibility mask) |
| `model.py` | `MotionNet` (correlation + motion regression + geometry) and U-Net (`flow` / `direct`) |
| `train.py` | training, PSNR/SSIM evaluation, inference timing |
| `train_colab.ipynb` | end-to-end Colab notebook (stage 1 + stage 2) |
| `infer.py` | local inference on generated samples or real camera frames |
| `config_webcam.json` | starting parameters for the USB webcams (no hardware sync) |
| `config_example.json` | parameter overrides, to tune to your cameras |

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
