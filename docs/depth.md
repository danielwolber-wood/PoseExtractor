# Monocular depth (Depth Anything V2, Depth Pro)

Photos from LiDAR/TrueDepth cameras carry a metric depth map, and the fitter uses it (see
[How the fit works](fitting.md), section 6). For every other photo, Clay Pose can estimate depth
from the image itself with one of two optional Core ML models. They are installed locally, never
downloaded by the app, and never required: with none installed, nothing changes.

| Backend | Id | Output | Size | Warm inference (M-series) | Licence |
|---|---|---|---|---|---|
| **Depth Anything V2 Small** | `depth-anything-v2-small` | Relative inverse depth (larger = nearer), no scale | 48 MB (Float16) | ~25 ms at 518 × 392 | Apache-2.0 |
| **Depth Pro** | `depth-pro` | Metric depth given a focal length, plus a field-of-view estimate | 1.9 GB (Float16; ~950 M parameters) | ~1.35 s at 1536 × 1536 (CPU + GPU) | Code: Apple sample-code licence. Weights: `apple-amlr` (Apple's research-model licence) |

Requirements: macOS 14+, Apple Silicon (Core ML uses the Neural Engine/GPU; it also runs on the
CPU, slowly), and for Depth Pro ~2 GB of free memory while it runs. The first use of a model compiles
it for this Mac (Depth Anything: ~30 s once); the compiled model is cached in
`~/Library/Caches/ClayPose/depth`, and later launches load it in well under a second.

Measured on an M-series Mac, one process, `clay-depth-selftest --bench examples/photos/012.jpg`:

| | Load (compiled) | First prediction | Warm prediction |
|---|---|---|---|
| Depth Anything V2 Small (Neural Engine) | 79 ms | 63 ms | 13 ms |
| Depth Pro, Float16 (CPU + GPU) | 2.1 s | 2.3 s | 1.35 s |

Depth Pro is set to CPU + GPU (`computeUnits` in its `depth.json`). Compiling a model this size for
the Neural Engine took over half an hour during conversion, and still hadn't finished when it was
stopped, so the Neural Engine isn't used for it.

## Install

Models live in `Models/depth/<id>/`, under any models directory the app searches, in this order:
`$CLAY_MODELS`, the pipeline's models directory, the app bundle's `Resources/Models`, `./Models`,
`Models` next to (or up to five levels above) the executable, and
`~/Library/Application Support/ClayStudio/Models`. The first match wins.

| Backend | Accepted file names (first found wins) |
|---|---|
| Depth Anything | `model.mlmodelc`, `model.mlpackage`, `DepthAnythingV2SmallF16.{mlmodelc,mlpackage}`, `…F16P6…`, `DepthAnythingV2SmallF32.…` — Float16 is preferred, and Apple's file names are also accepted directly in `Models/depth/` |
| Depth Pro | `model.{mlmodelc,mlpackage}`, `DepthProF16.…`, `DepthPro.…` |

A backend directory may hold a `depth.json` manifest overriding the built-in I/O contract (see below).

**Depth Anything V2 Small** is Apple's own Core ML export; no conversion is needed:

```bash
.cache/iqa-env/bin/python tools/convert_depth_models.py depth-anything-v2-small
```

This downloads `DepthAnythingV2SmallF16.mlpackage` from Hugging Face
(`apple/coreml-depth-anything-v2-small`), checks its inputs and outputs, and installs it into
`Models/depth/depth-anything-v2-small/`. Any Python with `huggingface_hub` and `coremltools` works;
you can also download the package by hand and drop it into that folder.

**Depth Pro** has no official Core ML release, so it is converted once, offline:

```bash
uv venv --python 3.11 .cache/depth-env
uv pip install --python .cache/depth-env/bin/python torch==2.5.1 torchvision==0.20.1 timm coremltools==8.3 \
    huggingface_hub pillow numpy "depth_pro @ git+https://github.com/apple/ml-depth-pro"
.cache/depth-env/bin/python tools/convert_depth_models.py depth-pro \
    --models-dir ~/Library/Application\ Support/ClayStudio/Models
```

Conversion takes 10–25 minutes and ~16 GB of memory (swap is used on a 16 GB Mac). The converter
downloads `depth_pro.pt` (1.8 GB) from `apple/DepthPro`, traces `DepthPro.forward`
at 1536 × 1536, converts it to an ML Program, and compares PyTorch against Core ML on the same input
(`--image photo.jpg`, or a synthetic image). It writes `model.mlpackage` and a `depth.json` that
records the contract and the parity numbers, and it refuses to install a model that fails parity.
The installed Float16 model matched PyTorch on `examples/photos/012.jpg`: canonical inverse depth
median relative error 0.02% (95th percentile 0.07%), field of view 12.195° vs 12.193°.
Installing into Application Support keeps the 1.9 GB model and its research-only weights out of the
app bundle: `scripts/make_app.sh` copies everything under `./Models` into the app. PyTorch's CPU
matrix multiply crashed once inside Accelerate during tracing (a segfault in `libBLAS`); rerunning worked.

Check any package against the contract with
`python tools/convert_depth_models.py check path/to/model.mlpackage`.

## Use

```bash
build/clay photo.jpg                                    # auto: the first installed backend (Depth Anything first)
build/clay photo.jpg --depth-backend depth-pro          # a specific backend
build/clay photo.jpg --no-monocular-depth               # none (same as --depth-backend none)
build/clay photo.jpg --depth-backend depth-pro --depth-model ~/Downloads/DepthPro.mlpackage --no-depth-fallback
build/clay --list-models                                # shows which depth models were found
```

In Clay Studio, the toolbar's **Depth** menu offers *No monocular depth*, *Automatic*, and each backend
(marked when not installed). Switching re-runs depth in the background and re-fits, keeping your
edits. A status line under the photo shows the backend, representation, timing and how many people
it was used for, or why it wasn't (e.g. a missing model).

Which depth is used, in priority order:

1. **Embedded metric depth** (LiDAR/TrueDepth). No monocular model runs.
2. **The selected backend** (`--depth-backend`, the Studio menu).
3. **The fallback:** if the selected backend isn't installed or fails to load, another installed one,
   unless `--no-depth-fallback` is given. `auto` tries Depth Anything, then Depth Pro, so the large
   model only loads when it's the only one installed or you choose it.
4. **No depth.** A missing or broken model is a warning, never an error.

Embedded *relative* depth (dual-camera Portrait disparity) is still not used, and doesn't stop a
monocular backend from running.

## What each backend's numbers mean

Nothing is called metres unless the model and its inputs make that true. Each estimate is labelled
with a `DepthRepresentation`:

- **`relativeInverseDepth`** (Depth Anything): value ≈ s / z + t with unknown s > 0 and t, where
  **larger = nearer**. The raw output is divided by its 99th percentile. That is a positive scale
  only, so ratios and the zero point survive. It is never called metric, however the tensor is named.
  If a `depth.json` declares `"outputKind": "metricDepth"` (e.g. a metric fine-tune), that is honoured.
- **`metricDepth`** (Depth Pro): Depth Pro predicts *canonical inverse depth*. The app converts it
  exactly as `depth_pro.infer` does: depth = 1 / (canonical × W / f), with f the focal length in pixels.
  f is the photo's EXIF focal length when present (source `exif`), else Depth Pro's own field-of-view
  estimate (`predicted`, f = W / 2 / tan(fov / 2)). With neither (no EXIF and a model without the fov
  output), the scale would just be the app's 50 mm guess, so the map is labelled
  **relative inverse depth** instead (source `assumed`). The predicted focal length is recorded
  even when EXIF is used, for comparison.
- **Confidence**: neither model predicts one. A per-pixel confidence is derived from local depth
  gradients: it is low at occlusion edges, where monocular depth bleeds between body and background.
  It is flagged `confidenceIsDerived`.

### Tensor contract

| | Depth Anything V2 Small (Apple) | Depth Pro (converted) |
|---|---|---|
| Input | `image`: RGB image 518 × 392; ImageNet normalisation is inside the model | `image`: RGB image 1536 × 1536; scaled to [-1, 1] inside the model |
| Resize | **Letterbox**: aspect kept, padded with the photo's mean colour, padding cropped off the output | **Stretch** to 1536 × 1536 (Depth Pro's own preprocessing) |
| Output | `depth`: Grayscale16Half image 518 × 392 | `canonical_inverse_depth`: 1 × 1 × 1536 × 1536; `fov_deg`: 1 |

The photo is made upright from its EXIF orientation before inference (`LoadedImage`), so the depth
map is always in the upright image's frame. Outputs are read with their strides (Float16 via vImage,
Float32/64, Int32, or one-component pixel buffers). They stay at the model's output resolution, minus
any letterbox padding, and still cover exactly the whole photo. `DepthMap` samples them in
image-pixel coordinates (bilinear, or robust window medians), so no full-resolution copy is made.
Flexible-size models get the enumerated size closest to the photo's aspect ratio, or, for a range,
the default size's long side at the photo's aspect.

A `depth.json` may set any of `input`, `output`, `fovOutput` (`""` for none), `outputKind`
(`metricDepth` | `canonicalInverseDepth` | `relativeInverseDepth` | `relativeDepth`),
`resize` (`stretch` | `letterbox`), `mean`/`std` (for multi-array inputs), and `computeUnits`
(`all` | `cpuAndGPU` | `cpuAndNeuralEngine` | `cpuOnly`). A model whose tensors don't match fails to
load with an explicit `incompatibleModel` error; a truncated or corrupt package fails with `loadFailed`.
The package structure is checked before compiling, because Core ML's compiler aborts the process on a
malformed `Manifest.json` instead of throwing.

## How depth enters the fit

The fit without monocular depth always runs first. Monocular depth then refines it, and the refined
fit is kept only if it passes every check below. Code: `Sources/ClayCore/Fitting/MonocularDepthFit.swift`
and `Sources/ClayCore/Depth/DepthCalibration.swift`.

1. **Samples.** The map is sampled at each person's shoulders, hips, pelvis, elbows, wrists, knees,
   ankles and nose. The nose is used rather than the head top, which hair makes unreliable. A sample
   is taken as the median of a small window. It is skipped if the keypoint is guessed or user-edited
   (a dragged joint may sit over background), lies within ~1.5% of body height of the mask's edge, or
   sits on a depth edge (window spread > 10%). Samples are down-weighted by keypoint confidence, the
   derived depth confidence, and the local spread.
2. **Joint-to-surface offsets.** The map sees skin and clothing, while the fitter moves joints. Each
   sample's offset is found by ray-casting the fitted mesh through the joint's projection, so it
   matches the model's rig, build and pose. Joints with more than 25 cm of body in front of them are
   occluded (e.g. an arm across the hip) and are dropped.
3. **Scale** (`DepthCalibration`), anchored to the fit without depth. That fit's distance comes from
   the 2D keypoints, the camera intrinsics, Vision's 3D pose and the body-size prior:
   - `metric`: used as is. The ratio to the prior distance is recorded; beyond ×0.67–1.5 the map may
     not move the body, only shape its limbs in depth.
   - `sharedScale` / `affine`: with several people, one scale (or scale + shift) for everyone, if they
     agree. People can then be placed relative to each other.
   - `perPersonScale`: one person. A shift-free scale (the model's zero at infinity) from their own
     prior distance. This is circular for distance, so it **can't move the body**; it only says which
     limbs are in front.
   - `unusable`: torso samples that disagree with each other by more than ~10 cm (hair, clothing or
     background at the torso). The map is not used.
   The result's `scaleEstimated` / `constrainsDistance` say which of these happened.
4. **Residuals.** The residuals are *relative* (each keypoint's depth relative to the torso, σ =
   4 cm + 1.5% of distance), plus, only when distance is observable, an *absolute* torso-distance term
   (σ = 10% of distance for Depth Pro, 12% for a shared relative scale). Depth Pro's zero-shot error is
   about that, no better than the adult body-size prior, so it nudges the distance rather than
   dictating it. Both use a Cauchy loss, so samples on hair, loose clothing or background barely pull.
   Body shape may change only with metric depth that agrees with the prior. The 2D keypoint,
   silhouette and pose-prior terms are unchanged.
5. **Accept or reject.** The depth-guided fit replaces the fit without depth only if:
   - 2D keypoint error doesn't grow (≤ 10% + 1 px);
   - the silhouette overlap doesn't get worse (outside ≤ +2%, IoU ≥ −0.02);
   - no user-edited joint moves more than 2 px;
   - the pose stays plausible;
   - height changes by less than 20%, or, without metric depth, distance by less than 15%;
   - the fit agrees better with the map;
   - the fit ends within 1.5σ of the map;
   - the map's limb ordering doesn't contradict Vision's 3D pose (correlation above −0.5).

   Each person's outcome and reason is reported in the CLI output and `body_params.json`
   (`monocularDepth`), and in the Studio status line.

Inference runs once per photo, in parallel with Vision's detection, and is shared by everyone in it.
The estimate and calibration are cached on the result (`ClayResult.depth`). Re-fits (switching body
model, editing joints) reuse them without running the model again. While a joint is dragged, the live
update uses the cached depth only if that person's last full fit accepted it. The full re-fit when the
drag ends checks again. `CLAY_DEBUG_DEPTH=1` prints each person's samples (map, body before, body after).

### Self-test results

`swift run -c release clay-selftest --model <id> --pose <pose>` builds synthetic monocular maps from
the render's true z-buffer, with the errors real models make: unknown scale and shift, blurred
occlusion edges, a ±4% low-frequency warp, and 1% noise. It fits them on the same detections as the
fit without depth. Mean joint error (MPJPE), no depth → relative (Depth Anything-like) → metric
(Depth Pro-like):

| Model | raise | reach | walk |
|---|---|---|---|
| SMPL | 9.8 → 8.0 → 6.4 cm | 6.4 → 5.3 → 4.8 cm | 6.2 → 4.7 → 4.3 cm |
| SMPL-X | 11.2 → 10.6 → 10.8 cm | 8.8 → 6.8 → 6.8 cm | 7.2 → 6.0 → 5.9 cm |
| Anny | 9.7 → 8.1 → 8.0 cm | 7.7 → 6.5 → 7.2 cm | 9.1 → rejected → 9.5 cm |

With metric depth the SMPL pelvis distance moves from 3.98–4.10 m towards the true 4.22 m
(4.10–4.15 m). A pure-noise map is always rejected. A depth-order-inverted map is rejected in 8 of 9
cases, by residual disagreement, the silhouette or the Vision-correlation check. The exception is SMPL
`raise`, where limbs barely vary in depth: there it is accepted and costs 1.2 cm. Dragged joints land
within 1 px of where they land without depth, and live drag updates take 1.1–1.5× as long as without
depth. The self-test exits non-zero if these guarantees break.

### Real models

The same comparison with the installed models, through the CLI. The input is the SMPL self-test
render (`out/selftest/smpl_neutral/`, true pelvis distance 4.22 m) and the sample photo:

| Input | No depth | Embedded metric (HEIC) | Depth Anything V2 Small | Depth Pro |
|---|---|---|---|---|
| Synthetic render, no lens info | 3.99 m | 4.17 m, height 1.76 m (truth 1.79) | rejected: map 29 cm off the body (flat-shaded clay isn't a natural image) | predicted lens 3.6× too long, so disagrees with the prior (×0.29): relative use only, accepted, 4.04 m |
| Synthetic render, `--focal-mm 50` (true) | 3.99 m | — | — | metric, agrees with the prior to ×1.03, accepted with the distance term: 4.00 m, depth disagreement 3.5 → 2.0 cm |
| `012.jpg`, no EXIF (50 mm assumed) | 1.86 m | — | rejected: torso depth inconsistent (23 cm; hedge right behind, long hair) | predicted 110 mm lens, disagrees with the prior (×0.49): relative only, accepted, reprojection 33.4 → 29.7 px |
| `012.jpg`, `--focal-mm 110` (Depth Pro's lens) | 3.79 m | — | — | metric, agrees with the prior to ×1.02; rejected: silhouette got worse (IoU 0.84 → 0.76) |

Depth Anything added 13–60 ms per photo, overlapping Vision's detection; Depth Pro added 1.3–2.7 s.

## Limitations

- **Monocular depth can't know absolute scale.** Depth Anything's relative output gives no distance at
  all for a single person. It only improves how limbs are placed in depth. Depth Pro's metric output
  is typically ~10% off, and that error passes into body size when the size prior was right (on the
  self-test, a 6% bias grew Anny by 3–4 cm).
- **Unknown shift.** With one person, the shift-free assumption (zero = infinity) holds roughly for
  outdoor scenes with sky or far background. With a near background (a wall or hedge behind the
  person) it exaggerates depth differences inside the body. The torso-consistency check then usually
  rejects the map, as on `examples/photos/012.jpg`.
- **A self-consistent but wrong map** (e.g. inverted) can only be caught by its disagreement with the
  body and with Vision's 3D pose. When limbs barely vary in depth, neither signal is strong enough.
- **Hair, loose clothing, reflective and transparent surfaces** mislead monocular depth. Mask-edge
  margins, the robust loss and the checks limit the damage; they can't recover what's hidden.
- **Lens.** Without EXIF, the pipeline assumes a 50 mm lens and Depth Pro predicts its own, and the
  two can disagree by 2–4× (on the render and the sample photo). Depth Pro's metric distance is then
  inconsistent with the fit's camera, so it's only used for relative depth. Its predicted focal length is
  reported (`focalLengthPredictedPixels`). Pass `--focal-mm` to fix the lens. Feeding the prediction
  back into Vision's intrinsics would need detection to wait for Depth Pro (~1.3 s); that isn't done.
- **Portrait-mode disparity** (relative, embedded) is still ignored. It could use the same relative
  calibration path.
- The first run of a model compiles it (~30 s for Depth Anything). Vision's first run after boot can
  be slowed by the same compiler service.

## Tests

`swift run -c release clay-depth-selftest` (no body or depth models needed, runs in CI) checks model
discovery and preference order, missing, incompatible and corrupt models, source priority and
fallback, orientation (an EXIF-rotated photo), letterbox and stretch resizing and output cropping,
tensor decoding (strides, Float16, pixel buffers), relative normalisation, metric vs relative
labelling (including Depth Pro's focal-length rules), depth sampling, scale estimation, and cold/warm
timing. The Core ML path runs on tiny generated fixtures that mimic both models' I/O contracts
(`Sources/clay-depth-selftest/Fixtures.swift`, from `tools/make_depth_test_fixtures.py`). The fitting
guarantees are checked by `clay-selftest`, which needs a body model.
