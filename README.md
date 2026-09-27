# Armature

Photo → people → SMPL bodies → clay figures, running natively on macOS (Apple Silicon).

```
image ──► Vision: human rectangles, instance masks, horizon
      ──► per-person crop (with camera intrinsics) ──► 3D body pose · 2D body pose · hand pose · face landmarks
      ──► optional monocular depth (Core ML: Depth Anything V2 / Depth Pro), once per photo, alongside Vision
      ──► SMPL fit (Levenberg–Marquardt, Swift/Accelerate): 3D joints → 2D keypoints (+ depth) → silhouette
      ──► SMPL mesh (blend shapes + LBS via BLAS) ──► SceneKit/Metal render, USDZ / OBJ / JSON
```

No Python, PyTorch or network access at runtime. Pose inference runs on the Neural Engine through
Vision. Fitting takes about 10 ms per person, and a warm end-to-end run takes about 0.3 s.

## Repository layout

```text
Sources/
  ArmatureCore/      Shared library: Models, Detection, Fitting, Rendering, Pipeline, Depth, Quality
  ArmatureApp/       SwiftUI desktop app
  armature/          Command-line app
  armature-selftest/ Synthetic end-to-end validation executable
  armature-depth-selftest/   Monocular depth checks (no models needed)
  armature-quality-selftest/ Image-quality checks (no models needed)
design/icon/         App icon (Icon Composer file v1.icon, plus source SVG drafts)
scripts/             App build and packaging scripts
tools/               Offline Python model converters
docs/                Implementation notes
examples/photos/     Local input photos (ignored; add your own)
data/archives/       Downloaded model archives (local, ignored)
data/datasets/       Evaluation datasets (local, ignored)
Models/              Converted runtime models (local, ignored)
build/               Packaged app, CLI, and icons (generated, ignored)
out/                 Rendered results and self-test output (generated, ignored)
```

SwiftPM discovers the nested source folders automatically. `.build/` holds SwiftPM
build intermediates; `.cache/` holds downloaded converter weights. Keep large inputs
under `data/` and generated files in their designated output folders.
See [data/README.md](data/README.md) for input placement and migration notes.

## Body models

| Model | Joints | Adds | Licence | How to get it |
|---|---|---|---|---|
| **SMPL** (neutral, male, female) | 24 | The standard | MPI, non-commercial | `SMPL_python_v.1.1.0.zip` in `data/archives/` |
| **SMPL-X** (neutral, male, female) | 55 | Fingers, jaw | MPI, non-commercial | Download from smpl-x.is.tue.mpg.de (needs a login) and drop the zip in `data/archives/` |
| **Anny** | 104 | All ages (infants to elders), articulated fingers | Apache-2.0, built on CC0 MakeHuman assets | Installed by the converter from PyPI (`anny`) |

- **Canonical format:** the converter turns each model into the same frame (y up, facing +z, metres) as a linear model.
- **Rig description:** each model also gets a rig description saying which joint is which, where each keypoint lives on the body, joint stiffness and hinge limits. The Swift fitter reads only that, so it has no model-specific code.
- **SMPL-X specifics:** nose, eyes and mouth come from SMPL-X's own landmark embedding, which is exact. Transferring them from SMPL put them 2–3.5 cm too low. Its relaxed mean hand is the default pose, so unobserved hands curl naturally instead of lying flat.
- **Anny's shape space:** Anny's phenotype space (age, gender, weight, muscle, height, proportions) is non-linear. It's linearised by PCA over 2,000 bodies sampled from Anny's WHO-calibrated shape distribution. 16 components capture essentially all of the variation, and a linear read-out recovers apparent age (R² 0.92), shown in the app and the JSON.

## Age

Anny's shape space covers all ages, so knowing a person's age helps it a lot. The age can be:
- **Given:** `--age 34` for everyone, `--age 0=34,1=6` per person, or the age field in each person's row in the app.
- **Estimated:** by an open-source age model, run natively through Core ML.

| Age model | Input | Licence | Notes |
|---|---|---|---|
| **MiVOLO v2** (default) | Face + body crops | Apache-2.0 | All ages (0–122). Works with small or turned-away faces through the body crop. |
| **FaceAge** (Harvard AIM) | Face crop | Research use only, not for clinical care or commercial use | Trained on IMDb-Wiki, tuned for ages 60+ |

**How the age is used:**
- **Shape prior:** the converter learns Anny's average shape, and its spread, for each age from 0 to 90 over 2,000 WHO-calibrated sampled bodies. The fitter starts from and pulls towards that age's shapes.
- **Read-out constraint:** the fitter also constrains the body's apparent age to the target.
- **Confidence:** a given age counts as ±1.5 years; an estimate as ±max(3, 12%) years.
- **Other models:** SMPL and SMPL-X are adult models, so for them the age is shown but not used.

**Checks:**
- **Estimators:** the Core ML versions match the original PyTorch and TensorFlow models on identical crops of the test photo (MiVOLO 22.5 vs 22.53, FaceAge 23.95 vs 23.86).
- **Conditioning:** the self-test (`armature-selftest --model anny --age <years>`) compares fits with and without the age.

| True age | Shape error, age not given | Shape error, age given | Height (given / truth) |
|---|---|---|---|
| 6 | 21.5 cm | 2.2 cm | 1.11 / 1.14 m |
| 12 | 0.7 cm | 1.2 cm | 1.48 / 1.50 m |
| 35 | 14.9 cm | 2.7 cm | 1.67 / 1.71 m |
| 75 | 14.8 cm | 1.0 cm | 1.69 / 1.70 m |

**Converting the age models** (throwaway Python environments; see the docstring in `tools/convert_age_models.py` for the pinned versions):

```bash
python tools/convert_age_models.py mivolo
```

```bash
python tools/convert_age_models.py faceage
```

- **MiVOLO:** its overlapping `fold` is rewritten as a transposed convolution for Core ML.
- **FaceAge:** its Python-3.6-pickled Lambda layers are rebuilt as explicit residual `ScaleSum` layers.
- **Parity:** both conversions are checked against the originals.
- **FAHR-FaceAge:** its weights aren't released yet. When they are, it can plug in the same way, since the app reads each model's crops, size, normalisation and output from its `age.json`.

## Depth

Photos with LiDAR/TrueDepth depth use it directly. For other photos, two optional Core ML models
can estimate depth. They are installed locally and never downloaded by the app; with neither
installed, nothing changes.

| Depth model | Output | Size | Licence | Install |
|---|---|---|---|---|
| **Depth Anything V2 Small** (default when installed) | Relative inverse depth (no scale) | 48 MB | Apache-2.0 | `tools/convert_depth_models.py depth-anything-v2-small` (Apple's Core ML export) |
| **Depth Pro** | Metric depth + field of view | 1.9 GB | Code: Apple sample-code licence; weights: `apple-amlr` (research) | `tools/convert_depth_models.py depth-pro` (offline conversion) |

- **Where:** `Models/depth/<id>/` in any models folder, e.g. `~/Library/Application Support/Armature/Models`,
  which the app searches but `make_app.sh` doesn't bundle.
- **Priority:** embedded metric depth, then the selected backend, then another installed one, then none.
  A missing model is a warning, never an error.
- **How it's used:** relative depth is never treated as metres. With one person it only says which limbs
  are in front. Depth Pro's metric distance counts about as much as the body-size prior. A depth-guided fit
  is kept only if it doesn't make the 2D keypoints, silhouette, edited joints or pose worse.
- **Self-test:** mean joint error drops by 0.5–3.4 cm on synthetic maps with realistic errors, and bad
  maps are rejected. Setup, tensor contracts, results and limits are in [Monocular depth](docs/depth.md).

## Requirements

- macOS 14 or later on Apple Silicon.
- Xcode Command Line Tools with Swift 5.10 or later (`xcode-select --install`).
- For model conversion: Python and [uv](https://docs.astral.sh/uv/).
- Body model downloads acquired separately; no model weights or datasets are included.

To compile without downloading models, run `swift build -c release`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development and validation commands.

## Setup (once)

```bash
uv run --with numpy --with scipy tools/convert_models.py
```

```bash
uv run --python 3.12 --with anny --with numpy --with scipy tools/convert_models.py --anny
```

```bash
./scripts/make_app.sh
```

- **First command:** converts SMPL, plus SMPL-X if its zip is present.
- **Second command:** also converts Anny. It downloads Anny and PyTorch into a throwaway environment; the app itself never needs Python.
- **Third command:** builds `build/Armature.app` and `build/armature`. Compiling the icon needs Xcode 26 or later (not just the Command Line Tools).
- **Reading the models:** the converter reads straight from the zips, so nothing needs to be unzipped. It stubs out `chumpy`, so that doesn't need installing either.

## Use

**App:** `open build/Armature.app`, then drop in a photo.

- **Masks:** the *Masks* checkbox shows each person's segmentation mask, and *Fit silhouette* toggles the silhouette stage. Each person's row shows how much of the body lies outside its mask.
- **Depth:** the toolbar's *Depth* menu picks the monocular depth backend (or none). A status line under the photo says what it did, or why it wasn't used.
- **Body model:** the toolbar dropdown lists every converted model, grouped by family. Switching re-fits the current detections, and your edits are kept. For Anny, each person's row shows their apparent age.
- **Views:** "Photo" composites the figures over the photo with a matching camera and ground shadows. "Studio" shows a three-quarter view on a backdrop. Drag to orbit and double-click to reset.
- **Fixing the pose:** drag any joint on the photo. The person is re-fit live (about 20 ms per re-fit), and edited joints turn yellow. Dashed joints are Vision's guesses (out of frame or hidden), so check those first. The fitter trusts their 2D position and ignores Vision's 3D guess for them. Each person's ⋯ menu has *Swap Left and Right* (a common Vision failure), *Reset Edits* and *Remove Person*. Changing the body model keeps your edits.
- **Materials:** clay (with tool marks), smooth clay, plastic, glazed ceramic, marble, bronze, chrome, carved wood and wireframe. The colour palette applies to clay, plastic, ceramic and wireframe.
- **Export (⌘E):** renders an image of exactly what's in the 3D view, including an orbited camera. You choose the size (window, 1080, 2048 or 4096 px), the format (PNG, JPEG or HEIC) and the background (scene, or transparent with shadows kept). The Export menu also offers USDZ, and OBJ with SMPL parameters.

**CLI:**

```bash
build/armature photo.jpg -o out/photo -m marble --size 4096 --transparent
```

Options:
- `-m`: clay, smooth, plastic, ceramic, marble, bronze, chrome, wood or wireframe
- `-p`: palette
- `-b <id>`: body model. `--list-models` shows the ids.
- `-g`: shorthand for the SMPL genders
- `--subdivision 0-3`
- `--focal-mm` (35 mm-equivalent lens: EXIF by default, else 50)
- `--no-silhouette` (skip the segmentation-mask stage)
- `--age <years | i=years,...>` and `--age-model mivolo|faceage|none`
- `--depth-backend none|auto|depth-anything-v2-small|depth-pro` (default `auto`: the first installed),
  `--no-monocular-depth`, `--depth-model <path>`, `--no-depth-fallback`
- `--no-usdz`

It writes these to the output folder:
- `clay_photo.png` and `clay_studio.png`
- `clay.usdz`
- `person_N.obj`
- `body_params.json`, with the model id, pose, shape, translation, keypoints, fit errors and (for Anny) phenotype, in the OpenCV camera frame. It also records the depth source and diagnostics (`depth`), and what monocular depth did to each person (`monocularDepth`).

**Self-test:** `swift run -c release armature-selftest --model <id>` works with any model. It renders a known pose, runs the whole pipeline on the render and reports the joint errors. It also checks:
- that left and right come out correctly
- that dragging a keypoint pulls the fitted body there, for both the full re-fit and the fast live update
- silhouette gains in shape and joint error
- the depth HEIC round trip
- synthetic monocular depth (relative and metric, plus noise and inverted maps that must be rejected), with depth priority and edited joints kept

`swift run -c release armature-depth-selftest` checks the depth plumbing without any model.

**Accuracy** (self-test, averaged over three poses, after the silhouette stage):

| Model | Joint error | Body-shape error | Posed-surface error | Live drag update |
|---|---|---|---|---|
| SMPL | 7.5 cm | 3.3 cm | 8.0 cm | 2 ms |
| SMPL-X | 9.1 cm | 3.5 cm | 12.0 cm | 14 ms |
| Anny | 8.8 cm | 2.2 cm | 11.4 cm | 8–24 ms |

- **Poses:** `raise`, `reach` and `walk`, chosen with `--pose`.
- **Noise:** results vary by about 0.5 cm between runs.
- **Surface error:** SMPL-X's is inflated by its detailed hands and face.

**Icon:** `design/icon/v1.icon` is an Icon Composer file. `make_app.sh` compiles it with Xcode's `actool` into `Assets.car` and `AppIcon.icns`.

## Image quality

Open **Image quality → Analyze Quality** below the source photo to score it with
MUSIQ, HyperIQA, NIMA, BRISQUE, CLIP-IQA, NIQE, ARNIQA, and LIQE. Export Scores saves
CSV or JSON. Analysis is independent of person detection and pose edits.

```bash
build/armature quality examples/photos -o out/quality.csv
```

This runs natively: six Core ML networks plus Swift statistical metrics. Convert the
weights once with `tools/convert_quality_models.py`; Python is only used offline.
The `align-one` name in the supplied script is not registered in PyIQA and is reported
as unavailable. See [image-quality setup, validation, and limits](docs/image-quality.md).

## Implementation

See [How the fit works](docs/fitting.md) for fitting stages, camera handling, depth, silhouettes, and priors,
and [Monocular depth](docs/depth.md) for the depth backends.

## Limitations and next steps

- **Accuracy depends on Vision's 3D pose.** On the synthetic self-test, Vision is about 19 cm off per joint and the fit lands at about 17 cm. Most of that error is depth. The image-plane alignment is tight, at about 11 px on a 620 px tall figure.
- **Body width comes from the silhouette, and clothing limits it.** Loose clothes, long hair, and arms resting against the body widen the mask. The stage ignores large overhangs, but can't see a slim body inside a baggy outfit. Try the female/male models too.
- **Lens.** Without EXIF, a 50 mm-equivalent lens is assumed. If the overlay is too big or small in depth, pass `--focal-mm`.
- **SMPL-X's jaw is pinned closed.** Vision's "chin" is the bottom of the face outline, lower than the mesh's chin, so a free jaw gets pulled open. Fitting facial expression would need proper mouth landmarks.
- **SMPL has no fingers.** On SMPL, hands are oriented from the knuckles only. SMPL-X and Anny articulate fingers from Vision's 21 hand keypoints. No model fits facial expression yet.
- **Anny is heavier to fit.** It has 104 joints: a full fit takes about 220 ms (SMPL: 60 ms), and live drag updates about 22 ms (SMPL: 2 ms). They stay near rest, which is fine for a clay look.
- **FLAME isn't used.** Fitting FLAME needs its landmark embedding (`flame_static_embedding.pkl`, not in your zip), plus an SMPL-X-style head swap. If you want expressive clay heads, the next step would be to add that file and fit FLAME to Vision's 76 face landmarks.
- **First launch is slow.** The very first run after boot spends about 10 s while macOS compiles Vision's models; later runs are fast.
- SMPL/FLAME are under non-commercial licences, and `Models/` is git-ignored. Don't redistribute the built app outside your own use.

## Source and model licensing

A source-code license has not yet been selected for this repository. Model weights,
datasets, and local photos are excluded from Git. Each model has its own terms;
see the model tables above and the terms distributed by its provider. The app
packaging script bundles locally installed models, so built apps are excluded too.
