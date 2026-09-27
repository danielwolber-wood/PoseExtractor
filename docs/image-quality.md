# Native image-quality analysis

ClayStudio's **Image quality → Analyze Quality** scores the source photo independently
of people, body models, or pose edits. Results stay attached to the open image until
another image is opened or Analyze Again is selected. Export Scores writes CSV or JSON.
The analysis runs off the main thread; cancellation takes effect between metrics and crop predictions.

The CLI also scores a file or recursively scans a folder, without loading body models:

```sh
build/clay quality examples/photos -o out/quality.csv
build/clay quality photo.jpg --metrics nima,brisque,niqe -o out/quality.json
```

Supported folder extensions are JPG/JPEG, PNG, HEIC/HEIF, TIFF, BMP, and WebP; actual
format decoding depends on ImageIO. Unreadable files are reported on stderr and skipped.
Individual metric failures are preserved in the report. An all-failed run exits nonzero.
CSV has a fixed schema even for empty input results, with raw, scaled, and error columns
for each metric from `vqa_script.py`. JSON retains the metric direction as well.

## Runtime assets and offline conversion

There is no Python, PyTorch, network request, or model download at application runtime.
Six learned networks use Core ML. BRISQUE and NIQE use Swift statistical algorithms
and exported reference parameters. The existing app packager includes `Models/quality`.

```sh
uv venv --python 3.11 .cache/iqa-env
uv pip install --python .cache/iqa-env/bin/python -r tools/quality-requirements.txt
.cache/iqa-env/bin/python tools/convert_quality_models.py all
./scripts/make_app.sh
```

Conversion downloads upstream weights into `.cache/iqa/`. Converted assets are kept in
`Models/quality/<metric>/` and excluded from Git. The float32 assets total approximately
1.1 GB. A `quality.json` manifest records the source version, input contract, scale,
and tensor parity checks. Failed conversions do not replace installed models.
The native runtime compiles and caches packages in `~/Library/Caches/ClayPose/quality`.
Weights retain their respective upstream terms; conversion does not relicense them.

| Script metric | Native implementation |
|---|---|
| `musiq` | Core ML transformer; Swift multiscale bicubic patches, positions, masks |
| `hyperiqa` | Core ML patch model; original 25-crop grid and score averaging |
| `nima` | Core ML Inception-ResNet-v2; aspect-preserving antialiased resize and center crop |
| `brisque` | Swift luminance, two-scale natural scene statistics, exported SVM |
| `clipiqa` | Core ML RN50 image encoder with precomputed original prompt features |
| `niqe` | Swift two-scale block statistics, exported pristine distribution, Accelerate pseudoinverse |
| `arniqa` | Core ML ResNet encoder at original and half size plus regressor |
| `liqe` | Core ML ViT-B/32, cached text features, original sampled patches and logit averaging |
| `align-one` | **Unavailable:** not a registered metric in PyIQA 0.1.14.1; no replacement is silently selected |

`all` converts the eight valid metrics. The original script catches the invalid
`align-one` name and skips it. Clay records that problem explicitly. If another model
was intended, its exact identity and weights are needed before adding it.

## Numerical behavior and limits

- Images are made upright using EXIF orientation and decoded to sRGB. Transparent
  pixels are composited over white. These are deliberate differences from the script's
  bare PIL `convert('RGB')`; compare already-upright sRGB images for numerical parity.
- Original dimensions are retained. Full-frame models are not silently resized.
  ARNIQA and CLIP-IQA accept dimensions from 32 to 8192. MUSIQ supports at most 16,384
  multiscale patches. Large images cost substantially more time and memory than pose fitting.
- LIQE requires both dimensions >=224. NIQE needs at least two complete 96×96 blocks;
  flat or degenerate images may lack valid statistical features. Failures stay per metric.
- Scaled values use PyIQA's approximate ranges and reverse lower-is-better metrics.
  They are not clamped or averaged; they are neither calibrated probabilities nor
  measures of pose-fitting accuracy. The UI shows raw values and direction.
- Swift double-precision statistical calculations can select slightly different
  0.001-spaced shape parameters than PyIQA's float32 operations.

## Validation

```sh
swift run -c release clay-quality-selftest
.cache/iqa-env/bin/python tools/validate_quality_models.py out/quality-validation
```

The converter checks tensor outputs against both the original PyIQA implementation
and the rewritten conversion wrapper before publishing. The separate validator checks
complete native file loading, preprocessing, inference, and aggregation against PyIQA
on a folder of PNG fixtures; it writes `native.json` and `parity.json`.
Use natural images, portrait/landscape orientations, odd dimensions, blur, and noise.
Model-independent self-tests cover metadata, CSV escaping, JSON, missing models,
cancellation, and corrupt images. These work with Xcode Command Line Tools alone.

Sources: [PyIQA](https://github.com/chaofengc/IQA-PyTorch),
[Core ML conversion](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html).
