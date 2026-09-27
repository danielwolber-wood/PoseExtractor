# Development

Use macOS 14 or later on Apple Silicon, with Xcode Command Line Tools providing
Swift 5.10 or later. Python converters are optional and are not runtime dependencies.

## Build and check without model downloads

```bash
swift build -c release
swift run -c release clay --help
python3 -m py_compile tools/convert_models.py tools/convert_age_models.py
zsh -n scripts/make_app.sh
```

`swift run -c release clay-depth-selftest` checks monocular-depth discovery, tensor handling, calibration
and error paths on tiny embedded Core ML fixtures; it needs no model downloads. After changing the
fixture generator, regenerate them with `.cache/iqa-env/bin/python tools/make_depth_test_fixtures.py`.

GitHub Actions runs these checks on macOS. They compile all executable targets and
check CLI startup and script syntax. `swift run -c release clay-quality-selftest` checks quality metadata, exports, and error handling without models. There is no XCTest target; model-dependent
rendering and inference are checked locally with the synthetic self-test.

## Model-dependent validation

Follow the model conversion instructions in [README.md](README.md), then run:

```bash
swift run -c release clay-selftest --model smpl_neutral
```

Use `--model smplx_neutral` or `--model anny` to exercise other installed models.
The self-test reports numerical errors and writes visual results under `out/selftest/`;
inspect both when changing fitting or rendering. It requires local model files and
macOS rendering/inference support, so it is not run in CI.

## Repository conventions

Keep Swift code within its existing target and responsibility folder. Put converter
code in `tools/`, packaging scripts in `scripts/`, and design notes in `docs/`.
Keep model weights, datasets, input photos, generated renders, and build products
out of commits. Model downloads and conversions remain local.

Describe behavior changes and the validation performed in pull requests. Source
licensing has not yet been selected; third-party model terms are separate.

Image-quality conversion and end-to-end PyIQA parity checks are documented in [docs/image-quality.md](docs/image-quality.md).
