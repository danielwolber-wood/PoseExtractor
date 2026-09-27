#!/usr/bin/env python3
"""One-time conversion of open-source age estimators to Core ML, for the app's age-conditioned fitting.

Each model is written to Models/age/<id>/ as:
    model.mlpackage   Core ML model (the app compiles and caches it on first use)
    age.json          how to feed it: which crops, size, colour order, normalisation, and how to read
                      the output (see Sources/ArmatureCore/Detection/AgeEstimation.swift)

Estimators:
  mivolo   MiVOLO v2 (Kuprashevich & Tolstykh, 2023-25) -- Apache-2.0 code and weights. Face + body crops,
           all ages (0-122). https://huggingface.co/iitolstykh/mivolo_v2
  faceage  FaceAge (Bontempi et al., Lancet Digital Health 2025) -- weights for research only, "not intended
           for clinical care or commercial use". Face crop; trained on IMDb-Wiki with 60+ curated.
           https://github.com/AIM-Harvard/FaceAge

Environments (throwaway; the app never needs Python):
  mivolo:  python 3.11 venv with torch==2.5.1 torchvision==0.20.1 timm==0.8.13.dev0 transformers==4.51.0
           accelerate==1.8.1 coremltools, plus `pip install --no-deps --no-build-isolation
           git+https://github.com/WildChlamydia/MiVOLO.git` (needs setuptools<81)
  faceage: python 3.11 venv with tensorflow==2.15.1 coremltools gdown

Usage:
    python tools/convert_age_models.py mivolo
    python tools/convert_age_models.py faceage [--weights path/to/faceage_model]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Models" / "age"


def write_meta(out_dir: Path, meta: dict):
    (out_dir / "age.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_dir}")


# ---------------------------------------------------------------------------------------------------
# MiVOLO v2


def convert_mivolo():
    import coremltools as ct
    import torch
    from transformers import AutoConfig, AutoModelForImageClassification

    repo = "iitolstykh/mivolo_v2"
    print(f"MiVOLO v2 ({repo})")
    config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    lo, hi, avg = float(config.min_age), float(config.max_age), float(config.avg_age)

    class Wrapper(torch.nn.Module):
        """6-channel input (face RGB | body RGB, each letterboxed to 384 and ImageNet-normalised) ->
        age in years and P(female)."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            head = self.net(x)                                   # (1, 3): gender logits (male, female), raw age
            age = head[:, 2:3] * (hi - lo) + avg
            female = torch.softmax(head[:, 0:2], dim=1)[:, 1:2]
            return age, female

    wrapper = Wrapper(model.mivolo.model).eval()
    # Reference outputs from the unmodified model, for the parity check below.
    references = []
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.randn(1, 6, 384, 384) * 0.8
        with torch.no_grad():
            references.append((x, float(wrapper(x)[0])))

    # VOLO's outlook attention folds overlapping windows (F.fold with stride < kernel), which Core ML can't
    # convert. An overlapping fold is exactly a grouped transposed convolution with a one-hot kernel, so
    # substitute that for tracing (verified against the original model by the parity check).
    original_fold = torch.nn.functional.fold

    def fold_as_conv_transpose(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
        pair = lambda v: (v, v) if isinstance(v, int) else tuple(v)
        (kh, kw), (ph, pw), (sh, sw), (H, W) = pair(kernel_size), pair(padding), pair(stride), pair(output_size)
        assert pair(dilation) == (1, 1)
        B, ckk, _ = x.shape
        C = ckk // (kh * kw)
        h, w = (H + 2 * ph - kh) // sh + 1, (W + 2 * pw - kw) // sw + 1
        weight = torch.zeros(ckk, 1, kh, kw, dtype=x.dtype)
        for i in range(kh * kw):
            weight[i::kh * kw, 0, i // kw, i % kw] = 1
        pad_out = (H - ((h - 1) * sh - 2 * ph + kh), W - ((w - 1) * sw - 2 * pw + kw))
        return torch.nn.functional.conv_transpose2d(x.reshape(B, ckk, h, w), weight, stride=(sh, sw),
                                                    padding=(ph, pw), output_padding=pad_out, groups=C)

    x0 = torch.randn(2, 4 * 9, 24 * 24)
    assert torch.allclose(original_fold(x0, (48, 48), 3, padding=1, stride=2),
                          fold_as_conv_transpose(x0, (48, 48), 3, padding=1, stride=2), atol=1e-5)
    torch.nn.functional.fold = fold_as_conv_transpose
    example = torch.randn(1, 6, 384, 384)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    torch.nn.functional.fold = original_fold

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 6, 384, 384))],
        outputs=[ct.TensorType(name="age"), ct.TensorType(name="female")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "MiVOLO v2 age & gender (face + body)"
    mlmodel.license = "Apache-2.0"

    # Parity check against PyTorch on a few inputs (FP16 conversion: expect a fraction of a year).
    errors = []
    for x, a in references:
        out = mlmodel.predict({"input": x.numpy()})
        errors.append(abs(float(np.asarray(out["age"]).ravel()[0]) - a))
    print(f"  Core ML vs PyTorch age difference: max {max(errors):.2f} years over {len(errors)} inputs")

    out_dir = OUT / "mivolo"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    mlmodel.save(str(out_dir / "model.mlpackage"))
    write_meta(out_dir, {
        "displayName": "MiVOLO v2 (face + body)",
        "licence": "Apache-2.0",
        "order": 10,
        "inputs": [
            # Channels 0-2: face crop; 3-5: body crop with every face box blacked out (MiVOLO's own recipe,
            # so the body branch reads build and posture, not the face). A missing crop is all zeros, normalised.
            {"crop": "face", "channels": [0, 3]},
            {"crop": "bodyFacesMasked", "channels": [3, 6]},
        ],
        "size": 384, "resize": "letterbox", "colour": "RGB",
        "normalise": {"mode": "fixed", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "input": "input", "output": "age", "outputKind": "years",
        "validAges": [lo, hi],
    })


# ---------------------------------------------------------------------------------------------------
# FaceAge


def convert_faceage(weights: Path | None):
    import coremltools as ct
    import tensorflow as tf

    print("FaceAge (AIM-Harvard)")
    if weights is None:
        import gdown
        cache = ROOT / ".cache" / "faceage"
        cache.mkdir(parents=True, exist_ok=True)
        # The Google Drive file is a Keras HDF5 model (~92 MB).
        weights = cache / "faceage_model.h5"
        if not weights.exists():
            gdown.download(id="1KBZBMKbeqDH95KMbPKJ6vFqnf3ze2aIn", output=str(weights), quiet=False)
    print(f"  loading {weights}")
    # The HDF5 stores its 21 residual "ScaleSum" layers as Keras Lambdas holding marshalled Python 3.6
    # bytecode, which modern Python can't load. Their bytecode is `inputs[0] + inputs[1] * scale` (the
    # Inception-ResNet residual; decoded by hand), so rebuild them as an explicit layer and load weights.
    import h5py

    class ScaleSum(tf.keras.layers.Layer):
        def __init__(self, scale=1.0, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale

        def call(self, inputs):
            return inputs[0] + inputs[1] * self.scale

        def get_config(self):
            return {**super().get_config(), "scale": self.scale}

    def replace_lambdas(node):
        if isinstance(node, dict):
            if node.get("class_name") == "Lambda":
                c = node["config"]
                node["class_name"] = "ScaleSum"
                node["config"] = {"name": c["name"], "trainable": False, "dtype": c.get("dtype", "float32"),
                                  "scale": c["arguments"]["scale"]}
            for v in node.values():
                replace_lambdas(v)
        elif isinstance(node, list):
            for v in node:
                replace_lambdas(v)

    with h5py.File(weights, "r") as h5:
        config = json.loads(h5.attrs["model_config"])
    replace_lambdas(config)
    model = tf.keras.models.model_from_json(json.dumps(config), custom_objects={"ScaleSum": ScaleSum})
    model.load_weights(str(weights))
    model.summary(print_fn=lambda s: None)
    print(f"  input {model.input_shape} -> output {model.output_shape}")

    input_name = model.input_names[0]
    mlmodel = ct.convert(
        model,
        inputs=[ct.TensorType(name=input_name, shape=(1, 160, 160, 3))],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    out_name = mlmodel.get_spec().description.output[0].name
    mlmodel.short_description = "FaceAge biological age (face)"
    mlmodel.license = "Research use only (AIM-Harvard FaceAge)"

    errors = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal((1, 160, 160, 3)).astype(np.float32)
        ref = float(np.asarray(model.predict(x, verbose=0)).ravel()[0])
        got = float(np.asarray(mlmodel.predict({input_name: x})[out_name]).ravel()[0])
        errors.append(abs(ref - got))
    print(f"  Core ML vs TensorFlow age difference: max {max(errors):.2f} years over {len(errors)} inputs")

    out_dir = OUT / "faceage"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    mlmodel.save(str(out_dir / "model.mlpackage"))
    write_meta(out_dir, {
        "displayName": "FaceAge (research only)",
        "licence": "Research use only — not for clinical care or commercial use (AIM-Harvard)",
        "order": 20,
        # FaceAge crops the MTCNN face box exactly, stretches it to 160x160 and standardises per image.
        "inputs": [{"crop": "face", "channels": [0, 3]}],
        "size": 160, "resize": "stretch", "colour": "RGB", "layout": "NHWC",
        "normalise": {"mode": "perImage"},
        "input": input_name, "output": out_name, "outputKind": "years",
        "validAges": [18, 100],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["mivolo", "faceage"])
    ap.add_argument("--weights", type=Path, default=None, help="faceage: local SavedModel/.h5 instead of downloading")
    args = ap.parse_args()
    if args.model == "mivolo":
        convert_mivolo()
    else:
        convert_faceage(args.weights)


if __name__ == "__main__":
    main()
