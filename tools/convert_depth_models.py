"""Installs the optional monocular depth models for Armature (offline; the app never runs Python).

    python tools/convert_depth_models.py depth-anything-v2-small
    python tools/convert_depth_models.py depth-pro [--precision float16|float32] [--image photo.jpg]
    python tools/convert_depth_models.py check Models/depth/depth-pro/model.mlpackage

depth-anything-v2-small
    Downloads Apple's own Core ML export (Hugging Face: apple/coreml-depth-anything-v2-small,
    DepthAnythingV2SmallF16.mlpackage, ~50 MB) into Models/depth/depth-anything-v2-small/.
    No conversion. Needs: huggingface_hub, coremltools (for the I/O check).

depth-pro
    Converts Apple Depth Pro (github.com/apple/ml-depth-pro; weights: Hugging Face apple/DepthPro,
    depth_pro.pt, ~1.9 GB) to an ML Program with:
      input  "image"                    RGB image 1536 x 1536, scaled to [-1, 1] inside the model
      output "canonical_inverse_depth"  1 x 1 x 1536 x 1536 (Float16 or Float32)
      output "fov_deg"                  horizontal field of view, degrees
    Writes Models/depth/depth-pro/model.mlpackage and depth.json (contract, source, parity).
    The Swift runtime turns canonical inverse depth into metres with the photo's EXIF focal length
    (or the predicted field of view): depth = 1 / (canonical * W / f_px), exactly as depth_pro.infer.
    Parity: the traced PyTorch model and the Core ML model are compared on the same input (a
    synthetic image, or --image), and the result is recorded in depth.json. A conversion that
    fails parity is not installed.
    Needs (throwaway environment, ~16 GB RAM for float32 tracing):
      uv venv --python 3.11 .cache/depth-env
      uv pip install --python .cache/depth-env/bin/python torch==2.5.1 timm coremltools==8.3 \
          huggingface_hub pillow "depth_pro @ git+https://github.com/apple/ml-depth-pro"
      .cache/depth-env/bin/python tools/convert_depth_models.py depth-pro

check
    Prints a package's inputs/outputs and whether they match the backend contract.

--models-dir DIR installs somewhere other than ./Models, e.g. the app's Application Support folder,
which the app searches but scripts/make_app.sh does not bundle — the right place for Depth Pro
(1.9 GB, research-only weights licence):
    --models-dir ~/Library/Application\ Support/Armature/Models
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPTH = ROOT / "Models" / "depth"  # replaced by --models-dir
DA_REPO = "apple/coreml-depth-anything-v2-small"
DA_FILE = "DepthAnythingV2SmallF16.mlpackage"
PRO_REPO = "apple/DepthPro"
PRO_CKPT = "depth_pro.pt"
PRO_SIZE = 1536


def describe(package: pathlib.Path) -> dict:
    import coremltools as ct

    spec = ct.utils.load_spec(str(package))
    def feature(f):
        t = f.type.WhichOneof("Type")
        if t == "imageType":
            return {"name": f.name, "kind": "image", "width": f.type.imageType.width, "height": f.type.imageType.height,
                    "colorSpace": ct.proto.FeatureTypes_pb2.ImageFeatureType.ColorSpace.Name(f.type.imageType.colorSpace)}
        if t == "multiArrayType":
            return {"name": f.name, "kind": "multiArray", "shape": list(f.type.multiArrayType.shape),
                    "dataType": ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(f.type.multiArrayType.dataType)}
        return {"name": f.name, "kind": t}
    return {"inputs": [feature(f) for f in spec.description.input],
            "outputs": [feature(f) for f in spec.description.output],
            "metadata": dict(spec.description.metadata.userDefined),
            "description": spec.description.metadata.shortDescription}


def check(package: pathlib.Path, backend: str) -> list:
    """Problems with a package's I/O for `backend` (empty when it matches the Swift contract)."""
    d = describe(package)
    problems = []
    images = [i for i in d["inputs"] if i["kind"] == "image"]
    if len(images) != 1:
        problems.append(f"expected one image input, found {d['inputs']}")
    spatial = [o for o in d["outputs"] if o["kind"] == "image" or (o["kind"] == "multiArray" and len(o["shape"]) >= 2)]
    if backend == "depth-pro":
        names = {o["name"] for o in d["outputs"]}
        if "canonical_inverse_depth" not in names:
            problems.append("no canonical_inverse_depth output")
        if "fov_deg" not in names:
            problems.append("no fov_deg output (metric depth then needs EXIF focal lengths)")
    elif len(spatial) != 1:
        problems.append(f"expected one 2D depth output, found {d['outputs']}")
    return problems


def install_depth_anything(args):
    from huggingface_hub import snapshot_download

    target = DEPTH / "depth-anything-v2-small"
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(snapshot_download(DA_REPO, allow_patterns=[f"{DA_FILE}/*"], local_dir=tmp))
        package = path / DA_FILE
        if not package.exists():
            sys.exit(f"{DA_REPO} has no {DA_FILE}")
        d = describe(package)
        print(json.dumps(d, indent=2))
        problems = check(package, "depth-anything-v2-small")
        if problems:
            sys.exit("not installed: " + "; ".join(problems))
        target.mkdir(parents=True, exist_ok=True)
        dest = target / DA_FILE
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(package, dest)
    # No depth.json needed: the built-in contract (relative inverse depth, letterboxed) applies.
    print(f"installed {dest}")


def install_depth_pro(args):
    import numpy as np
    import torch
    import coremltools as ct
    from huggingface_hub import hf_hub_download
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    ckpt = args.checkpoint or hf_hub_download(PRO_REPO, PRO_CKPT, cache_dir=str(ROOT / ".cache" / "depth"))
    config = DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = ckpt
    model, _ = depth_pro.create_model_and_transforms(config=config, device=torch.device("cpu"), precision=torch.float32)
    model.eval()

    class Forward(torch.nn.Module):
        """DepthPro.forward on a [-1, 1] image at 1536 x 1536: (canonical inverse depth, fov in degrees)."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            canonical, fov = self.m(x)
            return canonical, fov.reshape(1)

    # Parity input: a real photo (stretched to 1536 x 1536, as depth_pro.infer does) or a smooth synthetic scene.
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB").resize((PRO_SIZE, PRO_SIZE), Image.BILINEAR)
    else:
        from PIL import Image
        yy, xx = np.mgrid[0:PRO_SIZE, 0:PRO_SIZE] / PRO_SIZE
        rgb = np.stack([0.3 + 0.5 * yy, 0.4 + 0.3 * np.sin(6 * xx), 0.5 - 0.3 * yy * xx], -1)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255 * 2 - 1).permute(2, 0, 1)[None]

    wrapper = Forward(model).eval()
    t0 = time.time()
    with torch.no_grad():
        ref_cid, ref_fov = wrapper(x)
        traced = torch.jit.trace(wrapper, x)
    print(f"traced in {time.time() - t0:.0f} s; reference fov {ref_fov.item():.2f} deg")

    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    t0 = time.time()
    mlmodel = ct.convert(
        # skip_model_load: loading here would compile for the Neural Engine (all compute units), which for a
        # model this size takes tens of minutes; parity below loads it for CPU + GPU instead.
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=precision,
        skip_model_load=True,
        inputs=[ct.ImageType(name="image", shape=(1, 3, PRO_SIZE, PRO_SIZE), color_layout=ct.colorlayout.RGB,
                             scale=2 / 255.0, bias=[-1.0, -1.0, -1.0])],
        outputs=[ct.TensorType(name="canonical_inverse_depth"), ct.TensorType(name="fov_deg")])
    print(f"converted in {time.time() - t0:.0f} s")
    mlmodel.short_description = "Apple Depth Pro (canonical inverse depth + field of view), converted by Armature"
    mlmodel.user_defined_metadata["clay.outputKind"] = "canonicalInverseDepth"
    mlmodel.user_defined_metadata["clay.source"] = f"{PRO_REPO}/{PRO_CKPT}"

    with tempfile.TemporaryDirectory() as tmp:
        package = pathlib.Path(tmp) / "model.mlpackage"
        mlmodel.save(str(package))
        # Parity on the same pixels (Core ML applies the [-1, 1] scaling itself).
        loaded = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        t0 = time.time()
        out = loaded.predict({"image": img})
        coreml_seconds = time.time() - t0
        cid = np.asarray(out["canonical_inverse_depth"], dtype=np.float32).reshape(ref_cid.shape)
        ref = ref_cid.numpy()
        rel = np.abs(cid - ref) / np.maximum(np.abs(ref), 1e-6)
        fov = float(np.asarray(out["fov_deg"]).reshape(-1)[0])
        parity = {"input": args.image or "synthetic", "precision": args.precision,
                  "medianRelativeError": float(np.median(rel)), "p95RelativeError": float(np.percentile(rel, 95)),
                  "fovTorch": float(ref_fov.item()), "fovCoreML": fov, "coremlSeconds": coreml_seconds}
        print(json.dumps(parity, indent=2))
        ok = parity["medianRelativeError"] < 0.01 and parity["p95RelativeError"] < 0.05 and abs(fov - parity["fovTorch"]) < 0.5
        if not ok and not args.force:
            sys.exit("parity check failed; not installed (try --precision float32)")
        target = DEPTH / "depth-pro"
        target.mkdir(parents=True, exist_ok=True)
        dest = target / "model.mlpackage"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(package, dest)
        (target / "depth.json").write_text(json.dumps({
            "schemaVersion": 1, "backend": "depth-pro", "input": "image", "output": "canonical_inverse_depth",
            "fovOutput": "fov_deg", "outputKind": "canonicalInverseDepth", "resize": "stretch",
            "computeUnits": args.compute_units, "source": f"https://huggingface.co/{PRO_REPO} ({PRO_CKPT})",
            "licence": "see https://github.com/apple/ml-depth-pro (code and weights licences)", "parity": parity,
        }, indent=2) + "\n")
    print(f"installed {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["depth-anything-v2-small", "depth-pro", "check"])
    ap.add_argument("package", nargs="?", help="for check: the .mlpackage")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16")
    ap.add_argument("--compute-units", choices=["all", "cpuAndGPU", "cpuAndNeuralEngine", "cpuOnly"], default="cpuAndGPU")
    ap.add_argument("--image", help="photo for the Depth Pro parity check")
    ap.add_argument("--checkpoint", help="local depth_pro.pt instead of downloading it")
    ap.add_argument("--force", action="store_true", help="install even if parity fails")
    ap.add_argument("--models-dir", help="models directory to install into (default: ./Models)")
    args = ap.parse_args()
    global DEPTH
    if args.models_dir:
        DEPTH = pathlib.Path(args.models_dir).expanduser() / "depth"
    if args.what == "depth-anything-v2-small":
        install_depth_anything(args)
    elif args.what == "depth-pro":
        install_depth_pro(args)
    else:
        if not args.package:
            sys.exit("check needs a package path")
        package = pathlib.Path(args.package)
        backend = "depth-pro" if "depth-pro" in str(package) or "DepthPro" in package.name else "depth-anything-v2-small"
        print(json.dumps(describe(package), indent=2))
        problems = check(package, backend)
        print("OK" if not problems else "problems: " + "; ".join(problems))


if __name__ == "__main__":
    main()
