#!/usr/bin/env python3
"""Compare complete Swift quality scores to PyIQA on a folder of PNG fixtures.
Run with the conversion environment. Exit nonzero on missing/failed/mismatched scores.
The model converters' tensor checks do not replace this preprocessing/inference check.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from convert_quality_models import IDS  # configures isolated download caches before importing PyIQA
import pyiqa
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixtures', type=Path)
    p.add_argument('--armature', type=Path, default=ROOT / '.build/release/armature')
    p.add_argument('--metrics', nargs='+', default=IDS)
    args = p.parse_args()
    files = sorted(args.fixtures.glob('*.png'))
    if not files: raise RuntimeError('No PNG fixtures')
    output = args.fixtures / 'native.json'
    subprocess.run([str(args.armature), 'quality', str(args.fixtures), '--metrics', ','.join(args.metrics),
                    '--models', str(ROOT/'Models'), '-o', str(output)], check=True)
    native = {r['fileName']: {s['metric']: s for s in r['scores']} for r in json.loads(output.read_text())}
    results, failures = [], []
    # Statistical shape lookups are quantized to 0.001; float32 vs double may select adjacent bins.
    tolerance = {'brisque': 0.15, 'niqe': 0.025, 'musiq': 0.03, 'hyperiqa': 0.002,
                 'nima': 0.01, 'clipiqa': 0.002, 'arniqa': 0.002, 'liqe': 0.005}
    for name in args.metrics:
        metric = pyiqa.create_metric(name, device='cpu')
        for file in files:
            with Image.open(file) as image:
                x = to_tensor(image.convert('RGB')).unsqueeze(0)
            with torch.no_grad(): expected = float(metric(x).item())
            actual = native[file.name][name].get('raw')
            error = abs(actual - expected) if actual is not None else None
            ok = error is not None and np.isfinite(error) and error <= tolerance[name]
            row = dict(file=file.name, metric=name, pytorch=expected, native=actual,
                       absoluteError=error, tolerance=tolerance[name], passed=bool(ok))
            results.append(row)
            print(json.dumps(row), flush=True)
            if not ok: failures.append(row)
        del metric
    (args.fixtures/'parity.json').write_text(json.dumps(results, indent=2))
    if failures: raise RuntimeError(f'{len(failures)} parity checks failed')
    print(f'Passed {len(results)} end-to-end comparisons')

if __name__ == '__main__': main()
