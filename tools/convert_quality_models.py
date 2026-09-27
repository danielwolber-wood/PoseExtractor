#!/usr/bin/env python3
"""Offline PyIQA 0.1.14.1 -> native Clay quality assets. No Python is used at runtime.

Setup: uv venv --python 3.11 .cache/iqa-env
uv pip install --python .cache/iqa-env/bin/python -r tools/quality-requirements.txt
Run: .cache/iqa-env/bin/python tools/convert_quality_models.py all

Models are published only after Core ML/PyTorch parity succeeds. BRISQUE and
NIQE export statistical parameters for the Swift implementations instead.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import traceback
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/iqa/torch'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/iqa/huggingface'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache/iqa'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/iqa/matplotlib'))
import numpy as np
import torch
import coremltools as ct
import pyiqa
from pyiqa.default_model_configs import DEFAULT_CONFIGS

torch.set_num_threads(4)
IDS = ['musiq', 'hyperiqa', 'nima', 'brisque', 'clipiqa', 'niqe', 'arniqa', 'liqe']

class PatchModel(torch.nn.Module):
    def __init__(self, net, kind):
        super().__init__()
        self.net, self.kind = net, kind
        if kind == 'clipiqa':
            self.visual = net.clip_model[0].float().visual
            with torch.no_grad():
                t = net.clip_model[0].encode_text(net.prompt_pairs).float()
                self.register_buffer('text', t / t.norm(dim=-1, keepdim=True))
                self.register_buffer('scale', net.clip_model[0].logit_scale.exp().detach())
        elif kind == 'liqe':
            self.visual = net.clip_model.float().visual
            # Fixed 224px patches give 50 tokens: positional interpolation is exactly identity.
            import types
            def visual_forward(self, x, return_token=False, pos_embedding=True):
                x = self.conv1(x).reshape(x.shape[0], self.conv1.out_channels, -1).permute(0, 2, 1)
                cls = self.class_embedding.reshape(1, 1, -1).expand(x.shape[0], 1, -1)
                x = torch.cat([cls, x], dim=1) + self.positional_embedding
                x = self.ln_pre(x).permute(1, 0, 2)
                x = self.transformer(x).permute(1, 0, 2)
                return self.ln_post(x[:, 0, :]) @ self.proj
            self.visual.forward = types.MethodType(visual_forward, self.visual)
            self.register_buffer('text', net.text_features.float().detach())
            self.register_buffer('scale', net.clip_model.logit_scale.exp().detach())
        if kind in ('clipiqa', 'liqe'):
            self.register_buffer('mean', net.default_mean)
            self.register_buffer('std', net.default_std)
            # Only image encoder + precomputed prompt embeddings are needed at runtime.
            del self.net

    def forward(self, x):
        if self.kind == 'nima':
            x = (x - self.net.default_mean.to(x)) / self.net.default_std.to(x)
            dist = self.net.classifier(self.net.global_pool(self.net.base_model(x)[-1]))
            return (dist * torch.arange(1, 11).to(x)).sum(-1)
        if self.kind == 'arniqa':
            even = x[:, :, :x.shape[2]//2*2, :x.shape[3]//2*2]
            ds = torch.nn.functional.interpolate(even, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            f = []
            for image in (x, ds):
                image = (image - self.net.default_mean.to(x)) / self.net.default_std.to(x)
                f.append(torch.nn.functional.normalize(self.net.encoder(image), dim=1))
            return self.net._scale_score((torch.cat(f, dim=1).reshape(1, -1) @ self.net.regressor.weights.t() + self.net.regressor.biases.reshape(1))).reshape(-1)
        if self.kind == 'hyperiqa':
            return self.net.forward_patch(x).reshape(-1)
        if self.kind in ('clipiqa', 'liqe'):
            f = self.visual((x - self.mean) / self.std, pos_embedding=self.kind == 'liqe')
            f = f / f.norm(dim=-1, keepdim=True)
            logits = self.scale * f @ self.text.t()
            if self.kind == 'clipiqa':
                return logits.reshape(1, -1, 2).softmax(-1)[..., 0].mean(1)
            return logits.reshape(-1)
        return self.net(x).reshape(-1)


def convert(name, output):
    print(f'\n=== {name} ===', flush=True)
    metric = pyiqa.create_metric(name, device='cpu')
    net = metric.net.float().eval()
    config = DEFAULT_CONFIGS[name]
    bounds = [float(v.strip().replace('~', '')) for v in config['score_range'].split(',')]
    meta = dict(id=name, displayName=name.upper(), schemaVersion=1,
                lowerBetter=config.get('lower_better', False), scoreRange=bounds,
                source='pyiqa==0.1.14.1', sourceURL='https://github.com/chaofengc/IQA-PyTorch',
                conversionID=str(uuid.uuid4()), input='input', output='score')
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    if name in ('brisque', 'niqe'):
        if name == 'brisque':
            params = dict(sv=net.sv.tolist(), coefficients=net.sv_coef.tolist(), gamma=net.gamma, rho=net.rho)
        else:
            params = dict(mean=net.mu_pris_param.tolist(), covariance=net.cov_pris_param.tolist())
        (dest / 'parameters.json').write_text(json.dumps(params))
        meta['backend'] = 'statistics'
        (dest / 'quality.json').write_text(json.dumps(meta, indent=2))
        return
    meta['backend'] = 'coreml'
    size = 299 if name == 'nima' else 224
    if name == 'musiq':
        # Feed pre-extracted multiscale patches, bypassing Python image preparation.
        net.training = True  # only the outer forward; all child dropout/BN remain in eval mode
        shape = (1, 257, 3075)
        example = torch.rand(shape)
        example[:, :, -3:] = 0
        example[:, :, -1] = 1
        input_shape = (1, ct.RangeDim(194, 16384, default=257), 3075)
        meta['preprocessing'] = 'musiq'
    else:
        shape = (1, 3, size, size)
        example = torch.rand(shape)
        if name in ('arniqa', 'clipiqa'):
            input_shape = (1, 3, ct.RangeDim(32, 8192, default=size), ct.RangeDim(32, 8192, default=size))
        else:
            input_shape = shape
        meta['preprocessing'] = name
    # Record unmodified PyIQA outputs before replacing conversion-incompatible operations.
    references = []
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.rand(shape)
        if name == 'musiq':
            x[:, :, -3:] = example[:, :, -3:]
        elif name in ('arniqa', 'clipiqa') and seed:
            x = torch.rand(1, 3, 257 + seed * 32, 321 + seed * 32)
        with torch.no_grad():
            reference = net.forward_patch(x) if name == 'hyperiqa' else net(x)
        references.append((x, reference.detach().numpy().reshape(-1)))
    if name == 'musiq':
        # Core ML's stock GroupNorm converter embeds a symbolic batch in a constant.
        # Flatten spatial dimensions explicitly before reducing; same GroupNorm equation.
        import types
        def group_forward(self, x):
            z = x.reshape(x.shape[0], self.num_groups, -1)
            mean = z.mean(-1, keepdim=True)
            variance = ((z - mean) ** 2).mean(-1, keepdim=True)
            z = ((z - mean) / torch.sqrt(variance + self.eps)).reshape_as(x)
            return z * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        for module in net.modules():
            if isinstance(module, torch.nn.GroupNorm):
                module.forward = types.MethodType(group_forward, module)
    wrapper = PatchModel(net, name).eval()
    if name == 'musiq':
        wrapper.net.training = True
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    pipeline = ct.PassPipeline.DEFAULT
    if name == 'arniqa':
        pipeline = copy.deepcopy(pipeline)
        pipeline.remove_passes({'common::fuse_linear_bias'})
    converted = ct.convert(traced, pass_pipeline=pipeline, inputs=[ct.TensorType(name='input', shape=input_shape)],
                           outputs=[ct.TensorType(name='score')], minimum_deployment_target=ct.target.macOS14,
                           compute_precision=ct.precision.FLOAT32)
    errors = []
    for x, reference in references:
        with torch.no_grad():
            expected = wrapper(x).numpy()
        actual = converted.predict({'input': x.numpy()})['score']
        comparable = actual.reshape(-1)
        if name == 'liqe':
            probabilities = np.exp(comparable - comparable.max())
            comparable = np.array([(probabilities * np.arange(1, 6)).sum() / probabilities.sum()])
        np.testing.assert_allclose(comparable, reference, rtol=2e-3, atol=2e-3)
        error = float(np.max(np.abs(actual - expected)))
        errors.append(error)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    converted.short_description = f'PyIQA {name}; see quality.json for preprocessing'
    converted.save(str(dest / 'model.mlpackage'))
    meta['parity'] = dict(maxAbsoluteError=max(errors), inputs=3, precision='float32')
    (dest / 'quality.json').write_text(json.dumps(meta, indent=2))
    print(f'{name}: parity passed, max error {max(errors):.6g}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics', nargs='+', choices=['all', *IDS, 'align-one'])
    p.add_argument('--output', type=Path, default=ROOT / 'Models/quality')
    args = p.parse_args()
    failures = {}
    for name in IDS if 'all' in args.metrics else args.metrics:
        try:
            if name == 'align-one':
                raise ValueError('align-one is not a registered PyIQA metric; no substitute is selected')
            args.output.mkdir(parents=True, exist_ok=True)
            # Keep the installed model intact on conversion/parity failure.
            with tempfile.TemporaryDirectory(prefix='.conversion-', dir=args.output) as temporary:
                staging = Path(temporary)
                convert(name, staging)
                destination = args.output / name
                backup = staging / 'previous'
                if destination.exists():
                    destination.rename(backup)
                try:
                    (staging / name).rename(destination)
                except Exception:
                    if backup.exists(): backup.rename(destination)
                    raise
        except Exception as e:
            traceback.print_exc()
            failures[name] = str(e)
    print(json.dumps({'failures': failures}, indent=2))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
