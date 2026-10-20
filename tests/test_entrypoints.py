"""Exercise the real source scripts without custom commands or path setup."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from latent_bridge.checkpoints import build_bridge, load_bridge
from latent_bridge.geometry import unproject
from train import parse_args as train_args

ROOT = Path(__file__).resolve().parents[1]


def test_training_defaults_unchanged():
    args = train_args(['--config', 'c.json', '--manifest', 'm.jsonl', '--output', 'run', '--steps', '2'])
    assert vars(args) == dict(config='c.json', manifest='m.jsonl', output='run', steps=2,
                              batch_size=1, lr=2e-4, weight_decay=1e-4, save_every=1000,
                              seed=42, device='cuda', bf16=False)


def checkout(destination):
    destination.mkdir()
    shutil.copytree(ROOT / 'latent_bridge', destination / 'latent_bridge',
                    ignore=shutil.ignore_patterns('__pycache__'))
    for script in ('train.py', 'infer.py', 'prepare_data.py', 'evaluate.py', 'export_checkpoint.py'):
        shutil.copyfile(ROOT / script, destination / script)
    return destination


def run(source, script, *args):
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(source / script), *map(str, args)],
                            cwd=source.parent, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_fresh_source_training_and_export(bridge_case, tmp_path):
    config, latent = bridge_case
    source = checkout(tmp_path / 'source')
    cfg, manifest, output = tmp_path / 'config.json', tmp_path / 'train.jsonl', tmp_path / 'run'
    cfg.write_text(json.dumps(config))
    torch.save(dict(latent=latent[0], teacher_tokens=torch.randn(9, 4, 8)), tmp_path / 'pair.pt')
    manifest.write_text(json.dumps(dict(id='test', pair='pair.pt')) + '\n')
    run(source, 'train.py', '--config', cfg, '--manifest', manifest, '--output', output,
        '--steps', '2', '--save-every', '1', '--device', 'cpu')
    assert load_bridge(output / 'step_000002.pt')[2] == 2
    assert json.loads((output / 'TRAINING_COMPLETE.json').read_text())['steps'] == 2
    legacy, exported = tmp_path / 'legacy.pt', tmp_path / 'exported.pt'
    torch.save(dict(model=build_bridge(config).state_dict(), step=17), legacy)
    run(source, 'export_checkpoint.py', '--source', legacy, '--config', cfg, '--output', exported)
    assert load_bridge(exported)[2] == 17


def test_fresh_source_evaluation(tmp_path):
    source = checkout(tmp_path / 'source')
    depth = np.ones((3, 3, 3), dtype=np.float32)
    K = np.tile(np.eye(3), (3, 1, 1))
    c2w = np.tile(np.eye(4), (3, 1, 1))
    bundle, out = tmp_path / 'bundle.npz', tmp_path / 'metrics'
    np.savez(bundle, depth=depth, intrinsics=K, c2w=c2w,
             gt_xyz=unproject(depth, K, c2w), gt_mask=np.ones_like(depth, dtype=bool))
    run(source, 'evaluate.py', '--bundle', bundle, '--output', out, '--unit', 'native')
    score = json.loads((out / 'metrics.json').read_text())
    assert score['chamfer'] < 1e-6
    assert score['bundle_sha256']
    with np.load(out / 'prediction_mask.npz') as mask:
        assert mask['mask'].all()


def test_fresh_source_help(tmp_path):
    source = checkout(tmp_path / 'source')
    for name in ('train.py', 'infer.py', 'prepare_data.py', 'evaluate.py', 'export_checkpoint.py'):
        result = run(source, name, '--help')
        assert f'usage: python {name} ' in result.stdout
