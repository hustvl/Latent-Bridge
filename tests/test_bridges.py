import json
import pytest
import torch

from latent_bridge.checkpoints import build_bridge, bridge_tokens, load_bridge, export_checkpoint
from latent_bridge.losses import Omega35AlignmentLoss
from latent_bridge.models.wan import CausalTemporalPhaseUpsampler
from latent_bridge.models.vrae import prepend_reference_frame_padding, crop_reference_frame_padding
from train import main


def test_forward_backward(bridge_case):
    config, latent = bridge_case
    model = build_bridge(config).train()
    latent.requires_grad_(True)
    result = bridge_tokens(model, latent, config['route'])
    assert result.shape == (1, 9, 4, 8)
    teacher = torch.randn_like(result, requires_grad=True)
    loss = Omega35AlignmentLoss()(result, teacher)['loss']
    loss.backward()
    assert teacher.grad is None
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert any(p.grad is not None for p in model.parameters())


def test_causal_phase_order():
    x = torch.arange(3.).reshape(1, 1, 3, 1, 1)
    layer = CausalTemporalPhaseUpsampler(1, 4)
    assert layer(x).flatten().tolist() == [0., 1., 1., 1., 1., 2., 2., 2., 2.]


def test_reference_padding():
    x = torch.randn(1, 49, 3, 2, 2)
    padded = prepend_reference_frame_padding(x, compression_ratio=4)
    assert padded.shape[1] == 52
    torch.testing.assert_close(crop_reference_frame_padding(padded, compression_ratio=4), x)


def test_identity_loss():
    x = torch.randn(2, 9, 4, 8)
    assert abs(Omega35AlignmentLoss()(x, x)['loss'].item()) < 1e-6


def test_export_strict_and_metadata_stripping(bridge_case, tmp_path):
    config, latent = bridge_case
    model = build_bridge(config).eval()
    source, output, cfg = tmp_path / 'legacy.pt', tmp_path / 'public.pt', tmp_path / 'config.json'
    cfg.write_text(json.dumps(config))
    torch.save(dict(model=model.state_dict(), step=17, private_run_name='omit-me'), source)
    with pytest.raises(ValueError, match='disagrees'):
        export_checkpoint(source, cfg, output, step=18)
    export_checkpoint(source, cfg, output)
    payload = torch.load(output, weights_only=True)
    assert 'private_run_name' not in payload
    restored, loaded, step = load_bridge(output)
    assert step == 17 and loaded == config
    torch.testing.assert_close(bridge_tokens(model, latent, config['route']),
                               bridge_tokens(restored, latent, config['route']))
    with pytest.raises(FileExistsError):
        export_checkpoint(source, cfg, output)
    state = model.state_dict()
    state.pop(next(iter(state)))
    torch.save(dict(model=state, step=17), source)
    with pytest.raises(RuntimeError, match='Missing key'):
        export_checkpoint(source, cfg, tmp_path / 'bad.pt')


def test_two_step_cached_training(bridge_case, tmp_path):
    config, latent = bridge_case
    cfg, manifest, output = tmp_path / 'config.json', tmp_path / 'pairs.jsonl', tmp_path / 'run'
    cfg.write_text(json.dumps(config))
    torch.save(dict(latent=latent[0], teacher_tokens=torch.randn(9, 4, 8)), tmp_path / 'pair.pt')
    manifest.write_text(json.dumps(dict(id='synthetic', pair='pair.pt')) + '\n')
    main(['--config', str(cfg), '--manifest', str(manifest), '--output', str(output),
          '--steps', '2', '--save-every', '1', '--device', 'cpu'])
    assert json.loads((output / 'TRAINING_COMPLETE.json').read_text())['steps'] == 2
    assert load_bridge(output / 'step_000002.pt')[2] == 2
    with pytest.raises(OSError):
        main(['--config', str(cfg), '--manifest', str(manifest), '--output', str(output),
              '--steps', '2', '--device', 'cpu'])


def test_invalid_tokens_rejected(bridge_case):
    config, latent = bridge_case
    latent.flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        bridge_tokens(build_bridge(config), latent, config['route'])


def test_native_temporal_contract(bridge_case):
    config, latent = bridge_case
    if config['route'] == 'wan':
        config['model'].update(latent_frames=13, output_frames=49)
        latent = torch.randn(1, 2, 13, 4, 4)
    else:
        latent = torch.randn(1, 13, 8, 2, 2)
    with torch.no_grad():
        output = bridge_tokens(build_bridge(config).eval(), latent, config['route'])
    assert output.shape == (1, 49, 4, 8)
