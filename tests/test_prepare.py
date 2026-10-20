import argparse
import json
from pathlib import Path
import sys
import types

import av
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from latent_bridge import prepare as prep
from latent_bridge.encoders import OmegaPatchTeacher, WanVideoEncoder, VraeVideoEncoder, load_temporal_pool
from latent_bridge.train import FeaturePairs
from train import main as train_main


def video(path, frames=52):
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 20, 16, 'bgr0'
        for index in range(frames):
            pixels = np.zeros((16, 20, 3), dtype=np.uint8)
            pixels[..., 0] = index
            pixels[..., 1] = np.arange(20)[None, :]
            pixels[..., 2] = np.arange(16)[:, None]
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_decode_frame_order_crop_and_pixels(tmp_path):
    path = tmp_path / 'clip.mkv'
    video(path)
    images, info = prep.read_clip(path, start_frame=2, image_size=16)
    assert images.shape == (1, 49, 3, 16, 16)
    assert info['frame_indices'] == list(range(2, 51))
    assert info['crop_xyxy'] == [2, 0, 18, 16]
    assert images[0, :, 0, 0, 0].mul(255).round().tolist() == list(range(2, 51))
    assert images[0, 0, 1, 0].mul(255).round().tolist() == list(range(2, 18))
    assert images[0, 0, 2, :, 0].mul(255).round().tolist() == list(range(16))
    resized, _ = prep.read_clip(path, start_frame=2, image_size=8)
    array = images[0, 0].permute(1, 2, 0).mul(255).round().byte().numpy()
    expected = torch.from_numpy(np.array(Image.fromarray(array).resize((8, 8), Image.Resampling.BICUBIC))).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(resized[0, 0], expected)


def test_video_stride_and_short_clip(tmp_path):
    path = tmp_path / 'long.mkv'
    video(path, frames=100)
    _, info = prep.read_clip(path, start_frame=1, stride=2, image_size=8)
    assert info['frame_indices'] == list(range(1, 98, 2))
    short = tmp_path / 'short.mkv'
    video(short, frames=48)
    with pytest.raises(ValueError, match='insufficient frames'):
        prep.read_clip(short, image_size=8)
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'not a video')
    with pytest.raises(av.error.InvalidDataError):
        prep.read_clip(bad, image_size=8)


def test_collect_windows_and_fail_closed(tmp_path):
    first = tmp_path / 'one.mp4'
    second = tmp_path / 'two.mp4'
    first.touch()
    second.touch()
    rows = prep.collect_jobs(videos=tmp_path)
    assert len(rows) == 2
    assert rows == prep.collect_jobs(videos=tmp_path)
    manifest = tmp_path / 'windows.jsonl'
    manifest.write_text(json.dumps(dict(id='window', video='one.mp4', start_frame=10, stride=2)))
    jobs = prep.collect_jobs(manifest=manifest)
    assert jobs[0]['start_frame'] == 10 and jobs[0]['stride'] == 2
    assert jobs[0]['video'] == str(first)
    for identifier in ('../escape', '', 'a/b'):
        manifest.write_text(json.dumps(dict(id=identifier, video='one.mp4')))
        with pytest.raises(ValueError, match='safe'):
            prep.collect_jobs(manifest=manifest)
    row = json.dumps(dict(id='same', video='one.mp4')) + '\n'
    manifest.write_text(row + row)
    with pytest.raises(ValueError, match='Duplicate'):
        prep.collect_jobs(manifest=manifest)
    with pytest.raises(ValueError):
        prep.collect_jobs(videos=tmp_path, stride=0)


@pytest.fixture
def small_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'IMAGE_SIZE', 16)
    monkeypatch.setattr(prep, 'TEACHER_SHAPE', (49, 4, 8))
    monkeypatch.setattr(prep, 'LATENT_SHAPES', {'wan': (2, 13, 4, 4), 'vrae': (13, 8, 2, 2)})
    video_root = tmp_path / 'raw'
    video_root.mkdir()
    video(video_root / 'clip.mkv')
    config_file = tmp_path / 'config.json'
    config_file.write_text('{}')
    def setup(route):
        monkeypatch.setattr(prep, 'read_prepare_config', lambda path: (dict(route=route), {}))
        def load(config, device, dtype, trust_pickle=False):
            captured = []
            def encoder(images):
                captured.append(images.clone())
                return torch.ones((1, *prep.LATENT_SHAPES[route]))
            def teacher(images):
                torch.testing.assert_close(images, captured[-1])
                assert images.shape[1] == 49
                return torch.ones((1, *prep.TEACHER_SHAPE))
            return encoder, teacher
        monkeypatch.setattr(prep, 'load_encoders', load)
        return argparse.Namespace(config=str(config_file), videos=str(video_root), manifest=None,
                   output=str(tmp_path / f'cache_{route}'), device='cpu', bf16=False,
                   start_frame=0, stride=1, trust_legacy_pickle=False)
    return setup


@pytest.mark.parametrize('route', ['wan', 'vrae'])
def test_video_to_cache_to_two_training_steps(small_pipeline, route, tmp_path):
    args = small_pipeline(route)
    report = prep.prepare(args)
    output = Path(args.output)
    assert report['status'] == 'completed' and report['count'] == 1
    pairs = FeaturePairs(output / 'train.jsonl')
    assert len(pairs) == 1
    prep.validate_pair(pairs[0], route)
    metadata = json.loads(next((output / 'metadata').glob('*.json')).read_text())
    assert metadata['frame_indices'] == list(range(49)) and len(metadata['video_sha256']) == 64
    assert (output / 'COMPLETE.json').exists() and not (output / 'FAILED.json').exists()
    with pytest.raises(FileExistsError):
        prep.prepare(args)
    if route == 'wan':
        model_config = dict(latent_channels=2, latent_frames=13, latent_height=4, latent_width=4,
                            grid_height=2, grid_width=2, output_frames=49, hidden_dim=8,
                            output_dim=8, num_blocks=1, num_heads=2, mlp_ratio=2,
                            spatial_kernel_size=3, temporal_chunk_size=4)
    else:
        model_config = dict(dim=8, hidden_dim=16, temporal_heads=2, compression_ratio=4)
    cfg = tmp_path / 'train.json'
    cfg.write_text(json.dumps(dict(route=route, model=model_config)))
    train_main(['--config', str(cfg), '--manifest', str(output / 'train.jsonl'),
                '--output', str(tmp_path / 'trained'), '--steps', '2', '--device', 'cpu'])
    assert (tmp_path / 'trained' / 'step_000002.pt').exists()


def test_failure_never_publishes_training_manifest(small_pipeline):
    args = small_pipeline('wan')
    video(Path(args.videos) / 'z_short.mkv', frames=3)
    with pytest.raises(ValueError, match='insufficient frames'):
        prep.prepare(args)
    output = Path(args.output)
    assert len(list((output / 'pairs').glob('*.pt'))) == 1
    assert not (output / 'train.jsonl').exists()
    assert not (output / 'COMPLETE.json').exists()
    failure = json.loads((output / 'FAILED.json').read_text())
    assert failure['error_type'] == 'ValueError' and failure['id'].startswith('z_short')


def test_nonfinite_and_shape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'LATENT_SHAPES', {'wan': (1,)})
    monkeypatch.setattr(prep, 'TEACHER_SHAPE', (1,))
    for value in (torch.tensor([float('nan')]), torch.ones(2)):
        with pytest.raises(ValueError):
            prep.save_pair(tmp_path / 'bad.pt', dict(latent=value, teacher_tokens=torch.ones(1)), 'wan')
        assert not (tmp_path / 'bad.pt').exists()


def test_wan_range_and_no_double_normalization():
    encoder = WanVideoEncoder.__new__(WanVideoEncoder)
    class Backend:
        def encode(self, inputs):
            assert len(inputs) == 1 and inputs[0].shape == (3, 49, 2, 2)
            assert inputs[0].min() == -1 and inputs[0].max() == 1
            return [torch.full((16, 13, 1, 1), 7.)]
    encoder.backend = Backend()
    images = torch.zeros(1, 49, 3, 2, 2)
    images[:, -1] = 1
    assert torch.all(encoder(images) == 7)


def test_vrae_reference_padding_and_no_diffusion_normalizer():
    model = VraeVideoEncoder.__new__(VraeVideoEncoder)
    nn.Module.__init__(model)
    class Encoder(nn.Module):
        def forward(self, x):
            assert x.shape[1] == 52
            assert x[0, :4, 0, 0, 0].tolist() == [0.] * 4
            assert x[0, 4:, 0, 0, 0].tolist() == list(range(1, 49))
            return torch.full((1, 52, 8, 2, 2), 7.)
    class Pool(nn.Module):
        def forward(self, x):
            return x[:, ::4]
    model.encoder, model.pool = Encoder(), Pool()
    images = torch.arange(49.).view(1, 49, 1, 1, 1).expand(1, 49, 3, 2, 2)
    latent = model(images)
    assert latent.shape == (1, 13, 8, 2, 2) and torch.all(latent == 7)


def test_teacher_normalization_and_all49(tmp_path, monkeypatch):
    class Patch(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
        def forward(self, x):
            assert x.shape == (49, 3, 16, 16)
            expected = torch.tensor([-.485 / .229, -.456 / .224, -.406 / .225])
            torch.testing.assert_close(x[0, :, 0, 0], expected)
            return {'x_norm_patchtokens': torch.ones(49, 1, 1024)}
    module = types.ModuleType('vggt_omega.models.aggregator')
    module._build_patch_embed = lambda **kwargs: Patch()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / 'omega.pt'
    torch.save({'aggregator.patch_embed.weight': torch.ones(1)}, path)
    teacher = OmegaPatchTeacher(path)
    assert not teacher.training and not next(teacher.parameters()).requires_grad
    assert teacher(torch.zeros(1, 49, 3, 16, 16)).shape == (1, 49, 1, 1024)
    with pytest.raises(ValueError, match='49'):
        teacher(torch.zeros(1, 7, 3, 16, 16))


def test_pool_ema_is_explicit_and_strict(tmp_path):
    class Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.group_size, self.dim = 4, 1024
        @classmethod
        def from_config(cls, config, dim):
            return cls()
    payload = dict(resolved_config={'model': {'pooling': {'group_size': 4}}},
                   model_metadata={'hidden_size': 1024},
                   ema={'shadow': {'module.temporal_pool.weight': torch.tensor([3.])}},
                   model={'temporal_pool.weight': torch.tensor([8.])})
    path = tmp_path / 'vrae.pt'
    torch.save(payload, path)
    assert load_temporal_pool(Pool, path, 'ema').weight.item() == 3
    assert load_temporal_pool(Pool, path, 'model').weight.item() == 8
    payload.pop('ema')
    torch.save(payload, path)
    with pytest.raises(KeyError):
        load_temporal_pool(Pool, path, 'ema')


def test_config_paths_required_and_unknown_keys_rejected(tmp_path):
    path = tmp_path / 'prepare.json'
    path.write_text(json.dumps(dict(route='wan', bad='not-supported')))
    with pytest.raises(ValueError, match='Unknown'):
        prep.read_prepare_config(path)
    path.write_text(json.dumps(dict(route='wan')))
    with pytest.raises(ValueError, match='omega_checkpoint'):
        prep.read_prepare_config(path)


def test_default_cuda_resolves_an_explicit_device(monkeypatch, tmp_path):
    selected = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_device', selected.append)
    args = argparse.Namespace(device='cuda', bf16=False, config=str(tmp_path / 'missing.json'))
    with pytest.raises(FileNotFoundError):
        prep.prepare(args)
    assert selected == [torch.device('cuda:2')]
