"""Frozen feature encoders using explicitly provided external source and weights."""
from collections.abc import Mapping
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys

import torch
from torch import nn

from .models.vrae import prepend_reference_frame_padding


def tensor_state(payload):
    if not isinstance(payload, Mapping):
        raise TypeError('Checkpoint must be a mapping')
    for key in ('state_dict', 'model'):
        if isinstance(payload.get(key), Mapping):
            payload = payload[key]
            break
    return payload


def freeze(module):
    module.eval().requires_grad_(False)
    if any(not torch.isfinite(p).all() for p in module.parameters()):
        raise ValueError('Nonfinite encoder parameter')
    return module


class OmegaPatchTeacher(nn.Module):
    """Normalized DINO patch targets only; no temporal geometry decoder is run."""
    def __init__(self, checkpoint):
        super().__init__()
        from vggt_omega.models.aggregator import _build_patch_embed
        self.patch_embed = _build_patch_embed(patch_size=16, embed_dim=1024)
        state = tensor_state(torch.load(checkpoint, map_location='cpu', weights_only=True))
        prefix = 'aggregator.patch_embed.'
        selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        if not selected:
            raise ValueError('No aggregator.patch_embed weights in Omega checkpoint')
        self.patch_embed.load_state_dict(selected, strict=True)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1), persistent=False)
        freeze(self)

    @torch.no_grad()
    def forward(self, images):
        batch, frames, channels, height, width = images.shape
        if frames != 49 or channels != 3 or height % 16 or width % 16:
            raise ValueError('Teacher expects all 49 RGB frames and a multiple-of-16 grid')
        normalized = (images - self.mean) / self.std
        output = self.patch_embed(normalized.reshape(batch * frames, 3, height, width))
        if isinstance(output, Mapping):
            output = output['x_norm_patchtokens']
        expected = (batch * frames, (height // 16) * (width // 16), 1024)
        if tuple(output.shape) != expected:
            raise ValueError(f'Omega patch encoder returned {tuple(output.shape)}, expected {expected}')
        return output.reshape(batch, frames, expected[1], expected[2])


class WanVideoEncoder:
    def __init__(self, source_file, checkpoint, device, dtype):
        source_file = Path(source_file).resolve()
        spec = importlib.util.spec_from_file_location('_latent_bridge_external_wan_vae', source_file)
        if spec is None or spec.loader is None:
            raise ImportError('Cannot load the supplied Wan2.1 VAE source file')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.WanVAE
        options = dict(z_dim=16, vae_pth=str(checkpoint), device=str(device), dtype=dtype)
        if 'temporal_compression_ratio' in inspect.signature(cls).parameters:
            options['temporal_compression_ratio'] = 4
        self.backend = cls(**options)
        freeze(self.backend.model)

    @torch.no_grad()
    def __call__(self, images):
        # The upstream encode already applies native channel mean/std scaling.
        video = images.permute(0, 2, 1, 3, 4).mul(2).sub(1)
        encoded = self.backend.encode(list(video.unbind(0)))
        if not isinstance(encoded, (tuple, list)) or len(encoded) != len(images):
            raise ValueError('WanVAE.encode must return one native latent per input video')
        return torch.stack(encoded)


def import_vrae(module_name, source):
    root = Path(source).resolve()
    if not (root / 'vrae').is_dir():
        raise ValueError('vrae_source must be the Python source root containing vrae/')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ImportError('V-RAE is already imported from a different source tree')
    return module


def load_temporal_pool(pool_cls, checkpoint, selection='ema', trust_pickle=False):
    if selection not in ('ema', 'model'):
        raise ValueError('pooling_weights must be ema or model')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=not trust_pickle)
    config = payload['resolved_config']
    model_config = config.get('model', config)
    pool_config = model_config['pooling']
    dim = int(payload['model_metadata']['hidden_size'])
    pool = pool_cls.from_config(pool_config, dim=dim)
    if pool.group_size != 4 or pool.dim != 1024:
        raise ValueError('V-RAE requires pooling group_size=4 and dim=1024')
    # Do not substitute raw weights when EMA was explicitly selected.
    state = payload['ema']['shadow'] if selection == 'ema' else payload['model']
    selected = {}
    for name, value in state.items():
        while name.startswith(('module.', 'model.')):
            name = name.split('.', 1)[1]
        if name.startswith('temporal_pool.'):
            key = name[len('temporal_pool.'):]
            if key in selected:
                raise ValueError('Duplicate temporal pool key after prefix normalization')
            selected[key] = value
    if not selected:
        raise ValueError(f'No temporal_pool parameters in {selection} state')
    pool.load_state_dict(selected, strict=True)
    return freeze(pool)


class VraeVideoEncoder(nn.Module):
    def __init__(self, config, trust_pickle=False):
        super().__init__()
        adapter = import_vrae('vrae.models.encoders.dinov3', config['vrae_source']).DINOv3Adapter
        pool_cls = import_vrae('vrae.models.pooling', config['vrae_source']).TemporalAttentionPool
        encoder_config = dict(name='dinov3', variant='vit_l16',
                              layers=[11, 13, 15, 17, 19, 21, 23],
                              fusion='mean_plus_final_spatial_mean', hidden_size=1024,
                              num_blocks=24, patch_size=16, encoder_tubelet_size=1,
                              pixel_normalization='imagenet', image_size=[256, 256],
                              runtime_image_size=[560, 560], checkpoint=str(config['dinov3_checkpoint']))
        self.encoder = adapter(encoder_config, source_dir=config['dinov3_source'],
                               checkpoint_path=config['dinov3_checkpoint'])
        self.pool = load_temporal_pool(pool_cls, config['vrae_checkpoint'],
                                      config.get('pooling_weights', 'ema'), trust_pickle)
        freeze(self)

    @torch.no_grad()
    def forward(self, images):
        if images.shape[1] != 49:
            raise ValueError('V-RAE expects 49 visible frames before reference padding')
        padded = prepend_reference_frame_padding(images, compression_ratio=4)
        # This is the native clean feature grid, before diffusion normalization.
        return self.pool(self.encoder(padded))


def load_encoders(config, device, dtype, trust_pickle=False):
    teacher = OmegaPatchTeacher(config['omega_checkpoint']).to(device)
    if config['route'] == 'wan':
        encoder = WanVideoEncoder(config['wan_vae_source'], config['wan_vae_checkpoint'], device, dtype)
    else:
        encoder = VraeVideoEncoder(config, trust_pickle).to(device)
    return encoder, teacher
