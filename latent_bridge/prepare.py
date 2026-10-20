"""Decode original videos to matched, atomically published Bridge training pairs."""
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

from .checkpoints import sha256
from .encoders import load_encoders

FRAMES = 49
IMAGE_SIZE = 560
LATENT_SHAPES = {'wan': (16, 13, 70, 70), 'vrae': (13, 1024, 35, 35)}
TEACHER_SHAPE = (49, 1225, 1024)
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


def integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def read_prepare_config(path):
    config = json.loads(Path(path).read_text())
    route = config.get('route')
    files = ['omega_checkpoint']
    directories = []
    if route == 'wan':
        files += ['wan_vae_source', 'wan_vae_checkpoint']
    elif route == 'vrae':
        files += ['dinov3_checkpoint', 'vrae_checkpoint']
        directories = ['vrae_source', 'dinov3_source']
    else:
        raise ValueError('Preparation route must be wan or vrae')
    allowed = {'route', *files, *directories}
    if route == 'vrae':
        allowed.add('pooling_weights')
        if config.get('pooling_weights', 'ema') not in ('ema', 'model'):
            raise ValueError('pooling_weights must be ema or model')
    if set(config) - allowed:
        raise ValueError(f'Unknown preparation config keys: {sorted(set(config) - allowed)}')
    for key in files + directories:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Config requires path {key}')
        target = Path(value).expanduser().resolve()
        if not (target.is_file() if key in files else target.is_dir()):
            raise FileNotFoundError(f'{key}: {target}')
        config[key] = str(target)
    return config, {key: sha256(config[key]) for key in files}


def collect_jobs(videos=None, manifest=None, start_frame=0, stride=1):
    integer(start_frame, 'start_frame', 0)
    integer(stride, 'stride', 1)
    if (videos is None) == (manifest is None):
        raise ValueError('Specify exactly one of videos or manifest')
    jobs = []
    if videos is not None:
        root = Path(videos).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        for path in paths:
            relative = path.relative_to(root).as_posix()
            suffix = hashlib.sha256(relative.encode()).hexdigest()[:16]
            stem = re.sub(r'[^A-Za-z0-9_-]', '_', path.stem)[:60] or 'video'
            jobs.append(dict(id=f'{stem}_{suffix}', video=str(path), source=relative,
                             start_frame=start_frame, stride=stride))
    else:
        path = Path(manifest).resolve()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not {'id', 'video'} <= set(row) or set(row) - {'id', 'video', 'start_frame', 'stride'}:
                raise ValueError('Video manifest needs id/video and optional start_frame/stride')
            if not isinstance(row['video'], str) or not row['video']:
                raise ValueError('Manifest video must be a nonempty path')
            jobs.append(dict(id=row['id'], video=str((path.parent / row['video']).resolve()),
                             source=row['video'], start_frame=row.get('start_frame', start_frame),
                             stride=row.get('stride', stride)))
    if not jobs:
        raise ValueError('No input videos/windows found')
    seen = set()
    for job in jobs:
        identifier = job['id']
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier) or '..' in identifier:
            raise ValueError('Sample id must be a safe, nonempty ASCII filename')
        if identifier in seen:
            raise ValueError(f'Duplicate sample id: {identifier}')
        seen.add(identifier)
        integer(job['start_frame'], 'start_frame', 0)
        integer(job['stride'], 'stride', 1)
        if not Path(job['video']).is_file():
            raise FileNotFoundError(job['video'])
    return jobs


def read_clip(path, start_frame=0, stride=1, image_size=IMAGE_SIZE):
    import av
    from PIL import Image

    integer(start_frame, 'start_frame', 0)
    integer(stride, 'stride', 1)
    wanted = [start_frame + offset * stride for offset in range(FRAMES)]
    images, timestamps = [], []
    original_size = None
    crop = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Input has no video stream')
        # Sequential decoding preserves presentation order and does not approximate seeks.
        for index, frame in enumerate(container.decode(video=0)):
            if index != wanted[len(images)]:
                continue
            rgb = frame.to_image().convert('RGB')
            if original_size is None:
                original_size = rgb.size
                width, height = rgb.size
                side = min(width, height)
                left, top = (width - side) // 2, (height - side) // 2
                crop = (left, top, left + side, top + side)
            elif rgb.size != original_size:
                raise ValueError('Video resolution changes within the selected window')
            square = rgb.crop(crop).resize((image_size, image_size), Image.Resampling.BICUBIC)
            images.append(torch.from_numpy(np.array(square, copy=True)).permute(2, 0, 1).float().div(255))
            timestamps.append(None if frame.time is None else float(frame.time))
            if len(images) == FRAMES:
                break
    if len(images) != FRAMES:
        raise ValueError(f'Video has insufficient frames: selected {len(images)}/{FRAMES}; no padding/repeat allowed')
    metadata = dict(frame_indices=wanted, timestamps_seconds=timestamps,
                    original_size_wh=list(original_size), crop_xyxy=list(crop),
                    output_size_hw=[image_size, image_size], resize='PIL_BICUBIC',
                    color='RGB', pixel_range=[0, 1], fps_resampling=False)
    return torch.stack(images).unsqueeze(0), metadata


def validate_pair(pair, route):
    if set(pair) != {'latent', 'teacher_tokens'}:
        raise ValueError('A feature pair must contain exactly latent and teacher_tokens')
    for name, shape in [('latent', LATENT_SHAPES[route]), ('teacher_tokens', TEACHER_SHAPE)]:
        value = pair[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f'Invalid {name} shape; expected {shape}')
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'{name} must contain finite floating-point features')


def atomic_text(path, content):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_pair(path, pair, route):
    path = Path(path)
    validate_pair(pair, route)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(pair, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(temporary, map_location='cpu', weights_only=True)
    validate_pair(loaded, route)
    for key in pair:
        if not torch.equal(loaded[key], pair[key]):
            raise ValueError('Written feature pair failed exact round-trip verification')
    temporary.replace(path)
    return sha256(path)


@torch.inference_mode()
def prepare(args):
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported preparation devices: cpu or cuda')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    if args.bf16 and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 preparation requires a supported CUDA GPU')
    config, model_hashes = read_prepare_config(args.config)
    jobs = collect_jobs(args.videos, args.manifest, args.start_frame, args.stride)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identifier = None
    try:
        (output / 'pairs').mkdir()
        (output / 'metadata').mkdir()
        import av
        import PIL
        metadata = dict(protocol='video_feature_pairs_full49_v1', route=config['route'],
                        config_sha256=sha256(args.config), model_file_sha256=model_hashes,
                        windows=len(jobs), videos=len({job['video'] for job in jobs}),
                        frames=FRAMES, image_size=IMAGE_SIZE,
                        precision='bf16' if args.bf16 else 'fp32', device=str(device),
                        pooling_weights=config.get('pooling_weights'), trust_legacy_pickle=args.trust_legacy_pickle,
                        torch_version=torch.__version__, av_version=av.__version__, pillow_version=PIL.__version__,
                        teacher_target='omega_x_norm_patchtokens', geometry_decoder_run=False,
                        historical_parity_validated=False,
                        latent_convention='wan_native_encode_scaled' if config['route'] == 'wan' else 'vrae_k7_pool_native_no_diffusion_normalizer')
        atomic_json(output / 'preparation.json', metadata)
        encoder, teacher = load_encoders(config, device, torch.bfloat16 if args.bf16 else torch.float32,
                                        trust_pickle=args.trust_legacy_pickle)
        rows = []
        for job in jobs:
            identifier = job['id']
            print(json.dumps(dict(status='processing', id=identifier)), flush=True)
            before = Path(job['video']).stat()
            video_hash = sha256(job['video'])
            images, crop_metadata = read_clip(job['video'], job['start_frame'], job['stride'], IMAGE_SIZE)
            after = Path(job['video']).stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Source video changed during hashing/decoding')
            images = images.to(device)
            context = torch.autocast('cuda', dtype=torch.bfloat16) if args.bf16 else nullcontext()
            with context:
                latent = encoder(images)
                tokens = teacher(images)
            if latent.ndim != 5 or latent.shape[0] != 1 or tokens.ndim != 4 or tokens.shape[0] != 1:
                raise ValueError('Encoder output batch must be exactly one')
            pair = dict(latent=latent[0].detach().cpu().contiguous(),
                        teacher_tokens=tokens[0].detach().cpu().contiguous())
            path = output / 'pairs' / f'{identifier}.pt'
            pair_hash = save_pair(path, pair, config['route'])
            atomic_json(output / 'metadata' / f'{identifier}.json',
                        dict(id=identifier, source=job['source'], video_sha256=video_hash,
                             pair_sha256=pair_hash, route=config['route'], **crop_metadata,
                             shapes={key: list(value.shape) for key, value in pair.items()},
                             dtypes={key: str(value.dtype) for key, value in pair.items()}))
            rows.append(dict(id=identifier, pair=f'pairs/{identifier}.pt'))
            print(json.dumps(dict(status='validated', id=identifier, pair_sha256=pair_hash)), flush=True)
            del images, latent, tokens, pair
        # Publish the training manifest only after every pair and sidecar verifies.
        atomic_text(output / 'train.jsonl', ''.join(json.dumps(row) + '\n' for row in rows))
        report = dict(status='completed', count=len(rows), route=config['route'],
                      manifest_sha256=sha256(output / 'train.jsonl'))
        atomic_json(output / 'COMPLETE.json', report)
        print(json.dumps(report), flush=True)
        return report
    except Exception as error:
        try:
            atomic_json(output / 'FAILED.json', dict(status='failed', id=identifier,
                        error_type=type(error).__name__, error=str(error)))
        except OSError:
            pass
        raise


