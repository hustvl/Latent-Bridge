"""Decode native video latents to full49 depth, cameras and point clouds."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from latent_bridge.checkpoints import load_bridge, bridge_tokens, sha256
from latent_bridge.geometry import first_camera_gauge, unproject, write_ply, validate_camera


def dump(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@torch.inference_mode()
def infer(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent device fallback")
    if args.bf16 and device.type != 'cuda':
        raise ValueError("BF16 geometry inference requires CUDA")
    model, config, step = load_bridge(args.checkpoint, device)
    payload = torch.load(args.latent, map_location='cpu', weights_only=True)
    latent = payload.get('latent') if isinstance(payload, dict) else payload
    if not torch.is_tensor(latent) or latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError("Expected a tensor or {latent: tensor} with a batch dimension of 1")
    if config['route'] == 'wan' and tuple(latent.shape) != (1, 16, 13, 70, 70):
        raise ValueError("WAN input must be native [1,16,13,70,70], with training-consistent normalization")
    if config['route'] == 'vrae' and tuple(latent.shape) != (1, 13, 1024, 35, 35):
        raise ValueError("V-RAE input must be de-normalized [1,13,1024,35,35], including reference chunk")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from latent_bridge.omega import load_omega, predict_geometry
    omega = load_omega(args.omega_checkpoint, device)
    context = torch.autocast('cuda', dtype=torch.bfloat16) if args.bf16 else nullcontext()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with context:
        tokens = bridge_tokens(model, latent.to(device, dtype=torch.float32), config['route'])
        depth, w2c, intrinsics = predict_geometry(omega, tokens)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    depth, w2c, intrinsics = (x.cpu().numpy() for x in (depth, w2c, intrinsics))
    c2w = first_camera_gauge(w2c)
    validate_camera(c2w, intrinsics)
    np.save(output / 'depth.npy', depth)
    np.savez(output / 'cameras.npz', w2c_opencv=w2c, c2w=c2w, intrinsics=intrinsics)
    if args.export_ply:
        rgb = np.load(args.rgb_frames, allow_pickle=False) if args.rgb_frames else None
        if rgb is not None and (rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8):
            raise ValueError("RGB colors require uint8 [49,560,560,3] with identical crop/frame order")
        for frame in range(49):
            points = unproject(depth[frame:frame+1], intrinsics[frame:frame+1], c2w[frame:frame+1])[0]
            write_ply(output / f'frame_{frame+1:03d}.ply', points, None if rgb is None else rgb[frame])
    report = dict(status='completed', route=config['route'], bridge_step=step,
                  bridge_sha256=sha256(args.checkpoint), omega_sha256=sha256(args.omega_checkpoint),
                  latent_sha256=sha256(args.latent), depth_shape=list(depth.shape), frames_chunk_size=None,
                  camera_source='predicted_omega', coordinate_gauge='first_predicted_camera',
                  full_density=True, unit='model_native', geometry_seconds=elapsed,
                  timing_scope='latent upload + Bridge + Omega; excludes loading, IO and point-cloud export; no warmup',
                  device=str(device), precision='bf16' if args.bf16 else 'fp32',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
    dump(output / 'COMPLETE.json', report)
    print(json.dumps(report))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python infer.py', description=__doc__)
    for key in ('checkpoint', 'omega-checkpoint', 'latent', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--export-ply', action='store_true')
    parser.add_argument('--rgb-frames', help='Optional uint8 NPY; colors only, never geometry input')
    return parser.parse_args(argv)


def main(argv=None):
    return infer(parse_args(argv))


if __name__ == '__main__':
    main()
