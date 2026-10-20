"""Portable cached-token Bridge training; not a video-generator trainer."""
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .checkpoints import build_bridge, read_config, save_bridge, sha256
from .losses import Omega35AlignmentLoss, Omega35AlignmentLossCfg


class FeaturePairs(Dataset):
    def __init__(self, manifest):
        self.root = Path(manifest).resolve().parent
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows or any(set(row) != {'id', 'pair'} for row in self.rows):
            raise ValueError("Nonempty JSONL manifest needs exactly id and pair per row")
        if len({row['id'] for row in self.rows}) != len(self.rows):
            raise ValueError("Duplicate sample ids")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.root / self.rows[index]['pair']
        pair = torch.load(path, map_location='cpu', weights_only=True)
        if set(pair) != {'latent', 'teacher_tokens'}:
            raise ValueError("Feature pair must contain latent and teacher_tokens only")
        if pair['latent'].ndim != 4 or pair['teacher_tokens'].ndim != 3:
            raise ValueError("Feature-pair tensors must exclude the batch dimension")
        if any(not torch.isfinite(value).all() for value in pair.values()):
            raise ValueError("Nonfinite feature cache")
        return pair


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1 or args.lr <= 0:
        raise ValueError("steps, batch-size, save-every, and lr must be positive")
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    if args.bf16 and device.type != 'cuda':
        raise ValueError("This trainer only enables BF16 on CUDA")
    if world > 1:
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        torch.manual_seed(args.seed)
        config = read_config(args.config)
        output = Path(args.output)
        creation_error = [None]
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                creation_error[0] = str(error)
        if world > 1:
            torch.distributed.broadcast_object_list(creation_error, src=0)
        if creation_error[0] is not None:
            raise OSError(creation_error[0])
        model = build_bridge(config).to(device)
        module = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None) if world > 1 else model
        model.train()
        dataset = FeaturePairs(args.manifest)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=True)
        if len(loader) == 0:
            raise ValueError("Not enough feature pairs for one global batch")
        loss_config = Omega35AlignmentLossCfg(**config.get('loss', {}))
        criterion = Omega35AlignmentLoss(loss_config)
        optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        if rank == 0:
            metadata = dict(config=config, manifest_sha256=sha256(args.manifest), seed=args.seed,
                            global_batch=args.batch_size * world, steps=args.steps,
                            lr=args.lr, weight_decay=args.weight_decay, loss=asdict(loss_config),
                            schedule='constant', precision='bf16' if args.bf16 else 'fp32', exact_historical_replay=False)
            (output / 'training.json').write_text(json.dumps(metadata, indent=2))
        step, epoch = 0, 0
        while step < args.steps:
            sampler.set_epoch(epoch)
            for pair in loader:
                latent = pair['latent'].to(device, dtype=torch.float32)
                teacher = pair['teacher_tokens'].to(device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    if config['route'] == 'wan':
                        tokens = module(latent)
                    else:
                        from .models.vrae import crop_reference_frame_padding
                        tokens = crop_reference_frame_padding(module(latent), compression_ratio=model.compression_ratio)
                        tokens = tokens.permute(0, 1, 3, 4, 2).flatten(2, 3)
                    loss = criterion(tokens, teacher)['loss']
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if rank == 0:
                    print(json.dumps(dict(step=step, loss=float(loss.detach()))), flush=True)
                    if step % args.save_every == 0 or step == args.steps:
                        save_bridge(output / f'step_{step:06d}.pt', model, config, step)
                if step == args.steps:
                    break
            epoch += 1
        if world > 1:
            torch.distributed.barrier()
        if rank == 0:
            (output / 'TRAINING_COMPLETE.json').write_text(json.dumps(dict(steps=step)))
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()
