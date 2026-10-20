# Prepare training pairs from original videos

Run `python prepare_data.py` from the repository root. It produces exactly the
`latent` / `teacher_tokens` pairs consumed by `python train.py`. No generated RGB,
depth, object mask or Gaussian supervision is required by this cache builder.

## Install encoders

```bash
python -m pip install -e '.[prepare,test]'
```

Install the separately licensed Omega dependency as described in
[Dependencies](DEPENDENCIES.md#installation). Preparation also imports its
`vggt_omega.models.aggregator._build_patch_embed(patch_size=16, embed_dim=1024)`
factory, and strictly loads the `aggregator.patch_embed.*` checkpoint subset.

Update paths in `configs/prepare_wan.json` or `configs/prepare_vrae.json`.
Paths inside these configs are relative to the **current working directory**,
normally the repository root; they are placeholders, not download URLs.

| Route | Required source/weights |
|---|---|
| WAN | Trusted standalone Wan2.1 `wan/modules/vae.py` exposing `WanVAE`, its 16-channel 8x8x4 VAE checkpoint, Omega checkpoint. |
| V-RAE | V-RAE Python source root containing `vrae/`, DINOv3 source and ViT-L/16 checkpoint, V-RAE reconstruction checkpoint with temporal pooling, Omega checkpoint. |

The standalone WAN source imports PyTorch and einops. The script passes `z_dim=16`,
checkpoint, device and dtype to `WanVAE`; if exposed, `temporal_compression_ratio`
is set to 4. It calls the native list-based `encode`, retaining its built-in latent
mean/std scaling without applying a second normalization.

The V-RAE backend must expose `vrae.models.encoders.dinov3.DINOv3Adapter` and
`vrae.models.pooling.TemporalAttentionPool.from_config`. It uses the audited K7
configuration: layers 11/13/15/17/19/21/23, `mean_plus_final_spatial_mean` fusion,
1024 channels, patch16 and ImageNet pixel normalization. This is an explicit
backend contract, not a claim that every public V-RAE checkpoint is interchangeable.
Install the dependencies specified by those separately obtained source trees.

For this DINOv3 backend, use the **Meta-native** ViT-L/16 state dictionary
(`patch_embed.proj.*`, `blocks.*`, fused `attn.qkv.*` keys), not a Hugging Face
conversion (`embeddings.*`, `layer.*`, separate attention projections). The
filename alone does not identify the format. Strict loading must succeed; do not
use `strict=False` to ignore incompatible weights.

The V-RAE checkpoint must contain `resolved_config.model.pooling` (or
`resolved_config.pooling`), `model_metadata.hidden_size=1024`, and either
`ema.shadow` or `model` weights with a `temporal_pool.` prefix. Pooling group size
must be 4. `pooling_weights` defaults to `ema`; missing EMA is an error, not a switch
to model weights. Full legacy checkpoints with Python objects may require the
explicit `--trust-legacy-pickle` flag. Use it only on trusted files; no unsafe load
fallback is performed by this script. External source and weights must also be trusted.

No external model code/weights are bundled or downloaded. Historical
numerical parity remains a separate gate; mocked tests only verify interfaces.

## Directory input

First separate train/validation videos at the sequence level. Then run one route:

```bash
python prepare_data.py --config configs/prepare_wan.json \
  --videos data/raw/train --output data/wan_train --device cuda --bf16

python prepare_data.py --config configs/prepare_vrae.json \
  --videos data/raw/train --output data/vrae_train --device cuda --bf16
```

The script recursively scans MP4/MOV/MKV/AVI/WebM/M4V in sorted relative-path order.
Default: one 49-frame window starting at decoded frame 0 of each video, stride 1.
Use `--start-frame` and `--stride` for a shared alternative window. It decodes in
presentation order without FPS resampling or approximate seeking. Audio is ignored.
It does not automatically apply rotation metadata or scene-cut detection.

Each frame is RGB, square center-cropped, then resized to 560x560 with PIL bicubic,
matching the audited image-frame preprocessing. Both encoders receive the **same**
processed clip. Source video compression/color conversion can still differ from
separately stored JPEG training images; identical selection is not pixel parity.

Short/corrupt videos, changing frame sizes, nonfinite features, unsupported model
layouts and duplicate sample IDs fail explicitly. No repeated last frames, synthetic
padding to rescue short videos, skipped failures or automatic retries.

## Explicit windows

To choose multiple windows per video, provide a JSONL manifest. Video paths are
relative to the **manifest directory**. IDs must be unique safe ASCII filenames.
Split train/validation by original sequence, not by these windows.

```jsonl
{"id":"scene_a_window0","video":"raw/train/scene_a.mp4","start_frame":0,"stride":1}
{"id":"scene_a_window1","video":"raw/train/scene_a.mp4","start_frame":49,"stride":1}
```

Each JSON line is one object. For example, place the above rows in
`data/train_windows.jsonl`, then:

```bash
python prepare_data.py --config configs/prepare_wan.json \
  --manifest data/train_windows.jsonl --output data/wan_windows --device cuda --bf16
```

Per-row start/stride override the CLI defaults. Every window still has 49 visible
frames. Sampling random windows across epochs is not reproduced by one fixed cache;
declare all desired windows rather than describing repeated training as new videos.

## Encoding conventions

- WAN: `[1,49,3,560,560]` RGB in `[0,1]` becomes `[1,3,49,560,560]` in `[-1,1]`; native VAE encode produces `[1,16,13,70,70]`.
- V-RAE: prepend three copies of frame 0, making 52 frames; K7 image features and the selected trained temporal pool produce `[1,13,1024,35,35]`. No diffusion normalizer is applied to these native clean features.
- Teacher: ImageNet-normalize the original 49 RGB frames, call the frozen Omega patch encoder in one 49-image batch and retain `x_norm_patchtokens[1,49,1225,1024]`.

The teacher image encoder is frame-wise. This step does not run the Omega temporal
aggregator or geometry heads; it is not seven-frame geometry decoding. V-RAE alone
gets reference padding. WAN's internal causal 1+4+4 VAE encoding is its codec design,
not a change to teacher context. Geometry inference remains full49 elsewhere.

## Outputs and integrity

```text
data/wan_train/
  pairs/<sample_id>.pt
  metadata/<sample_id>.json
  preparation.json
  train.jsonl
  COMPLETE.json
```

Each PT has exactly `latent` and `teacher_tokens`, without batch dimensions.
Storage preserves the computed dtype (BF16 where autocast produces BF16, otherwise
FP32); nothing is implicitly quantized. A full teacher record alone occupies
about 117 MiB at BF16 or 234 MiB at FP32, plus the latent. Reserve disk capacity
before a large run. No numerical cache parity claim follows from shape checks.

Pairs are written through temporary files, tensor-only reloaded, checked for exact
round-trip equality, then atomically renamed. Sidecars record video/pair SHA-256,
frame indices/timestamps, crop, tensor shapes/dtypes; `preparation.json` records
model-file/config hashes, route, pooling selection and software versions.
`train.jsonl` is published only after **all** requested windows succeed, followed
by `COMPLETE.json`. Training reads that manifest directly:

```bash
python train.py --config configs/wan.json --manifest data/wan_train/train.jsonl \
  --output runs/wan_cache_gate --steps 2 --save-every 1 --batch-size 1 \
  --lr 2e-4 --device cuda --bf16
```

On failure, inspect the first exception and `FAILED.json`. Earlier validated pairs
remain for diagnosis, but no final manifest/completion marker is published for
an incomplete run. Outputs must be new directories; cache resume is not implemented.
Metadata may identify local input paths and is experiment data, not public source.

The cache PT is a training pair, not the batch-shaped latent-only input of
`infer.py`. See [Data](DATA.md) for the distinction. Preparation does not generate
new videos, evaluate geometry quality, or certify dataset licensing/split isolation.

## CPU verification

```bash
python -m pytest tests/test_prepare.py -q
```

These tests decode real synthetic video files, verify frame order/crop/channels,
exercise failure publication, and train both tiny Bridges on freshly written
caches using mocked external encoders. They do not verify the real CUDA networks.
