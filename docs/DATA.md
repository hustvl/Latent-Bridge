# Public data and feature contracts

No dataset samples, private bucket paths, source identities or real evaluation
manifests are distributed here. Obtain DL3DV, CO3Dv2, WildRGB-D, RE10K and/or
TartanAir from their authorized sources and follow their individual terms.

## Feature-pair training

Generate these pairs from original videos with `python prepare_data.py`; see
[preparation instructions](PREPARATION.md). Its output `train.jsonl` is directly
accepted by `python train.py`. The default is one fixed 49-frame window per video;
use a video-window manifest for explicit additional windows.

JSONL manifest, resolved relative to the manifest file (not the current directory):

```json
{"id":"unique_scene_window_id","pair":"pairs/sample.pt"}
```

Each tensor-only PT file contains exactly:

```python
{
    "latent": native_latent_without_batch_dimension,
    "teacher_tokens": teacher_tokens_without_batch_dimension,
}
```

WAN sample latent: `[16,13,70,70]`. V-RAE: `[13,1024,35,35]`.
Teacher: `[49,1225,1024]`, the Omega image encoder's `x_norm_patchtokens`,
**not** globally aggregated tokens or predicted depth. The teacher is frozen.
Read the same 49 visible RGB frames and use exactly the same resize/crop for
both latent preparation and teacher feature extraction. Do not train only on
foreground masks when the intended output is full-scene geometry.

For WAN, use the VAE's training-consistent latent/channel normalization; the CLI
does not guess conversions. For V-RAE, remove the generator's latent normalization,
concatenate reference and target chunks, and keep the feature-channel order.
V-RAE encoding of 49 frames uses three extra copies of the first frame before
encoding, yielding 13 chunks. Padding utilities live in `models/vrae.py`.

The image encoder can process images as an image batch, but any teacher geometry
aggregation/depth validation must receive the complete 49-frame sequence at once.
The feature cache is not a substitute for raw-data licensing or split auditing.

## Inference inputs

`latent.pt`: a tensor with batch dimension or a mapping with the explicit key
`latent`. There is no heuristic guessing among arbitrary checkpoint tensors.
Optional `rgb_uint8.npy` has shape `[49,560,560,3]`, RGB channel order, uint8,
and the exact generated-frame order/crop. No vertical flip is applied.

`cameras.npz`: `w2c_opencv[T,3,4]` is the raw predicted extrinsic; `c2w[T,4,4]`
is first-predicted-camera-relative; `intrinsics[T,3,3]` is in pixel units.
Depth is positive camera-axis Z, not distance along a normalized ray.

## Evaluation NPZ

Required arrays: `depth[T,H,W]`, `intrinsics[T,3,3]`, `c2w[T,4,4]`,
`gt_xyz[T,H,W,3]`, `gt_mask[T,H,W]` (boolean). Optional `gt_c2w` is mandatory
for GT-assisted pose correction. The arrays must refer to identical scene,
frames, pixels and crops; GT geometry and GT cameras must share a coordinate system.

Convert source depth encoding, validity/object masks, GT clipping and dataset
units **before** producing this bundle. The evaluator cannot infer whether a
CO3D PNG stores float16 bits or whether an integer depth image uses millimetres.
No automatic global-depth fitting, confidence filtering or mismatched-frame repair.

Use sequence-level holdouts before selecting temporal windows. Record model hash,
manifest hash, crop, source frame indices, sampling seed, camera protocol and units
beside each experiment. Never count repeated windows as independent videos.
