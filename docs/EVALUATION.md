# Evaluation boundaries

## Prepare an evaluation bundle

Run the following Python code from the repository root after inference. It packs
saved predictions with independently prepared, pixel-aligned dataset GT; it does
not generate GT. Required GT inputs are `gt_xyz.npy[T,H,W,3]` and boolean
`gt_mask.npy[T,H,W]`. They must match prediction scene, frames, resize and crop.

```python
from pathlib import Path
import numpy as np

prediction = Path("runs/wan_infer")
depth = np.load(prediction / "depth.npy", allow_pickle=False)
gt_xyz = np.load("data/gt_xyz.npy", allow_pickle=False)
gt_mask = np.load("data/gt_mask.npy", allow_pickle=False)
assert gt_xyz.shape == (*depth.shape, 3)
assert gt_mask.shape == depth.shape and gt_mask.dtype == bool
with np.load(prediction / "cameras.npz", allow_pickle=False) as cameras:
    np.savez_compressed("data/wan_eval_case.npz", depth=depth,
                        intrinsics=cameras["intrinsics"], c2w=cameras["c2w"],
                        gt_xyz=gt_xyz, gt_mask=gt_mask)
```

For V-RAE change the prediction directory to `runs/vrae_infer` and output to
`data/vrae_eval_case.npz`. Share GT only when frame selection/crop actually match.
The GT mask defines scoring support, not a foreground mask predicted by the model.
For GT-assisted camera correction also include `gt_c2w[T,4,4]`, sharing the GT point
coordinate system. Follow the [data conventions](DATA.md) for decoding dataset depth
and units before packaging. The evaluator cannot infer them from the input files.

## Metrics

The portable evaluator reports `accuracy = mean_P min_G ||p-g||`,
`completeness = mean_G min_P ||g-p||`, and their arithmetic mean `chamfer`.
They are distances (lower is better), not classification accuracy.

It fits an SVD rotation with reflection correction, a **centered RMS-radius
ratio** scale, and translation from dense pixel correspondences. This is not
standard least-squares Umeyama scale and not ICP. Fewer than three correspondences
or zero spatial variance fail explicitly. Alignment is refit after filtering.
Prediction support and GT scoring support remain separate: removing a prediction
does not remove the corresponding GT point from completeness scoring.

Default `sampling=all` uses all retained points and SciPy KD-trees.
`voxel_fps` optionally applies Open3D voxel size0.005 followed by a deterministic
PyTorch3D FPS cap20,000 per cloud. This sampling applies to metrics only, never
to the PLY exporter. If dependencies are missing the command fails, not falls back.

## Filtering

- `none`: finite positive depth within the provided evaluation candidate mask.
- `log_mad`: per-frame median and robust spread of log-depth; threshold3 and
  MAD multiplier1.4826. The portable implementation uses NumPy median.
- `log_mad_multiview`: log-MAD plus reprojection to neighboring frames of radius2,
  relative depth threshold0.10 and at least2 supporting views. Neighbor lookup
  uses its valid depth, not its foreground/filter acceptance.
- `log_mad_multiview_safe_v1`: whole-sequence fallback to log-MAD if strict
  multi-view filtering retains fewer than3 points. Never label strict runs safe.

## Cameras and units

`omega` uses predicted poses and intrinsics. `gt_rotation_direction` explicitly
uses first-camera-relative GT rotations and translation directions but retains
predicted translation radii, intrinsics and depth. Stationary directions retain
the predicted translations. It is GT-assisted, not a pure prediction result or
full GT-metric translation replacement. Metadata preserves the selected mode.

Use `native` for uncalibrated model/GT reconstruction units. Request `meters`
only when the bundle has independently established metre units. No unit conversion
is performed by the flag. A field name in an old results file is not calibration.

## Historical reproduction is a separate gate

Every output is labeled `portable_rms_correspondence_v1` and
`historical_parity_validated=false`. Full numerical parity with prior paper tables
has not been established: library versions, NumPy versus Torch median conventions,
pixel reprojection details, precision and voxel/FPS order can affect values.
The original data preparation/mask/unit protocol must also be reproduced.
Do not append these metrics to historical result rows as interchangeable numbers.

Evaluate fixed scene lists; report CO3D and WildRGB-D separately, with actual
valid case counts. Qualitative best-seed selection is not a new benchmark result.
Cross-method comparisons must disclose pose assistance, native sequence lengths,
inference budgets, hardware and filter variants rather than selecting each method's
best protocol after observing the outcomes.
