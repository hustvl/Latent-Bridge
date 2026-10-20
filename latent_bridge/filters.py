"""Prediction-side depth filters; GT scoring support is never edited here."""
import numpy as np
from .geometry import unproject


def filter_depth(depth, candidate, intrinsics, c2w, mode='none', mad_threshold=3.0,
                 relative_threshold=0.10, neighbor_radius=2, min_support=2):
    if depth.ndim != 3 or candidate.shape != depth.shape or candidate.dtype != bool:
        raise ValueError("Depth and boolean candidate mask must have equal [T,H,W] shape")
    if mode not in ('none', 'log_mad', 'log_mad_multiview', 'log_mad_multiview_safe_v1'):
        raise ValueError("Unknown filter")
    if relative_threshold <= 0 or mad_threshold <= 0 or min_support < 1 or neighbor_radius < 1:
        raise ValueError("Filter thresholds/radius/support must be positive")
    valid = candidate & np.isfinite(depth) & (depth > 0)
    if mode == 'none':
        return valid
    mad = np.zeros_like(valid)
    for frame, mask in enumerate(valid):
        if not mask.any():
            continue
        values = np.log(np.maximum(depth[frame][mask], 1e-8))
        center = np.median(values)
        spread = max(float(1.4826 * np.median(np.abs(values - center))), 1e-6)
        mad[frame][mask] = np.abs(values - center) <= mad_threshold * spread
    if mode == 'log_mad':
        return mad
    xyz = unproject(depth, intrinsics, c2w)
    w2c = np.linalg.inv(c2w)
    result = np.zeros_like(valid)
    frames, height, width = depth.shape
    for frame in range(frames):
        yy, xx = np.where(mad[frame])
        points = xyz[frame, yy, xx]
        votes = np.zeros(len(points), dtype=np.int32)
        for neighbor in range(max(0, frame - neighbor_radius), min(frames, frame + neighbor_radius + 1)):
            if neighbor == frame:
                continue
            camera = points @ w2c[neighbor, :3, :3].T + w2c[neighbor, :3, 3]
            z = camera[:, 2]
            positive = np.isfinite(camera).all(-1) & (z > 1e-8)
            uv = camera @ intrinsics[neighbor].T
            uv = uv[:, :2] / np.where(positive, z, 1)[:, None]
            inside = positive & np.isfinite(uv).all(-1)
            # Round only finite, bounded coordinates to avoid integer overflow.
            inside &= (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
            inside &= (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5)
            ids = np.flatnonzero(inside)
            ij = np.rint(uv[ids]).astype(int)
            observed = depth[neighbor, ij[:, 1], ij[:, 0]]
            consistent = np.isfinite(observed) & (observed > 0)
            consistent &= np.abs(observed - z[ids]) / np.maximum(observed, 1e-8) <= relative_threshold
            votes[ids] += consistent
        result[frame, yy, xx] = votes >= min_support
    if mode.endswith('safe_v1') and result.sum() < 3:
        return mad
    return result
