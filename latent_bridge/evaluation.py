"""Portable dense-correspondence evaluation, with explicit protocol labels."""
import numpy as np
from scipy.spatial import cKDTree

from .filters import filter_depth
from .geometry import unproject, rotation_direction_correction


def rms_align(prediction, target, correspondence):
    p, g = prediction[correspondence], target[correspondence]
    if len(p) < 3:
        raise ValueError("At least three finite corresponding points are required")
    pm, gm = p.mean(0), g.mean(0)
    a, b = p - pm, g - gm
    variance = np.square(a).sum()
    if variance <= 1e-16 or np.square(b).sum() <= 1e-16:
        raise ValueError("Degenerate alignment point cloud")
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    scale = float(np.sqrt(np.square(b).sum() / variance))
    translation = gm - scale * (rotation @ pm)
    return scale * (prediction @ rotation.T) + translation, dict(scale=scale, rotation=rotation.tolist(), translation=translation.tolist())


def distances(prediction, target):
    if not len(prediction) or not len(target):
        raise ValueError("Cannot evaluate empty point clouds")
    accuracy = float(cKDTree(target).query(prediction, workers=1)[0].mean())
    completeness = float(cKDTree(prediction).query(target, workers=1)[0].mean())
    return dict(accuracy=accuracy, completeness=completeness, chamfer=(accuracy + completeness) / 2)


def voxel_fps(points, voxel_size, max_points):
    try:
        import open3d as o3d
        import torch
        from pytorch3d.ops import sample_farthest_points
    except ImportError as error:
        raise ImportError("voxel_fps requires separately installed Open3D and PyTorch3D; no silent fallback") from error
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    points = np.asarray(cloud.voxel_down_sample(voxel_size).points).astype(np.float32)
    if len(points) > max_points:
        tensor = torch.from_numpy(points)[None]
        points = sample_farthest_points(tensor, K=max_points, random_start_point=False)[0][0].numpy()
    return points


def evaluate_bundle(bundle, *, filter_mode='none', camera_mode='omega', sampling='all', unit='native', seed=42):
    required = ('depth', 'intrinsics', 'c2w', 'gt_xyz', 'gt_mask')
    if any(key not in bundle for key in required):
        raise ValueError(f"Evaluation NPZ requires {required}")
    depth, intrinsics, c2w, target, gt_mask = (bundle[key] for key in required)
    if target.shape != (*depth.shape, 3) or gt_mask.shape != depth.shape or gt_mask.dtype != bool:
        raise ValueError("GT shape/boolean mask mismatch")
    if camera_mode == 'gt_rotation_direction':
        if 'gt_c2w' not in bundle:
            raise ValueError("GT-assisted camera protocol requires gt_c2w")
        c2w = rotation_direction_correction(c2w, bundle['gt_c2w'])
    elif camera_mode != 'omega':
        raise ValueError("Unknown camera protocol")
    prediction = unproject(depth, intrinsics, c2w)
    gt_valid = gt_mask & np.isfinite(target).all(-1)
    candidate = gt_valid & np.isfinite(prediction).all(-1)
    keep = filter_depth(depth, candidate, intrinsics, c2w, filter_mode)
    aligned, transform = rms_align(prediction, target, keep & gt_valid)
    p, g = aligned[keep], target[gt_valid]
    if sampling == 'voxel_fps':
        p, g = voxel_fps(p, 0.005, 20000), voxel_fps(g, 0.005, 20000)
    elif sampling != 'all':
        raise ValueError("Unknown sampling protocol")
    score = distances(p, g)
    return dict(protocol='portable_rms_correspondence_v1', historical_parity_validated=False,
                filter=filter_mode, cameras=camera_mode, sampling=sampling, unit=unit, seed=seed,
                input_prediction_points=int(candidate.sum()), retained_prediction_points=int(keep.sum()),
                gt_points=int(gt_valid.sum()), scored_prediction_points=len(p), scored_gt_points=len(g),
                alignment=transform, **score), keep
