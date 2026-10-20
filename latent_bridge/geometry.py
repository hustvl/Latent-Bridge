"""Explicit OpenCV camera conventions and full-density geometry utilities."""
from pathlib import Path
import numpy as np


def validate_camera(c2w, intrinsics):
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or intrinsics.shape != (len(c2w), 3, 3):
        raise ValueError("Expected c2w[T,4,4], intrinsics[T,3,3]")
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Nonfinite camera")
    if not np.allclose(c2w[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous camera matrix")
    rotation = c2w[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("Camera rotation must be a proper orthonormal matrix")
    if not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-5) or not np.allclose(intrinsics[:, 0, 1], 0, atol=1e-5) or not np.allclose(intrinsics[:, 1, 0], 0, atol=1e-5):
        raise ValueError("Expected zero-skew OpenCV intrinsics")
    if (intrinsics[:, [0, 1], [0, 1]] <= 0).any():
        raise ValueError("Focal lengths must be positive")


def first_camera_gauge(w2c):
    if w2c.ndim != 3 or w2c.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("Expected w2c[T,3,4] or [T,4,4]")
    matrix = np.broadcast_to(np.eye(4), (len(w2c), 4, 4)).copy()
    matrix[:, :3] = w2c[:, :3]
    return (matrix[0] @ np.linalg.inv(matrix)).astype(np.float32)


def unproject(depth, intrinsics, c2w):
    validate_camera(c2w, intrinsics)
    if depth.ndim != 3 or len(depth) != len(c2w):
        raise ValueError("Expected depth[T,H,W] matching cameras")
    yy, xx = np.indices(depth.shape[1:], dtype=np.float32)
    z = depth.astype(np.float32)
    x = (xx - intrinsics[:, 0, 2, None, None]) * z / intrinsics[:, 0, 0, None, None]
    y = (yy - intrinsics[:, 1, 2, None, None]) * z / intrinsics[:, 1, 1, None, None]
    points = np.stack((x, y, z), axis=-1)
    return np.einsum("tij,thwj->thwi", c2w[:, :3, :3], points) + c2w[:, None, None, :3, 3]


def rotation_direction_correction(predicted_c2w, gt_c2w):
    if predicted_c2w.shape != gt_c2w.shape:
        raise ValueError("Camera sequences must match")
    pred = np.linalg.inv(predicted_c2w[0]) @ predicted_c2w
    gt = np.linalg.inv(gt_c2w[0]) @ gt_c2w
    corrected = pred.copy()
    corrected[:, :3, :3] = gt[:, :3, :3]
    length = np.linalg.norm(gt[:, :3, 3], axis=-1)
    moving = length > 1e-8
    corrected[moving, :3, 3] = gt[moving, :3, 3] / length[moving, None] * np.linalg.norm(pred[moving, :3, 3], axis=-1, keepdims=True)
    return corrected


def write_ply(path, xyz, rgb=None):
    points = np.asarray(xyz).reshape(-1, 3)
    valid = np.isfinite(points).all(-1)
    fields = [(name, '<f4') for name in ('x', 'y', 'z')]
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.shape != xyz.shape:
            raise ValueError("RGB must be uint8 and match the dense XYZ shape")
        fields += [(name, 'u1') for name in ('red', 'green', 'blue')]
    packed = np.empty(int(valid.sum()), dtype=fields)
    for index, key in enumerate(('x', 'y', 'z')):
        packed[key] = points[valid, index]
    if rgb is not None:
        for index, key in enumerate(('red', 'green', 'blue')):
            packed[key] = rgb.reshape(-1, 3)[valid, index]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(packed)}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    if rgb is not None:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    with Path(path).open('xb') as stream:
        stream.write((header + 'end_header\n').encode('ascii'))
        stream.write(packed.tobytes())
