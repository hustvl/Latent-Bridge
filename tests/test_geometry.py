import numpy as np
import pytest
from latent_bridge.geometry import unproject, first_camera_gauge, write_ply, rotation_direction_correction
from latent_bridge.evaluation import rms_align, evaluate_bundle, distances
from latent_bridge.filters import filter_depth


def scene(frames=3):
    depth = np.ones((frames, 3, 3), dtype=np.float32)
    k = np.tile(np.eye(3), (frames, 1, 1))
    pose = np.tile(np.eye(4), (frames, 1, 1))
    return depth, k, pose


def test_unprojection_and_gauge():
    depth, k, pose = scene()
    xyz = unproject(depth, k, pose)
    np.testing.assert_allclose(xyz[0, 2, 1], [1, 2, 1])
    pose[:, 0, 3] = [2, 3, 4]
    gauge = first_camera_gauge(np.linalg.inv(pose))
    np.testing.assert_allclose(gauge[:, 0, 3], [0, 1, 2])


def test_rms_similarity():
    points = np.random.default_rng(1).normal(size=(20, 3))
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2 * points @ rotation.T + [3, 2, 1]
    aligned, transform = rms_align(points, target, np.ones(20, dtype=bool))
    np.testing.assert_allclose(aligned, target, atol=1e-12)
    assert transform['scale'] == pytest.approx(2)


def test_distances_direction():
    result = distances(np.array([[0, 0, 0.]]), np.array([[0, 0, 0.], [2, 0, 0.]]))
    assert result == dict(accuracy=0., completeness=1., chamfer=.5)


@pytest.mark.parametrize('mode', ['none', 'log_mad', 'log_mad_multiview', 'log_mad_multiview_safe_v1'])
def test_identity_plane_filters(mode):
    depth, k, pose = scene()
    assert filter_depth(depth, np.ones_like(depth, dtype=bool), k, pose, mode).all()


def test_outlier_and_safe_fallback():
    depth, k, pose = scene(1)
    depth[0, 0, 0] = 100
    mask = np.ones_like(depth, dtype=bool)
    mad = filter_depth(depth, mask, k, pose, 'log_mad')
    assert mad.sum() == 8 and not mad[0, 0, 0]
    assert not filter_depth(depth, mask, k, pose, 'log_mad_multiview').any()
    np.testing.assert_array_equal(filter_depth(depth, mask, k, pose, 'log_mad_multiview_safe_v1'), mad)


def test_filter_never_removes_gt_scoring_points():
    depth, k, pose = scene()
    target = unproject(depth, k, pose)
    depth[:, 0, 0] = 100
    bundle = dict(depth=depth, intrinsics=k, c2w=pose, gt_xyz=target, gt_mask=np.ones_like(depth, dtype=bool))
    score, keep = evaluate_bundle(bundle, filter_mode='log_mad')
    assert keep.sum() == 24 and score['gt_points'] == 27
    assert score['accuracy'] < 1e-6 and score['completeness'] > 0
    assert score['historical_parity_validated'] is False
    with pytest.raises(ValueError, match='gt_c2w'):
        evaluate_bundle(bundle, camera_mode='gt_rotation_direction')


def test_correction_keeps_predicted_radius():
    _, _, pred = scene()
    gt = pred.copy()
    pred[:, 0, 3] = [0, 2, 4]
    gt[:, 1, 3] = [0, 20, 40]
    corrected = rotation_direction_correction(pred, gt)
    np.testing.assert_allclose(corrected[:, 1, 3], [0, 2, 4])
    np.testing.assert_allclose(corrected[:, 0, 3], 0)


def test_full_density_ply(tmp_path):
    depth, k, pose = scene()
    points = unproject(depth, k, pose)
    path = tmp_path / 'full.ply'
    write_ply(path, points, np.full(points.shape, 127, dtype=np.uint8))
    header, data = path.read_bytes().split(b'end_header\n')
    assert b'element vertex 27\n' in header
    assert len(data) == 27 * 15
    with pytest.raises(FileExistsError):
        write_ply(path, points)
