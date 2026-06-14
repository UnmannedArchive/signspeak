"""Unit tests for the pure landmark math (no MediaPipe needed)."""
import numpy as np
import pytest

from asl import config as C
from asl.landmarks import normalize_landmarks


def _make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    pose = rng.random((C.NUM_POSE, 4)).astype(np.float32)
    # ensure shoulders are well separated so width is non-trivial
    pose[C.L_SHOULDER, :2] = [0.4, 0.5]
    pose[C.R_SHOULDER, :2] = [0.6, 0.5]
    lh = rng.random((C.NUM_HAND, 3)).astype(np.float32)
    rh = rng.random((C.NUM_HAND, 3)).astype(np.float32)
    return pose, lh, rh


def test_feature_size_and_dtype():
    pose, lh, rh = _make_inputs()
    vec = normalize_landmarks(pose, lh, rh)
    assert vec.shape == (C.FEATURE_DIM,)
    assert vec.dtype == np.float32


def test_translation_invariance():
    pose, lh, rh = _make_inputs()
    base = normalize_landmarks(pose, lh, rh)

    shift = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    pose_s = pose.copy()
    pose_s[:, :3] += shift          # shift xyz, leave visibility
    lh_s = lh + shift
    rh_s = rh + shift
    shifted = normalize_landmarks(pose_s, lh_s, rh_s)

    np.testing.assert_allclose(base, shifted, atol=1e-5)


def test_scale_invariance():
    pose, lh, rh = _make_inputs()
    base = normalize_landmarks(pose, lh, rh)

    s = 2.5
    pose_s = pose.copy()
    pose_s[:, :3] *= s              # scale xyz, leave visibility
    lh_s = lh * s
    rh_s = rh * s
    scaled = normalize_landmarks(pose_s, lh_s, rh_s)

    np.testing.assert_allclose(base, scaled, atol=1e-5)


def test_missing_hand_is_zero_filled():
    pose, lh, rh = _make_inputs()
    vec = normalize_landmarks(pose, lh, rh, rh_present=False)
    assert np.all(vec[C.RH_SLICE] == 0.0)
    assert np.any(vec[C.LH_SLICE] != 0.0)   # present hand is not zero


def test_no_pose_returns_zeros():
    pose, lh, rh = _make_inputs()
    vec = normalize_landmarks(pose, lh, rh, pose_present=False)
    assert vec.shape == (C.FEATURE_DIM,)
    assert np.all(vec == 0.0)
