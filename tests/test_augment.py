"""Tests for landmark-space augmentation."""
import numpy as np

from asl import augment, config as C


def _seq(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((C.SEQ_LEN, C.FEATURE_DIM)).astype(np.float32)


def test_augment_preserves_shape_and_dtype():
    rng = np.random.default_rng(1)
    out = augment.augment_sequence(_seq(), rng)
    assert out.shape == (C.SEQ_LEN, C.FEATURE_DIM)
    assert out.dtype == np.float32


def test_each_op_preserves_shape():
    rng = np.random.default_rng(2)
    s = _seq()
    for fn in (augment.flip_horizontal,):
        assert fn(s).shape == s.shape
    for fn in (augment.scale, augment.translate, augment.rotate2d,
               augment.jitter, augment.time_warp):
        assert fn(s, rng).shape == s.shape


def test_flip_is_an_involution():
    s = _seq(3)
    twice = augment.flip_horizontal(augment.flip_horizontal(s))
    np.testing.assert_allclose(twice, s, atol=1e-5)


def test_flip_swaps_hands():
    # Left hand present, right hand absent -> after mirroring, right is present.
    s = np.zeros((C.SEQ_LEN, C.FEATURE_DIM), np.float32)
    s[:, C.LH_SLICE] = 1.0
    flipped = augment.flip_horizontal(s)
    assert np.all(flipped[:, C.LH_SLICE] == 0.0)
    assert np.any(flipped[:, C.RH_SLICE] != 0.0)
