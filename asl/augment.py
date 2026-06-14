"""Landmark-space augmentation for training.

With only ~10 clips per sign, augmentation is the biggest accuracy lever: it
synthesizes plausible variants of each clip so the model sees many more "signers"
and camera conditions than we actually downloaded. Everything operates on a
normalized (T, FEATURE_DIM) sequence and returns the same shape, so it slots
straight into the training loop. Pure numpy — unit-tested without a model.
"""
from __future__ import annotations

import numpy as np

from . import config as C

# MediaPipe pose left<->right symmetric landmark pairs (33-landmark model).
_POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
              (27, 28), (29, 30), (31, 32)]


def _split(seq: np.ndarray):
    T = seq.shape[0]
    pose = seq[:, C.POSE_SLICE].reshape(T, C.NUM_POSE, 4).copy()
    lh = seq[:, C.LH_SLICE].reshape(T, C.NUM_HAND, 3).copy()
    rh = seq[:, C.RH_SLICE].reshape(T, C.NUM_HAND, 3).copy()
    return pose, lh, rh


def _merge(pose, lh, rh) -> np.ndarray:
    T = pose.shape[0]
    return np.concatenate(
        [pose.reshape(T, -1), lh.reshape(T, -1), rh.reshape(T, -1)], axis=1
    ).astype(np.float32)


def flip_horizontal(seq: np.ndarray) -> np.ndarray:
    """Mirror the signer: negate x, swap symmetric pose landmarks, swap hands.

    Turns a right-handed performance into a left-handed one, which helps the
    model generalize across signers of either handedness.
    """
    pose, lh, rh = _split(seq)
    pose[..., 0] *= -1.0
    lh[..., 0] *= -1.0
    rh[..., 0] *= -1.0
    for a, b in _POSE_SWAP:
        pose[:, [a, b]] = pose[:, [b, a]]
    return _merge(pose, rh, lh)  # left/right hands swapped


def scale(seq, rng, lo=0.85, hi=1.15):
    s = float(rng.uniform(lo, hi))
    pose, lh, rh = _split(seq)
    pose[..., :3] *= s
    lh *= s
    rh *= s
    return _merge(pose, lh, rh)


def translate(seq, rng, sigma=0.05):
    off = rng.normal(0, sigma, 3).astype(np.float32)
    pose, lh, rh = _split(seq)
    pose[..., :3] += off
    lh += off
    rh += off
    return _merge(pose, lh, rh)


def rotate2d(seq, rng, max_deg=12.0):
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pose, lh, rh = _split(seq)
    for arr in (pose, lh, rh):
        arr[..., :2] = arr[..., :2] @ rot.T
    return _merge(pose, lh, rh)


def jitter(seq, rng, sigma=0.015):
    return (seq + rng.normal(0, sigma, seq.shape)).astype(np.float32)


def time_warp(seq, rng, lo=0.8, hi=1.2):
    """Speed the sign up or slow it down, then resample back to T frames."""
    T = seq.shape[0]
    n = max(2, int(round(T * float(rng.uniform(lo, hi)))))
    up = seq[np.linspace(0, T - 1, n).round().astype(int)]
    return up[np.linspace(0, n - 1, T).round().astype(int)].astype(np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random subset of augmentations to one (T, FEATURE_DIM) clip."""
    out = seq.astype(np.float32)
    if rng.random() < 0.5:
        out = flip_horizontal(out)
    if rng.random() < 0.7:
        out = scale(out, rng)
    if rng.random() < 0.6:
        out = translate(out, rng)
    if rng.random() < 0.5:
        out = rotate2d(out, rng)
    if rng.random() < 0.7:
        out = time_warp(out, rng)
    if rng.random() < 0.8:
        out = jitter(out, rng)
    return out.astype(np.float32)
