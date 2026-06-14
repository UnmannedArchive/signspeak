"""Smoke test: the training pipeline must be able to learn separable signs.

We synthesize 3 classes, each a distinct landmark-sequence pattern plus noise,
then assert train_model overfits them. This exercises model.py + train.py end to
end without needing WLASL.
"""
import numpy as np

from asl import config as C
from asl.dataset import resample_sequence
from asl.train import train_model


def _synthetic_dataset(n_per_class=24, classes=3, T=16, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=(classes, T, C.FEATURE_DIM)).astype(np.float32)
    X, y = [], []
    for c in range(classes):
        for _ in range(n_per_class):
            sample = base[c] + rng.normal(0, 0.15, size=(T, C.FEATURE_DIM)).astype(np.float32)
            X.append(sample)
            y.append(c)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def test_pipeline_learns_separable_classes():
    X, y = _synthetic_dataset()
    result = train_model(X, y, num_classes=3, epochs=60, lr=1e-3, seed=0)
    assert result["train_acc"] >= 0.9, f"pipeline failed to learn: {result['train_acc']}"
    assert result["val_acc"] >= 0.7, f"poor generalization on toy data: {result['val_acc']}"


def test_resample_sequence_shapes():
    F = C.FEATURE_DIM
    short = np.ones((5, F), dtype=np.float32)
    long = np.ones((90, F), dtype=np.float32)
    empty = np.zeros((0, F), dtype=np.float32)

    assert resample_sequence(short, 32).shape == (32, F)
    assert resample_sequence(long, 32).shape == (32, F)
    assert resample_sequence(empty, 32).shape == (32, F)
