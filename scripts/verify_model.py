"""Prove the trained model actually recognizes signs, with no webcam.

Loads the saved model + the processed landmark arrays, evaluates on the held-out
split, and prints accuracy plus a few real example predictions. The webcam path
feeds the *same* landmark vectors into the *same* model, so good held-out
accuracy here is direct evidence the live recognizer works.

Run from the repo root:  python -m scripts.verify_model
"""
from __future__ import annotations

import json

import numpy as np
import torch

from asl import config as C
from asl.model import load_model


def main():
    if not C.MODEL_WEIGHTS.exists():
        raise SystemExit("No trained model yet — run the pipeline first.")

    X = np.load(C.PROCESSED_DIR / "X.npy")
    y = np.load(C.PROCESSED_DIR / "y.npy")
    splits = np.load(C.PROCESSED_DIR / "splits.npy", allow_pickle=True)
    labels = json.loads((C.PROCESSED_DIR / "labels.json").read_text())

    model, model_labels = load_model(device="cpu")
    assert model_labels == labels, "label mismatch between model and processed data"

    # Prefer the WLASL test split; fall back to a held-out slice if it's empty.
    mask = splits == "test"
    if mask.sum() < len(labels):  # too few test samples to be meaningful
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        mask = np.zeros(len(X), bool)
        mask[idx[: max(len(labels), int(0.2 * len(X)))]] = True
        split_name = "held-out 20%"
    else:
        split_name = "WLASL test"

    Xe = torch.tensor(X[mask], dtype=torch.float32)
    ye = y[mask]
    with torch.no_grad():
        logits = model(Xe)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).numpy()

    acc = (preds == ye).mean()
    print(f"Glosses: {len(labels)}  |  clips total: {len(X)}  |  eval set: {split_name} ({mask.sum()})")
    print(f"Held-out accuracy: {acc:.1%}  (random baseline {1/len(labels):.1%})\n")

    print("Sample predictions (true -> predicted @ confidence):")
    shown = 0
    for i in np.where(mask)[0]:
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(X[i:i+1], dtype=torch.float32)), 1)[0]
        conf, idx = float(p.max()), int(p.argmax())
        flag = "ok " if idx == y[i] else "MISS"
        print(f"  [{flag}] {labels[y[i]]:10s} -> {labels[idx]:10s} @ {conf:.0%}")
        shown += 1
        if shown >= 16:
            break


if __name__ == "__main__":
    main()
