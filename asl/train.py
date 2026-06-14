"""Train the sign classifier on prepared landmark sequences.

`train_model` is a pure function over numpy arrays so tests can drive it with
synthetic data. `main` wires it to the WLASL arrays produced by dataset.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .augment import augment_sequence
from .model import SignLSTM, save_model


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    num_classes: int | None = None,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
    augment: bool = False,
) -> dict:
    """Train SignLSTM on (N, T, F) inputs. Returns the best model + metrics.

    If no validation set is given, a small slice of X is held out so we always
    report an honest val number.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if num_classes is None:
        num_classes = int(y.max()) + 1

    if X_val is None:
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.2 * len(X)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_val, y_val = X[val_idx], y[val_idx]
        X, y = X[tr_idx], y[tr_idx]
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int64)

    # Train data stays in numpy so we can augment per batch; val stays fixed.
    Xt_eval = torch.tensor(X, device=device)
    yt_eval = torch.tensor(y, device=device)
    Xv = torch.tensor(X_val, device=device)
    yv = torch.tensor(y_val, device=device)

    model = SignLSTM(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(X), batch_size):
            bidx = perm[i : i + batch_size]
            xb = X[bidx]
            if augment:
                xb = np.stack([augment_sequence(s, rng) for s in xb])
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(y[bidx], dtype=torch.int64, device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb_t), yb_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_acc = _accuracy(model(Xt_eval), yt_eval)
            val_acc = _accuracy(model(Xv), yv)
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train {train_acc:.3f}  val {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train = _accuracy(model(Xt_eval), yt_eval)
    return {
        "model": model,
        "num_classes": num_classes,
        "train_acc": final_train,
        "val_acc": best_val,
        "history": history,
    }


def _save_confusion_png(cm, labels, test_acc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"ASL sign recognizer — test confusion (acc {test_acc:.0%})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = C.BASE_DIR / "docs" / "confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out}")


def main():
    proc = C.PROCESSED_DIR
    X = np.load(proc / "X.npy")
    y = np.load(proc / "y.npy")
    splits = np.load(proc / "splits.npy", allow_pickle=True)
    labels = json.loads((proc / "labels.json").read_text())

    is_train = splits != "test"
    is_test = splits == "test"
    print(f"Loaded {len(X)} clips, {len(labels)} glosses, "
          f"{is_train.sum()} train/val, {is_test.sum()} test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = train_model(
        X[is_train], y[is_train], num_classes=len(labels),
        device=device, verbose=True, augment=True, epochs=250,
    )
    model = result["model"]
    print(f"\nBest val accuracy: {result['val_acc']:.3f}")

    if is_test.any():
        from sklearn.metrics import confusion_matrix
        with torch.no_grad():
            preds = model(torch.tensor(X[is_test], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
        test_acc = (preds == y[is_test]).mean()
        print(f"Test accuracy:     {test_acc:.3f}\n")
        cm = confusion_matrix(y[is_test], preds, labels=list(range(len(labels))))
        print("Confusion matrix (rows = true, cols = pred):")
        print("        " + " ".join(f"{i:>3d}" for i in range(len(labels))))
        for i, row in enumerate(cm):
            print(f"{labels[i][:7]:>7} " + " ".join(f"{v:>3d}" for v in row))
        _save_confusion_png(cm, labels, test_acc)

    save_model(model, labels)
    print(f"\nSaved weights -> {C.MODEL_WEIGHTS}")
    print(f"Saved labels  -> {C.LABELS_JSON}")


if __name__ == "__main__":
    main()
