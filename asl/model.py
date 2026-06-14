"""The sign classifier: an LSTM over landmark sequences.

Kept deliberately small and swappable -- the rest of the pipeline only depends
on the (batch, SEQ_LEN, FEATURE_DIM) -> (batch, num_classes) contract, so an MLP
or transformer could drop in here unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from . import config as C


class SignLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = C.FEATURE_DIM,
        hidden: int = C.LSTM_HIDDEN,
        layers: int = C.LSTM_LAYERS,
        dropout: float = C.LSTM_DROPOUT,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, num_classes),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)   # average over time — robust for short clips
        return self.head(pooled)


def save_model(model: SignLSTM, labels: list[str], weights_path: Path = C.MODEL_WEIGHTS,
               labels_path: Path = C.LABELS_JSON) -> None:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(labels)}, weights_path)
    Path(labels_path).write_text(json.dumps(labels, indent=2))


def load_model(weights_path: Path = C.MODEL_WEIGHTS, labels_path: Path = C.LABELS_JSON,
               device: str = "cpu") -> tuple[SignLSTM, list[str]]:
    labels = json.loads(Path(labels_path).read_text())
    ckpt = torch.load(weights_path, map_location=device)
    model = SignLSTM(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, labels
