"""The live demo: webcam -> landmarks -> LSTM -> word on screen.

Run from the repo root:  python -m asl.infer_live
Press q (or Esc) to quit.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import config as C
from .landmarks import HolisticExtractor, draw_overlay


class Debouncer:
    """Commits a word only after it has been the top guess for N consistent,
    confident frames -- this kills the per-frame flicker."""

    def __init__(self, n: int = C.DEBOUNCE_FRAMES, conf: float = C.CONF_THRESHOLD):
        self.n = n
        self.conf = conf
        self.recent: deque[int] = deque(maxlen=n)
        self.committed: str | None = None

    def update(self, label_idx: int, confidence: float, labels: list[str]) -> str | None:
        if confidence < self.conf:
            self.recent.clear()
            return self.committed
        self.recent.append(label_idx)
        if len(self.recent) == self.n and len(set(self.recent)) == 1:
            word = labels[label_idx]
            if word != self.committed:
                self.committed = word
        return self.committed


def _softmax_top(model, buffer, device):
    import torch

    x = torch.tensor(np.stack(buffer)[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx.item()), float(conf.item())


def main():
    import cv2
    import torch

    from .model import load_model

    if not C.MODEL_WEIGHTS.exists():
        raise SystemExit(
            f"No trained model at {C.MODEL_WEIGHTS}.\n"
            "Build the dataset and train first:\n"
            "  python -m asl.dataset\n  python -m asl.train"
        )

    device = "cpu"  # tiny model; CPU avoids per-frame host<->device transfer cost
    model, labels = load_model(device=device)
    print(f"Loaded model with {len(labels)} glosses: {labels}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    extractor = HolisticExtractor(running_mode="VIDEO")
    buffer: deque[np.ndarray] = deque(maxlen=C.SEQ_LEN)
    debouncer = Debouncer()
    start = time.time()
    prev_t = start
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, so it feels natural
            now = time.time()
            ts_ms = int((now - start) * 1000)

            vec, result, pose_present = extractor(frame, timestamp_ms=ts_ms)
            if pose_present:
                buffer.append(vec)

            live_label, live_conf = "...", 0.0
            if len(buffer) == C.SEQ_LEN and pose_present:
                idx, conf = _softmax_top(model, buffer, device)
                live_label, live_conf = labels[idx], conf
                debouncer.update(idx, conf, labels)

            draw_overlay(frame, result)
            _draw_hud(cv2, frame, debouncer.committed, live_label, live_conf,
                      len(buffer), fps)

            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.imshow("ASL sign recognizer  (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


def _draw_hud(cv2, frame, committed, live_label, live_conf, buf_len, fps):
    h, w = frame.shape[:2]
    # translucent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    word = committed or "—"
    cv2.putText(frame, word, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)

    sub = f"live: {live_label} ({live_conf:.0%})"
    cv2.putText(frame, sub, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)

    # buffer fill bar
    bar_w = int((buf_len / C.SEQ_LEN) * 200)
    cv2.rectangle(frame, (20, h - 30), (220, h - 18), (80, 80, 80), 1)
    cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 18), (80, 220, 80), -1)
    cv2.putText(frame, f"{fps:4.0f} fps", (w - 110, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
