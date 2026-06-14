"""The live demo: webcam -> landmarks -> LSTM -> word on screen.

Phase 2 adds a sentence layer: committed signs accumulate into a phrase, and
after a short pause (or when you press space) the glosses are sent to Claude and
rendered as fluent English. Translation runs on a background thread so the video
never stalls.

Run from the repo root:  python -m asl.infer_live
Keys:  space = translate now   c = clear sentence   q / Esc = quit
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from . import config as C
from .landmarks import HolisticExtractor, draw_overlay
from .translator import Translator


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
            self.committed = labels[label_idx]
        return self.committed


class SentenceBuilder:
    """Accumulates committed glosses into a phrase and decides when it's done.

    Pure logic (no camera/model), so it's unit-tested directly. A new gloss is
    appended only when the committed word *changes*, and the phrase is ready to
    finalize once no new gloss has arrived for `pause_s` seconds.
    """

    def __init__(self, pause_s: float = C.FINALIZE_PAUSE_S):
        self.pause_s = pause_s
        self.glosses: list[str] = []
        self.last_commit: float | None = None
        self.finalized = False
        self._last_word: str | None = None

    def note_commit(self, word: str | None, now: float) -> None:
        if word and word != self._last_word:
            self.glosses.append(word)
            self._last_word = word
            self.last_commit = now
            self.finalized = False

    def should_finalize(self, now: float) -> bool:
        return (
            bool(self.glosses)
            and not self.finalized
            and self.last_commit is not None
            and (now - self.last_commit) >= self.pause_s
        )

    def clear(self) -> None:
        self.glosses = []
        self.last_commit = None
        self.finalized = False
        self._last_word = None


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

    translator = Translator()
    print("Translator:", "Claude" if translator.available else "offline fallback")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (index 0).")

    extractor = HolisticExtractor(running_mode="VIDEO")
    buffer: deque[np.ndarray] = deque(maxlen=C.SEQ_LEN)
    debouncer = Debouncer()
    builder = SentenceBuilder()

    # Shared state between the loop and the translation worker thread.
    state_lock = threading.Lock()
    sentence_text = ""
    translating = False

    def translate_async(glosses: list[str]) -> None:
        nonlocal sentence_text, translating
        with state_lock:
            translating = True
            sentence_text = ""
        result = translator.to_sentence(glosses)
        with state_lock:
            sentence_text = result
            translating = False

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
                committed = debouncer.update(idx, conf, labels)
                builder.note_commit(committed, now)

            # Auto-finalize after a pause.
            if builder.should_finalize(now):
                builder.finalized = True
                threading.Thread(
                    target=translate_async, args=(list(builder.glosses),), daemon=True
                ).start()

            with state_lock:
                cur_sentence, cur_translating = sentence_text, translating

            draw_overlay(frame, result)
            _draw_hud(cv2, frame, debouncer.committed, live_label, live_conf,
                      len(buffer), fps, builder.glosses, cur_sentence, cur_translating)

            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.imshow("ASL sign recognizer  (space=translate  c=clear  q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                builder.clear()
                with state_lock:
                    sentence_text = ""
                    translating = False
            elif key == ord(" ") and builder.glosses:
                builder.finalized = True
                threading.Thread(
                    target=translate_async, args=(list(builder.glosses),), daemon=True
                ).start()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


def _draw_hud(cv2, frame, committed, live_label, live_conf, buf_len, fps,
              glosses, sentence, translating):
    h, w = frame.shape[:2]

    # top banner: current committed word + the building gloss sequence
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 96), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    word = committed or "—"
    cv2.putText(frame, word, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3, cv2.LINE_AA)
    phrase = " ".join(glosses) if glosses else ""
    cv2.putText(frame, phrase, (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180, 220, 180), 1, cv2.LINE_AA)

    # bottom banner: the translated English sentence
    if translating or sentence:
        b = frame.copy()
        cv2.rectangle(b, (0, h - 70), (w, h), (15, 30, 45), -1)
        cv2.addWeighted(b, 0.6, frame, 0.4, 0, frame)
        text = "translating…" if translating else sentence
        color = (150, 180, 210) if translating else (255, 255, 255)
        cv2.putText(frame, text, (20, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2, cv2.LINE_AA)

    # live guess + buffer fill + fps
    sub = f"live: {live_label} ({live_conf:.0%})"
    cv2.putText(frame, sub, (20, h - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    bar_w = int((buf_len / C.SEQ_LEN) * 180)
    cv2.rectangle(frame, (w - 200, 18), (w - 20, 30), (80, 80, 80), 1)
    cv2.rectangle(frame, (w - 200, 18), (w - 200 + bar_w, 30), (80, 220, 80), -1)
    cv2.putText(frame, f"{fps:4.0f} fps", (w - 110, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
