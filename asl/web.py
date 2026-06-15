"""Local web UI for the ASL recognizer (FastAPI).

Two modes in one page (asl/static/index.html):
  * Recognize -> English: annotated webcam feed + live top-3 + Claude sentence.
  * Speak -> Teach me: say/type a sentence, watch the real reference clip for
    each sign in ASL gloss order.

Both reuse the exact desktop pipeline, so predictions match. Run from repo root:
    python -m asl.web        then open  http://localhost:8000
The server process needs camera permission — run it from your own terminal and
allow camera + mic access in the browser when asked.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from . import config as C

STATIC = Path(__file__).resolve().parent / "static"


def _placeholder_jpeg(msg: str) -> bytes:
    import cv2

    img = np.full((480, 640, 3), 18, np.uint8)
    for i, line in enumerate(_wrap(msg, 44)):
        cv2.putText(img, line, (24, 220 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


class Engine:
    """Owns the camera + inference loop on a background thread; exposes the latest
    annotated frame and prediction state to the web layer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = _placeholder_jpeg("starting camera…")
        self.state = {"word": None, "topk": [], "phrase": [], "sentence": "",
                      "translating": False, "camera": False, "fps": 0.0}
        self._sentence = ""
        self._translating = False
        self.want_camera = True   # toggled by /camera/on|off

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def set_camera(self, on: bool):
        self.want_camera = on
        if not on:
            self._set_jpeg(_placeholder_jpeg("camera is off — turn it on to start recognizing"))

    def _set_jpeg(self, jpeg):
        with self.lock:
            self.latest_jpeg = jpeg

    def _run(self):
        import cv2

        from .infer_live import Debouncer, SentenceBuilder, _softmax_topk
        from .landmarks import HolisticExtractor, draw_overlay
        from .model import load_model
        from .translator import Translator

        if not C.MODEL_WEIGHTS.exists():
            self._set_jpeg(_placeholder_jpeg("No trained model — run python -m asl.train first."))
            return

        model, labels = load_model(device="cpu")
        translator = Translator()
        extractor = HolisticExtractor(running_mode="VIDEO")
        buffer: deque[np.ndarray] = deque(maxlen=C.SEQ_LEN)
        debouncer = Debouncer()
        builder = SentenceBuilder()

        def translate_async(glosses):
            with self.lock:
                self._translating = True
                self._sentence = ""
            result = translator.to_sentence(glosses)
            with self.lock:
                self._sentence = result
                self._translating = False

        cap = None
        start = time.time()
        prev = start
        fps = 0.0
        try:
            while True:
                # Camera off: release the device and idle until toggled back on.
                if not self.want_camera:
                    if cap is not None:
                        cap.release(); cap = None
                        buffer.clear()
                    with self.lock:
                        self.state["camera"] = False
                        self.state["fps"] = 0.0
                    time.sleep(0.1)
                    continue
                if cap is None:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        cap = None
                        self.want_camera = False
                        self._set_jpeg(_placeholder_jpeg(
                            "Camera unavailable — run `python -m asl.web` in your "
                            "terminal and allow camera access, then turn it on."))
                        continue

                ok, frame = cap.read()
                if not ok:
                    cap.release(); cap = None
                    continue
                frame = cv2.flip(frame, 1)
                now = time.time()
                vec, result, pose_present = extractor(frame, timestamp_ms=int((now - start) * 1000))
                if pose_present:
                    buffer.append(vec)

                topk = []
                if len(buffer) == C.SEQ_LEN and pose_present:
                    topk = _softmax_topk(model, buffer, "cpu", k=3)
                    idx, conf = topk[0]
                    debouncer.update(idx, conf, labels)
                    builder.note_commit(debouncer.committed, now)

                if builder.should_finalize(now):
                    builder.finalized = True
                    threading.Thread(target=translate_async,
                                     args=(list(builder.glosses),), daemon=True).start()

                draw_overlay(frame, result)
                jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 / dt

                with self.lock:
                    self.latest_jpeg = jpeg
                    self.state = {
                        "word": debouncer.committed,
                        "topk": [[labels[i], c] for i, c in topk],
                        "phrase": list(builder.glosses),
                        "sentence": self._sentence,
                        "translating": self._translating,
                        "camera": True,
                        "fps": round(fps, 1),
                    }
        finally:
            if cap is not None:
                cap.release()
            extractor.close()


engine = Engine()
app = FastAPI(title="ASL sign recognizer")


@app.on_event("startup")
def _startup():
    engine.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


@app.get("/state")
def state():
    with engine.lock:
        return JSONResponse(engine.state)


@app.get("/video")
def video():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with engine.lock:
                jpeg = engine.latest_jpeg
            yield boundary + jpeg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera/{state}")
def camera(state: str):
    engine.set_camera(state == "on")
    return JSONResponse({"camera": engine.want_camera})


@app.get("/teach")
def teach(text: str = ""):
    from . import teach as teach_mod

    return JSONResponse(teach_mod.plan(text))


@app.get("/clip/{video_id}.mp4")
def clip(video_id: str):
    if not video_id.isdigit():
        return JSONResponse({"error": "bad id"}, status_code=400)
    path = C.WLASL_VIDEO_DIR / f"{video_id}.mp4"
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


def main():
    import uvicorn
    print("ASL web UI -> http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
