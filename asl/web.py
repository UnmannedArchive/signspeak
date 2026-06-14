"""Local web UI for the ASL recognizer (FastAPI).

The browser shows the annotated webcam feed plus a live panel with the top-3
predictions, the building phrase, and the translated English sentence. It reuses
the *exact* Python pipeline behind the desktop demo (same landmarks, same model,
same debouncing, same Claude translator), so predictions match.

Run:  python -m asl.web      then open  http://localhost:8000
Note: the server process needs camera permission — run it from your own terminal
and allow camera access when macOS asks.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import config as C


def _placeholder_jpeg(msg: str) -> bytes:
    import cv2

    img = np.full((480, 640, 3), 22, np.uint8)
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

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._set_jpeg(_placeholder_jpeg(
                "Camera unavailable. Run `python -m asl.web` from your terminal "
                "and allow camera access (System Settings > Privacy > Camera)."))
            return

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

        start = time.time()
        prev = start
        fps = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
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
            cap.release()
            extractor.close()

    def clear_sentence(self):
        # called from the web layer; safe no-op signal via state
        with self.lock:
            self._sentence = ""


engine = Engine()
app = FastAPI(title="ASL sign recognizer")


@app.on_event("startup")
def _startup():
    engine.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


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


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASL sign recognizer</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0d1117; color:#e6edf3; }
  header { padding:18px 24px; border-bottom:1px solid #21262d; }
  header h1 { margin:0; font-size:18px; letter-spacing:.2px; }
  header p { margin:4px 0 0; color:#8b949e; font-size:13px; }
  .wrap { display:grid; grid-template-columns: minmax(0,1fr) 320px; gap:20px; padding:24px; max-width:1100px; margin:0 auto; }
  @media (max-width: 820px){ .wrap{ grid-template-columns:1fr; } }
  .video { background:#000; border-radius:12px; overflow:hidden; border:1px solid #21262d; aspect-ratio:4/3; }
  .video img { width:100%; height:100%; object-fit:cover; display:block; }
  .panel { display:flex; flex-direction:column; gap:16px; }
  .card { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:16px; }
  .label { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#8b949e; margin-bottom:8px; }
  .word { font-size:34px; font-weight:700; line-height:1.1; }
  .phrase { color:#7ee787; font-size:14px; min-height:18px; word-break:break-word; }
  .row { display:flex; justify-content:space-between; align-items:center; font-variant-numeric:tabular-nums; padding:3px 0; }
  .row .nm { color:#c9d1d9; }
  .bar { height:6px; background:#21262d; border-radius:4px; overflow:hidden; margin-top:3px; }
  .bar > div { height:100%; background:#58a6ff; }
  .sentence { font-size:20px; line-height:1.35; min-height:28px; }
  .muted { color:#8b949e; font-size:12px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; }
  .ok { background:#1f6f3f33; color:#7ee787; } .off { background:#6e252533; color:#ff7b72; }
  button { background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:8px;
           padding:6px 12px; font-size:13px; cursor:pointer; }
  button:hover { background:#30363d; }
</style></head>
<body>
  <header>
    <h1>ASL sign recognizer <span id="cam" class="pill off">camera off</span></h1>
    <p>18 signs · bidirectional LSTM on MediaPipe landmarks · sentence layer via Claude</p>
  </header>
  <div class="wrap">
    <div class="video"><img src="/video" alt="webcam feed"></div>
    <div class="panel">
      <div class="card">
        <div class="label">Recognized</div>
        <div class="word" id="word">—</div>
        <div class="phrase" id="phrase"></div>
      </div>
      <div class="card">
        <div class="label">Live top-3</div>
        <div id="topk"></div>
      </div>
      <div class="card">
        <div class="label">English</div>
        <div class="sentence" id="sentence"></div>
      </div>
      <div class="muted"><span id="fps">0</span> fps</div>
    </div>
  </div>
<script>
async function tick(){
  try {
    const s = await (await fetch('/state')).json();
    document.getElementById('word').textContent = s.word || '—';
    document.getElementById('phrase').textContent = (s.phrase||[]).join(' ');
    document.getElementById('sentence').textContent = s.translating ? 'translating…' : (s.sentence||'');
    document.getElementById('fps').textContent = s.fps ?? 0;
    const cam = document.getElementById('cam');
    cam.textContent = s.camera ? 'camera on' : 'camera off';
    cam.className = 'pill ' + (s.camera ? 'ok' : 'off');
    const tk = (s.topk||[]).map(([n,c]) =>
      `<div class="row"><span class="nm">${n}</span><span>${Math.round(c*100)}%</span></div>
       <div class="bar"><div style="width:${Math.round(c*100)}%"></div></div>`).join('');
    document.getElementById('topk').innerHTML = tk || '<div class="muted">…</div>';
  } catch(e){}
}
setInterval(tick, 200); tick();
</script>
</body></html>"""


def main():
    import uvicorn
    print("ASL web UI -> http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
