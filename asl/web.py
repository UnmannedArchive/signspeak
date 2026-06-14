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
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

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
  .tabs { display:flex; gap:6px; padding:12px 24px 0; }
  .tab { background:#161b22; border:1px solid #21262d; border-bottom:none; color:#8b949e;
         padding:8px 16px; border-radius:8px 8px 0 0; cursor:pointer; font-size:14px; }
  .tab.active { background:#0d1117; color:#e6edf3; border-color:#30363d; }
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
           padding:8px 14px; font-size:14px; cursor:pointer; }
  button:hover { background:#30363d; }
  .teach { padding:24px; max-width:1100px; margin:0 auto; }
  .mic { font-size:16px; padding:12px 22px; background:#1f6feb; border-color:#1f6feb; color:#fff; }
  .mic:disabled { opacity:.6; }
  .ask { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0 4px; }
  #sentence-in { flex:1; min-width:240px; background:#0d1117; border:1px solid #30363d;
                 color:#e6edf3; border-radius:8px; padding:10px 12px; font-size:15px; }
  .glossline { color:#d2a8ff; font-size:15px; letter-spacing:.5px; margin:14px 0; min-height:20px; }
  .steps { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; }
  .step { background:#161b22; border:1px solid #21262d; border-radius:12px; padding:10px; }
  .step video { width:100%; border-radius:8px; background:#000; aspect-ratio:1/1; object-fit:cover; }
  .step .fs { display:flex; align-items:center; justify-content:center; aspect-ratio:1/1;
              border:1px dashed #30363d; border-radius:8px; color:#8b949e; text-align:center; }
  .step .sg { font-weight:700; margin:8px 2px 4px; }
  .step .sh { color:#8b949e; font-size:12px; line-height:1.4; }
</style></head>
<body>
  <header>
    <h1>ASL sign recognizer <span id="cam" class="pill off">camera off</span></h1>
    <p>18 signs · bidirectional LSTM on MediaPipe landmarks · Claude sentence + teaching layer</p>
  </header>
  <div class="tabs">
    <div class="tab active" data-tab="recognize" onclick="showTab('recognize')">Recognize → English</div>
    <div class="tab" data-tab="teach" onclick="showTab('teach')">Speak → Teach me to sign</div>
  </div>

  <div id="tab-recognize">
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
  </div>

  <div id="tab-teach" style="display:none">
    <div class="teach">
      <div class="ask">
        <button class="mic" id="mic">🎤 Speak a sentence</button>
        <input id="sentence-in" placeholder="…or type a sentence, e.g. I want to drink water">
        <button id="go">Show me</button>
      </div>
      <div class="muted" id="teach-status">Click the mic and say something, or type it and hit “Show me”.</div>
      <div class="glossline" id="teach-gloss"></div>
      <div class="steps" id="steps"></div>
    </div>
  </div>

<script>
let RECOG_ACTIVE = true;
function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.tab===name));
  document.getElementById('tab-recognize').style.display = name==='recognize'?'':'none';
  document.getElementById('tab-teach').style.display = name==='teach'?'':'none';
  RECOG_ACTIVE = (name==='recognize');
}

async function tick(){
  if(!RECOG_ACTIVE) return;
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

function setStatus(t){ document.getElementById('teach-status').textContent = t; }
function renderTeach(d){
  document.getElementById('teach-gloss').textContent = (d.gloss||[]).join('   ·   ');
  document.getElementById('steps').innerHTML = (d.steps||[]).map(s => {
    const media = s.video_id
      ? `<video src="/clip/${s.video_id}.mp4" autoplay muted loop playsinline></video>`
      : `<div class="fs">✋ fingerspell<br><b>${s.gloss}</b></div>`;
    return `<div class="step">${media}<div class="sg">${s.gloss}</div>
            <div class="sh">${s.how_to||''}</div></div>`;
  }).join('') || '<div class="muted">No signs found — try another sentence.</div>';
}
async function runTeach(text){
  if(!text || !text.trim()) return;
  setStatus('Planning the signs for “'+text+'” …');
  document.getElementById('steps').innerHTML = '';
  try {
    const d = await (await fetch('/teach?text='+encodeURIComponent(text))).json();
    setStatus('“'+d.sentence+'”  →  ' +
      (d.source==='claude' ? 'reordered into ASL grammar by Claude'
                           : 'word order (set ANTHROPIC_API_KEY for ASL grammar + notes)'));
    renderTeach(d);
  } catch(e){ setStatus('Sorry — could not plan the signs.'); }
}
document.getElementById('go').onclick = () => runTeach(document.getElementById('sentence-in').value);
document.getElementById('sentence-in').addEventListener('keydown', e=>{ if(e.key==='Enter') runTeach(e.target.value); });

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
document.getElementById('mic').onclick = () => {
  if(!SR){ setStatus('Speech recognition is not available in this browser — type your sentence instead.'); return; }
  const rec = new SR(); rec.lang='en-US'; rec.interimResults=false; rec.maxAlternatives=1;
  const btn = document.getElementById('mic'); btn.disabled = true; setStatus('Listening… speak now.');
  rec.onresult = e => { const t = e.results[0][0].transcript;
                        document.getElementById('sentence-in').value = t; runTeach(t); };
  rec.onerror = e => setStatus('Mic error: ' + e.error + ' (you can type instead).');
  rec.onend = () => { btn.disabled = false; };
  rec.start();
};
</script>
</body></html>"""


def main():
    import uvicorn
    print("ASL web UI -> http://localhost:8000  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
