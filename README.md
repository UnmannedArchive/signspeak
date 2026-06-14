# ASL Word-Level Sign Recognizer

A local, real-time tool that watches you sign through your webcam and prints the
recognized word on screen. It recognizes a fixed set of ~30–40 whole American
Sign Language signs.

This is a **portfolio v1**, scoped on purpose. Full ASL translation (fluent,
connected signing, with grammar carried by facial expression and spatial
referencing) is an open research problem; this project does the part that
genuinely works well — *isolated word-level recognition* — and does it cleanly.

See [`docs/specs/2026-06-14-asl-translator-design.md`](docs/specs/2026-06-14-asl-translator-design.md)
for the full design and the honest list of trade-offs.

## How it works

```
webcam / WLASL video ─► MediaPipe Holistic ─► normalized skeleton ─► LSTM ─► word
                         (same code both sides)
```

The trick that makes a model trained on the WLASL dataset recognize *you*: we
never feed pixels to the model. Every frame — recorded or live — is reduced to a
normalized skeleton of hand + upper-body points, which strips away signer
appearance, background, and resolution. That shared feature extractor lives in
[`asl/landmarks.py`](asl/landmarks.py).

## Setup

Requires Python 3.12.

```bash
cd ~/projects/asl-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/get_holistic_model.sh        # downloads the MediaPipe model (~13 MB)
```

## Run the tests (no dataset needed)

```bash
pytest -q
```

These verify the landmark math (shape, translation/scale invariance, zero-fill)
and that the training pipeline can learn separable signs from synthetic data.

## Train on WLASL

1. **Pick your vocabulary.** Edit `GLOSSES` in [`asl/config.py`](asl/config.py).
   Words without enough downloaded videos are skipped automatically
   (`MIN_SAMPLES_PER_GLOSS`).

2. **Get the videos.** Two options:

   - **No-auth downloader (recommended).** The WLASL metadata `WLASL_v0.3.json`
     comes from the [WLASL GitHub repo](https://github.com/dxli94/WLASL); the
     original per-clip URLs have rotted, but several dictionary sources and
     YouTube still serve video. `scripts/download_wlasl.py` pulls a subset for
     your `GLOSSES` from those live sources and validates each file:

     ```bash
     curl -sSL -o data/wlasl/WLASL_v0.3.json \
       https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json
     python -m scripts.download_wlasl   # needs yt-dlp (in requirements.txt)
     ```

   - **Kaggle mirror.** `risangbaskoro/wlasl-processed` has the full video set;
     arrange it as `data/wlasl/WLASL_v0.3.json` + `data/wlasl/videos/<id>.mp4`.

3. **Build the landmark dataset** (runs MediaPipe over every clip — slow):

   ```bash
   python -m asl.dataset
   ```

4. **Train** (prints validation accuracy + a confusion matrix so you can prune
   confusable signs):

   ```bash
   python -m asl.train
   ```

5. **Verify recognition without a webcam** — evaluates the trained model on
   held-out real clips:

   ```bash
   python -m scripts.verify_model
   ```

## Run the live demo

```bash
python -m asl.infer_live
```

Keys: **space** = translate the current phrase now · **c** = clear · **q / Esc** = quit.

The window shows your skeleton, the committed word (debounced so it doesn't
flicker), the live top guess + confidence, a buffer-fill bar, and — once a phrase
is built — the English sentence.

### Sentence translation (phase 2)

Committed signs accumulate into a phrase. After a short pause (or when you press
space) the glosses go to **Claude** and come back as fluent English (e.g.
`ME WANT COFFEE` → "I'd like a coffee"). It runs on a background thread, so the
video never stalls.

Set a key to enable it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Without a key it falls back to a plain word-join, so the demo still runs offline.
The model is `claude-opus-4-8` by default; switch `LLM_MODEL` in
[`asl/config.py`](asl/config.py) to `claude-haiku-4-5` for lower latency.

## Project layout

| File | Job |
|---|---|
| `asl/config.py` | Constants, paths, vocabulary, thresholds |
| `asl/landmarks.py` | Frame → normalized landmark vector (shared spine) |
| `asl/dataset.py` | WLASL videos → landmark-sequence arrays |
| `asl/model.py` | The LSTM classifier (save/load) |
| `asl/train.py` | Training loop + metrics |
| `asl/infer_live.py` | The OpenCV real-time demo |
| `tests/` | Landmark-math unit tests + a training smoke test |

## Phase 2 (not built yet)

- A language-model layer to render recognized glosses as fluent English
  (e.g. `WANT COFFEE` → "I'd like a coffee").
- Fingerspelling fallback for out-of-vocabulary words.
- Larger vocabulary; a few self-recorded samples per word to close the
  WLASL→webcam domain gap.
- A web UI and deployment.
