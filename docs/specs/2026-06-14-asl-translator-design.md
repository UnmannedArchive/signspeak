# ASL Word-Level Sign Recognizer — Design

**Date:** 2026-06-14
**Status:** Approved, building v1
**Author:** Joseph (with Claude)

## What this is

A local, real-time tool that watches you sign through your webcam and prints the
recognized word on screen. v1 recognizes a fixed set of ~30–40 whole signs from
American Sign Language. It is a portfolio piece — the bar is "works reliably and
looks great on camera," not "translates fluent ASL."

## What this is *not* (and why)

Full ASL translation is an open research problem. ASL grammar lives in facial
expression, eyebrow position, mouth morphemes, spatial referencing, and
classifiers — not just hand shapes — and continuous (connected) signing has never
been translated reliably by anyone. General vision-language models are also weak
at fast temporal hand motion and add latency. So v1 deliberately scopes down to
**isolated, word-level recognition** of a fixed vocabulary, which is genuinely
solvable and demos well.

## Decisions (locked)

| Axis | Choice |
|---|---|
| Purpose | Portfolio / demo piece |
| Recognition target | Word-level signs, fixed vocabulary (~30–40 glosses) |
| Training data | WLASL (Word-Level ASL) — a small, sample-rich subset |
| Feature representation | MediaPipe landmarks (NOT raw pixels) |
| Model | Sliding-window LSTM/GRU over landmark sequences |
| Local UI | OpenCV window with overlaid prediction + skeleton |
| Language stack | Python 3.12, mediapipe, opencv, numpy, PyTorch |
| VLM sentence layer | Deferred to phase 2 |

## The key idea that makes it work

The **same feature-extraction code runs on both the WLASL training videos and the
live webcam.** Every frame — recorded or live — is reduced to a normalized
skeleton of hand + upper-body points. Because we train on geometry, not pixels,
the model is insulated from differences in signer appearance, background, and
resolution between WLASL and your camera. Normalization (centering on the
shoulder midpoint, scaling by shoulder width) adds invariance to *where* and *how
big* the signer appears in frame.

## Architecture

Three stages sharing one feature extractor.

### 1. Data prep (offline) — `asl/dataset.py`
- Input: `WLASL_v0.3.json` (gloss → instances) + a folder of downloaded videos.
- Select the subset: the configured gloss list, keeping only glosses with enough
  available videos.
- For each video: read frames (OpenCV) → MediaPipe Holistic per frame →
  normalized landmark vector → resample/pad to fixed length `T`.
- Output: `X` (N, T, F) and `y` (N,) `.npy` arrays, a `labels.json` map, and the
  WLASL train/val/test split.

### 2. Train (offline) — `asl/train.py` + `asl/model.py`
- Load prepared sequences → train the LSTM/GRU classifier → save best weights +
  label map. Report validation accuracy and a confusion matrix so we can see
  which signs collide (and swap them out of the vocab if needed).

### 3. Live demo (the deliverable) — `asl/infer_live.py`
- OpenCV webcam loop → same landmark extraction → rolling buffer of the last `T`
  frames → model prediction → **debounce** (commit a word only after K
  consistent, confident frames) → overlay the word, confidence, and drawn
  skeleton on the video.

## Feature vector

Per frame, `F = 258` values:
- Pose: 33 landmarks × (x, y, z, visibility) = 132
- Left hand: 21 × (x, y, z) = 63
- Right hand: 21 × (x, y, z) = 63

Face landmarks are omitted in v1 (large and mostly noise for distinguishing these
glosses; revisit if non-manual markers become necessary). A missing hand is
zero-filled, using the *same* convention in training and live inference.

Normalization is a pure numpy function (no MediaPipe dependency) so it can be unit
tested: translate by the shoulder midpoint, scale by shoulder width.

## Modules (each one job)

- `asl/config.py` — sequence length `T`, feature size, the gloss list, paths,
  confidence + debounce thresholds.
- `asl/landmarks.py` — frame → normalized landmark vector. Splits into a
  MediaPipe extraction part and a **pure, testable** normalization part. Shared by
  data prep and live inference.
- `asl/dataset.py` — WLASL subset selection + video → saved landmark sequences.
- `asl/model.py` — defines / saves / loads the classifier.
- `asl/train.py` — training loop, metrics, saves weights + label map. Exposes a
  `train_model(...)` function callable from tests on synthetic data.
- `asl/infer_live.py` — the OpenCV real-time loop.

## Robustness

- No hand detected → show "…", do not predict.
- Confidence below threshold → show "?".
- One hand off-screen → zero-fill that hand (consistent with training).
- Debounce: require K consecutive consistent predictions above threshold before
  committing a word; require a low-activity gap before the same word repeats.

## Testing

- **Unit:** landmark vector shape is `F`; normalization is invariant to
  translation and scale; zero-fill lands in the right slots.
- **Smoke:** generate a synthetic 3-class dataset of distinct landmark-sequence
  patterns and confirm the training pipeline overfits it (proves model + loop
  learn) — runs without WLASL.
- **Eval:** held-out validation accuracy + confusion matrix on the real subset.
- **Live sanity:** sign a known word, watch it appear.

## Phasing

- **Phase 1 (built):** WLASL subset, landmark pipeline, LSTM, OpenCV live demo,
  tests.
- **Phase 2 — sentence layer (built):** `asl/translator.py` renders a phrase of
  committed glosses as fluent English via Claude (`claude-opus-4-8` by default),
  with a plain word-join fallback when no `ANTHROPIC_API_KEY` is set. The live
  loop accumulates committed signs, finalizes on a pause (or space), and
  translates on a background thread so the video never stalls. `SentenceBuilder`
  is unit-tested.
- **Phase 2 — remaining (later):** fingerspelling fallback; larger vocabulary;
  web UI and deployment.

## Risks / honest caveats

- **WLASL link rot:** the original video URLs decay; the practical source is the
  Kaggle "wlasl-processed" mirror. Documented in the README; `dataset.py` works
  off a local video folder and skips missing files.
- **Domain gap:** even on landmarks, WLASL→webcam transfer is imperfect.
  Mitigations: normalization, picking visually distinct glosses, and the
  confusion matrix to prune confusable signs. If transfer is poor, the same
  pipeline supports adding a few self-recorded samples per word (phase-2 hybrid).
- **MediaPipe install** on macOS arm64 / Python 3.12 is verified as step 0.
