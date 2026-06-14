"""Turn WLASL videos into landmark-sequence training arrays.

Reads WLASL_v0.3.json, keeps the configured glosses that actually have videos on
disk, runs HolisticExtractor over each clip, resamples to SEQ_LEN frames, and
writes X.npy / y.npy / splits.npy / labels.json into data/processed.

See the README for how to obtain the WLASL videos (Kaggle "wlasl-processed").
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config as C
from .landmarks import HolisticExtractor


def resample_sequence(seq: np.ndarray, length: int = C.SEQ_LEN) -> np.ndarray:
    """Resample a (n, F) sequence to exactly (length, F) by uniform index sampling.

    Upsamples short clips (repeats frames) and downsamples long ones. Empty
    input yields zeros.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"expected (n, F) sequence, got shape {seq.shape}")
    n = seq.shape[0]
    if n == 0:
        return np.zeros((length, seq.shape[1] if seq.ndim == 2 else C.FEATURE_DIM), np.float32)
    if n == length:
        return seq
    idx = np.linspace(0, n - 1, length).round().astype(int)
    return seq[idx]


def _video_path(video_id: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):
        p = C.WLASL_VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def _select_instances() -> dict[str, list[dict]]:
    """Map each wanted gloss -> list of instances whose video file exists."""
    entries = json.loads(C.WLASL_JSON.read_text())
    wanted = set(C.GLOSSES)
    available: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        gloss = entry["gloss"]
        if gloss not in wanted:
            continue
        for inst in entry["instances"]:
            if _video_path(inst["video_id"]) is not None:
                available[gloss].append(inst)
    return {g: insts for g, insts in available.items()
            if len(insts) >= C.MIN_SAMPLES_PER_GLOSS}


def _clip_to_sequence(extractor: HolisticExtractor, video_path: Path,
                      frame_start: int, frame_end: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    vectors = []
    frame_idx = 0
    ts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx < frame_start:
            continue
        if frame_end != -1 and frame_idx > frame_end:
            break
        vec, _result, pose_present = extractor(frame, timestamp_ms=ts)
        ts += 33  # ~30 fps spacing for monotonic timestamps
        if pose_present:
            vectors.append(vec)
    cap.release()
    if not vectors:
        return np.zeros((0, C.FEATURE_DIM), np.float32)
    return resample_sequence(np.stack(vectors), C.SEQ_LEN)


def build():
    from tqdm import tqdm

    if not C.WLASL_JSON.exists():
        raise SystemExit(f"Missing {C.WLASL_JSON}. See README for WLASL setup.")

    selected = _select_instances()
    if not selected:
        raise SystemExit(
            "No glosses had enough downloaded videos. Check data/wlasl/videos "
            "and config.GLOSSES / MIN_SAMPLES_PER_GLOSS."
        )

    labels = sorted(selected.keys())
    label_to_idx = {g: i for i, g in enumerate(labels)}
    print(f"Building dataset for {len(labels)} glosses: {labels}")

    extractor = HolisticExtractor(running_mode="VIDEO")
    X, y, splits = [], [], []
    try:
        for gloss in labels:
            for inst in tqdm(selected[gloss], desc=gloss, leave=False):
                vpath = _video_path(inst["video_id"])
                seq = _clip_to_sequence(
                    extractor, vpath,
                    int(inst.get("frame_start", 1)),
                    int(inst.get("frame_end", -1)),
                )
                if seq.shape[0] != C.SEQ_LEN:
                    continue  # no pose detected anywhere in the clip
                X.append(seq)
                y.append(label_to_idx[gloss])
                splits.append(inst.get("split", "train"))
    finally:
        extractor.close()

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    splits = np.array(splits, dtype=object)

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.PROCESSED_DIR / "X.npy", X)
    np.save(C.PROCESSED_DIR / "y.npy", y)
    np.save(C.PROCESSED_DIR / "splits.npy", splits)
    (C.PROCESSED_DIR / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"\nSaved {len(X)} clips -> {C.PROCESSED_DIR}")
    print(f"Splits: {dict(zip(*np.unique(splits, return_counts=True)))}")


if __name__ == "__main__":
    build()
