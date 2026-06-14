"""Frame -> normalized landmark vector.

This module is the spine of the project: the *exact same* feature extraction runs
on WLASL training videos and on the live webcam, so the model only ever sees
normalized geometry, never pixels.

It is split into two layers:
  * `normalize_landmarks` / `arrays_from_result` are pure numpy + plain data and
    are unit-tested without MediaPipe.
  * `HolisticExtractor` wraps the MediaPipe Tasks HolisticLandmarker.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from . import config as C

# Standard MediaPipe hand topology (21 landmarks) for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]

# Upper-body pose connections (subset of the 33-point pose model) for drawing.
POSE_CONNECTIONS = [
    (11, 12),            # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24), (23, 24),  # torso
]


# --- Pure, testable feature math ------------------------------------------
def normalize_landmarks(
    pose: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    pose_present: bool = True,
    lh_present: bool = True,
    rh_present: bool = True,
) -> np.ndarray:
    """Turn raw landmark arrays into a normalized feature vector.

    Args:
        pose: (33, 4) array of [x, y, z, visibility].
        lh:   (21, 3) array of [x, y, z] for the left hand.
        rh:   (21, 3) array of [x, y, z] for the right hand.
        *_present: whether each group was actually detected this frame.

    Returns:
        (FEATURE_DIM,) float32 vector. All zeros if the pose is absent (no
        reference to normalize against). A missing hand is zero-filled in its
        own slice -- the same convention used everywhere.

    Normalization makes the vector invariant to where the signer is in frame
    (subtract the shoulder midpoint) and how large they appear (divide by
    shoulder width).
    """
    if not pose_present:
        return np.zeros(C.FEATURE_DIM, dtype=np.float32)

    pose = np.asarray(pose, dtype=np.float32)
    lh = np.asarray(lh, dtype=np.float32)
    rh = np.asarray(rh, dtype=np.float32)

    ref = (pose[C.L_SHOULDER, :3] + pose[C.R_SHOULDER, :3]) / 2.0
    width = float(np.linalg.norm(pose[C.L_SHOULDER, :2] - pose[C.R_SHOULDER, :2]))
    if width < 1e-6:
        width = 1.0

    pose_xyz = (pose[:, :3] - ref) / width
    pose_feat = np.concatenate([pose_xyz, pose[:, 3:4]], axis=1).reshape(-1)

    if lh_present:
        lh_feat = ((lh - ref) / width).reshape(-1)
    else:
        lh_feat = np.zeros(C.LH_DIM, dtype=np.float32)

    if rh_present:
        rh_feat = ((rh - ref) / width).reshape(-1)
    else:
        rh_feat = np.zeros(C.RH_DIM, dtype=np.float32)

    return np.concatenate([pose_feat, lh_feat, rh_feat]).astype(np.float32)


def arrays_from_result(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
    """Pull plain numpy arrays + presence flags out of a HolisticLandmarkerResult."""
    pose = np.zeros((C.NUM_POSE, 4), dtype=np.float32)
    lh = np.zeros((C.NUM_HAND, 3), dtype=np.float32)
    rh = np.zeros((C.NUM_HAND, 3), dtype=np.float32)
    pose_present = lh_present = rh_present = False

    if getattr(result, "pose_landmarks", None):
        pts = result.pose_landmarks
        if len(pts) == C.NUM_POSE:
            pose = np.array(
                [[p.x, p.y, p.z, getattr(p, "visibility", 0.0) or 0.0] for p in pts],
                dtype=np.float32,
            )
            pose_present = True

    if getattr(result, "left_hand_landmarks", None):
        pts = result.left_hand_landmarks
        if len(pts) == C.NUM_HAND:
            lh = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
            lh_present = True

    if getattr(result, "right_hand_landmarks", None):
        pts = result.right_hand_landmarks
        if len(pts) == C.NUM_HAND:
            rh = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
            rh_present = True

    return pose, lh, rh, pose_present, lh_present, rh_present


# --- MediaPipe wrapper -----------------------------------------------------
class HolisticExtractor:
    """Runs MediaPipe HolisticLandmarker and returns normalized feature vectors.

    Import of MediaPipe/OpenCV is deferred to construction time so the pure
    functions above can be imported (and tested) without them installed.
    """

    def __init__(self, model_path=None, running_mode: str = "VIDEO", min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        model_path = str(model_path or C.HOLISTIC_TASK)
        mode = {"IMAGE": RunningMode.IMAGE, "VIDEO": RunningMode.VIDEO}[running_mode]
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mode,
            min_pose_detection_confidence=min_conf,
            min_pose_landmarks_confidence=min_conf,
            min_hand_landmarks_confidence=min_conf,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)
        self.running_mode = running_mode

    def __call__(self, frame_bgr, timestamp_ms: int | None = None):
        """Returns (feature_vector, raw_result, pose_present)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("VIDEO mode requires a timestamp_ms")
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = self.landmarker.detect(mp_image)

        pose, lh, rh, pp, lhp, rhp = arrays_from_result(result)
        vector = normalize_landmarks(pose, lh, rh, pp, lhp, rhp)
        return vector, result, pp

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_overlay(frame, result):
    """Draw pose + hand skeletons on a BGR frame (for the live demo)."""
    import cv2

    h, w = frame.shape[:2]

    def to_px(lm):
        return int(lm.x * w), int(lm.y * h)

    pose = getattr(result, "pose_landmarks", None)
    if pose and len(pose) == C.NUM_POSE:
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, to_px(pose[a]), to_px(pose[b]), (255, 200, 0), 2)
        for lm in pose:
            cv2.circle(frame, to_px(lm), 3, (255, 200, 0), -1)

    for hand, color in (
        (getattr(result, "left_hand_landmarks", None), (80, 255, 80)),
        (getattr(result, "right_hand_landmarks", None), (80, 180, 255)),
    ):
        if hand and len(hand) == C.NUM_HAND:
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, to_px(hand[a]), to_px(hand[b]), color, 2)
            for lm in hand:
                cv2.circle(frame, to_px(lm), 4, color, -1)

    return frame
