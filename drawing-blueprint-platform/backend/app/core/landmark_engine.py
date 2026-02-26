"""
Anatomical landmark detection engine.

Uses MediaPipe Pose when available; otherwise falls back to a proportional
placeholder based on the classical 8-head figure model so the rest of the
pipeline still functions.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Attempt MediaPipe import — graceful fallback
try:
    import mediapipe as mp  # type: ignore[import-untyped]
    _mp_pose = mp.solutions.pose
    _MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe Pose available — using real landmark detection")
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed — using placeholder landmarks")

# Canonical body landmark names (subset of MediaPipe's 33)
LANDMARK_NAMES: List[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _detect_with_mediapipe(image: np.ndarray) -> List[Dict]:
    """Real MediaPipe Pose detection → list of landmark dicts."""
    import cv2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with _mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(rgb)
    if not results.pose_landmarks:
        return []
    landmarks: List[Dict] = []
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"landmark_{idx}"
        landmarks.append({"name": name, "x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility)})
    return landmarks


def _detect_placeholder(image: np.ndarray) -> List[Dict]:
    """Synthetic landmarks from the 8-head proportional model."""
    proportions: Dict[str, tuple] = {
        "nose": (0.50, 0.08), "left_eye_inner": (0.47, 0.06), "left_eye": (0.45, 0.06),
        "left_eye_outer": (0.43, 0.06), "right_eye_inner": (0.53, 0.06), "right_eye": (0.55, 0.06),
        "right_eye_outer": (0.57, 0.06), "left_ear": (0.40, 0.07), "right_ear": (0.60, 0.07),
        "mouth_left": (0.47, 0.10), "mouth_right": (0.53, 0.10),
        "left_shoulder": (0.35, 0.20), "right_shoulder": (0.65, 0.20),
        "left_elbow": (0.28, 0.37), "right_elbow": (0.72, 0.37),
        "left_wrist": (0.25, 0.50), "right_wrist": (0.75, 0.50),
        "left_hip": (0.40, 0.52), "right_hip": (0.60, 0.52),
        "left_knee": (0.38, 0.72), "right_knee": (0.62, 0.72),
        "left_ankle": (0.37, 0.92), "right_ankle": (0.63, 0.92),
    }
    return [
        {"name": name, "x": proportions[name][0], "y": proportions[name][1], "z": 0.0, "visibility": 1.0}
        for name in LANDMARK_NAMES
    ]


def detect_landmarks(image: np.ndarray) -> List[Dict]:
    """
    Detect anatomical landmarks in *image*.
    Returns list of dicts with keys: name, x, y, z, visibility.
    """
    if _MEDIAPIPE_AVAILABLE:
        return _detect_with_mediapipe(image)
    return _detect_placeholder(image)
