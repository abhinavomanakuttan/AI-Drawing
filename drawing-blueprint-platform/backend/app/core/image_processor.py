"""
Image preprocessing for the Step-by-Step Drawing Guide.

Responsibilities:
- Resize to fixed working resolution (preserving aspect ratio)
- Convert to grayscale
- Apply noise reduction (Gaussian blur)
- Perform Canny edge detection
- Return processed arrays for downstream contour analysis
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# Working resolution — contour analysis happens at this size
WORKING_SIZE: int = 1024


def load_image_from_bytes(raw: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR NumPy array."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image from provided bytes")
    return image


def resize_preserve_aspect(image: np.ndarray, max_side: int = WORKING_SIZE) -> np.ndarray:
    """
    Resize so the longest side equals *max_side*, preserving aspect ratio.
    Never stretches or distorts the image.
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR → single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before edge detection."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 1.4)


def detect_edges(gray: np.ndarray, low: int = 30, high: int = 100) -> np.ndarray:
    """
    Canny edge detection tuned for drawing contour extraction.
    Lower thresholds than typical to capture softer edges.
    """
    blurred = reduce_noise(gray)
    return cv2.Canny(blurred, low, high)


def preprocess(raw_bytes: bytes) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline for the step generator.

    Returns
    -------
    dict with keys:
        original  : resized BGR image (aspect-preserved)
        grayscale : single-channel grayscale
        edges     : Canny edge map
        shape     : (height, width) of the resized image
    """
    image = load_image_from_bytes(raw_bytes)
    resized = resize_preserve_aspect(image, WORKING_SIZE)
    gray = to_grayscale(resized)
    edges = detect_edges(gray)

    return {
        "original": resized,
        "grayscale": gray,
        "edges": edges,
        "shape": resized.shape[:2],
    }
