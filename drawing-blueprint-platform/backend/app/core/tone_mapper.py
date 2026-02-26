"""
Tone mapper — converts a grayscale image into discrete tone regions
for the shading step of the drawing guide.

Segments the image into 3–5 tone bands using threshold segmentation:
- **Deep shadow** : darkest areas (value 0)
- **Mid tone**    : middle range (value 1)
- **Highlight**   : lightest areas (value 2)

Output is a single-channel mask where each pixel's value indicates its
tone band.  This mask is used by ``canvas_renderer`` to overlay
hatching patterns in the final shading step.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# Default tone band thresholds (on 0–255 grayscale)
DEFAULT_BANDS: List[Tuple[str, int, int, int]] = [
    # (label, low_threshold, high_threshold, mask_value)
    ("deep_shadow", 0, 80, 0),
    ("mid_tone", 81, 180, 1),
    ("highlight", 181, 255, 2),
]


def generate_tone_mask(
    grayscale: np.ndarray,
    bands: List[Tuple[str, int, int, int]] | None = None,
    blur_ksize: int = 7,
) -> np.ndarray:
    """
    Segment a grayscale image into discrete tone regions.

    Parameters
    ----------
    grayscale : np.ndarray
        Single-channel uint8 grayscale image.
    bands : list of (label, low, high, mask_value) tuples
        Custom tone band definitions.  Defaults to 3-band shadow/mid/highlight.
    blur_ksize : int
        Gaussian blur kernel size applied before segmentation to smooth
        noise and create cleaner tone regions.

    Returns
    -------
    np.ndarray
        Single-channel mask (uint8) where each pixel's value is the
        assigned band's mask_value.  Values 0 = deep shadow, 1 = mid tone,
        2 = highlight.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    # Ensure uint8
    if grayscale.dtype != np.uint8:
        if grayscale.max() <= 1.0:
            grayscale = (grayscale * 255).astype(np.uint8)
        else:
            grayscale = grayscale.astype(np.uint8)

    # Smooth to create cleaner regions
    smoothed = cv2.GaussianBlur(grayscale, (blur_ksize, blur_ksize), 0)

    # Build mask
    h, w = smoothed.shape[:2]
    mask = np.full((h, w), 2, dtype=np.uint8)  # default to highlight

    for label, low, high, val in bands:
        region = cv2.inRange(smoothed, low, high)
        mask[region > 0] = val

    return mask


def tone_summary(mask: np.ndarray) -> dict:
    """
    Quick summary of tone distribution.

    Returns dict like: {"deep_shadow": 25.3, "mid_tone": 45.1, "highlight": 29.6}
    (values are percentages of total pixels)
    """
    total = mask.size
    return {
        "deep_shadow": round(float(np.sum(mask == 0)) / total * 100, 1),
        "mid_tone": round(float(np.sum(mask == 1)) / total * 100, 1),
        "highlight": round(float(np.sum(mask == 2)) / total * 100, 1),
    }
