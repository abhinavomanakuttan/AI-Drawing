"""
Shading mapper — maps grayscale intensities to classified shadow zones.

Bands: deep_shadow (0–63), mid_tone (64–191), highlight (192–255).
Uses OpenCV connected-component analysis to find region bounding boxes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

DEFAULT_BANDS: List[Tuple[str, int, int]] = [
    ("deep_shadow", 0, 63),
    ("mid_tone", 64, 191),
    ("highlight", 192, 255),
]


def _regions_for_band(gray: np.ndarray, label: str, low: int, high: int, offset: int) -> List[Dict]:
    """Find connected-component regions within an intensity band."""
    mask = cv2.inRange(gray, low, high)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = gray.shape
    regions: List[Dict] = []

    for i in range(1, num_labels):  # skip background label 0
        px_count = int(stats[i, cv2.CC_STAT_AREA])
        if px_count < (h * w * 0.001):  # skip noise < 0.1%
            continue
        x_min = int(stats[i, cv2.CC_STAT_LEFT])
        y_min = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        regions.append({
            "region_id": offset + len(regions),
            "label": label,
            "intensity_range": [float(low), float(high)],
            "pixel_count": px_count,
            "bounding_box": {
                "x_min": round(x_min / w, 6), "y_min": round(y_min / h, 6),
                "x_max": round((x_min + cw) / w, 6), "y_max": round((y_min + ch) / h, 6),
            },
        })
    return regions


def map_shading(grayscale: np.ndarray, bands: List[Tuple[str, int, int]] | None = None) -> List[Dict]:
    """
    Map grayscale image → shadow zone regions.

    Returns list of dicts compatible with ShadingRegion schema.
    """
    if bands is None:
        bands = DEFAULT_BANDS

    if grayscale.dtype != np.uint8:
        grayscale = (grayscale * 255).astype(np.uint8) if grayscale.max() <= 1.0 else grayscale.astype(np.uint8)

    all_regions: List[Dict] = []
    offset = 0
    for label, low, high in bands:
        regions = _regions_for_band(grayscale, label, low, high, offset)
        offset += len(regions)
        all_regions.extend(regions)
    return all_regions
