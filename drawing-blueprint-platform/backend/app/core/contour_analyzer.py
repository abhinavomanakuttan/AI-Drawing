"""
Contour analyzer — hierarchical contour extraction and intelligent ordering.

Uses ``cv2.findContours`` with ``RETR_TREE`` to capture the full contour
hierarchy.  Contours are classified into three tiers:

- **Outer contours**  : top-level shapes (hierarchy depth 0)
- **Inner contours**  : holes and internal structures (depth 1–2)
- **Detail contours** : fine textures and small features (depth 3+)

Ordering is NOT simply by area.  Contours are ranked by a composite score
that considers:
1. Hierarchical depth (outer shapes first)
2. Area (larger first within same depth)
3. Structural importance (perimeter-to-area ratio as a proxy)

All coordinates are **normalised to [0, 1]** relative to the image dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class AnalyzedContour:
    """A single contour with computed metrics and normalised points."""
    points_normalised: np.ndarray   # Nx2 float array in [0,1] space
    area: float                     # normalised area (fraction of image)
    perimeter: float                # normalised perimeter
    depth: int                      # hierarchy depth (0 = outermost)
    parent_idx: int                 # index of parent contour (-1 if none)
    centroid: Tuple[float, float]   # normalised (x, y) centroid
    tier: str                       # "outer" | "inner" | "detail"
    importance_score: float         # composite ranking score


def _compute_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """Compute the centroid of a contour using moments."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        # Fallback — mean of all points
        return (float(contour[:, 0, 0].mean()), float(contour[:, 0, 1].mean()))
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def _compute_depth(hierarchy: np.ndarray, idx: int) -> int:
    """Walk up the parent chain to compute the hierarchy depth of a contour."""
    depth = 0
    current = idx
    while hierarchy[0, current, 3] != -1:
        current = hierarchy[0, current, 3]
        depth += 1
    return depth


def _classify_tier(depth: int) -> str:
    """Assign a tier label based on hierarchy depth."""
    if depth == 0:
        return "outer"
    elif depth <= 2:
        return "inner"
    return "detail"


def analyze_contours(
    edges: np.ndarray,
    min_area_fraction: float = 0.0005,
) -> List[AnalyzedContour]:
    """
    Extract, analyse, and order all contours from an edge map.

    Parameters
    ----------
    edges : np.ndarray
        Binary edge image (uint8, from Canny).
    min_area_fraction : float
        Minimum contour area as a fraction of total image area.
        Contours smaller than this are discarded as noise.

    Returns
    -------
    list[AnalyzedContour]
        Ordered from most important (draw first) to least important.
    """
    h, w = edges.shape[:2]
    image_area = h * w

    # --- Dilate edges slightly to close small gaps ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.dilate(edges, kernel, iterations=1)

    # --- Find contours with full hierarchy ---
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return []

    min_area_px = image_area * min_area_fraction
    analyzed: List[AnalyzedContour] = []

    for idx, contour in enumerate(contours):
        area_px = cv2.contourArea(contour)

        # Skip noise contours
        if area_px < min_area_px:
            continue

        perimeter_px = cv2.arcLength(contour, closed=True)
        depth = _compute_depth(hierarchy, idx)
        tier = _classify_tier(depth)
        parent_idx = int(hierarchy[0, idx, 3])

        # Normalise area and perimeter
        area_norm = area_px / image_area
        perimeter_norm = perimeter_px / max(w, h)

        # Centroid in pixel space → normalised
        cx_px, cy_px = _compute_centroid(contour)
        centroid = (cx_px / w, cy_px / h)

        # Normalise contour points to [0, 1]
        pts = contour.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h

        # --- Composite importance score ---
        # Higher is more important (drawn first)
        # Prioritises: shallow depth > large area > structural significance
        depth_weight = max(0, 5 - depth) / 5.0          # depth 0 → 1.0, depth 5+ → 0.0
        area_weight = min(area_norm * 10, 1.0)           # scale up, cap at 1
        # Structural importance: how "significant" the shape is
        # (longer perimeter relative to area = more complex shape)
        structure_weight = min(perimeter_norm * 2, 1.0)

        importance = (
            0.50 * depth_weight
            + 0.35 * area_weight
            + 0.15 * structure_weight
        )

        analyzed.append(AnalyzedContour(
            points_normalised=pts,
            area=area_norm,
            perimeter=perimeter_norm,
            depth=depth,
            parent_idx=parent_idx,
            centroid=centroid,
            tier=tier,
            importance_score=importance,
        ))

    # --- Sort: highest importance first ---
    analyzed.sort(key=lambda c: c.importance_score, reverse=True)

    return analyzed
