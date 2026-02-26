"""
Drawing complexity scoring model.

Weighted-sum scoring (0–100) based on three features:
  1. landmark_count  (weight 0.35)
  2. grid_density    (weight 0.25)
  3. edge_variance   (weight 0.40)

Designed to be easily swappable with a trained ML model later.
"""

from __future__ import annotations

import numpy as np

W_LANDMARKS: float = 0.35
W_GRID_DENSITY: float = 0.25
W_EDGE_VARIANCE: float = 0.40

MAX_LANDMARK_COUNT: int = 33
MAX_GRID_DENSITY: float = 100.0
MAX_EDGE_VARIANCE: float = 5000.0


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def compute_edge_variance(edge_map: np.ndarray) -> float:
    """Variance of pixel intensities in the edge map — higher = more detail."""
    return float(np.var(edge_map.astype(np.float32)))


def score_complexity(landmark_count: int, grid_density: float, edge_variance: float) -> float:
    """Compute normalised complexity score [0, 100]."""
    n_lm = min(landmark_count / MAX_LANDMARK_COUNT, 1.0)
    n_gd = min(grid_density / MAX_GRID_DENSITY, 1.0)
    n_ev = min(edge_variance / MAX_EDGE_VARIANCE, 1.0)
    raw = (W_LANDMARKS * n_lm + W_GRID_DENSITY * n_gd + W_EDGE_VARIANCE * n_ev) * 100.0
    return round(_clamp(raw), 2)


class ComplexityModel:
    """
    Class wrapper for dependency injection + future ML model swap.

    >>> model = ComplexityModel()
    >>> model.predict(landmark_count=23, grid_density=45.0, edge_variance=1200.0)
    """

    def __init__(self, w_lm: float = W_LANDMARKS, w_gd: float = W_GRID_DENSITY, w_ev: float = W_EDGE_VARIANCE):
        self.w_lm = w_lm
        self.w_gd = w_gd
        self.w_ev = w_ev

    def predict(self, landmark_count: int, grid_density: float, edge_variance: float) -> float:
        n_lm = min(landmark_count / MAX_LANDMARK_COUNT, 1.0)
        n_gd = min(grid_density / MAX_GRID_DENSITY, 1.0)
        n_ev = min(edge_variance / MAX_EDGE_VARIANCE, 1.0)
        raw = (self.w_lm * n_lm + self.w_gd * n_gd + self.w_ev * n_ev) * 100.0
        return round(_clamp(raw), 2)
