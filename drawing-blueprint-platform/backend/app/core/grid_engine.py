"""
Proportional grid generation engine.

Generates a grid overlay based on landmark bounding box with configurable
difficulty: beginner (4×4), intermediate (8×8), advanced (16×16).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

DIFFICULTY_GRID_SIZE: Dict[str, Tuple[int, int]] = {
    "beginner": (4, 4),
    "intermediate": (8, 8),
    "advanced": (16, 16),
}


def _landmark_bounding_box(landmarks: List[Dict]) -> Dict[str, float]:
    """Compute bounding box around landmarks with 5% padding."""
    if not landmarks:
        return {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
    xs = [lm["x"] for lm in landmarks]
    ys = [lm["y"] for lm in landmarks]
    pad = 0.05
    return {
        "x_min": max(0.0, min(xs) - pad),
        "y_min": max(0.0, min(ys) - pad),
        "x_max": min(1.0, max(xs) + pad),
        "y_max": min(1.0, max(ys) + pad),
    }


def generate_grid(
    landmarks: List[Dict],
    difficulty: str = "intermediate",
    image_shape: Tuple[int, int] | None = None,
) -> List[Dict]:
    """
    Generate grid cells proportional to the landmark bounding box.

    Returns list of dicts: row, col, x_start, y_start, x_end, y_end, contains_landmark.
    """
    rows, cols = DIFFICULTY_GRID_SIZE.get(difficulty, (8, 8))
    bbox = _landmark_bounding_box(landmarks)

    cell_w = (bbox["x_max"] - bbox["x_min"]) / cols
    cell_h = (bbox["y_max"] - bbox["y_min"]) / rows

    cells: List[Dict] = []
    for r in range(rows):
        for c in range(cols):
            x_start = bbox["x_min"] + c * cell_w
            y_start = bbox["y_min"] + r * cell_h
            x_end = x_start + cell_w
            y_end = y_start + cell_h

            has_landmark = any(
                x_start <= lm["x"] <= x_end and y_start <= lm["y"] <= y_end
                for lm in landmarks
            )
            cells.append({
                "row": r, "col": c,
                "x_start": round(x_start, 6), "y_start": round(y_start, 6),
                "x_end": round(x_end, 6), "y_end": round(y_end, 6),
                "contains_landmark": has_landmark,
            })
    return cells


def grid_density(grid: List[Dict]) -> float:
    """Percentage of grid cells containing at least one landmark."""
    if not grid:
        return 0.0
    return round((sum(1 for c in grid if c["contains_landmark"]) / len(grid)) * 100, 2)
