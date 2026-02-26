"""
Canvas renderer — draws step images on a white canvas with grid lines.

Responsibilities:
- Create a white canvas at a given paper size (A4, A3, or custom)
- Draw grid lines dynamically (resolution-independent)
- Convert normalised [0,1] coordinates to paper pixel coordinates
- Render previous contours in a standard stroke colour
- Render new (current-step) contours in a highlight colour
- Add "Start Here" marker using the primary contour centroid
- Save each step as a PNG file

Rendering is fully separated from step planning logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.contour_analyzer import AnalyzedContour

# ---------------------------------------------------------------------------
# Paper sizes in pixels at 150 DPI (suitable for screen display)
# ---------------------------------------------------------------------------
PAPER_SIZES: Dict[str, Tuple[int, int]] = {
    "A4": (1240, 1754),       # 210mm × 297mm at 150 DPI
    "A3": (1754, 2480),       # 297mm × 420mm at 150 DPI
    "square": (1500, 1500),   # square format
}

# Colours (BGR for OpenCV)
WHITE = (255, 255, 255)
GRID_COLOR = (220, 220, 220)         # light grey grid lines
PREV_STROKE_COLOR = (80, 80, 80)     # dark grey for previous strokes
NEW_STROKE_COLOR = (30, 30, 200)     # red-ish for new strokes (highlight)
MARKER_COLOR = (0, 140, 255)         # orange for "start here" marker
SHADING_COLORS = {
    "deep_shadow": (180, 180, 180),
    "mid_tone": (220, 220, 220),
    "highlight": (245, 245, 245),
}

# Stroke widths
GRID_THICKNESS = 1
PREV_STROKE_THICKNESS = 2
NEW_STROKE_THICKNESS = 3
MARKER_RADIUS = 12


def _create_white_canvas(size: Tuple[int, int]) -> np.ndarray:
    """Create a blank white canvas of the given (width, height)."""
    w, h = size
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_grid(
    canvas: np.ndarray,
    grid_cells: int = 8,
    margin_fraction: float = 0.05,
) -> Tuple[int, int, int, int]:
    """
    Draw evenly-spaced grid lines on the canvas.

    Returns the drawing area bounds: (x_start, y_start, x_end, y_end)
    accounting for margins.
    """
    h, w = canvas.shape[:2]

    # Margin so grid doesn't touch the edges
    mx = int(w * margin_fraction)
    my = int(h * margin_fraction)
    draw_w = w - 2 * mx
    draw_h = h - 2 * my

    # Vertical lines
    for i in range(grid_cells + 1):
        x = mx + int(i * draw_w / grid_cells)
        cv2.line(canvas, (x, my), (x, my + draw_h), GRID_COLOR, GRID_THICKNESS)

    # Horizontal lines
    for i in range(grid_cells + 1):
        y = my + int(i * draw_h / grid_cells)
        cv2.line(canvas, (mx, y), (mx + draw_w, y), GRID_COLOR, GRID_THICKNESS)

    # Outer border (slightly thicker)
    cv2.rectangle(canvas, (mx, my), (mx + draw_w, my + draw_h), (200, 200, 200), 2)

    return mx, my, mx + draw_w, my + draw_h


def _normalised_to_canvas(
    points: np.ndarray,
    draw_area: Tuple[int, int, int, int],
    image_aspect: float,
) -> np.ndarray:
    """
    Convert normalised [0,1] contour points to canvas pixel coordinates.

    Centers the drawing within the draw area while preserving aspect ratio.
    """
    x_start, y_start, x_end, y_end = draw_area
    area_w = x_end - x_start
    area_h = y_end - y_start
    area_aspect = area_w / area_h

    # Fit image aspect ratio into the drawing area
    if image_aspect > area_aspect:
        # Image is wider — fit to width
        render_w = area_w
        render_h = int(area_w / image_aspect)
    else:
        # Image is taller — fit to height
        render_h = area_h
        render_w = int(area_h * image_aspect)

    # Center within draw area
    offset_x = x_start + (area_w - render_w) // 2
    offset_y = y_start + (area_h - render_h) // 2

    # Scale and translate
    scaled = points.copy()
    scaled[:, 0] = (points[:, 0] * render_w + offset_x).astype(np.int32)
    scaled[:, 1] = (points[:, 1] * render_h + offset_y).astype(np.int32)

    return scaled.astype(np.int32)


def _draw_contours(
    canvas: np.ndarray,
    contours: List[AnalyzedContour],
    draw_area: Tuple[int, int, int, int],
    image_aspect: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a set of contours onto the canvas."""
    for c in contours:
        pts = _normalised_to_canvas(c.points_normalised, draw_area, image_aspect)
        pts_cv = pts.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_cv], isClosed=True, color=color, thickness=thickness,
                      lineType=cv2.LINE_AA)


def _draw_start_marker(
    canvas: np.ndarray,
    centroid: Tuple[float, float],
    draw_area: Tuple[int, int, int, int],
    image_aspect: float,
) -> None:
    """Draw a 'Start Here' circle + text at the given normalised centroid."""
    pts = np.array([[centroid[0], centroid[1]]])
    canvas_pts = _normalised_to_canvas(pts, draw_area, image_aspect)
    cx, cy = int(canvas_pts[0, 0]), int(canvas_pts[0, 1])

    # Circle marker
    cv2.circle(canvas, (cx, cy), MARKER_RADIUS, MARKER_COLOR, 3, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, MARKER_COLOR, -1, lineType=cv2.LINE_AA)

    # "Start Here" text
    cv2.putText(
        canvas, "Start Here", (cx + MARKER_RADIUS + 6, cy + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, MARKER_COLOR, 2, cv2.LINE_AA,
    )


def _draw_step_label(canvas: np.ndarray, step_number: int, phase_name: str, description: str) -> None:
    """Draw step number and description at the top of the canvas."""
    h, w = canvas.shape[:2]
    label = f"Step {step_number}: {phase_name}"
    cv2.putText(canvas, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(canvas, description, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)


def render_step(
    step_number: int,
    phase_name: str,
    description: str,
    previous_contours: List[AnalyzedContour],
    new_contours: List[AnalyzedContour],
    image_aspect: float,
    start_centroid: Optional[Tuple[float, float]] = None,
    shading_mask: Optional[np.ndarray] = None,
    paper_size: str = "A4",
    grid_cells: int = 8,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Render a single drawing step image.

    Parameters
    ----------
    step_number, phase_name, description : step metadata
    previous_contours : contours from all prior steps (drawn in grey)
    new_contours : contours introduced in THIS step (drawn in highlight)
    image_aspect : width/height ratio of the original image
    start_centroid : optional (x, y) normalised centroid for "Start Here" marker
    shading_mask : optional grayscale mask for shading overlay
    paper_size : "A4", "A3", or "square"
    grid_cells : number of grid divisions
    output_path : if set, save PNG to this path

    Returns
    -------
    np.ndarray : the rendered canvas image (BGR)
    """
    size = PAPER_SIZES.get(paper_size, PAPER_SIZES["A4"])
    canvas = _create_white_canvas(size)

    # Draw grid
    draw_area = _draw_grid(canvas, grid_cells=grid_cells)

    # Draw previous strokes (all steps so far, in grey)
    _draw_contours(canvas, previous_contours, draw_area, image_aspect,
                   PREV_STROKE_COLOR, PREV_STROKE_THICKNESS)

    # Draw new strokes for this step (in highlight colour)
    _draw_contours(canvas, new_contours, draw_area, image_aspect,
                   NEW_STROKE_COLOR, NEW_STROKE_THICKNESS)

    # Shading overlay (for the final shading step)
    if shading_mask is not None:
        _overlay_shading(canvas, shading_mask, draw_area, image_aspect)

    # Start marker on step 0 or step 1
    if start_centroid and step_number <= 1:
        _draw_start_marker(canvas, start_centroid, draw_area, image_aspect)

    # Step label
    _draw_step_label(canvas, step_number, phase_name, description)

    # Save to disk if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, canvas)

    return canvas


def _overlay_shading(
    canvas: np.ndarray,
    shading_mask: np.ndarray,
    draw_area: Tuple[int, int, int, int],
    image_aspect: float,
) -> None:
    """
    Overlay shading zones onto the canvas as semi-transparent hatching.

    The shading_mask should have pixel values:
      0 = deep shadow, 1 = mid tone, 2 = highlight, 3+ = skip
    """
    x_start, y_start, x_end, y_end = draw_area
    area_w = x_end - x_start
    area_h = y_end - y_start
    area_aspect = area_w / area_h

    if image_aspect > area_aspect:
        render_w = area_w
        render_h = int(area_w / image_aspect)
    else:
        render_h = area_h
        render_w = int(area_h * image_aspect)

    offset_x = x_start + (area_w - render_w) // 2
    offset_y = y_start + (area_h - render_h) // 2

    # Resize shading mask to render dimensions
    mask_resized = cv2.resize(shading_mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

    # Draw crosshatch for shadow regions
    overlay = canvas.copy()
    for y in range(render_h):
        for x in range(0, render_w, 3):
            val = mask_resized[y, x]
            if val == 0:  # deep shadow — dense hatching
                if (x + y) % 6 < 3:
                    canvas[offset_y + y, offset_x + x] = (180, 180, 180)
            elif val == 1:  # mid tone — lighter hatching
                if (x + y) % 10 < 2:
                    canvas[offset_y + y, offset_x + x] = (210, 210, 210)
            # highlight (val 2+) — leave white
