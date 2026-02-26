"""
Step planner — groups contours into logical drawing phases.

Phases
------
1. Primary outer shapes   (the main silhouette)
2. Large internal structures  (holes, major internal divisions)
3. Medium details         (secondary shapes, features)
4. Fine lines             (small textures, thin strokes)
5. Texture                (very fine details)
6. Shading                (tone-based shading zones — handled separately)

Adaptive batching: early steps contain fewer contours so the user
isn't overwhelmed; later steps batch more contours together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.contour_analyzer import AnalyzedContour


@dataclass
class DrawingStep:
    """A single planned drawing step with its batch of contours."""
    step_number: int
    phase_name: str
    description: str
    contours: List[AnalyzedContour]
    is_shading_step: bool = False


# ---------------------------------------------------------------------------
# Phase definitions — contour tier + importance thresholds
# ---------------------------------------------------------------------------
_PHASES = [
    {
        "name": "Primary Outline",
        "description": "Draw the main outer shape — the overall silhouette",
        "tiers": ["outer"],
        "min_importance": 0.6,
        "max_contours": 3,
    },
    {
        "name": "Secondary Shapes",
        "description": "Add large internal structures and secondary outlines",
        "tiers": ["outer", "inner"],
        "min_importance": 0.4,
        "max_contours": 5,
    },
    {
        "name": "Internal Details",
        "description": "Draw inner features and medium-sized details",
        "tiers": ["inner"],
        "min_importance": 0.2,
        "max_contours": 8,
    },
    {
        "name": "Fine Details",
        "description": "Add fine lines and smaller shapes",
        "tiers": ["inner", "detail"],
        "min_importance": 0.1,
        "max_contours": 12,
    },
    {
        "name": "Texture & Finishing",
        "description": "Add textures, tiny details, and finishing touches",
        "tiers": ["detail"],
        "min_importance": 0.0,
        "max_contours": 999,  # all remaining
    },
]


def plan_steps(
    contours: List[AnalyzedContour],
    include_shading: bool = True,
) -> List[DrawingStep]:
    """
    Group ordered contours into drawing steps.

    Parameters
    ----------
    contours : list[AnalyzedContour]
        Ordered contour list from ``contour_analyzer.analyze_contours``.
    include_shading : bool
        Whether to append a shading step at the end.

    Returns
    -------
    list[DrawingStep]
        Planned steps, each containing a batch of contours.
    """
    # Track which contours have been assigned to a step
    assigned: set = set()
    steps: List[DrawingStep] = []
    step_num = 1

    for phase in _PHASES:
        batch: List[AnalyzedContour] = []

        for i, c in enumerate(contours):
            if i in assigned:
                continue

            # Check tier match
            if c.tier not in phase["tiers"]:
                continue

            # Check importance threshold
            if c.importance_score < phase["min_importance"]:
                continue

            batch.append(c)
            assigned.add(i)

            if len(batch) >= phase["max_contours"]:
                break

        if batch:
            steps.append(DrawingStep(
                step_number=step_num,
                phase_name=phase["name"],
                description=phase["description"],
                contours=batch,
            ))
            step_num += 1

    # --- Catch any remaining unassigned contours ---
    remaining = [contours[i] for i in range(len(contours)) if i not in assigned]
    if remaining:
        steps.append(DrawingStep(
            step_number=step_num,
            phase_name="Additional Details",
            description="Complete any remaining small details",
            contours=remaining,
        ))
        step_num += 1

    # --- Shading step (empty contours — will be handled by tone_mapper) ---
    if include_shading:
        steps.append(DrawingStep(
            step_number=step_num,
            phase_name="Shading",
            description="Add shading — fill in shadow and mid-tone regions",
            contours=[],
            is_shading_step=True,
        ))

    # --- Ensure at least a blank grid step at the start ---
    # Insert Step 0: blank grid with starting point marker
    primary_centroid = (0.5, 0.5)
    if contours:
        primary_centroid = contours[0].centroid

    grid_step = DrawingStep(
        step_number=0,
        phase_name="Preparation",
        description=f"Your blank grid — start drawing at the marked point",
        contours=[],
    )

    # Re-number all existing steps
    for s in steps:
        s.step_number += 1

    return [grid_step] + steps


def total_contour_count(steps: List[DrawingStep]) -> int:
    """Total number of contours across all steps (excluding shading)."""
    return sum(len(s.contours) for s in steps if not s.is_shading_step)
