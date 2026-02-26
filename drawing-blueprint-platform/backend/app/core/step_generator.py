"""
Step generator — the orchestrator module.

Flow:
1. Preprocess image (resize, grayscale, edge detection)
2. Extract and analyse contours (hierarchical)
3. Generate ordered contour list
4. Plan contour batches into drawing steps
5. For each step: render canvas → overlay contours → save PNG
6. Return list of file paths + metadata

This is the single entry point called by the API.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.core.canvas_renderer import render_step
from app.core.contour_analyzer import AnalyzedContour, analyze_contours
from app.core.image_processor import preprocess
from app.core.step_planner import DrawingStep, plan_steps
from app.core.tone_mapper import generate_tone_mask, tone_summary

# Output directory for generated step images
STEPS_OUTPUT_DIR: Path = Path(os.getenv("STEPS_OUTPUT_DIR", "./static/steps"))


@dataclass
class StepResult:
    """Metadata for a single generated step image."""
    step_number: int
    phase_name: str
    description: str
    image_filename: str
    image_url: str
    new_contour_count: int
    total_contours_so_far: int
    is_shading_step: bool


@dataclass
class GenerationResult:
    """Complete result of the step generation pipeline."""
    session_id: str
    total_steps: int
    steps: List[StepResult]
    image_width: int
    image_height: int
    tone_distribution: Dict[str, float]


def generate_drawing_steps(
    raw_bytes: bytes,
    paper_size: str = "A4",
    grid_cells: int = 8,
    include_shading: bool = True,
) -> GenerationResult:
    """
    Full step-by-step drawing guide generation pipeline.

    Parameters
    ----------
    raw_bytes : bytes
        Raw uploaded image file content.
    paper_size : str
        Canvas paper size: "A4", "A3", or "square".
    grid_cells : int
        Number of grid divisions on each axis.
    include_shading : bool
        Whether to include a final shading step.

    Returns
    -------
    GenerationResult
        Contains session_id, total steps, and per-step metadata with
        image URLs.
    """
    # --- 1. Preprocess ---
    processed = preprocess(raw_bytes)
    img_h, img_w = processed["shape"]
    image_aspect = img_w / img_h

    # --- 2. Extract and analyse contours ---
    contours = analyze_contours(processed["edges"])

    # --- 3. Plan steps ---
    steps = plan_steps(contours, include_shading=include_shading)

    # --- 4. Generate tone mask for shading ---
    tone_mask = generate_tone_mask(processed["grayscale"])
    tones = tone_summary(tone_mask)

    # --- 5. Render each step ---
    session_id = uuid.uuid4().hex[:12]
    session_dir = STEPS_OUTPUT_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Track cumulative contours across steps
    all_previous_contours: List[AnalyzedContour] = []
    step_results: List[StepResult] = []

    # Find primary centroid for the "Start Here" marker
    primary_centroid = contours[0].centroid if contours else (0.5, 0.5)

    for step in steps:
        filename = f"step_{step.step_number:02d}.png"
        output_path = str(session_dir / filename)

        # Determine shading mask for the shading step
        shading = tone_mask if step.is_shading_step else None

        # Render this step
        render_step(
            step_number=step.step_number,
            phase_name=step.phase_name,
            description=step.description,
            previous_contours=all_previous_contours,
            new_contours=step.contours,
            image_aspect=image_aspect,
            start_centroid=primary_centroid,
            shading_mask=shading,
            paper_size=paper_size,
            grid_cells=grid_cells,
            output_path=output_path,
        )

        # Build URL for this step image
        image_url = f"/static/steps/{session_id}/{filename}"

        total_so_far = len(all_previous_contours) + len(step.contours)

        step_results.append(StepResult(
            step_number=step.step_number,
            phase_name=step.phase_name,
            description=step.description,
            image_filename=filename,
            image_url=image_url,
            new_contour_count=len(step.contours),
            total_contours_so_far=total_so_far,
            is_shading_step=step.is_shading_step,
        ))

        # Accumulate contours for subsequent steps
        all_previous_contours.extend(step.contours)

    return GenerationResult(
        session_id=session_id,
        total_steps=len(step_results),
        steps=step_results,
        image_width=img_w,
        image_height=img_h,
        tone_distribution=tones,
    )
