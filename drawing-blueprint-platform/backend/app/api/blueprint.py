"""
Blueprint (Step Generation) API.

POST /api/blueprint/generate
  → Runs the full step-by-step drawing guide pipeline
  → Returns list of step image URLs + metadata
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.step_generator import generate_drawing_steps
from app.db import get_db
from app.db.models import Blueprint, Drawing
from app.db.schemas import GenerateRequest, GenerateResponse, StepInfo

router = APIRouter(prefix="/api/blueprint", tags=["Blueprint"])


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_201_CREATED)
async def generate_steps_endpoint(
    body: GenerateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Generate step-by-step drawing guide images.

    Pipeline:
    1. Load drawing record → read image bytes
    2. Preprocess (resize, grayscale, edges)
    3. Extract hierarchical contours
    4. Plan drawing phases / steps
    5. Render each step on a white grid canvas
    6. Return step image URLs + metadata
    """
    # 1 — Load drawing
    result = await db.execute(select(Drawing).where(Drawing.id == body.drawing_id))
    drawing = result.scalar_one_or_none()
    if drawing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Drawing not found")

    image_path = Path(drawing.file_path)
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file missing — please re-upload")
    raw_bytes = image_path.read_bytes()

    # 2–5 — Run the full step generation pipeline
    gen_result = generate_drawing_steps(
        raw_bytes=raw_bytes,
        paper_size=body.paper_size,
        grid_cells=body.grid_cells,
        include_shading=body.include_shading,
    )

    # 6 — Persist blueprint record
    bp = Blueprint(
        drawing_id=drawing.id,
        complexity_score=0.0,
        proportion_accuracy=0.0,
        layers={"session_id": gen_result.session_id, "total_steps": gen_result.total_steps},
        difficulty_level=body.paper_size,
    )
    db.add(bp)
    await db.flush()
    await db.refresh(bp)

    # Build response
    steps = [
        StepInfo(
            step_number=s.step_number,
            phase_name=s.phase_name,
            description=s.description,
            image_url=s.image_url,
            new_contour_count=s.new_contour_count,
            total_contours_so_far=s.total_contours_so_far,
            is_shading_step=s.is_shading_step,
        )
        for s in gen_result.steps
    ]

    return {
        "session_id": gen_result.session_id,
        "drawing_id": drawing.id,
        "total_steps": gen_result.total_steps,
        "image_width": gen_result.image_width,
        "image_height": gen_result.image_height,
        "paper_size": body.paper_size,
        "grid_cells": body.grid_cells,
        "tone_distribution": gen_result.tone_distribution,
        "steps": steps,
    }
