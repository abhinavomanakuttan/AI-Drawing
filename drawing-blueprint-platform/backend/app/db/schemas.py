"""
Pydantic v2 schemas for the Step-by-Step Drawing Guide.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Drawing (upload)
# ═══════════════════════════════════════════════════════════════════════════
class DrawingResponse(BaseModel):
    id: str
    filename: str
    file_size_bytes: int
    mime_type: str
    is_processed: bool
    created_at: datetime
    model_config = {"from_attributes": True}


# ═══════════════════════════════════════════════════════════════════════════
# Step-based blueprint generation
# ═══════════════════════════════════════════════════════════════════════════
class GenerateRequest(BaseModel):
    """Request body for step generation."""
    drawing_id: str
    paper_size: str = Field("A4", pattern="^(A4|A3|square)$")
    grid_cells: int = Field(8, ge=4, le=20)
    include_shading: bool = True


class StepInfo(BaseModel):
    """Metadata for a single drawing step."""
    step_number: int
    phase_name: str
    description: str
    image_url: str
    new_contour_count: int
    total_contours_so_far: int
    is_shading_step: bool


class GenerateResponse(BaseModel):
    """Response containing all step image URLs and metadata."""
    session_id: str
    drawing_id: str
    total_steps: int
    image_width: int
    image_height: int
    paper_size: str
    grid_cells: int
    tone_distribution: Dict[str, float]
    steps: List[StepInfo]


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
