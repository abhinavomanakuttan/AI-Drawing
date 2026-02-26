"""
SQLAlchemy ORM models for the prototype.

Only Drawing and Blueprint — User/Subscription/Payment deferred to later.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Drawing(Base):
    """Record of an uploaded image."""

    __tablename__ = "drawings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    blueprints: Mapped[List["Blueprint"]] = relationship(
        back_populates="drawing", cascade="all, delete-orphan"
    )


class Blueprint(Base):
    """Generated blueprint result derived from a drawing."""

    __tablename__ = "blueprints"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    drawing_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("drawings.id", ondelete="CASCADE"), nullable=False
    )
    complexity_score: Mapped[float] = mapped_column(Float, default=0.0)
    proportion_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    shading_regions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    landmark_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    grid_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    layers: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    difficulty_level: Mapped[str] = mapped_column(String(50), default="intermediate")
    feedback: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Relationships
    drawing: Mapped["Drawing"] = relationship(back_populates="blueprints")

    __table_args__ = (Index("ix_blueprints_drawing_id", "drawing_id"),)
