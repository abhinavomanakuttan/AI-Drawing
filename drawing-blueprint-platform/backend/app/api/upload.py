"""
Image upload API — no authentication required for the prototype.

POST /api/upload — accepts an image, saves locally, preprocesses, persists record.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.image_processor import preprocess
from app.db import get_db
from app.db.models import Drawing
from app.db.schemas import DrawingResponse

UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_MB: int = 20
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

router = APIRouter(prefix="/api/upload", tags=["Upload"])


def _mime_to_ext(mime: str) -> str:
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/bmp": ".bmp"}.get(mime, ".bin")


@router.post("", response_model=DrawingResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(file: UploadFile, db: AsyncSession = Depends(get_db)) -> Drawing:
    """
    Upload an image and run initial preprocessing.

    Steps:
    1. Validate MIME type and file size
    2. Save file to local disk
    3. Run OpenCV preprocessing (resize, grayscale, edge detection)
    4. Persist a Drawing record in the database
    """
    # Validate MIME
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type: {file.content_type}. Allowed: {ALLOWED_MIME_TYPES}",
        )

    # Read bytes
    raw_bytes: bytes = await file.read()
    file_size = len(raw_bytes)

    # Validate size
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    # Save to disk
    file_id = str(uuid.uuid4())
    ext = _mime_to_ext(file.content_type or "image/jpeg")
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    dest.write_bytes(raw_bytes)

    # Preprocess — validates the image is decodable
    try:
        _ = preprocess(raw_bytes)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Persist DB record
    drawing = Drawing(
        filename=file.filename or f"{file_id}{ext}",
        file_path=str(dest),
        file_size_bytes=file_size,
        mime_type=file.content_type or "image/jpeg",
        is_processed=True,
    )
    db.add(drawing)
    await db.flush()
    await db.refresh(drawing)
    return drawing
