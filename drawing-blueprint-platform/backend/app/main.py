"""
FastAPI entry point for the Step-by-Step Drawing Guide.

- Registers upload + blueprint routers
- Mounts /static/steps/ for serving generated step images
- Serves frontend HTML at /
- Creates DB tables on startup
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.blueprint import router as blueprint_router
from app.api.upload import router as upload_router
from app.db import Base, engine
from app.db.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_HTML = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"
STATIC_STEPS_DIR = Path("./static/steps")
STATIC_STEPS_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating database tables")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready")
    yield
    await engine.dispose()


app = FastAPI(
    title="Step-by-Step Drawing Guide",
    description="Upload an image → get progressive drawing step images on a white grid canvas.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving generated step images
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(upload_router)
app.include_router(blueprint_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend() -> HTMLResponse:
    if FRONTEND_HTML.exists():
        return HTMLResponse(content=FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Drawing Guide</h1><p>Frontend not found.</p>")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> dict:
    return {"status": "ok", "version": "0.2.0"}
