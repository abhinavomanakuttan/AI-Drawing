# Step-by-Step Drawing Guide Generator

Upload a reference image → get progressive drawing step images on a white grid canvas.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000**

## How It Works

```
1. UPLOAD    → Drop a reference image (apple, face, object, etc.)
2. CONFIGURE → Pick paper size (A4/A3/Square) and grid density (4–16)
3. GENERATE  → System analyzes contours and renders step-by-step images
4. FOLLOW    → Navigate steps one-by-one with Previous/Next buttons
```

Each step image shows:
- White canvas with grid lines
- All previous strokes in **grey**
- New strokes for this step in **red** (highlighted)
- "Start Here" marker on the first step

## Architecture

```
frontend/index.html → FastAPI → Core Pipeline → PNG step images

Core Pipeline:
  image_processor  → resize, grayscale, edge detection
  contour_analyzer → hierarchical RETR_TREE extraction, composite scoring
  step_planner     → phase-based batching (outline → details → shading)
  canvas_renderer  → white canvas + grid + contour rendering (A4/A3)
  tone_mapper      → 3-band shading segmentation
  step_generator   → orchestrator (ties everything together)
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Frontend |
| `POST` | `/api/upload` | Upload image |
| `POST` | `/api/blueprint/generate` | Generate step images |
| `GET` | `/static/steps/{session}/{file}` | Serve step PNGs |
| `GET` | `/docs` | Swagger UI |

## Drawing Phases

| Phase | What Gets Drawn |
|-------|----------------|
| Preparation | Blank grid + "Start Here" marker |
| Primary Outline | Main outer silhouette |
| Secondary Shapes | Large internal structures |
| Internal Details | Medium features |
| Fine Details | Small shapes and lines |
| Texture | Very fine details |
| Shading | Tone-based shadow zones |
