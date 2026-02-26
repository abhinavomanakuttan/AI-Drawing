"""Quick end-to-end test of the step generation pipeline."""
import httpx, json, glob, os

BASE = "http://localhost:8000"

# Find a test image from uploads
imgs = glob.glob("uploads/*")
if not imgs:
    print("No test images found in uploads/")
    exit(1)

test_img = imgs[0]
print(f"Using test image: {test_img}")

# 1. Upload
with open(test_img, "rb") as f:
    ext = os.path.splitext(test_img)[1]
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    r = httpx.post(f"{BASE}/api/upload", files={"file": (f"test{ext}", f, mime)})

print(f"UPLOAD: {r.status_code}")
if r.status_code != 201:
    print(f"  Error: {r.text}")
    exit(1)

data = r.json()
drawing_id = data["id"]
print(f"  Drawing ID: {drawing_id}")

# 2. Generate steps
r2 = httpx.post(f"{BASE}/api/blueprint/generate", json={
    "drawing_id": drawing_id,
    "paper_size": "A4",
    "grid_cells": 8,
}, timeout=60)

print(f"GENERATE: {r2.status_code}")
if r2.status_code != 201:
    print(f"  Error: {r2.text}")
    exit(1)

g = r2.json()
print(f"  Session: {g.get('session_id')}")
print(f"  Total steps: {g.get('total_steps')}")
print(f"  Tones: {g.get('tone_distribution')}")

for s in g.get("steps", []):
    print(f"  Step {s['step_number']}: {s['phase_name']} - {s['description']} ({s['new_contour_count']} new, {s['total_contours_so_far']} total)")
    print(f"    Image URL: {s['image_url']}")

# 3. Check step images exist
import os
for s in g.get("steps", []):
    path = "." + s["image_url"].replace("/static/", "/static/")
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING'}: {path}")

print("\nDONE - All checks passed!" if g.get("total_steps", 0) > 0 else "\nFAILED - No steps generated")
