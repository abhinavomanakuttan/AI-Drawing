"""
Blueprint generation engine.

Converts landmarks + grid + shading into structured blueprint layers.
Scores proportion accuracy against the classical 8-head model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

# Ideal proportional distances (normalised) for the 8-head model
_IDEAL_PROPORTIONS: Dict[str, float] = {
    "shoulder_width": 0.30,
    "torso_length": 0.32,
    "upper_arm": 0.17,
    "forearm": 0.17,
    "thigh": 0.20,
    "shin": 0.20,
    "head_to_shoulder": 0.12,
}


def _dist(a: Dict, b: Dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def _lm_by_name(landmarks: List[Dict], name: str) -> Dict | None:
    for lm in landmarks:
        if lm["name"] == name:
            return lm
    return None


def _score_proportions(landmarks: List[Dict]) -> float:
    """Compare measured vs ideal proportions → accuracy % (0–100)."""
    pairs = [
        ("left_shoulder", "right_shoulder", "shoulder_width"),
        ("left_shoulder", "left_hip", "torso_length"),
        ("left_shoulder", "left_elbow", "upper_arm"),
        ("left_elbow", "left_wrist", "forearm"),
        ("left_hip", "left_knee", "thigh"),
        ("left_knee", "left_ankle", "shin"),
        ("nose", "left_shoulder", "head_to_shoulder"),
    ]
    errors: List[float] = []
    for a_name, b_name, key in pairs:
        a, b = _lm_by_name(landmarks, a_name), _lm_by_name(landmarks, b_name)
        if a is None or b is None:
            continue
        measured = _dist(a, b)
        ideal = _IDEAL_PROPORTIONS[key]
        errors.append(min(abs(measured - ideal) / max(ideal, 1e-6), 1.0))

    if not errors:
        return 50.0
    return round((1.0 - sum(errors) / len(errors)) * 100, 2)


def _build_landmark_layer(landmarks: List[Dict]) -> Dict[str, Any]:
    return {"layer_name": "Anatomical Landmarks", "layer_type": "landmark", "data": {"count": len(landmarks), "points": landmarks}}


def _build_grid_layer(grid: List[Dict]) -> Dict[str, Any]:
    return {"layer_name": "Proportional Grid", "layer_type": "grid", "data": {"total_cells": len(grid), "occupied_cells": sum(1 for c in grid if c["contains_landmark"]), "cells": grid}}


def _build_shading_layer(shading_regions: List[Dict]) -> Dict[str, Any]:
    return {"layer_name": "Shading Map", "layer_type": "shading", "data": {"region_count": len(shading_regions), "regions": shading_regions}}


def _build_outline_layer(landmarks: List[Dict]) -> Dict[str, Any]:
    """Wireframe silhouette connecting landmark pairs."""
    connections = [
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ]
    edges: List[Dict] = []
    for a_name, b_name in connections:
        a, b = _lm_by_name(landmarks, a_name), _lm_by_name(landmarks, b_name)
        if a and b:
            edges.append({"from": a_name, "to": b_name, "from_xy": [a["x"], a["y"]], "to_xy": [b["x"], b["y"]]})

    return {"layer_name": "Wireframe Outline", "layer_type": "outline", "data": {"edge_count": len(edges), "edges": edges}}


def generate_blueprint(
    landmarks: List[Dict],
    grid: List[Dict],
    shading_regions: List[Dict],
    complexity_score: float,
    difficulty_level: str = "intermediate",
) -> Dict[str, Any]:
    """Assemble a full blueprint from all intermediate results."""
    proportion_accuracy = _score_proportions(landmarks)
    layers = [
        _build_landmark_layer(landmarks),
        _build_grid_layer(grid),
        _build_shading_layer(shading_regions),
        _build_outline_layer(landmarks),
    ]
    return {
        "complexity_score": complexity_score,
        "proportion_accuracy": proportion_accuracy,
        "difficulty_level": difficulty_level,
        "landmarks": landmarks,
        "grid": grid,
        "shading_regions": shading_regions,
        "layers": layers,
    }
