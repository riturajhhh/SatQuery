"""Unit tests for Phase 6 — Bi-Temporal Change Detection specialist models."""

import os
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.models.base import ModelInput, TaskType
from app.models.change_detection import (
    RSChangeDetection_Fallback,
    _align_and_resample_pair,
)
from app.services.analysis_service import _determine_task_type


def test_align_and_resample_pair():
    """Verify that images of differing dimensions are aligned to match."""
    img1 = Image.new("RGB", (120, 80), color=(100, 100, 100))
    img2 = Image.new("RGB", (200, 150), color=(150, 150, 150))

    aligned1, aligned2 = _align_and_resample_pair(img1, img2)
    assert aligned1.size == (120, 80)
    assert aligned2.size == (120, 80)


def test_change_detection_identical_images():
    """Verify change detection reports minimal/no change when comparing identical scenes."""
    arr = np.full((64, 64, 3), 120, dtype=np.uint8)
    img1 = Image.fromarray(arr)
    img2 = Image.fromarray(arr)

    model = RSChangeDetection_Fallback()
    output = model.predict(ModelInput(images=[img1, img2], query="Detect changes"))

    stats = output.evidence["statistics"]
    assert stats["changed_percentage"] == 0.0
    assert stats["changed_pixels"] == 0
    assert "stability" in output.answer.lower() or "minimal change" in output.answer.lower()
    assert output.confidence >= 0.8


def test_change_detection_vegetation_loss():
    """Verify detection of significant vegetation clearing/loss."""
    # T1: Green forest covering half the scene
    t1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t1[:50, :50] = [20, 210, 40]

    # T2: Forest cleared to bare soil/ground
    t2 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t2[:50, :50] = [180, 140, 90]

    model = RSChangeDetection_Fallback()
    metadata = {"resolution": {"x": 10.0, "y": 10.0}}
    output = model.predict(
        ModelInput(
            images=[Image.fromarray(t1), Image.fromarray(t2)],
            query="Analyze bi-temporal changes in land cover",
            metadata=metadata,
        )
    )

    stats = output.evidence["statistics"]
    assert stats["changed_percentage"] > 15.0
    assert stats["changed_pixels"] > 500
    assert stats["changed_area_hectares"] > 0
    assert "Vegetation Loss" in stats["dominant_transition"]
    assert Path(output.evidence["overlay_path"]).exists()


def test_change_detection_task_routing():
    """Verify task routing for comparative and change detection queries."""
    assert _determine_task_type("Detect changes between two images", num_files=2) == TaskType.CHANGE_DETECTION
    assert _determine_task_type("Show change map over time", num_files=2) == TaskType.CHANGE_DETECTION
    assert _determine_task_type("Compare difference between these dates", num_files=2) == TaskType.CHANGE_DETECTION
    assert _determine_task_type("Measure deforestation area", num_files=2) == TaskType.CHANGE_DETECTION
    # Default to change detection when 2 files are uploaded
    assert _determine_task_type("Satellite observation analysis", num_files=2) == TaskType.CHANGE_DETECTION
