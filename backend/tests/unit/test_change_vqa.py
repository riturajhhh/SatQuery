"""Unit tests for Phase 7 — Bi-Temporal Change Visual Question Answering (Change VQA)."""

import numpy as np
import pytest
from PIL import Image

from app.models.base import ModelInput, TaskType
from app.models.change_vqa import RSChangeVQA_Fallback
from app.services.analysis_service import _determine_task_type


def test_change_vqa_vegetation_loss_query():
    """Verify answering specific question regarding forest canopy decrease/expansion."""
    # T1: Green vegetation
    t1 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t1[:40, :40] = [20, 210, 40]

    # T2: Cleared land
    t2 = np.full((100, 100, 3), 100, dtype=np.uint8)
    t2[:40, :40] = [190, 140, 80]

    model = RSChangeVQA_Fallback()
    metadata = {"resolution": {"x": 10.0, "y": 10.0}}
    output = model.predict(
        ModelInput(
            images=[Image.fromarray(t1), Image.fromarray(t2)],
            query="Did the forest area decrease or expand between these two dates?",
            metadata=metadata,
        )
    )

    assert "decrease" in output.answer.lower() or "shrinkage" in output.answer.lower()
    assert "hectares" in output.answer.lower() or "pixels" in output.answer.lower()
    assert output.confidence >= 0.85
    assert "statistics" in output.evidence
    assert "vqa_reasoning" in output.evidence
    assert output.evidence["vqa_reasoning"]["detected_intent"] == "vegetation"


def test_change_vqa_water_inundation_query():
    """Verify answering question about water body dynamics."""
    # T1: Dry ground
    t1 = np.full((100, 100, 3), 120, dtype=np.uint8)

    # T2: Surface water inundation (blue)
    t2 = np.full((100, 100, 3), 120, dtype=np.uint8)
    t2[:40, :40] = [15, 60, 220]

    model = RSChangeVQA_Fallback()
    output = model.predict(
        ModelInput(
            images=[Image.fromarray(t1), Image.fromarray(t2)],
            query="What happened to the water body?",
        )
    )

    assert "water body expanded" in output.answer.lower() or "inundation" in output.answer.lower()
    assert output.evidence["vqa_reasoning"]["detected_intent"] == "water"


def test_change_vqa_task_routing():
    """Verify distinction between Change VQA questions and Change Detection commands."""
    # Questions -> CHANGE_VQA
    assert _determine_task_type("Did the forest canopy decrease?", num_files=2) == TaskType.CHANGE_VQA
    assert _determine_task_type("What happened to the lake between these dates?", num_files=2) == TaskType.CHANGE_VQA
    assert _determine_task_type("How much vegetation was lost?", num_files=2) == TaskType.CHANGE_VQA
    assert _determine_task_type("Are there any new buildings built?", num_files=2) == TaskType.CHANGE_VQA
    assert _determine_task_type("Is there deforestation observed?", num_files=2) == TaskType.CHANGE_VQA

    # Imperative commands -> CHANGE_DETECTION
    assert _determine_task_type("Detect changes", num_files=2) == TaskType.CHANGE_DETECTION
    assert _determine_task_type("Show difference map", num_files=2) == TaskType.CHANGE_DETECTION
    assert _determine_task_type("Generate change overlay", num_files=2) == TaskType.CHANGE_DETECTION
