"""Unit tests for remote-sensing image captioning models and task routing."""

import numpy as np
from PIL import Image
import pytest

from app.models import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    ModelOutput,
    ModelRegistry,
    RSCaptioning_Fallback,
    TaskType,
)
from app.services.analysis_service import _determine_task_type


@pytest.fixture
def sample_forest_image() -> Image.Image:
    """Create a predominantly green satellite image (dense forest)."""
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :, 0] = 25
    arr[:, :, 1] = 170
    arr[:, :, 2] = 35
    return Image.fromarray(arr)


@pytest.fixture
def sample_urban_image() -> Image.Image:
    """Create an image with high checkerboard/edge contrast (urban/built-up)."""
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    # Checkerboard pattern to create high spatial edge frequency
    for y in range(80):
        for x in range(80):
            if ((x // 8) + (y // 8)) % 2 == 0:
                arr[y, x, :] = 220
            else:
                arr[y, x, :] = 50
    return Image.fromarray(arr)


def test_captioning_forest_scene(sample_forest_image: Image.Image):
    model = RSCaptioning_Fallback()
    inp = ModelInput(images=[sample_forest_image], query="Describe this image")
    out = model.predict(inp)

    assert out.answer is not None
    assert "vegetation" in out.answer.lower() or "canopy" in out.answer.lower()
    assert out.confidence >= 0.8
    assert out.confidence_level == ConfidenceLevel.HIGH
    assert out.is_fallback is True

    explanation = model.explain(inp, out)
    assert "land_cover_breakdown" in explanation
    assert explanation["land_cover_breakdown"]["vegetation_percent"] > 50.0


def test_captioning_urban_scene(sample_urban_image: Image.Image):
    model = RSCaptioning_Fallback()
    inp = ModelInput(images=[sample_urban_image], query="Caption this satellite image")
    out = model.predict(inp)

    assert out.answer is not None
    assert "urban" in out.answer.lower() or "built-up" in out.answer.lower() or "structural" in out.answer.lower()
    assert out.confidence >= 0.8


def test_query_task_routing():
    # Captioning triggers
    assert _determine_task_type("Describe this image") == TaskType.CAPTIONING
    assert _determine_task_type("Caption this satellite scene") == TaskType.CAPTIONING
    assert _determine_task_type("Provide a scene description") == TaskType.CAPTIONING
    assert _determine_task_type("Summarize what this image shows") == TaskType.CAPTIONING
    assert _determine_task_type("what does this scene depict?") == TaskType.CAPTIONING

    # VQA questions
    assert _determine_task_type("What is the dominant land cover?") == TaskType.VQA
    assert _determine_task_type("Is there a water body?") == TaskType.VQA
    assert _determine_task_type("Count the number of buildings") == TaskType.VQA
