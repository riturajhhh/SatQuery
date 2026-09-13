"""Unit tests for Model Registry and VQA specialist adapters."""

import numpy as np
from PIL import Image
import pytest

from app.models import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    ModelOutput,
    ModelRegistry,
    RSVQA_Fallback,
    TaskType,
    get_model_registry,
)


@pytest.fixture
def mock_vegetation_image() -> Image.Image:
    """Create a predominantly green satellite image (high vegetation)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 30   # Low red
    arr[:, :, 1] = 160  # High green
    arr[:, :, 2] = 40   # Low blue
    return Image.fromarray(arr)


@pytest.fixture
def mock_water_image() -> Image.Image:
    """Create a predominantly blue satellite image (water body)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :, 0] = 20   # Very low red
    arr[:, :, 1] = 40   # Low green
    arr[:, :, 2] = 170  # High blue
    return Image.fromarray(arr)


def test_model_registry_registration():
    registry = ModelRegistry()
    model = RSVQA_Fallback()
    registry.register(model)

    found = registry.find_models(TaskType.VQA, InputType.SINGLE_OPTICAL)
    assert len(found) >= 1
    assert found[0].info.name == "rs-spectral-vqa-cpu"
    assert found[0].info.is_fallback is True


def test_vqa_vegetation_query(mock_vegetation_image: Image.Image):
    model = RSVQA_Fallback()
    inp = ModelInput(images=[mock_vegetation_image], query="Is there vegetation in this area?")
    out = model.predict(inp)

    assert out.answer is not None
    assert "vegetation" in out.answer.lower()
    assert out.confidence >= 0.8
    assert out.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
    assert out.is_fallback is True

    explanation = model.explain(inp, out)
    assert "spatial_indices" in explanation
    assert explanation["spatial_indices"]["vegetation_cover_percent"] > 50.0


def test_vqa_water_query(mock_water_image: Image.Image):
    model = RSVQA_Fallback()
    inp = ModelInput(images=[mock_water_image], query="Are there any water bodies or lakes?")
    out = model.predict(inp)

    assert out.answer is not None
    assert "water" in out.answer.lower()
    assert out.confidence >= 0.8
    assert out.is_fallback is True


def test_vqa_invalid_input():
    model = RSVQA_Fallback()
    with pytest.raises(ValueError):
        model.predict(ModelInput(images=[], query="What is here?"))

    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError):
        model.predict(ModelInput(images=[img], query="   "))
