"""Unit tests for visual grounding specialist model and task routing."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.models import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    ModelOutput,
    RSGrounding_Fallback,
    TaskType,
)
from app.models.grounding import _find_component_bboxes
from app.services.analysis_service import _determine_task_type


@pytest.fixture
def image_with_water_reservoir() -> Image.Image:
    """Create an optical image with a prominent blue water reservoir in the top-left."""
    arr = np.full((120, 120, 3), 110, dtype=np.uint8)  # Grayish background
    # Reservoir in quadrant: [10:50, 10:50]
    arr[10:50, 10:50, 0] = 10   # Red
    arr[10:50, 10:50, 1] = 40   # Green
    arr[10:50, 10:50, 2] = 190  # High Blue
    return Image.fromarray(arr)


def test_find_component_bboxes():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True

    boxes = _find_component_bboxes(mask, min_pixel_area=10)
    assert len(boxes) >= 1
    box = boxes[0]
    xmin, ymin, xmax, ymax = box["box_pixel"]
    assert xmin <= 20 and ymin <= 20
    assert xmax >= 39 and ymax >= 39


def test_grounding_water_target(image_with_water_reservoir: Image.Image):
    model = RSGrounding_Fallback()
    inp = ModelInput(images=[image_with_water_reservoir], query="Highlight the water body")
    out = model.predict(inp)

    assert out.answer is not None
    assert "located and highlighted" in out.answer.lower()
    assert out.confidence >= 0.8
    assert out.confidence_level == ConfidenceLevel.HIGH

    ev = out.evidence
    assert ev is not None
    assert ev["box_count"] >= 1
    assert "overlay_url" in ev
    assert "overlay_path" in ev
    assert Path(ev["overlay_path"]).exists()

    box = ev["boxes"][0]
    assert box["label"] == "Water Body"
    assert box["confidence"] >= 0.8


def test_grounding_task_routing():
    assert _determine_task_type("Highlight the water body") == TaskType.GROUNDING
    assert _determine_task_type("Locate the dense forest") == TaskType.GROUNDING
    assert _determine_task_type("Find the buildings in this scene") == TaskType.GROUNDING
    assert _determine_task_type("Where is the airport runway?") == TaskType.GROUNDING
    assert _determine_task_type("Bounding box for the agricultural field") == TaskType.GROUNDING
    assert _determine_task_type("Show me the river") == TaskType.GROUNDING
