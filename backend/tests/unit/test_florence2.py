import numpy as np
import pytest
from PIL import Image

from app.models.base import ConfidenceLevel, InputType, ModelInput, TaskType
from app.models.florence2 import RSFlorence2_Unified
from app.models.registry import get_model_registry


@pytest.fixture
def sample_image():
    """Generate synthetic optical satellite test image."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [40, 150, 45]  # Green forest
    arr[40:80, 40:80] = [210, 200, 190]  # Urban structure
    return Image.fromarray(arr)


def test_florence2_info_and_tasks():
    model = RSFlorence2_Unified()
    info = model.info
    assert info.name == "florence2-rs-unified"
    assert TaskType.CAPTIONING in info.supported_tasks
    assert TaskType.GROUNDING in info.supported_tasks
    assert TaskType.VQA in info.supported_tasks
    assert InputType.SINGLE_OPTICAL in info.supported_inputs
    assert not info.is_fallback


def test_florence2_input_validation():
    model = RSFlorence2_Unified()
    with pytest.raises(ValueError, match="requires at least one"):
        model.validate_input(ModelInput(images=[]))


def test_florence2_captioning_inference(sample_image):
    model = RSFlorence2_Unified()
    model_input = ModelInput(
        images=[sample_image],
        query="Describe this satellite scene in detail",
    )
    output = model.predict(model_input)
    assert output.answer is not None
    assert len(output.answer) > 20
    assert output.confidence >= 0.85
    assert output.evidence["evidence_type"] == "scene_description"
    assert output.model_info.name == "florence2-rs-unified"


def test_florence2_grounding_inference(sample_image):
    model = RSFlorence2_Unified()
    model_input = ModelInput(
        images=[sample_image],
        query="Locate the urban building structure",
    )
    output = model.predict(model_input)
    assert output.answer is not None
    assert output.evidence["evidence_type"] == "grounding_overlay"
    assert "bounding_boxes" in output.evidence
    boxes = output.evidence["bounding_boxes"]
    assert len(boxes) >= 1
    # Check normalized coordinates [ymin, xmin, ymax, xmax]
    box = boxes[0]
    assert len(box) == 4
    for coord in box:
        assert 0.0 <= coord <= 1.0


def test_florence2_vqa_inference(sample_image):
    model = RSFlorence2_Unified()
    model_input = ModelInput(
        images=[sample_image],
        query="What is the primary land cover visible in this area?",
    )
    output = model.predict(model_input)
    assert output.answer is not None
    assert output.confidence >= 0.85
    assert output.evidence["evidence_type"] == "vqa_reasoning"


def test_florence2_registry_lookup():
    registry = get_model_registry()
    model = registry.get_model("florence2-rs-unified")
    assert model is not None
    assert TaskType.GROUNDING in model.info.supported_tasks
    assert TaskType.CAPTIONING in model.info.supported_tasks
    assert TaskType.VQA in model.info.supported_tasks
