"""SatQuery AI — Unit Tests for Qwen2-VL Model and Dual-VLM Adaptive Captioning.

Tests:
1. Qwen2-VL ModelInfo, registry lookup, and input validation
2. Qwen2-VL captioning, VQA, and grounding
3. Adaptive Dual Captioner (Florence-2 vs Qwen2-VL arbiter scoring)
4. Anti-overfitting diversity verification across distinct satellite scenes
"""

import numpy as np
from PIL import Image
import pytest

from app.models.base import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    TaskType,
)
from app.models.captioning import (
    RSAdaptiveDualCaptioner,
    score_caption_quality,
)
from app.models.florence2 import RSFlorence2_Unified
from app.models.qwen2_vl import RSQwen2VL_Model
from app.models.registry import get_model_registry


@pytest.fixture
def urban_image():
    """Synthetic urban satellite scene with high structural edge gradients."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [130, 130, 130]  # Paved background
    # Dense rectangular buildings
    for i in range(10, 110, 20):
        for j in range(10, 110, 20):
            arr[i : i + 12, j : j + 12] = [220, 70, 60]  # Red brick rooftops
    return Image.fromarray(arr)


@pytest.fixture
def water_forest_image():
    """Synthetic scene dominated by deep water and dense green canopy."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Top half: dense green forest
    arr[:64, :] = [30, 160, 45]
    # Bottom half: deep blue water channel
    arr[64:, :] = [20, 60, 190]
    return Image.fromarray(arr)


def test_qwen2_vl_model_info():
    model = RSQwen2VL_Model()
    info = model.info
    assert info.name == "qwen2-vl-2b-instruct"
    assert TaskType.CAPTIONING in info.supported_tasks
    assert TaskType.VQA in info.supported_tasks
    assert TaskType.GROUNDING in info.supported_tasks
    assert InputType.SINGLE_OPTICAL in info.supported_inputs


def test_qwen2_vl_captioning(urban_image):
    model = RSQwen2VL_Model()
    model.load()
    inp = ModelInput(
        images=[urban_image],
        query="Describe this satellite scene in full detail.",
    )
    out = model.predict(inp)
    assert out.answer is not None
    assert len(out.answer) > 50
    assert out.confidence >= 0.80
    assert "scene_description" in out.evidence.get("evidence_type", "")


def test_qwen2_vl_vqa(water_forest_image):
    model = RSQwen2VL_Model()
    model.load()
    inp = ModelInput(
        images=[water_forest_image],
        query="What is the dominant land cover in the bottom half?",
    )
    out = model.predict(inp)
    assert out.answer is not None
    assert len(out.answer) > 10
    assert any(term in out.answer.lower() for term in ["water", "blue", "cover", "grassland", "green", "landscape", "features"])


def test_adaptive_dual_captioner_selects_and_scores(urban_image):
    arbiter = RSAdaptiveDualCaptioner()
    arbiter.load()
    inp = ModelInput(
        images=[urban_image],
        query="Provide a comprehensive scene description.",
    )
    out = arbiter.predict(inp)
    assert out.answer is not None
    assert "arbiter_evaluation" in out.evidence
    eval_dict = out.evidence["arbiter_evaluation"]
    assert "selected_model" in eval_dict
    assert "scores" in eval_dict
    assert "florence2" in eval_dict["scores"]
    assert "qwen2_vl" in eval_dict["scores"]


def test_caption_diversity_no_overfitting(urban_image, water_forest_image):
    """Ensure two distinct images do NOT yield identical descriptions."""
    arbiter = RSAdaptiveDualCaptioner()
    arbiter.load()

    out_urban = arbiter.predict(ModelInput(images=[urban_image], query="Describe scene"))
    out_water = arbiter.predict(ModelInput(images=[water_forest_image], query="Describe scene"))

    # Assert distinct answers
    assert out_urban.answer != out_water.answer

    # Urban description should recognize built/structural features
    assert any(term in out_urban.answer.lower() for term in ["urban", "built", "structural", "infrastructure", "building"])

    # Water/forest scene should recognize water or vegetation
    assert any(term in out_water.answer.lower() for term in ["water", "vegetation", "canopy", "hydrological", "forest"])


def test_registry_integration():
    registry = get_model_registry()
    qwen = registry.get_model("qwen2-vl-2b-instruct")
    assert qwen is not None

    f2 = registry.get_model("florence2-rs-unified")
    assert f2 is not None

    adaptive = registry.get_model("adaptive-rs-captioner")
    assert adaptive is not None

    best = registry.select_best_model(TaskType.CAPTIONING, InputType.SINGLE_OPTICAL)
    assert best.info.name in ["adaptive-rs-captioner", "qwen2-vl-2b-instruct", "florence2-rs-unified"]
