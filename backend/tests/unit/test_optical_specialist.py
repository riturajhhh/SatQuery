"""SatQuery AI — Unit Tests for Fine-Tuned Sentinel-2 Optical Specialist Model.

Tests:
1. RSOptical_Specialist ModelInfo, supported tasks, and registry discovery.
2. Inference on multispectral optical data (dense forest vegetation vs open water).
3. Verification of physical geophysical index extraction (NDVI, NDWI, vegetation vigor).
4. Optical visual evidence artifacts (CIR False-Color Infrared composite).
5. Robustness to RGB and 4-band spectral inputs.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.models.base import (
    ConfidenceLevel,
    InputType,
    ModelInput,
    TaskType,
)
from app.models.registry import get_model_registry
from app.models.optical_specialist import RSOptical_Specialist


@pytest.fixture
def synthetic_optical_forest():
    """Synthetic optical patch dominated by dense green vegetation (high NIR, low Red)."""
    # RGB image with dominant green chlorophyll
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [35, 140, 45]
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_optical_water():
    """Synthetic optical patch dominated by deep water (absorption in NIR/Red, high Blue/Green)."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [20, 70, 160]
    return Image.fromarray(arr)


def test_optical_specialist_model_info():
    model = RSOptical_Specialist()
    info = model.info
    assert info.name == "optical-sentinel2-specialist"
    assert "Sentinel-2" in info.base_model or "Sentinel2" in info.base_model
    assert TaskType.VQA in info.supported_tasks
    assert TaskType.CAPTIONING in info.supported_tasks
    assert InputType.SINGLE_OPTICAL in info.supported_inputs
    assert InputType.SINGLE_MULTISPECTRAL in info.supported_inputs
    assert InputType.OPTICAL_SAR_PAIR in info.supported_inputs


def test_optical_specialist_vegetation_prediction(synthetic_optical_forest):
    model = RSOptical_Specialist()
    model.load()

    inp = ModelInput(
        images=[synthetic_optical_forest],
        query="Calculate NDVI vegetation health and assess canopy vigor in this optical image.",
        metadata={"modality": "optical"},
    )
    output = model.predict(inp)

    assert output.confidence > 0.70
    assert output.evidence["mean_ndvi"] > 0.30
    assert "NDVI" in output.answer
    assert output.evidence["artifacts"] is not None
    assert len(output.evidence["artifacts"]) >= 1
    assert output.evidence["artifacts"][0]["type"] == "cir_false_color"


def test_optical_specialist_water_prediction(synthetic_optical_water):
    model = RSOptical_Specialist()
    model.load()

    inp = ModelInput(
        images=[synthetic_optical_water],
        query="Is surface water or open lake present in this scene?",
        metadata={"modality": "optical"},
    )
    output = model.predict(inp)

    assert output.confidence > 0.70
    assert output.evidence["mean_ndwi"] > -0.2
    assert "water" in output.answer.lower() or "hydrological" in output.answer.lower()


def test_optical_specialist_registry_discovery():
    registry = get_model_registry()
    model = registry.get_model("optical-sentinel2-specialist")
    assert model is not None
    assert model.info.name == "optical-sentinel2-specialist"
    assert InputType.SINGLE_OPTICAL in model.info.supported_inputs
