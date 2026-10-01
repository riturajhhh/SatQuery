"""SatQuery AI — Unit Tests for Fine-Tuned Sentinel-1 SAR Model.

Tests:
1. RSSAR_Specialist ModelInfo, supported tasks, and registry discovery.
2. Inference on dual-polarization Sentinel-1 SAR data (water vs urban double-bounce).
3. Verification of calibrated dB extraction (VV, VH, VV-VH ratio).
4. SAR visual evidence artifacts (polarimetric false-color, double-bounce boxes).
5. Explainability and physical parameter accuracy.
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
from app.models.sar_specialist import RSSAR_Specialist


@pytest.fixture
def synthetic_sar_water():
    """Synthetic SAR scene dominated by specular water absorption (very low backscatter)."""
    # RGB representation of SAR: R=VV, G=VH, B=Ratio
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [25, 20, 10]  # Very low VV and VH return (~ -23 dB)
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_sar_urban():
    """Synthetic SAR scene with intense dihedral double-bounce corner reflectors."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = [160, 140, 50]  # Background land
    # Dense high double-bounce building reflections (bright gold response)
    for i in range(20, 100, 20):
        for j in range(20, 100, 20):
            arr[i : i + 12, j : j + 12] = [250, 220, 100]
    return Image.fromarray(arr)


def test_sar_specialist_model_info():
    model = RSSAR_Specialist()
    info = model.info
    assert info.name == "sar-sentinel1-specialist"
    assert "Sentinel-1" in info.description
    assert TaskType.VQA in info.supported_tasks
    assert TaskType.CAPTIONING in info.supported_tasks
    assert InputType.SINGLE_SAR in info.supported_inputs
    assert InputType.OPTICAL_SAR_PAIR in info.supported_inputs


def test_sar_specialist_water_prediction(synthetic_sar_water):
    model = RSSAR_Specialist()
    model.load()
    inp = ModelInput(
        images=[synthetic_sar_water],
        query="What is the primary land-cover type detected in this SAR radar scene?",
        metadata={"modality": "sar"},
    )
    out = model.predict(inp)
    assert out.answer is not None
    assert out.confidence >= 0.75
    assert "polarimetric_parameters" in out.evidence
    params = out.evidence["polarimetric_parameters"]
    assert "mean_vv_backscatter_db" in params
    assert "mean_vh_backscatter_db" in params
    assert params["mean_vv_backscatter_db"] < -15.0  # Low backscatter for water


def test_sar_specialist_urban_double_bounce(synthetic_sar_urban):
    model = RSSAR_Specialist()
    model.load()
    inp = ModelInput(
        images=[synthetic_sar_urban],
        query="Describe this radar scene and locate double-bounce building structures.",
        metadata={"modality": "sar"},
    )
    out = model.predict(inp)
    assert out.answer is not None
    assert out.confidence >= 0.80
    assert "polarimetric_parameters" in out.evidence
    params = out.evidence["polarimetric_parameters"]
    assert params["double_bounce_percent"] > 0.0
    assert "overlay_path" in out.evidence
    assert Path(out.evidence["overlay_path"]).exists()


def test_sar_specialist_explain(synthetic_sar_urban):
    model = RSSAR_Specialist()
    model.load()
    inp = ModelInput(images=[synthetic_sar_urban], query="Analyze SAR backscatter")
    out = model.predict(inp)
    explanation = model.explain(inp, out)
    assert explanation["model"] == "sar-sentinel1-specialist"
    assert "polarimetric_scattering_summary" in explanation
    summary = explanation["polarimetric_scattering_summary"]
    assert "vv_surface_roughness" in summary
    assert "vh_volume_scattering" in summary


def test_sar_registry_lookup():
    registry = get_model_registry()
    sar_model = registry.get_model("sar-sentinel1-specialist")
    assert sar_model is not None
    assert sar_model.info.name == "sar-sentinel1-specialist"

    # Find models supporting SINGLE_SAR
    models = registry.find_models(TaskType.VQA, InputType.SINGLE_SAR)
    assert any(m.info.name == "sar-sentinel1-specialist" for m in models)
