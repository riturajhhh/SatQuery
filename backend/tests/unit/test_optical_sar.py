"""Unit tests for Phase 8 — Optical-SAR Cross-Modal Fusion specialist models."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.models.base import ModelInput, TaskType
from app.models.optical_sar import (
    RSOpticalSAR_Fallback,
    _align_and_prep_optical_sar,
)
from app.services.analysis_service import _determine_task_type


def test_align_and_prep_optical_sar():
    """Verify that Optical and SAR images of differing resolutions are properly aligned."""
    opt = Image.new("RGB", (120, 80), color=(100, 150, 200))
    sar = Image.new("L", (160, 100), color=128)

    aligned_opt, aligned_sar = _align_and_prep_optical_sar(opt, sar)
    assert aligned_opt.size == (120, 80)
    assert aligned_sar.size == (120, 80)
    assert aligned_opt.mode == "RGB"
    assert aligned_sar.mode == "L"


def test_optical_sar_cloud_penetration():
    """Verify detection of optical clouds and SAR penetration revealing underlying structures."""
    # 1. Optical scene: cloudy in upper half (high albedo white, low saturation)
    opt = np.full((100, 100, 3), 80, dtype=np.uint8)
    opt[:45, :] = [235, 235, 240]  # Thick cloud deck

    # 2. Co-registered SAR scene: high radar backscatter (urban infrastructure) under the cloud
    sar = np.full((100, 100), 70, dtype=np.uint8)
    sar[:30, :50] = 220  # Bright double-bounce scatterers under cloud
    sar[30:45, :50] = 20  # Calm water body under cloud

    model = RSOpticalSAR_Fallback()
    metadata = {"resolution": {"x": 10.0, "y": 10.0}}
    output = model.predict(
        ModelInput(
            images=[Image.fromarray(opt), Image.fromarray(sar, mode="L")],
            query="Analyze surface structures obscured by cloud cover using optical and SAR fusion",
            metadata=metadata,
        )
    )

    stats = output.evidence["statistics"]
    assert stats["cloud_cover_percent"] > 40.0
    assert stats["under_cloud_urban_pixels"] > 0
    assert stats["under_cloud_water_pixels"] > 0
    assert "cloud" in output.answer.lower()
    assert "radar" in output.answer.lower() or "sar" in output.answer.lower()
    assert output.confidence >= 0.85
    assert Path(output.evidence["overlay_path"]).exists()


def test_optical_sar_task_routing():
    """Verify task routing for multi-sensor Optical-SAR queries."""
    assert (
        _determine_task_type(
            "Analyze surface features using optical and SAR fusion",
            num_files=2,
            is_optical_sar=True,
        )
        == TaskType.OPTICAL_SAR
    )
    assert (
        _determine_task_type(
            "Pierce cloud cover with radar backscatter",
            num_files=2,
            is_optical_sar=False,
        )
        == TaskType.OPTICAL_SAR
    )
    assert (
        _determine_task_type(
            "What is under the cloud deck?",
            num_files=2,
            is_optical_sar=True,
        )
        == TaskType.OPTICAL_SAR
    )
