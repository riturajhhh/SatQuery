"""Unit tests for BuildingCounter and accurate Land Cover description generation."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from app.models.base import ModelInput, ConfidenceLevel, TaskType
from app.models.building_counter import BuildingCounter
from app.models.captioning import RSCaptioning_BLIP2, RSCaptioning_Fallback, analyze_scene_elements, synthesize_scene_description
from app.models.vqa import RSVQA_BLIP2, RSVQA_Fallback


@pytest.fixture
def synthetic_forest_image() -> Image.Image:
    """Predominantly forest canopy (green reflectance)."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:, :, 0] = 30
    arr[:, :, 1] = 165
    arr[:, :, 2] = 40
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_water_image() -> Image.Image:
    """Predominantly water (strong blue excess)."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:, :, 0] = 20
    arr[:, :, 1] = 70
    arr[:, :, 2] = 180
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_urban_with_buildings() -> Image.Image:
    """Scene with vegetation background and 3 distinct high-contrast building footprints."""
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    # Background: green terrain
    arr[:, :, 0] = 50
    arr[:, :, 1] = 140
    arr[:, :, 2] = 55

    # Building 1: Northwest (20x20 pixels)
    arr[15:35, 15:35, :] = [210, 205, 195]

    # Building 2: Northeast (25x20 pixels)
    arr[20:40, 75:100, :] = [190, 185, 180]

    # Building 3: Southeast (22x22 pixels)
    arr[75:97, 75:97, :] = [225, 220, 215]

    return Image.fromarray(arr)


# ---- 1. Building Counter Tests ----

def test_building_counter_detects_structures(synthetic_urban_with_buildings: Image.Image):
    res = BuildingCounter.detect_and_count(synthetic_urban_with_buildings)

    assert res["count"] >= 2, f"Expected at least 2 detected buildings, got {res['count']}"
    assert len(res["boxes"]) == res["count"]
    assert res["built_coverage_pct"] > 0
    assert "discrete buildings" in res["answer"].lower()
    assert Path(res["overlay_path"]).exists()
    assert res["overlay_url"].startswith("/api/files/evidence/")

    for b in res["boxes"]:
        assert len(b["box_2d"]) == 4
        assert len(b["box_pixel"]) == 4
        assert b["confidence"] >= 0.80
        ymin, xmin, ymax, xmax = b["box_2d"]
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0


def test_building_counter_zero_on_forest(synthetic_forest_image: Image.Image):
    res = BuildingCounter.detect_and_count(synthetic_forest_image)

    assert res["count"] == 0
    assert len(res["boxes"]) == 0
    assert "0 buildings" in res["answer"].lower()
    assert res["veg_pct"] > 60.0


def test_building_counter_zero_on_water(synthetic_water_image: Image.Image):
    res = BuildingCounter.detect_and_count(synthetic_water_image)

    assert res["count"] == 0
    assert len(res["boxes"]) == 0
    assert "0 buildings" in res["answer"].lower()
    assert res["water_pct"] > 60.0


# ---- 2. Land Cover Description Tests ----

def test_captioning_blip2_accurate_land_cover(synthetic_forest_image: Image.Image):
    model = RSCaptioning_BLIP2()
    inp = ModelInput(images=[synthetic_forest_image], query="Describe the land cover")
    out = model.predict(inp)

    assert out.answer is not None
    assert "vegetat" in out.answer.lower() or "canopy" in out.answer.lower()
    assert "vehicle" not in out.answer.lower(), "Should not contain hallucinated vehicle captions"

    breakdown = out.evidence["land_cover_breakdown"]
    assert breakdown["vegetation_percent"] > 60.0
    assert breakdown["water_percent"] < 10.0

    exp = model.explain(inp, out)
    assert exp["evidence_type"] == "spectral_indices"
    assert exp["spatial_indices"]["vegetation_cover_percent"] > 60.0


def test_captioning_water_scene_description(synthetic_water_image: Image.Image):
    model = RSCaptioning_Fallback()
    inp = ModelInput(images=[synthetic_water_image], query="Describe this scene")
    out = model.predict(inp)

    assert "water" in out.answer.lower() or "hydrological" in out.answer.lower()
    breakdown = out.evidence["land_cover_breakdown"]
    assert breakdown["water_percent"] > 60.0


# ---- 3. VQA Building Count and Land Cover Tests ----

def test_vqa_fallback_building_count_query(synthetic_urban_with_buildings: Image.Image):
    model = RSVQA_Fallback()
    inp = ModelInput(images=[synthetic_urban_with_buildings], query="How many buildings are in this satellite image?")
    out = model.predict(inp)

    assert out.confidence_level == ConfidenceLevel.HIGH
    assert "discrete buildings" in out.answer.lower()
    assert out.evidence is not None
    assert out.evidence["building_count"] >= 2
    assert "boxes" in out.evidence
    assert len(out.evidence["boxes"]) >= 2

    exp = model.explain(inp, out)
    assert exp["evidence_type"] == "grounding_overlay"
    assert len(exp["boxes"]) >= 2


def test_vqa_blip2_building_count_query(synthetic_urban_with_buildings: Image.Image):
    model = RSVQA_BLIP2()
    inp = ModelInput(images=[synthetic_urban_with_buildings], query="Count the buildings in this scene")
    out = model.predict(inp)

    assert "discrete buildings" in out.answer.lower() or "buildings" in out.answer.lower()
    assert out.evidence is not None
    assert "boxes" in out.evidence
    assert out.evidence["building_count"] >= 2

    exp = model.explain(inp, out)
    assert exp["evidence_type"] == "grounding_overlay"


def test_vqa_land_cover_query(synthetic_forest_image: Image.Image):
    model = RSVQA_BLIP2()
    inp = ModelInput(images=[synthetic_forest_image], query="What is the land cover of this image?")
    out = model.predict(inp)

    assert "vegetat" in out.answer.lower() or "forest" in out.answer.lower() or "canopy" in out.answer.lower()
    assert out.evidence is not None
    assert "land_cover_breakdown" in out.evidence
    assert out.evidence["land_cover_breakdown"]["vegetation_percent"] > 50.0
