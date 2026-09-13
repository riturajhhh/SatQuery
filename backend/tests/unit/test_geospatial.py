"""Tests for geospatial metadata extraction, modality detection, and compatibility."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_origin

from app.geospatial import (
    extract_metadata,
    detect_modality,
    generate_preview_and_thumbnail,
    validate_pair_compatibility,
)


@pytest.fixture
def sample_geotiff(tmp_path: Path) -> Path:
    """Create a temporary 3-band georeferenced GeoTIFF."""
    file_path = tmp_path / "sample_optical.tif"
    transform = from_origin(72.5, 23.5, 0.001, 0.001)
    data = np.ones((3, 50, 50), dtype=np.uint8) * 120

    with rasterio.open(
        str(file_path),
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def sample_sar_geotiff(tmp_path: Path) -> Path:
    """Create a temporary 1-band SAR GeoTIFF."""
    file_path = tmp_path / "sample_sar.tif"
    transform = from_origin(72.5, 23.5, 0.001, 0.001)
    data = np.random.uniform(0.0, 1.0, (1, 50, 50)).astype(np.float32)

    with rasterio.open(
        str(file_path),
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return file_path


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    """Create a standard non-georeferenced PNG."""
    file_path = tmp_path / "photo.png"
    img = Image.new("RGB", (60, 40), color=(200, 100, 50))
    img.save(file_path)
    return file_path


def test_extract_metadata_geotiff(sample_geotiff: Path):
    meta = extract_metadata(sample_geotiff)
    assert meta["width"] == 50
    assert meta["height"] == 50
    assert meta["bands"] == 3
    assert meta["is_georeferenced"] is True
    assert "EPSG:4326" in meta["crs"]
    assert meta["bounds"] is not None
    assert meta["bounds"]["left"] == pytest.approx(72.5)
    assert meta["bounds"]["top"] == pytest.approx(23.5)


def test_extract_metadata_png(sample_png: Path):
    meta = extract_metadata(sample_png)
    assert meta["width"] == 60
    assert meta["height"] == 40
    assert meta["bands"] == 3
    assert meta["is_georeferenced"] is False
    assert meta["crs"] is None


def test_detect_modality(sample_geotiff: Path, sample_sar_geotiff: Path):
    meta_opt = extract_metadata(sample_geotiff)
    assert detect_modality(meta_opt, filename="sample_optical.tif") == "optical"

    meta_sar = extract_metadata(sample_sar_geotiff)
    assert detect_modality(meta_sar, filename="sample_sar.tif") == "sar"

    # 4 bands -> multispectral
    assert detect_modality({"bands": 4}, filename="s2.tif") == "multispectral"


def test_preview_generation(sample_geotiff: Path, tmp_path: Path):
    processed_dir = tmp_path / "processed"
    preview_path, thumb_path = generate_preview_and_thumbnail(
        file_path=sample_geotiff,
        output_dir=processed_dir,
        file_id="test1234",
    )
    assert preview_path.exists()
    assert thumb_path.exists()
    assert preview_path.suffix == ".png"

    with Image.open(preview_path) as p_img:
        assert p_img.size[0] <= 1024


def test_pair_compatibility_overlapping(sample_geotiff: Path, sample_sar_geotiff: Path):
    meta_a = extract_metadata(sample_geotiff)
    meta_b = extract_metadata(sample_sar_geotiff)

    result = validate_pair_compatibility(meta_a, meta_b, "optical", "sar")
    assert result["valid"] is True
    assert result["details"]["workflow_recommendation"] == "optical_sar_fusion"
    assert result["details"]["spatial_overlap_percent"] == 100.0


def test_pair_compatibility_non_overlapping(tmp_path: Path):
    meta_a = {
        "bounds": {"left": 72.0, "bottom": 23.0, "right": 73.0, "top": 24.0},
        "resolution": {"x": 10.0, "y": 10.0},
    }
    meta_b = {
        "bounds": {"left": 80.0, "bottom": 12.0, "right": 81.0, "top": 13.0},
        "resolution": {"x": 10.0, "y": 10.0},
    }
    result = validate_pair_compatibility(meta_a, meta_b, "optical", "optical")
    assert result["valid"] is False
    assert result["details"]["spatial_overlap_percent"] == 0.0
