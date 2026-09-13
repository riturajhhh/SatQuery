"""API integration tests for Phase 8 — Optical-SAR Cross-Modal Fusion workflow."""

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from starlette.testclient import TestClient

from app.main import app


def _create_optical_geotiff(bounds: tuple[float, float, float, float]) -> io.BytesIO:
    """Create in-memory 3-band Optical GeoTIFF with synthetic cloud cover."""
    width, height = 64, 64
    transform = from_bounds(*bounds, width, height)

    data = np.full((3, height, width), 80, dtype=np.uint8)
    # Clouds in upper half (high albedo white)
    data[:, :30, :] = 240

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    buf.seek(0)
    return buf


def _create_sar_geotiff(bounds: tuple[float, float, float, float]) -> io.BytesIO:
    """Create in-memory 1-band SAR GeoTIFF with radar scatterers."""
    width, height = 64, 64
    transform = from_bounds(*bounds, width, height)

    data = np.full((1, height, width), 70, dtype=np.uint8)
    # Strong double-bounce corner reflectors under the cloud
    data[0, :20, :30] = 220

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    buf.seek(0)
    return buf


@pytest.fixture
def client():
    return TestClient(app)


def test_end_to_end_optical_sar_workflow(client):
    """Test full upload of Optical + SAR pair -> fusion analysis -> verify evidence & map download."""
    bounds = (-122.4, 37.7, -122.3, 37.8)
    opt_buf = _create_optical_geotiff(bounds)
    sar_buf = _create_sar_geotiff(bounds)

    # 1. Upload Optical + SAR Pair
    upload_res = client.post(
        "/api/upload",
        files=[
            ("files", ("scene_optical.tif", opt_buf.getvalue(), "image/tiff")),
            ("files", ("scene_sar.tif", sar_buf.getvalue(), "image/tiff")),
        ],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]
    file_ids = [f["file_id"] for f in upload_data["files"]]
    assert len(file_ids) == 2

    # Verify compatibility recommendation
    assert upload_data["compatibility"]["details"]["workflow_recommendation"] == "optical_sar_fusion"

    # 2. Submit Optical-SAR Fusion Analysis
    analyze_res = client.post(
        "/api/analyze",
        json={
            "upload_id": upload_id,
            "file_ids": file_ids,
            "query": "Perform optical-sar fusion to pierce cloud cover and identify surface features",
        },
    )
    assert analyze_res.status_code == 200
    analysis_id = analyze_res.json()["analysis_id"]

    # 3. Retrieve analysis results
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result = result_res.json()

    assert result["task"] == "optical_sar_analysis"
    assert result["input_type"] == "optical_sar_pair"
    assert result["status"] == "complete"
    assert result["answer"]["confidence"] >= 0.85
    assert "cloud" in result["answer"]["text"].lower()

    # 4. Verify evidence & statistics
    evidence = result["evidence"]
    assert "optical_sar_fusion" in evidence
    fusion_meta = evidence["optical_sar_fusion"]
    stats = fusion_meta["statistics"]
    assert stats["cloud_cover_percent"] > 30.0
    assert stats["under_cloud_urban_pixels"] > 0
    assert "change_map_url" in fusion_meta

    # 5. Verify image download over HTTP
    map_url = fusion_meta["change_map_url"]
    img_res = client.get(map_url)
    assert img_res.status_code == 200
    assert len(img_res.content) > 0
