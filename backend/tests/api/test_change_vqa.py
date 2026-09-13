"""API integration tests for Phase 7 — Bi-Temporal Change VQA workflow."""

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from starlette.testclient import TestClient

from app.main import app


def _create_synthetic_geotiff(
    bounds: tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    is_t2: bool = False,
) -> io.BytesIO:
    """Create an in-memory 3-band GeoTIFF with synthetic temporal variation."""
    width, height = 64, 64
    transform = from_bounds(*bounds, width, height)

    data = np.full((3, height, width), 80, dtype=np.uint8)
    if not is_t2:
        # T1: Forest canopy in upper quadrant
        data[0, :32, :32] = 20
        data[1, :32, :32] = 200
        data[2, :32, :32] = 30
    else:
        # T2: Land clearance (higher red/blue, low green)
        data[0, :32, :32] = 180
        data[1, :32, :32] = 130
        data[2, :32, :32] = 90

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    buf.seek(0)
    return buf


@pytest.fixture
def client():
    return TestClient(app)


def test_end_to_end_change_vqa_workflow(client):
    """Test full upload of 2 temporal images -> ask change question -> verify answer & evidence."""
    # 1. Prepare 2 overlapping temporal GeoTIFFs
    bounds = (-122.4, 37.7, -122.3, 37.8)
    tif1 = _create_synthetic_geotiff(bounds, crs="EPSG:4326", is_t2=False)
    tif2 = _create_synthetic_geotiff(bounds, crs="EPSG:4326", is_t2=True)

    # 2. Upload pair
    upload_res = client.post(
        "/api/upload",
        files=[
            ("files", ("pre_event_t1.tif", tif1.getvalue(), "image/tiff")),
            ("files", ("post_event_t2.tif", tif2.getvalue(), "image/tiff")),
        ],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]
    file_ids = [f["file_id"] for f in upload_data["files"]]
    assert len(file_ids) == 2

    # 3. Submit change VQA question
    analyze_res = client.post(
        "/api/analyze",
        json={
            "upload_id": upload_id,
            "file_ids": file_ids,
            "query": "Did the forest area decrease or expand between these observations?",
        },
    )
    assert analyze_res.status_code == 200
    analyze_data = analyze_res.json()
    analysis_id = analyze_data["analysis_id"]

    # 4. Fetch analysis results
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result = result_res.json()

    assert result["task"] == "change_vqa"
    assert result["input_type"] == "bi_temporal"
    assert result["status"] == "complete"
    assert result["answer"]["confidence"] >= 0.80

    answer_text = result["answer"]["text"].lower()
    assert "decrease" in answer_text or "shrinkage" in answer_text or "loss" in answer_text

    # 5. Verify evidence & statistics
    evidence = result["evidence"]
    assert "change_detection_map" in evidence
    change_map_meta = evidence["change_detection_map"]
    assert "statistics" in change_map_meta
    stats = change_map_meta["statistics"]
    assert stats["changed_percentage"] > 10.0
    assert "change_map_url" in change_map_meta

    # 6. Verify image retrieval via HTTP
    map_url = change_map_meta["change_map_url"]
    img_res = client.get(map_url)
    assert img_res.status_code == 200
    assert len(img_res.content) > 0
