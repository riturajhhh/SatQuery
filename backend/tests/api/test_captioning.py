"""API integration tests for remote-sensing image captioning workflow."""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
import rasterio
from rasterio.transform import from_origin

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_test_geotiff() -> io.BytesIO:
    buf = io.BytesIO()
    transform = from_origin(73.8, 18.5, 0.001, 0.001)
    data = np.zeros((3, 50, 50), dtype=np.uint8)
    data[1, :, :] = 160  # Green canopy
    with rasterio.open(
        buf,
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
    buf.seek(0)
    return buf


def test_end_to_end_captioning_workflow(client: TestClient):
    # 1. Upload satellite image
    buf = create_test_geotiff()
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("pune_rural_scene.tif", buf, "image/tiff"))],
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Submit captioning request
    query = "Describe this satellite image and its major land-cover components."
    analyze_res = client.post(
        "/api/analyze",
        json={"upload_id": upload_id, "query": query},
    )
    assert analyze_res.status_code == 200
    analysis_id = analyze_res.json()["analysis_id"]

    # 3. Retrieve analysis result
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result = result_res.json()

    assert result["task"] == "captioning"
    assert result["status"] == "complete"
    assert result["answer"] is not None
    # Must be a coherent multi-word description
    assert len(result["answer"]["text"].split()) >= 10
    assert result["answer"]["confidence"] >= 0.8
    assert result["models_used"][0]["task"] == "captioning"

    # 4. Check trace
    trace_res = client.get(f"/api/analysis/{analysis_id}/trace")
    assert trace_res.status_code == 200
    steps = trace_res.json()["trace"]["steps"]
    step2 = next(s for s in steps if s["step"] == 2)
    assert "captioning" in step2["details"].lower()
