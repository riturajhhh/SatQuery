"""API integration tests for analysis submission, polling, execution trace, and models endpoints."""

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
    transform = from_origin(75.0, 25.0, 0.001, 0.001)
    data = np.full((3, 50, 50), 120, dtype=np.uint8)
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


def test_list_models_endpoint(client: TestClient):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["total"] >= 1
    assert any(m["name"] == "rs-spectral-vqa-cpu" for m in data["models"])


def test_analyze_nonexistent_upload(client: TestClient):
    response = client.post(
        "/api/analyze",
        json={"upload_id": "non-existent-upload-id-999", "query": "What is here?"},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "not_found"


def test_end_to_end_vqa_analysis(client: TestClient):
    # 1. Upload satellite raster
    buf = create_test_geotiff()
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("test_scene.tif", buf, "image/tiff"))],
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Submit query
    query = "What type of land cover is dominant in this satellite imagery?"
    analyze_res = client.post(
        "/api/analyze",
        json={"upload_id": upload_id, "query": query},
    )
    assert analyze_res.status_code == 200
    analysis_id = analyze_res.json()["analysis_id"]

    # 3. Retrieve analysis result
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result_data = result_res.json()

    assert result_data["analysis_id"] == analysis_id
    assert result_data["status"] == "complete"
    assert result_data["task"] == "vqa"
    assert result_data["answer"] is not None
    assert len(result_data["answer"]["text"]) > 10
    assert result_data["answer"]["confidence"] > 0.6
    assert result_data["answer"]["confidence_level"] in ("HIGH", "MEDIUM")
    assert len(result_data["models_used"]) >= 1

    # 4. Check execution trace
    trace_res = client.get(f"/api/analysis/{analysis_id}/trace")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    steps = trace_data["trace"]["steps"]
    assert len(steps) >= 4
    assert all(s["status"] == "success" for s in steps)

    # 5. Check evidence
    evidence_res = client.get(f"/api/analysis/{analysis_id}/evidence")
    assert evidence_res.status_code == 200
    evidence_data = evidence_res.json()
    assert len(evidence_data["evidence"]) >= 1
