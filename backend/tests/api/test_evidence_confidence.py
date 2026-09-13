"""API integration tests for Phase 10 — Evidence & Confidence System."""

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from starlette.testclient import TestClient

from app.main import app


def _create_synthetic_geotiff(
    bounds: tuple[float, float, float, float] = (-122.4, 37.7, -122.3, 37.8),
    crs: str = "EPSG:4326",
) -> io.BytesIO:
    """Create an in-memory 3-band GeoTIFF with vegetation spectral profile."""
    width, height = 64, 64
    transform = from_bounds(*bounds, width, height)

    data = np.full((3, height, width), 50, dtype=np.uint8)
    data[0, :, :] = 30   # Red
    data[1, :, :] = 190  # Green
    data[2, :, :] = 40   # Blue

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


def test_end_to_end_evidence_confidence_workflow(client):
    """Test analysis produces visual evidence, calibrated confidence, audit findings, and trace steps."""
    tif = _create_synthetic_geotiff()

    # 1. Upload GeoTIFF
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("forest_scene.tif", tif.getvalue(), "image/tiff"))],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]
    file_ids = [f["file_id"] for f in upload_data["files"]]

    # 2. Submit Question
    analyze_res = client.post(
        "/api/analyze",
        json={
            "upload_id": upload_id,
            "file_ids": file_ids,
            "query": "Describe the vegetation and canopy cover in this satellite image",
        },
    )
    assert analyze_res.status_code == 200
    analysis_id = analyze_res.json()["analysis_id"]

    # 3. Retrieve Analysis
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result = result_res.json()

    assert result["status"] == "complete"
    assert result["answer"]["confidence"] > 0.0
    assert result["answer"]["confidence_level"] in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN")

    # Verify Calibrated Confidence metadata in Answer
    answer_meta = result["answer"]
    assert "calibrated_confidence" in answer_meta
    assert answer_meta["calibrated_confidence"] is not None
    calib = answer_meta["calibrated_confidence"]
    assert "score" in calib
    assert "entropy" in calib
    assert "consistency_score" in calib

    # Verify Evidence item persisted
    evidence = result["evidence"]
    assert "confidence_calibration" in evidence
    calib_ev = evidence["confidence_calibration"]
    assert calib_ev["is_consistent"] is True

    # Verify Execution Trace Audit Step
    trace_res = client.get(f"/api/analysis/{analysis_id}/trace")
    assert trace_res.status_code == 200
    trace_steps = trace_res.json()["trace"]["steps"]
    actions = [s["action"] for s in trace_steps]
    assert any("Evidence Consistency & Hallucination Audit" in a for a in actions)
