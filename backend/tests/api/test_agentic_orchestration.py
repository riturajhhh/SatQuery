"""API integration tests for Phase 9 — Agentic Orchestration workflow."""

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
        # T2: Land clearance
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


def test_agentic_multi_step_workflow(client):
    """Test full upload -> multi-step decomposed query (grounding + change detection) -> verify trace and multiple evidence records."""
    # 1. Prepare 2 temporal GeoTIFFs
    bounds = (-122.4, 37.7, -122.3, 37.8)
    tif1 = _create_synthetic_geotiff(bounds, crs="EPSG:4326", is_t2=False)
    tif2 = _create_synthetic_geotiff(bounds, crs="EPSG:4326", is_t2=True)

    # 2. Upload pair
    upload_res = client.post(
        "/api/upload",
        files=[
            ("files", ("t1.tif", tif1.getvalue(), "image/tiff")),
            ("files", ("t2.tif", tif2.getvalue(), "image/tiff")),
        ],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]
    file_ids = [f["file_id"] for f in upload_data["files"]]
    assert len(file_ids) == 2

    # 3. Submit composite query: Grounding + Change Detection
    query = "Locate the forest and analyze temporal changes over time"
    analyze_res = client.post(
        "/api/analyze",
        json={
            "upload_id": upload_id,
            "file_ids": file_ids,
            "query": query,
        },
    )
    assert analyze_res.status_code == 200
    analyze_data = analyze_res.json()
    analysis_id = analyze_data["analysis_id"]

    # 4. Fetch analysis results
    result_res = client.get(f"/api/analysis/{analysis_id}")
    assert result_res.status_code == 200
    result = result_res.json()

    assert result["task"] == "agentic_multi_task"
    assert result["status"] == "complete"
    assert result["answer"]["confidence"] >= 0.75


    # Verify multiple models were coordinated
    models_used = result["models_used"]
    assert len(models_used) >= 2
    tasks_run = [m["task"] for m in models_used]
    assert "grounding" in tasks_run
    assert "change_detection" in tasks_run

    # Verify multiple evidence items persisted
    evidence = result["evidence"]
    assert "grounding_overlay" in evidence
    assert "change_detection_map" in evidence

    # Verify execution trace endpoint
    trace_res = client.get(f"/api/analysis/{analysis_id}/trace")
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    steps = trace_data["trace"]["steps"]
    actions = [s["action"] for s in steps]
    assert any("Agentic Planning" in a for a in actions)
    assert any("Tool Execution" in a for a in actions)
    assert any("Cross-Task Synthesis" in a for a in actions)

    # Verify synthesized answer contains combined details
    answer = result["answer"]["text"]
    assert len(answer) > 20
    assert "located" in answer.lower() or "vegetation" in answer.lower()
