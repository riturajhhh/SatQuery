"""API integration tests for visual grounding workflow and overlay delivery."""

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


def create_grounding_geotiff() -> io.BytesIO:
    buf = io.BytesIO()
    transform = from_origin(74.0, 15.0, 0.001, 0.001)
    data = np.full((3, 60, 60), 100, dtype=np.uint8)
    # High blue water region in center [15:45, 15:45]
    data[0, 15:45, 15:45] = 10
    data[1, 15:45, 15:45] = 30
    data[2, 15:45, 15:45] = 200

    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=60,
        width=60,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    buf.seek(0)
    return buf


def test_end_to_end_grounding_workflow(client: TestClient):
    # 1. Upload satellite image
    buf = create_grounding_geotiff()
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("reservoir_scene.tif", buf, "image/tiff"))],
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.json()["upload_id"]

    # 2. Submit visual grounding query
    query = "Highlight the water body visible in this satellite imagery."
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

    assert result["task"] == "grounding"
    assert result["status"] == "complete"
    assert "located and highlighted" in result["answer"]["text"].lower()

    # Evidence checks
    evidence = result["evidence"]
    assert evidence is not None
    # Check that overlay_url is present in grounding evidence
    grounding_ev = evidence.get("grounding_overlay") or evidence
    overlay_url = grounding_ev.get("overlay_url")
    assert overlay_url is not None
    assert overlay_url.startswith("/api/files/evidence/")
    assert len(grounding_ev.get("boxes", [])) >= 1

    # 4. Verify overlay image is accessible over HTTP
    img_res = client.get(overlay_url)
    assert img_res.status_code == 200
    assert len(img_res.content) > 100
