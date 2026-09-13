"""API tests for Phase 13 Report & Audit endpoints.

Tests:
- GET /api/analysis/{id}/report?format=html
- GET /api/analysis/{id}/report?format=pdf
- GET /api/analysis/{id}/report?format=invalid
- GET /api/analysis/history
"""

import io
import pytest
from starlette.testclient import TestClient
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from app.main import app


def _create_synthetic_geotiff(
    bounds=(-122.4, 37.7, -122.3, 37.8),
    crs: str = "EPSG:4326",
) -> io.BytesIO:
    """Create a minimal in-memory GeoTIFF."""
    width, height = 64, 64
    transform = from_bounds(*bounds, width, height)
    data = np.random.randint(20, 200, size=(3, height, width), dtype=np.uint8)

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


@pytest.fixture
def completed_analysis_id(client):
    """Fixture that performs an upload and query to generate a valid completed analysis."""
    tif = _create_synthetic_geotiff()
    upload_res = client.post(
        "/api/upload",
        files=[("files", ("audit_scene.tif", tif.getvalue(), "image/tiff"))],
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    upload_id = upload_data["upload_id"]

    analyze_res = client.post(
        "/api/analyze",
        json={
            "upload_id": upload_id,
            "query": "Describe the scene layout and detect key land features.",
        },
    )
    assert analyze_res.status_code == 200
    return analyze_res.json()["analysis_id"]


def test_get_analysis_history(client, completed_analysis_id):
    """Test retrieving paginated list of historical analyses."""
    res = client.get("/api/analysis/history?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()

    assert "total" in data
    assert "items" in data
    assert data["total"] >= 1
    assert len(data["items"]) >= 1

    first = data["items"][0]
    assert "analysis_id" in first
    assert "query" in first
    assert "status" in first
    assert "confidence" in first


def test_get_analysis_html_report(client, completed_analysis_id):
    """Test downloading or viewing HTML audit report."""
    res = client.get(f"/api/analysis/{completed_analysis_id}/report?format=html")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    html_text = res.text
    assert "<!DOCTYPE html>" in html_text
    assert "SatQuery AI" in html_text
    assert completed_analysis_id in html_text


def test_get_analysis_pdf_report(client, completed_analysis_id):
    """Test downloading PDF audit report."""
    res = client.get(f"/api/analysis/{completed_analysis_id}/report?format=pdf")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert "attachment" in res.headers.get("content-disposition", "")
    assert res.content.startswith(b"%PDF-")
    assert len(res.content) > 1000


def test_get_analysis_report_invalid_format(client, completed_analysis_id):
    """Test invalid format request returns 400."""
    res = client.get(f"/api/analysis/{completed_analysis_id}/report?format=docx")
    assert res.status_code == 400
    assert "Unsupported report format" in res.text


def test_get_analysis_report_not_found(client):
    """Test report for non-existent analysis ID returns 404."""
    res = client.get("/api/analysis/nonexistent_id_404/report?format=pdf")
    assert res.status_code == 404
