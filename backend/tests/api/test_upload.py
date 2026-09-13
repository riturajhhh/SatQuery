"""API integration tests for the /api/upload endpoint."""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
import rasterio
from rasterio.transform import from_origin

from app.main import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def create_mock_geotiff(
    left: float = 72.5,
    top: float = 23.5,
    pixel_size: float = 0.001,
    bands: int = 3,
) -> io.BytesIO:
    """Create in-memory GeoTIFF file."""
    buf = io.BytesIO()
    transform = from_origin(left, top, pixel_size, pixel_size)
    data = np.full((bands, 40, 40), 100, dtype=np.uint8)

    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=bands,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    buf.seek(0)
    return buf


def test_upload_invalid_extension(client: TestClient):
    """Attempting to upload an unsupported format should return HTTP 400."""
    response = client.post(
        "/api/upload",
        files=[("files", ("malicious.exe", b"binary content", "application/octet-stream"))],
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "validation_error"
    assert "Unsupported file format" in data["message"]


def test_upload_single_geotiff(client: TestClient):
    """Uploading a valid GeoTIFF extracts metadata and returns 200."""
    buf = create_mock_geotiff()
    response = client.post(
        "/api/upload",
        files=[("files", ("satellite_image.tif", buf, "image/tiff"))],
    )
    assert response.status_code == 200
    data = response.json()

    assert "upload_id" in data
    assert len(data["files"]) == 1
    file_info = data["files"][0]

    assert file_info["original_name"] == "satellite_image.tif"
    assert file_info["format"] == "GeoTIFF"
    assert file_info["width"] == 40
    assert file_info["height"] == 40
    assert file_info["bands"] == 3
    assert file_info["crs"] == "EPSG:4326"
    assert file_info["modality"] == "optical"
    assert file_info["preview_url"] is not None
    assert data["input_type"] == "single"
    assert data["compatibility"]["valid"] is True


def test_upload_paired_compatible(client: TestClient):
    """Uploading two overlapping images returns 200 and compatibility check."""
    buf1 = create_mock_geotiff(left=72.5, top=23.5)
    buf2 = create_mock_geotiff(left=72.505, top=23.505)

    response = client.post(
        "/api/upload",
        files=[
            ("files", ("image_t1.tif", buf1, "image/tiff")),
            ("files", ("image_t2.tif", buf2, "image/tiff")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2
    assert data["compatibility"]["valid"] is True
    assert data["compatibility"]["details"]["spatial_overlap_percent"] > 50


def test_upload_paired_incompatible(client: TestClient):
    """Uploading two non-overlapping images returns 422 compatibility error."""
    buf1 = create_mock_geotiff(left=72.0, top=23.0)
    buf2 = create_mock_geotiff(left=85.0, top=12.0)

    response = client.post(
        "/api/upload",
        files=[
            ("files", ("img_a.tif", buf1, "image/tiff")),
            ("files", ("img_b.tif", buf2, "image/tiff")),
        ],
    )
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "compatibility_error"
    assert "non-overlapping" in data["message"]
