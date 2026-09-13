"""Unit tests for Phase 12 — ISRO/SAC Evaluation Readiness.

Validates:
1. Cartosat-2S / Cartosat-3 optical imagery ingestion (11-bit uint16, 4-band VNIR, PAN 0.65m).
2. RISAT-1 / RISAT-1A (EOS-04) SAR imagery ingestion (C-band, hybrid RH/RV & linear pol, sigma0 dB calibration).
3. Indian Coordinate Reference Systems (Indian UTM Zones 42N-46N, Kalianpur 1975, LCC India).
4. Co-registration verification for Cartosat bi-temporal and Cartosat-RISAT optical-SAR pairs.
5. Resilient handling of unconventional evaluation rasters (unprojected pixel grids, NaN nodata, float32).
"""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.geospatial.isro import (
    ISROPlatform,
    ISROPolarization,
    identify_isro_metadata,
    calibrate_isro_radiometry,
    verify_coregistration,
)
from app.geospatial.metadata import extract_metadata
from app.geospatial.modality import detect_modality
from app.geospatial.preview import generate_preview_and_thumbnail
from app.geospatial.compatibility import validate_pair_compatibility


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "isro_test_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_mock_raster(
    path: Path,
    width: int = 128,
    height: int = 128,
    count: int = 4,
    dtype: str = "uint16",
    crs: str = "EPSG:32643",  # Indian UTM Zone 43N
    transform=None,
    data_gen=None,
    tags=None,
    nodata=None,
):
    """Helper to write test GeoTIFF rasters with custom metadata."""
    if transform is None:
        # Origin somewhere in Western/Central India (e.g. Gujarat/Ahmedabad vicinity: UTM 43N)
        transform = from_origin(300000, 2500000, 1.6, 1.6)

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }

    with rasterio.open(str(path), "w", **meta) as dst:
        for i in range(1, count + 1):
            if data_gen:
                arr = data_gen(i, height, width).astype(dtype)
            else:
                max_val = 250 if dtype == "uint8" else 1800
                arr = np.random.randint(10, max_val, size=(height, width), dtype=dtype)
            dst.write(arr, i)


        if tags:
            dst.update_tags(**tags)


class TestCartosatReadiness:
    """Test suite for Cartosat series optical imagery."""

    def test_cartosat_2s_multispectral_4band_uint16(self, temp_dir: Path):
        """Cartosat-2S 4-band VNIR (B1:Blue, B2:Green, B3:Red, B4:NIR) in Indian UTM Zone 43N."""
        file_path = temp_dir / "Cartosat_2S_MX_Ahmedabad.tif"

        # Simulate 11-bit radiance (0 to 2047)
        def gen(band_idx, h, w):
            base = band_idx * 300
            return np.clip(base + np.random.randint(20, 400, size=(h, w)), 0, 2047)

        create_mock_raster(
            file_path,
            width=256,
            height=256,
            count=4,
            dtype="uint16",
            crs="EPSG:32643",
            data_gen=gen,
            tags={
                "SPACECRAFT_NAME": "CARTOSAT-2S",
                "SENSOR": "4-BAND VNIR",
                "RESOLUTION": "1.6",
            },
        )

        meta = extract_metadata(file_path)

        assert meta["is_georeferenced"] is True
        assert meta["crs"] == "EPSG:32643"
        assert meta["bands"] == 4
        assert meta["dtype"] == "uint16"
        assert meta["bounds"] is not None

        # Verify ISRO identification
        isro = meta.get("isro")
        assert isro is not None
        assert isro["is_isro"] is True
        assert isro["platform"] == ISROPlatform.CARTOSAT_2S.value
        assert isro["gsd_nominal_m"] == 1.6
        assert isro["bit_depth_effective"] == 11
        assert isro["rgb_band_indices"] == [3, 2, 1]  # Red=B3, Green=B2, Blue=B1
        assert "Zone 43N" in isro["indian_crs_zone"]

        # Verify modality detection
        modality = detect_modality(meta, filename=file_path.name)
        assert modality == "multispectral"

        # Verify preview generation runs and stretches 11-bit properly
        prev, thumb = generate_preview_and_thumbnail(file_path, temp_dir, "carto_test")
        assert prev.exists()
        assert thumb.exists()
        assert prev.stat().st_size > 0

    def test_cartosat_2s_panchromatic_highres(self, temp_dir: Path):
        """Cartosat-2S 0.65m Panchromatic single-band uint16."""
        file_path = temp_dir / "c2s_pan_0.65m.tif"

        transform = from_origin(450000, 2100000, 0.65, 0.65)
        create_mock_raster(
            file_path,
            width=128,
            height=128,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=transform,
            tags={"SPACECRAFT_NAME": "Cartosat-2S", "SENSOR": "PAN"},
        )

        meta = extract_metadata(file_path)
        assert meta["bands"] == 1
        assert meta["isro"]["platform"] == ISROPlatform.CARTOSAT_2S.value
        assert meta["isro"]["gsd_nominal_m"] == 0.65

        # Should detect as optical panchromatic, NOT sar
        modality = detect_modality(meta, filename=file_path.name)
        assert modality == "optical"


class TestRISATSARReadiness:
    """Test suite for RISAT-1 / RISAT-1A (EOS-04) SAR imagery."""

    def test_risat_1a_cband_hybrid_polarization(self, temp_dir: Path):
        """RISAT-1A (EOS-04) C-band SAR with Circular/Hybrid (RH/RV) polarimetry."""
        file_path = temp_dir / "RISAT1A_EOS04_FRS_Ahmedabad.tif"

        # 2 bands: RH (Right circular transmit, Horizontal receive), RV (Vertical receive)
        def gen(band_idx, h, w):
            # Rayleigh-distributed SAR speckle
            return np.random.exponential(scale=200.0, size=(h, w)) + 50

        create_mock_raster(
            file_path,
            width=128,
            height=128,
            count=2,
            dtype="uint16",
            crs="EPSG:32644",  # Indian UTM Zone 44N
            data_gen=gen,
            tags={
                "MISSION": "RISAT-1A",
                "PAYLOAD": "C-BAND SAR",
                "BEAM_MODE": "FRS-1",
                "POLARISATION": "RH/RV",
                "CALIBRATION_CONSTANT": "40.2",
            },
        )

        meta = extract_metadata(file_path)
        isro = meta.get("isro")
        assert isro is not None
        assert isro["platform"] == ISROPlatform.RISAT_1A_EOS04.value
        assert isro["polarization"] == ISROPolarization.CIRCULAR_HYBRID_RH_RV.value
        assert "Zone 44N" in isro["indian_crs_zone"]

        modality = detect_modality(meta, filename=file_path.name)
        assert modality == "sar"

        # Test SAR radiometry sigma0 dB calibration
        raw_band = np.array([[10, 100], [500, 2000]], dtype=np.uint16)
        calibrated = calibrate_isro_radiometry(
            raw_band, is_sar=True, calibration_constant=40.2
        )
        assert calibrated.shape == (2, 2)
        assert calibrated.dtype == np.uint8
        # Higher DN should result in higher calibrated output
        assert calibrated[1, 1] > calibrated[0, 0]

    def test_risat_linear_pol_single_band(self, temp_dir: Path):
        """RISAT-1 linear polarization (HH/HV) single-band backscatter."""
        file_path = temp_dir / "risat1_frs_hh_cal.tif"
        create_mock_raster(
            file_path,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            tags={"SATELLITE": "RISAT-1", "POLARISATION": "HH"},
        )

        meta = extract_metadata(file_path)
        assert meta["isro"]["platform"] == ISROPlatform.RISAT_1.value
        modality = detect_modality(meta, filename=file_path.name)
        assert modality == "sar"


class TestCoRegistrationAndPairing:
    """Test suite for sub-pixel co-registration verification and cross-modal pair processing."""

    def test_pre_coregistered_cartosat_pair(self, temp_dir: Path):
        """Two Cartosat acquisitions over the exact same footprint (t1 and t2)."""
        transform = from_origin(310000, 2510000, 1.6, 1.6)

        t1_path = temp_dir / "cartosat2s_2023_t1.tif"
        t2_path = temp_dir / "cartosat2s_2024_t2.tif"

        create_mock_raster(t1_path, width=100, height=100, count=4, transform=transform)
        create_mock_raster(t2_path, width=100, height=100, count=4, transform=transform)

        meta_t1 = extract_metadata(t1_path)
        meta_t2 = extract_metadata(t2_path)

        coreg = verify_coregistration(meta_t1, meta_t2)
        assert coreg["is_coregistered"] is True
        assert coreg["alignment_type"] == "sub_pixel_coregistered"
        assert coreg["pixel_offset_estimate"] == 0.0

        compat = validate_pair_compatibility(meta_t1, meta_t2, "multispectral", "multispectral")
        assert compat["valid"] is True
        assert compat["details"]["is_coregistered"] is True

    def test_optical_sar_pair_cartosat_and_risat(self, temp_dir: Path):
        """Co-registered Cartosat-2S Optical and RISAT-1A SAR image pair."""
        transform = from_origin(310000, 2510000, 2.0, 2.0)

        opt_path = temp_dir / "cartosat_opt.tif"
        sar_path = temp_dir / "risat_sar.tif"

        create_mock_raster(opt_path, width=150, height=150, count=4, transform=transform)
        create_mock_raster(sar_path, width=150, height=150, count=1, transform=transform)

        meta_opt = extract_metadata(opt_path)
        meta_sar = extract_metadata(sar_path)

        compat = validate_pair_compatibility(meta_opt, meta_sar, "multispectral", "sar")
        assert compat["valid"] is True
        assert compat["details"]["workflow_recommendation"] == "optical_sar_fusion"
        assert compat["details"]["is_coregistered"] is True


class TestEvaluationResilience:
    """Test resilience against unconventional evaluation formats (zero crashes)."""

    def test_unprojected_evaluation_tile(self, temp_dir: Path):
        """Evaluation raster with no CRS or geotransform (raw pixel grid)."""
        file_path = temp_dir / "unprojected_cartosat_tile_01.tif"

        # Create raster with None CRS and identity-like pixel transform
        pixel_transform = from_origin(0, 200, 1.0, 1.0)
        create_mock_raster(
            file_path,
            width=200,
            height=200,
            count=3,
            crs="",
            dtype="uint8",
            transform=pixel_transform,
        )

        meta = extract_metadata(file_path)
        assert meta["is_georeferenced"] is False
        assert meta["crs"] is None
        assert meta["bounds"] is not None
        assert meta["bounds"]["right"] == 200.0
        assert meta["bounds"]["left"] == 0.0


        # Preview generation should succeed without error
        prev, thumb = generate_preview_and_thumbnail(file_path, temp_dir, "unproj_test")
        assert prev.exists()
        assert thumb.exists()

    def test_unprojected_pair_coregistration(self, temp_dir: Path):
        """Two unprojected evaluation tiles with matching matrix dimensions."""
        p1 = temp_dir / "unproj_t1.tif"
        p2 = temp_dir / "unproj_t2.tif"

        create_mock_raster(p1, width=128, height=128, count=3, crs="")
        create_mock_raster(p2, width=128, height=128, count=3, crs="")

        m1 = extract_metadata(p1)
        m2 = extract_metadata(p2)

        coreg = verify_coregistration(m1, m2)
        assert coreg["is_coregistered"] is True
        assert coreg["alignment_type"] == "pixel_grid_exact"

    def test_float32_with_nan_nodata(self, temp_dir: Path):
        """Raster with Float32 backscatter or index data containing NaNs and infinite values."""
        file_path = temp_dir / "risat_sigma0_float_nan.tif"

        def nan_gen(band_idx, h, w):
            arr = np.random.uniform(-30.0, 5.0, size=(h, w)).astype(np.float32)
            arr[0:10, 0:10] = np.nan
            arr[10:15, 10:15] = -9999.0
            return arr

        create_mock_raster(
            file_path,
            width=64,
            height=64,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            data_gen=nan_gen,
            tags={"SATELLITE": "RISAT-1A"},
        )

        meta = extract_metadata(file_path)
        assert meta["dtype"] == "float32"

        # Preview generation must handle NaNs without raising exceptions
        prev, thumb = generate_preview_and_thumbnail(file_path, temp_dir, "nan_test")
        assert prev.exists()
        assert thumb.exists()
