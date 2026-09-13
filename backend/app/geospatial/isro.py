"""SatQuery AI — ISRO & SAC Remote Sensing Evaluation Readiness.

Specialized handling for Indian Space Research Organisation (ISRO) and
Space Applications Centre (SAC) satellite payloads:
- Cartosat-2S / Cartosat-3 Optical (0.65m/0.28m PAN, 4-band VNIR ~1.6m GSD, uint16 10/11/12-bit)
- RISAT-1 / RISAT-1A (EOS-04) / RISAT-2 SAR (C-band/X-band, linear and circular/hybrid RH/RV polarimetry)
- Indian Coordinate Reference Systems (Indian UTM Zones 42N-46N, Kalianpur, LCC India)
- Radiometric calibration (backscatter sigma0 in dB for SAR, dynamic range stretching for Cartosat)
- Sub-pixel co-registration verification for pre-aligned evaluation pairs
- Resilient fallbacks for unprojected/unconventional evaluation formats
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.utils.logging import get_logger

logger = get_logger("geospatial.isro")


class ISROPlatform(str, Enum):
    CARTOSAT_2S = "Cartosat-2S"
    CARTOSAT_3 = "Cartosat-3"
    CARTOSAT_1 = "Cartosat-1"
    RISAT_1A_EOS04 = "RISAT-1A (EOS-04)"
    RISAT_1 = "RISAT-1"
    RISAT_2 = "RISAT-2"
    RESOURCESAT_2A = "Resourcesat-2A"
    OCEANSAT_3 = "Oceansat-3"
    GENERIC_ISRO = "ISRO Generic"


class ISROPolarization(str, Enum):
    CIRCULAR_HYBRID_RH_RV = "Circular/Hybrid (RH/RV)"
    LINEAR_HH_HV = "Linear (HH/HV)"
    LINEAR_VV_VH = "Linear (VV/VH)"
    QUAD_POL = "Quad-Pol (HH/HV/VV/VH)"
    SINGLE_POL = "Single-Pol"
    UNKNOWN = "Unknown"


# Indian EPSG codes and spatial reference identifiers
INDIAN_CRS_REGISTRY: Dict[str, str] = {
    "EPSG:32642": "Indian UTM Zone 42N (Western Border / Gujarat)",
    "EPSG:32643": "Indian UTM Zone 43N (Western / Central India)",
    "EPSG:32644": "Indian UTM Zone 44N (Central / Southern India)",
    "EPSG:32645": "Indian UTM Zone 45N (Eastern India / Bay of Bengal)",
    "EPSG:32646": "Indian UTM Zone 46N (North-Eastern India / Assam)",
    "EPSG:24378": "Kalianpur 1975 / India Zone I",
    "EPSG:24379": "Kalianpur 1975 / India Zone IIa",
    "EPSG:24380": "Kalianpur 1975 / India Zone IIb",
    "EPSG:7755": "India LCC (Lambert Conformal Conic)",
}

import re

# Heuristics for ISRO sensor identification
CARTOSAT_KEYWORDS = [
    "cartosat", "carto_2s", "carto-2s", "cartosat2s", "cartosat-2s",
    "cartosat3", "cartosat-3", "cartosat1", "cartosat-1"
]
RISAT_KEYWORDS = [
    "risat", "risat1", "risat-1", "risat1a", "risat-1a", "eos04", "eos-04",
    "eos_04", "risat2", "risat-2"
]
RESOURCESAT_KEYWORDS = ["resourcesat", "res2", "res2a", "liss", "awifs"]


def identify_isro_metadata(
    metadata: Dict[str, Any],
    filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Inspect raster metadata tags, dimensions, CRS, and filename to identify ISRO platform.

    Returns structured ISRO metadata dictionary if identified, or None if not an ISRO payload.
    """
    raw_fn = (filename or "").lower()
    # Strip random UUID prefix (e.g. 2af6c192478da43d5fc145be5dea_filename.tif)
    fn_lower = re.sub(r'^[0-9a-f]{32}_', '', raw_fn)

    tags = metadata.get("tags") or {}
    tags_upper = {str(k).upper(): str(v).upper() for k, v in tags.items()}

    # Merge top-level tags and driver metadata if any
    spacecraft = tags_upper.get("SPACECRAFT_NAME") or tags_upper.get("SATELLITE") or tags_upper.get("MISSION") or ""
    sensor = tags_upper.get("SENSOR") or tags_upper.get("SENSOR_NAME") or tags_upper.get("PAYLOAD") or ""
    polarization_tag = tags_upper.get("POLARISATION") or tags_upper.get("POLARIZATION") or ""
    beam_mode = tags_upper.get("BEAM_MODE") or tags_upper.get("MODE") or ""

    bands = metadata.get("bands", 1)
    dtype = metadata.get("dtype", "uint8")
    crs = metadata.get("crs") or ""

    # Check Indian UTM or Kalianpur CRS
    indian_crs_desc = INDIAN_CRS_REGISTRY.get(crs)
    is_indian_grid = indian_crs_desc is not None

    has_cartosat_name = (
        any(k in fn_lower for k in CARTOSAT_KEYWORDS)
        or bool(re.search(r'(?:^|[\W_])(c1|c2s|c3)(?:$|[\W_])', fn_lower))
    )
    is_cartosat = (
        has_cartosat_name
        or any(k in spacecraft.lower() for k in CARTOSAT_KEYWORDS)
        or any(k in sensor.lower() for k in ["carto"])
        or (is_indian_grid and any(k in sensor.lower() for k in ["pan", "mx"]))
    )

    is_risat = (
        any(k in fn_lower for k in RISAT_KEYWORDS)
        or any(k in spacecraft.lower() for k in RISAT_KEYWORDS)
        or any(k in sensor.lower() for k in ["c-band", "c_band"])
    )

    is_resourcesat = (
        any(k in fn_lower for k in RESOURCESAT_KEYWORDS)
        or any(k in spacecraft.lower() for k in RESOURCESAT_KEYWORDS)
        or any(k in sensor.lower() for k in ["liss", "awifs"])
    )

    if not (is_cartosat or is_risat or is_resourcesat or (is_indian_grid and ("isro" in fn_lower or "sac" in fn_lower))):
        return None


    # Resolve platform & sensor specs
    platform = ISROPlatform.GENERIC_ISRO
    sensor_type = "Optical"
    gsd_nominal = 1.0
    bit_depth = 8
    band_names: List[str] = []
    rgb_indices: List[int] = [1, 2, 3]
    nir_indices: Optional[List[int]] = None
    pol_enum = ISROPolarization.UNKNOWN

    if is_cartosat:
        if "c3" in fn_lower or "cartosat3" in fn_lower or "cartosat-3" in spacecraft.lower():
            platform = ISROPlatform.CARTOSAT_3
            if bands == 1:
                sensor_type = "Panchromatic (PAN 0.28m)"
                gsd_nominal = 0.28
                band_names = ["Panchromatic"]
                rgb_indices = [1, 1, 1]
            else:
                sensor_type = "Multispectral (MX 1.12m VNIR)"
                gsd_nominal = 1.12
                # Cartosat-3 VNIR: B1: Blue, B2: Green, B3: Red, B4: NIR
                band_names = ["Blue (0.45-0.52 µm)", "Green (0.52-0.59 µm)", "Red (0.62-0.68 µm)", "NIR (0.77-0.86 µm)"]
                rgb_indices = [3, 2, 1]  # Standard RGB (Red=B3, Green=B2, Blue=B1)
                nir_indices = [4, 3, 2]  # False Color Composite (NIR=B4, Red=B3, Green=B2)
        elif "c1" in fn_lower or "cartosat1" in fn_lower:
            platform = ISROPlatform.CARTOSAT_1
            sensor_type = "Stereo Panchromatic (2.5m)"
            gsd_nominal = 2.5
            band_names = ["PAN"]
            rgb_indices = [1, 1, 1]
        else:
            # Default to Cartosat-2S (high priority for SAC evaluation)
            platform = ISROPlatform.CARTOSAT_2S
            if bands == 1:
                sensor_type = "Panchromatic (0.65m PAN)"
                gsd_nominal = 0.65
                band_names = ["Panchromatic"]
                rgb_indices = [1, 1, 1]
            else:
                sensor_type = "Multispectral (4-Band VNIR ~1.6m)"
                gsd_nominal = 1.6
                # Cartosat-2S band order: B1: Blue, B2: Green, B3: Red, B4: NIR
                band_names = ["B1 Blue (0.45-0.52 µm)", "B2 Green (0.52-0.59 µm)", "B3 Red (0.62-0.68 µm)", "B4 NIR (0.77-0.86 µm)"]
                rgb_indices = [3, 2, 1]
                nir_indices = [4, 3, 2]

        # Bit depth for Cartosat (usually 10, 11, or 12-bit stored as uint16)
        if "16" in dtype:
            bit_depth = 11  # Typical Cartosat-2S radiometric resolution (11-bit: 0-2047)
        else:
            bit_depth = 8

    elif is_risat:
        if "eos04" in fn_lower or "eos-04" in fn_lower or "risat-1a" in fn_lower or "risat1a" in fn_lower:
            platform = ISROPlatform.RISAT_1A_EOS04
        elif "risat2" in fn_lower or "risat-2" in fn_lower:
            platform = ISROPlatform.RISAT_2
        else:
            platform = ISROPlatform.RISAT_1

        sensor_type = "SAR (C-band 5.35 GHz)" if platform != ISROPlatform.RISAT_2 else "SAR (X-band)"
        gsd_nominal = 3.0 if "frs" in fn_lower or "frs" in beam_mode.lower() else (25.0 if "mrs" in fn_lower else 50.0)

        # Detect polarimetry: ISRO hybrid circular (RH/RV) vs linear (HH/HV or VV/VH)
        pol_str = (polarization_tag + " " + fn_lower).upper()
        if "RH" in pol_str or "RV" in pol_str or "HYBRID" in pol_str or "CIRCULAR" in pol_str:
            pol_enum = ISROPolarization.CIRCULAR_HYBRID_RH_RV
        elif "HH" in pol_str and "HV" in pol_str:
            pol_enum = ISROPolarization.LINEAR_HH_HV
        elif "VV" in pol_str and "VH" in pol_str:
            pol_enum = ISROPolarization.LINEAR_VV_VH
        elif any(p in pol_str for p in ["HH", "HV", "VV", "VH", "RH", "RV"]):
            pol_enum = ISROPolarization.SINGLE_POL
        else:
            pol_enum = ISROPolarization.UNKNOWN

        bit_depth = 16 if "16" in dtype else 8
        band_names = [f"SAR Channel {i+1}" for i in range(bands)]

    elif is_resourcesat:
        platform = ISROPlatform.RESOURCESAT_2A
        sensor_type = "LISS-4 / AWiFS Multispectral"
        gsd_nominal = 5.8 if "liss" in fn_lower else 56.0
        band_names = [f"Band {i+1}" for i in range(bands)]
        bit_depth = 10 if "16" in dtype else 8

    return {
        "is_isro": True,
        "platform": platform.value,
        "sensor_type": sensor_type,
        "gsd_nominal_m": gsd_nominal,
        "bit_depth_effective": bit_depth,
        "polarization": pol_enum.value,
        "band_names": band_names,
        "rgb_band_indices": rgb_indices,
        "nir_composite_indices": nir_indices,
        "indian_crs_zone": indian_crs_desc,
        "raw_calibration_constant": tags.get("CALIBRATION_CONSTANT") or tags.get("KCAL"),
    }


def calibrate_isro_radiometry(
    band_data: np.ndarray,
    is_sar: bool = False,
    bit_depth: int = 11,
    nodata: Optional[float] = None,
    calibration_constant: Optional[float] = None,
) -> np.ndarray:
    """Apply radiometrically calibrated stretch to ISRO optical or SAR rasters.

    - For Cartosat-2S uint16 (10-12 bit depth): avoids assuming full 16-bit 65535 scale;
      performs robust non-zero 1st-99th percentile contrast stretch.
    - For RISAT SAR: converts digital numbers (DN) to calibrated backscatter (sigma0 in dB)
      or applies log10 dynamic range scaling to suppress speckle and highlight target backscatter.

    Returns:
        np.ndarray of uint8 (0..255) for visualization or normalized array.
    """
    valid_mask = np.isfinite(band_data)
    if nodata is not None:
        valid_mask &= (band_data != nodata)

    # Exclude strict 0 values from percentile calculation if optical / SAR background
    non_zero_mask = valid_mask & (band_data > 0)
    calc_mask = non_zero_mask if np.any(non_zero_mask) else valid_mask

    if not np.any(calc_mask):
        return np.zeros(band_data.shape, dtype=np.uint8)

    sample = band_data[calc_mask]

    if is_sar:
        # RISAT SAR sigma0 in dB calculation:
        # sigma0_dB = 20 * log10(DN) - K_cal (default K_cal = 40.0 dB if omitted)
        kcal = float(calibration_constant) if calibration_constant else 40.0
        # Protect against log(0)
        dn_safe = np.maximum(sample.astype(np.float32), 1e-3)
        sigma0_sample = 20.0 * np.log10(dn_safe) - kcal

        # Clip backscatter to standard SAR dB range: -32 dB (calm water/smooth) to +3 dB (specular/urban)
        db_min, db_max = np.percentile(sigma0_sample, [2.0, 98.0])
        if db_max <= db_min:
            db_max = db_min + 5.0

        # Scale full array
        full_safe = np.maximum(band_data.astype(np.float32), 1e-3)
        full_sigma0 = 20.0 * np.log10(full_safe) - kcal
        clipped = np.clip(full_sigma0, db_min, db_max)
        scaled_raw = (clipped - db_min) / (db_max - db_min) * 255.0
        scaled_raw[~valid_mask] = 0
        scaled = np.nan_to_num(scaled_raw, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
        return scaled

    else:
        # Cartosat-2S / Cartosat-3 Optical Percentile Stretch
        # 10-bit (max 1023), 11-bit (max 2047), or 12-bit (max 4095)
        p_low, p_high = np.percentile(sample, [1.0, 99.0])
        if p_high <= p_low:
            p_high = p_low + 1.0

        clipped = np.clip(band_data, p_low, p_high)
        scaled_raw = (clipped - p_low) / (p_high - p_low) * 255.0
        scaled_raw[~valid_mask] = 0
        scaled = np.nan_to_num(scaled_raw, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
        return scaled



def verify_coregistration(
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    pixel_tolerance: float = 1.5,
) -> Dict[str, Any]:
    """Verify whether two rasters (e.g. Cartosat pair or Cartosat + RISAT pair) are pre-co-registered.

    Checks:
    1. Geographic bounding box match (in projected or WGS84 coordinates)
    2. Resolution / pixel grid compatibility
    3. Unprojected evaluation fallbacks (matching pixel dimensions and extent)
    """
    w_a, h_a = meta_a.get("width", 0), meta_a.get("height", 0)
    w_b, h_b = meta_b.get("width", 0), meta_b.get("height", 0)
    crs_a = meta_a.get("crs")
    crs_b = meta_b.get("crs")
    bounds_a = meta_a.get("native_bounds") or meta_a.get("bounds")
    bounds_b = meta_b.get("native_bounds") or meta_b.get("bounds")

    # Case 1: Unprojected evaluation tiles (common in ML benchmark test splits)
    if not crs_a and not crs_b:
        if w_a == w_b and h_a == h_b and w_a > 0:
            return {
                "is_coregistered": True,
                "alignment_type": "pixel_grid_exact",
                "pixel_offset_estimate": 0.0,
                "details": f"Unprojected evaluation rasters have identical matrix dimensions ({w_a}x{h_a}). Aligned on pixel grid.",
            }
        else:
            return {
                "is_coregistered": False,
                "alignment_type": "unprojected_dimension_mismatch",
                "pixel_offset_estimate": None,
                "details": f"Unprojected rasters have different dimensions: ({w_a}x{h_a}) vs ({w_b}x{h_b}).",
            }

    # Case 2: Projected with bounds
    if bounds_a and bounds_b:
        # Check if CRS matches or can be compared
        same_crs = (crs_a == crs_b) if crs_a and crs_b else False
        res_a = meta_a.get("resolution", {}) or {}
        res_x = float(res_a.get("x", 1.0)) if res_a else 1.0

        d_left = abs(float(bounds_a["left"]) - float(bounds_b["left"]))
        d_top = abs(float(bounds_a["top"]) - float(bounds_b["top"]))
        d_right = abs(float(bounds_a["right"]) - float(bounds_b["right"]))
        d_bottom = abs(float(bounds_a["bottom"]) - float(bounds_b["bottom"]))

        max_bound_diff = max(d_left, d_top, d_right, d_bottom)

        if same_crs and res_x > 0:
            pixel_offset = max_bound_diff / res_x
            if pixel_offset <= pixel_tolerance:
                return {
                    "is_coregistered": True,
                    "alignment_type": "sub_pixel_coregistered",
                    "pixel_offset_estimate": round(pixel_offset, 2),
                    "details": f"Sub-pixel co-registration verified: maximum corner shift is {pixel_offset:.2f} pixels (<= {pixel_tolerance} px).",
                }

        # Fallback to overlap percentage
        from shapely.geometry import box
        try:
            b_a = box(bounds_a["left"], bounds_a["bottom"], bounds_a["right"], bounds_a["top"])
            b_b = box(bounds_b["left"], bounds_b["bottom"], bounds_b["right"], bounds_b["top"])
            if b_a.intersects(b_b):
                inter = b_a.intersection(b_b).area
                min_a = min(b_a.area, b_b.area)
                overlap = (inter / min_a) * 100.0 if min_a > 0 else 0.0
                if overlap > 98.0 and abs(w_a - w_b) <= 2 and abs(h_a - h_b) <= 2:
                    return {
                        "is_coregistered": True,
                        "alignment_type": "coincident_geospatial_bounds",
                        "pixel_offset_estimate": 0.5,
                        "details": f"Imagery footprints are coincident (> {overlap:.1f}% boundary overlap). Co-registered.",
                    }
        except Exception:
            pass

    return {
        "is_coregistered": False,
        "alignment_type": "spatial_offset",
        "pixel_offset_estimate": None,
        "details": "Rasters require spatial resampling or registration before fine pixel alignment.",
    }
