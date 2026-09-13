"""SatQuery AI — Imagery Modality Detection.

Infers the remote sensing sensor modality (optical, multispectral, SAR)
from band count, data distribution, metadata tags, and filename heuristics.
"""

from typing import Any, Dict, Optional
from pathlib import Path


SAR_KEYWORDS = [
    "sar", "s1", "sentinel1", "sentinel-1", "risat", "risat1", "risat1a", "eos04", "eos-04",
    "alos", "terrasar", "radarsat", "slc", "grd", "vv", "vh", "hh", "hv", "rh", "rv"
]
OPTICAL_KEYWORDS = [
    "optical", "s2", "sentinel2", "sentinel-2", "landsat", "planet",
    "cartosat", "cartosat2s", "cartosat-2s", "cartosat3", "cartosat-3", "c2s", "c3",
    "resourcesat", "naip", "spot"
]


def detect_modality(
    metadata: Dict[str, Any],
    filename: Optional[str] = None,
    user_hint: Optional[str] = None,
) -> str:
    """Detect image modality.

    Args:
        metadata: Metadata dictionary extracted by extract_metadata.
        filename: Original file name.
        user_hint: Optional explicit modality hint from caller.

    Returns:
        One of 'optical', 'multispectral', 'sar'.
    """
    if user_hint and user_hint.lower() in ("optical", "multispectral", "sar"):
        return user_hint.lower()

    bands = metadata.get("bands", 3)
    name_lower = (filename or "").lower()
    tags = metadata.get("tags") or {}
    tags_str = " ".join(f"{k} {v}" for k, v in tags.items()).lower()

    # Check filename and tags for SAR or Optical indicators
    has_sar_keyword = any(k in name_lower or k in tags_str for k in SAR_KEYWORDS)
    has_optical_keyword = any(k in name_lower or k in tags_str for k in OPTICAL_KEYWORDS)

    # If identified as ISRO payload, leverage sensor type
    isro_info = metadata.get("isro")
    if isro_info:
        stype = (isro_info.get("sensor_type") or "").lower()
        if "sar" in stype:
            return "sar"
        elif "multispectral" in stype or bands >= 4:
            return "multispectral"
        else:
            return "optical"

    # 1-band images: PAN optical or SAR
    if bands == 1:
        if has_optical_keyword and not has_sar_keyword:
            # e.g., Cartosat-2S PAN 0.65m
            return "optical"
        if has_sar_keyword:
            return "sar"
        return "sar"

    # 2-band images: often dual-pol SAR (e.g. RISAT RH/RV or HH/HV)
    if bands == 2:
        if has_sar_keyword or not has_optical_keyword:
            return "sar"
        return "optical"

    # 3-band images
    if bands == 3:
        if has_sar_keyword and not has_optical_keyword:
            # e.g., SAR false color composite (VV/VH/ratio or RH/RV/m-chi)
            return "sar"
        return "optical"

    # 4 or more bands
    if bands >= 4:
        if has_sar_keyword and not has_optical_keyword and bands == 4:
            # Quad-pol SAR (HH, HV, VH, VV)
            return "sar"
        return "multispectral"

    return "optical"
