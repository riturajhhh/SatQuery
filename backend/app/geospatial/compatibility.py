"""SatQuery AI — Image Pair Compatibility Validator.

Validates spatial, temporal, and sensor compatibility between two uploaded
satellite images for bi-temporal change detection and cross-modal Optical-SAR workflows.
"""

from typing import Any, Dict, Optional
from shapely.geometry import box

from app.geospatial.isro import verify_coregistration


def validate_pair_compatibility(
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    modality_a: str,
    modality_b: str,
) -> Dict[str, Any]:
    """Validate spatial, resolution, and modality compatibility between two images.

    Args:
        meta_a: Metadata dict for image A.
        meta_b: Metadata dict for image B.
        modality_a: Modality of image A (optical, multispectral, sar).
        modality_b: Modality of image B (optical, multispectral, sar).

    Returns:
        Dict with 'valid' (bool), 'message' (str), and detailed checks.
    """
    # Verify sub-pixel co-registration or pixel-grid alignment
    coreg_check = verify_coregistration(meta_a, meta_b)

    details: Dict[str, Any] = {
        "modality_a": modality_a,
        "modality_b": modality_b,
        "spatial_overlap_percent": None,
        "resolution_ratio": None,
        "workflow_recommendation": None,
        "coregistration": coreg_check,
        "is_coregistered": coreg_check.get("is_coregistered", False),
    }


    # Modality analysis
    is_cross_modal = (modality_a == "sar" and modality_b in ("optical", "multispectral")) or \
                     (modality_b == "sar" and modality_a in ("optical", "multispectral"))

    if is_cross_modal:
        details["workflow_recommendation"] = "optical_sar_fusion"
    else:
        details["workflow_recommendation"] = "bi_temporal_change"

    bounds_a = meta_a.get("bounds")
    bounds_b = meta_b.get("bounds")

    # If both images have geospatial bounding boxes
    if bounds_a and bounds_b:
        try:
            box_a = box(bounds_a["left"], bounds_a["bottom"], bounds_a["right"], bounds_a["top"])
            box_b = box(bounds_b["left"], bounds_b["bottom"], bounds_b["right"], bounds_b["top"])

            if not box_a.intersects(box_b):
                return {
                    "valid": False,
                    "message": "Images have non-overlapping geographic coordinates. Paired analysis requires spatially corresponding areas.",
                    "details": {
                        **details,
                        "spatial_overlap_percent": 0.0,
                        "bounds_a": bounds_a,
                        "bounds_b": bounds_b,
                    },
                }

            intersection_area = box_a.intersection(box_b).area
            min_area = min(box_a.area, box_b.area)
            overlap_pct = round((intersection_area / min_area) * 100.0, 1) if min_area > 0 else 0.0
            details["spatial_overlap_percent"] = overlap_pct

            if overlap_pct < 20.0:
                return {
                    "valid": False,
                    "message": f"Low spatial overlap ({overlap_pct}%). Images must share significant geographic coverage for change or fusion analysis.",
                    "details": details,
                }
        except Exception as e:
            details["spatial_check_warning"] = f"Could not calculate exact overlap: {e}"

    # Resolution check
    res_a = meta_a.get("resolution")
    res_b = meta_b.get("resolution")
    if res_a and res_b and res_a.get("x") and res_b.get("x"):
        scale_ratio = max(res_a["x"], res_b["x"]) / min(res_a["x"], res_b["x"])
        details["resolution_ratio"] = round(scale_ratio, 2)
        if scale_ratio > 10.0:
            details["resolution_warning"] = (
                f"Large resolution discrepancy ({round(scale_ratio, 1)}x). "
                "Coarser imagery will be upsampled during processing."
            )

    # Return valid status
    workflow_desc = (
        "Cross-modal Optical + SAR workflow"
        if is_cross_modal
        else "Bi-temporal change detection workflow"
    )

    overlap_msg = (
        f" with {details['spatial_overlap_percent']}% spatial overlap"
        if details.get("spatial_overlap_percent") is not None
        else ""
    )

    return {
        "valid": True,
        "message": f"Compatible image pair accepted for {workflow_desc}{overlap_msg}.",
        "details": details,
    }
