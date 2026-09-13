"""SatQuery AI — Geospatial Metadata Extraction.

Extracts comprehensive metadata from satellite imagery (GeoTIFF, standard TIFF,
PNG, JPEG) using Rasterio with fallback to Pillow.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

from app.geospatial.isro import identify_isro_metadata
from app.utils.logging import get_logger

logger = get_logger("geospatial.metadata")


def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from an image file.

    Attempts rasterio first for full geospatial metadata (CRS, bounds, resolution).
    Falls back to Pillow for standard non-georeferenced images (PNG, JPEG).

    Args:
        file_path: Path to the image file.

    Returns:
        Dictionary containing metadata properties.
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    suffix = file_path.suffix.lower()

    # Try rasterio first (supports GeoTIFF, TIFF, and common raster formats)
    try:
        with rasterio.open(str(file_path)) as src:
            width = src.width
            height = src.height
            count = src.count
            dtypes = [str(d) for d in src.dtypes]
            primary_dtype = dtypes[0] if dtypes else "uint8"
            nodata = src.nodata
            driver = src.driver

            # Transform matrix
            transform = list(src.transform) if src.transform else None

            # Resolution (pixel size)
            res = src.res
            resolution = {"x": float(res[0]), "y": float(res[1])} if res else None

            # CRS extraction
            crs_str = None
            if src.crs:
                if src.crs.is_epsg_code:
                    crs_str = f"EPSG:{src.crs.to_epsg()}"
                else:
                    crs_str = src.crs.to_string()

            # Bounds extraction
            native_bounds = None
            wgs84_bounds = None
            if src.bounds:
                native_bounds = {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                }

                # Attempt conversion to WGS84 lat/long if CRS is defined
                if src.crs:
                    try:
                        w_left, w_bottom, w_right, w_top = transform_bounds(
                            src.crs, "EPSG:4326",
                            src.bounds.left, src.bounds.bottom,
                            src.bounds.right, src.bounds.top,
                            densify_pts=21
                        )
                        wgs84_bounds = {
                            "left": float(w_left),
                            "bottom": float(w_bottom),
                            "right": float(w_right),
                            "top": float(w_top),
                        }
                    except Exception as e:
                        logger.warning("bounds_reprojection_failed", error=str(e))
                        wgs84_bounds = native_bounds

            # Pixel bounds fallback for unprojected rasters
            pixel_bounds = {
                "left": 0.0,
                "bottom": float(height),
                "right": float(width),
                "top": 0.0,
            }

            is_georeferenced = crs_str is not None
            format_name = "GeoTIFF" if driver in ("GTiff", "COG") and is_georeferenced else driver

            meta: Dict[str, Any] = {
                "format": format_name,
                "driver": driver,
                "is_georeferenced": is_georeferenced,
                "width": width,
                "height": height,
                "bands": count,
                "dtype": primary_dtype,
                "nodata": nodata,
                "crs": crs_str,
                "bounds": wgs84_bounds or native_bounds or pixel_bounds,
                "native_bounds": native_bounds or pixel_bounds,
                "resolution": resolution,
                "transform": transform,
                "file_size_bytes": file_size,
                "tags": dict(src.tags()),
            }

            # ISRO & SAC platform detection
            isro_info = identify_isro_metadata(meta, filename=file_path.name)
            meta["isro"] = isro_info

            # If unprojected but identified as ISRO, supplement nominal GSD resolution
            if not resolution and isro_info:
                nom_gsd = float(isro_info.get("gsd_nominal_m", 1.0))
                meta["resolution"] = {"x": nom_gsd, "y": nom_gsd}

            return meta

    except Exception as e:
        logger.info("rasterio_open_failed_trying_pil", file=file_path.name, error=str(e))

    # Pillow fallback for standard images without geospatial tags
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            bands = len(mode) if mode not in ("1", "L", "P") else 1
            if mode == "RGB":
                bands = 3
            elif mode == "RGBA":
                bands = 4

            fmt = img.format or suffix.replace(".", "").upper()

            pixel_bounds = {
                "left": 0.0,
                "bottom": float(height),
                "right": float(width),
                "top": 0.0,
            }

            meta = {
                "format": fmt,
                "driver": "PIL",
                "is_georeferenced": False,
                "width": width,
                "height": height,
                "bands": bands,
                "dtype": "uint8",
                "nodata": None,
                "crs": None,
                "bounds": pixel_bounds,
                "native_bounds": pixel_bounds,
                "resolution": None,
                "transform": None,
                "file_size_bytes": file_size,
                "tags": {},
            }

            isro_info = identify_isro_metadata(meta, filename=file_path.name)
            meta["isro"] = isro_info
            if isro_info:
                nom_gsd = float(isro_info.get("gsd_nominal_m", 1.0))
                meta["resolution"] = {"x": nom_gsd, "y": nom_gsd}

            return meta
    except Exception as pil_err:
        logger.error("metadata_extraction_failed", file=file_path.name, error=str(pil_err))
        raise ValueError(f"Unable to read image metadata for {file_path.name}: {pil_err}") from pil_err

