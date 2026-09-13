"""SatQuery AI — Web Preview & Thumbnail Generation.

Generates web-friendly 8-bit RGB PNG previews and thumbnails from satellite
imagery (supporting 16-bit, float, multi-band, and SAR rasters) with
percentile contrast stretching.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import rasterio

from app.geospatial.isro import calibrate_isro_radiometry, identify_isro_metadata
from app.utils.logging import get_logger

logger = get_logger("geospatial.preview")


def _normalize_band(band: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Normalize a single 2D raster band to 0-255 uint8 with percentile clipping."""
    # Mask out non-finite or nodata values if any
    valid_mask = np.isfinite(band)
    if not np.any(valid_mask):
        return np.zeros(band.shape, dtype=np.uint8)

    valid_data = band[valid_mask]
    v_min, v_max = np.percentile(valid_data, [p_low, p_high])

    if v_max <= v_min:
        v_max = v_min + 1e-5

    clipped = np.clip(band, v_min, v_max)
    scaled = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    return scaled


def generate_preview_and_thumbnail(
    file_path: Path,
    output_dir: Path,
    file_id: str,
    max_preview_dim: int = 1024,
    thumbnail_dim: int = 256,
) -> Tuple[Path, Path]:
    """Generate preview and thumbnail PNG images for an uploaded satellite raster.

    Args:
        file_path: Path to the source raster image.
        output_dir: Root processed directory where 'previews' and 'thumbnails' will be stored.
        file_id: Unique identifier for file naming.
        max_preview_dim: Max width or height for preview image.
        thumbnail_dim: Max width or height for thumbnail image.

    Returns:
        Tuple of (preview_path, thumbnail_path).
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    previews_dir = output_dir / "previews"
    thumbnails_dir = output_dir / "thumbnails"
    previews_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    preview_path = previews_dir / f"{file_id}_preview.png"
    thumbnail_path = thumbnails_dir / f"{file_id}_thumbnail.png"

    # Try reading through rasterio first
    rgb_img: Optional[Image.Image] = None

    try:
        with rasterio.open(str(file_path)) as src:
            count = src.count
            width, height = src.width, src.height
            tags = dict(src.tags())

            # Detect ISRO specs if present
            mini_meta = {"bands": count, "dtype": str(src.dtypes[0]), "tags": tags, "crs": str(src.crs)}
            isro_info = identify_isro_metadata(mini_meta, filename=file_path.name)

            # Determine downsampling factor if the raster is gigantic
            overview_factor = max(1, int(max(width, height) / (max_preview_dim * 1.5)))
            out_shape = (
                max(1, height // overview_factor),
                max(1, width // overview_factor),
            )

            if count >= 3:
                # Check for ISRO Cartosat VNIR band ordering (Red=3, Green=2, Blue=1)
                rgb_indices = [1, 2, 3]
                if isro_info and isro_info.get("rgb_band_indices"):
                    rgb_indices = isro_info["rgb_band_indices"]

                r = src.read(rgb_indices[0], out_shape=out_shape)
                g = src.read(rgb_indices[1], out_shape=out_shape)
                b = src.read(rgb_indices[2], out_shape=out_shape)

                if isro_info:
                    r_norm = calibrate_isro_radiometry(r, is_sar=False, bit_depth=isro_info.get("bit_depth_effective", 11), nodata=src.nodata)
                    g_norm = calibrate_isro_radiometry(g, is_sar=False, bit_depth=isro_info.get("bit_depth_effective", 11), nodata=src.nodata)
                    b_norm = calibrate_isro_radiometry(b, is_sar=False, bit_depth=isro_info.get("bit_depth_effective", 11), nodata=src.nodata)
                    rgb_arr = np.dstack([r_norm, g_norm, b_norm])
                elif src.dtypes[0] == "uint8":
                    rgb_arr = np.dstack([r, g, b])
                else:
                    r_norm = _normalize_band(r)
                    g_norm = _normalize_band(g)
                    b_norm = _normalize_band(b)
                    rgb_arr = np.dstack([r_norm, g_norm, b_norm])

                rgb_img = Image.fromarray(rgb_arr, mode="RGB")

            elif count == 1:
                # Single band (SAR / Panchromatic / grayscale)
                band = src.read(1, out_shape=out_shape)
                is_sar_modality = isro_info and "sar" in isro_info.get("sensor_type", "").lower()

                if isro_info:
                    band_norm = calibrate_isro_radiometry(
                        band,
                        is_sar=bool(is_sar_modality),
                        bit_depth=isro_info.get("bit_depth_effective", 11),
                        nodata=src.nodata,
                        calibration_constant=isro_info.get("raw_calibration_constant"),
                    )
                else:
                    band_norm = _normalize_band(band)

                rgb_arr = np.dstack([band_norm, band_norm, band_norm])
                rgb_img = Image.fromarray(rgb_arr, mode="RGB")

            elif count == 2:
                # 2 bands: e.g. dual-pol SAR (RH/RV or HH/HV)
                b1 = src.read(1, out_shape=out_shape)
                b2 = src.read(2, out_shape=out_shape)
                if isro_info:
                    b1_n = calibrate_isro_radiometry(b1, is_sar=True, nodata=src.nodata)
                    b2_n = calibrate_isro_radiometry(b2, is_sar=True, nodata=src.nodata)
                else:
                    b1_n = _normalize_band(b1)
                    b2_n = _normalize_band(b2)
                # False-color composite: R=b1, G=b2, B=ratio(b1/(b2+1e-5))
                ratio = np.clip((b1_n.astype(np.float32) / (b2_n.astype(np.float32) + 1e-4)) * 128.0, 0, 255).astype(np.uint8)
                rgb_arr = np.dstack([b1_n, b2_n, ratio])
                rgb_img = Image.fromarray(rgb_arr, mode="RGB")

    except Exception as e:
        logger.info("rasterio_preview_failed_fallback_pil", error=str(e), file=file_path.name)

    # Fallback to standard Pillow loading if rasterio didn't produce an image
    if rgb_img is None:
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")

    # Resize for preview (maintain aspect ratio)
    preview_img = rgb_img.copy()
    preview_img.thumbnail((max_preview_dim, max_preview_dim), Image.Resampling.LANCZOS)
    preview_img.save(preview_path, format="PNG", optimize=True)

    # Resize for thumbnail
    thumb_img = rgb_img.copy()
    thumb_img.thumbnail((thumbnail_dim, thumbnail_dim), Image.Resampling.LANCZOS)
    thumb_img.save(thumbnail_path, format="PNG", optimize=True)

    logger.info("preview_generated", preview=str(preview_path), thumbnail=str(thumbnail_path))
    return preview_path, thumbnail_path
