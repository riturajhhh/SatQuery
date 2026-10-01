"""SatQuery AI — File Upload & Geospatial Ingestion API Route.

Handles single and paired satellite imagery uploads, geospatial validation,
metadata extraction, modality classification, preview generation, and database storage.
"""

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_database, get_app_settings
from app.database import crud
from app.geospatial import (
    extract_metadata,
    detect_modality,
    generate_preview_and_thumbnail,
    validate_pair_compatibility,
)
from app.schemas.common import FileInfo, UploadResponse, ErrorResponse
from app.utils.config import Settings
from app.utils.logging import get_logger

logger = get_logger("routes.upload")

router = APIRouter(tags=["upload"])


def _sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal and unsafe characters."""
    clean = re.sub(r"[^\w\-_\.]", "_", Path(name).name)
    return clean or "unnamed_raster"


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error or unsupported format"},
        422: {"model": ErrorResponse, "description": "Incompatible image pair"},
    },
    summary="Upload satellite imagery",
    description=(
        "Accepts up to 2 satellite images (GeoTIFF, TIFF, PNG, JPEG). "
        "Extracts geospatial metadata, infers sensor modality, generates web previews, "
        "and checks pair compatibility."
    ),
)
async def upload_satellite_images(
    files: List[UploadFile] = File(..., description="1 or 2 satellite image files"),
    input_type: Optional[str] = Form(None, description="Optional override: single, bi_temporal, optical_sar"),
    db: Session = Depends(get_database),
    settings: Settings = Depends(get_app_settings),
):
    # 1. Validate file count
    if not files or len(files) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": "At least one satellite image file must be uploaded.",
                "details": {"reason": "no_files"},
            },
        )

    if len(files) > 2:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": "Maximum of 2 files allowed per upload.",
                "details": {"reason": "too_many_files", "count": len(files)},
            },
        )

    # 2. Validate file extensions
    allowed_exts = tuple(settings.allowed_extensions_list)
    for file in files:
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        if not ext or ext not in allowed_exts:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": f"Unsupported file format: {ext or 'none'}. Accepted formats: {settings.allowed_extensions}",
                    "details": {"file": filename, "reason": "unsupported_format"},
                },
            )

    upload_id = uuid.uuid4().hex
    upload_dir = Path(settings.upload_dir)
    processed_dir = Path(settings.processed_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    file_infos: List[FileInfo] = []
    saved_metadata: List[Dict[str, Any]] = []
    modalities: List[str] = []

    # 3. Save, process, and ingest each file
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    for idx, file in enumerate(files):
        file_id = uuid.uuid4().hex
        clean_name = _sanitize_filename(file.filename or f"image_{idx}.tif")
        stored_name = f"{file_id}_{clean_name}"
        stored_path = upload_dir / stored_name

        # Stream save file to disk while enforcing size limit
        bytes_written = 0
        try:
            with open(stored_path, "wb") as f_out:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    bytes_written += len(chunk)
                    if bytes_written > max_size_bytes:
                        # Clean up partial file
                        stored_path.unlink(missing_ok=True)
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={
                                "error": "validation_error",
                                "message": f"File {clean_name} exceeds maximum size limit of {settings.max_upload_size_mb}MB.",
                                "details": {"file": clean_name, "reason": "file_too_large"},
                            },
                        )
                    f_out.write(chunk)
        except Exception as e:
            stored_path.unlink(missing_ok=True)
            logger.error("file_save_failed", file=clean_name, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file {clean_name}: {e}",
            )

        # 4. Extract geospatial metadata
        try:
            meta = extract_metadata(stored_path)
        except Exception as e:
            stored_path.unlink(missing_ok=True)
            logger.error("metadata_extraction_error", file=clean_name, error=str(e))
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "validation_error",
                    "message": f"Failed to parse image data from {clean_name}: {e}",
                    "details": {"file": clean_name, "reason": "corrupted_or_invalid_image"},
                },
            )

        # 5. Detect modality
        modality = detect_modality(meta, filename=clean_name)
        modalities.append(modality)
        saved_metadata.append(meta)

        # 6. Generate previews & thumbnails
        try:
            prev_path, thumb_path = generate_preview_and_thumbnail(
                file_path=stored_path,
                output_dir=processed_dir,
                file_id=file_id,
            )
            preview_url = f"/api/files/processed/previews/{prev_path.name}"
            thumbnail_url = f"/api/files/processed/thumbnails/{thumb_path.name}"
        except Exception as e:
            logger.warning("preview_generation_error", file=clean_name, error=str(e))
            prev_path, thumb_path = None, None
            preview_url, thumbnail_url = None, None

        # 7. Persist record to database
        temporal_label = "t1" if (len(files) > 1 and idx == 0) else "t2" if (len(files) > 1 and idx == 1) else None

        crud.create_uploaded_file(
            db=db,
            id=file_id,
            upload_id=upload_id,
            original_name=file.filename or clean_name,
            stored_name=stored_name,
            stored_path=str(stored_path),
            file_format=meta.get("format"),
            file_size_bytes=meta.get("file_size_bytes"),
            width=meta.get("width"),
            height=meta.get("height"),
            bands=meta.get("bands"),
            dtype=meta.get("dtype"),
            crs=meta.get("crs"),
            bounds=meta.get("bounds"),
            resolution=meta.get("resolution"),
            transform=meta.get("transform"),
            modality=modality,
            temporal_label=temporal_label,
            preview_path=str(prev_path) if prev_path else None,
            thumbnail_path=str(thumb_path) if thumb_path else None,
            metadata_json=meta,
        )

        file_infos.append(
            FileInfo(
                file_id=file_id,
                original_name=file.filename or clean_name,
                stored_name=stored_name,
                format=meta.get("format"),
                width=meta.get("width"),
                height=meta.get("height"),
                bands=meta.get("bands"),
                crs=meta.get("crs"),
                bounds=meta.get("bounds"),
                resolution=meta.get("resolution"),
                modality=modality,
                file_size_bytes=meta.get("file_size_bytes"),
                preview_url=preview_url,
                thumbnail_url=thumbnail_url,
                metadata=meta,
            )
        )

    # 8. Pair compatibility check if 2 files
    compatibility = {"valid": True, "message": "Single image upload accepted."}

    if len(files) == 2:
        compatibility = validate_pair_compatibility(
            meta_a=saved_metadata[0],
            meta_b=saved_metadata[1],
            modality_a=modalities[0],
            modality_b=modalities[1],
        )

        # If pair is completely incompatible (e.g., non-overlapping geographical bounds)
        if not compatibility["valid"]:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "compatibility_error",
                    "message": compatibility["message"],
                    "details": compatibility.get("details", {}),
                },
            )

    # 9. Determine overall input_type
    if not input_type:
        if len(files) == 1:
            input_type = "single"
        else:
            rec = compatibility.get("details", {}).get("workflow_recommendation")
            input_type = "optical_sar" if rec == "optical_sar_fusion" else "bi_temporal"
    elif input_type in ("optical_sar_fusion", "optical_sar"):
        input_type = "optical_sar"

    if input_type == "optical_sar" and len(file_infos) == 2:
        # Ensure files reflect optical and SAR modalities in file_infos
        if "sar" not in modalities:
            file_infos[1].modality = "sar"
            modalities[1] = "sar"
            from app.database.models import UploadedFile
            db_f2 = db.query(UploadedFile).filter(UploadedFile.id == file_infos[1].file_id).first()
            if db_f2:
                db_f2.modality = "sar"
                db.commit()

    return UploadResponse(
        upload_id=upload_id,
        files=file_infos,
        input_type=input_type,
        compatibility=compatibility,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/sample-pairs",
    summary="Get available sample image pairs",
    description="Returns pre-bundled sample datasets for single, bi-temporal, and optical-sar analysis.",
)
def get_sample_pairs():
    return {
        "datasets": [
            {
                "id": "sentinel1_sar_sample",
                "title": "Sentinel-1 Dual-Pol SAR (Trained Polarimetric Model)",
                "mode": "single",
                "files": [
                    {
                        "filename": "sentinel1_sar_urban_water.tif",
                        "url": "/api/files/sample_images/sentinel1_sar_urban_water.tif",
                        "label": "Sentinel-1 C-Band SAR (VV + VH Dual-Pol)",
                        "modality": "sar",
                    }
                ],
                "recommended_queries": [
                    "Compute calibrated VV and VH backscatter in decibels (dB), evaluate the polarimetric ratio, and analyze surface roughness.",
                    "Identify surface water bodies and flood inundation zones using specular radar scattering regardless of cloud cover.",
                    "Detect high-intensity double-bounce corner reflections from high-rise buildings and metallic infrastructure.",
                    "Detect maritime vessels, ships, and offshore structures via radar dielectric contrast against dark open sea.",
                ],
            },
            {
                "id": "levir_cd_sample",
                "title": "LEVIR-CD Urban Building Expansion (0.5m VHR)",
                "mode": "bitemporal",
                "files": [
                    {
                        "filename": "levir_cd_t1_pre.tif",
                        "url": "/api/files/sample_images/levir_cd_t1_pre.tif",
                        "label": "T1 • Baseline (Rural / Agricultural)",
                        "modality": "optical",
                    },
                    {
                        "filename": "levir_cd_t2_post.tif",
                        "url": "/api/files/sample_images/levir_cd_t2_post.tif",
                        "label": "T2 • Follow-up (New Built-up Structures)",
                        "modality": "optical",
                    },
                ],
                "recommended_queries": [
                    "Detect and delineate all new residential and industrial building structures between T1 and T2.",
                    "Quantify the total footprint area, hectares, and spatial cluster count of newly constructed buildings.",
                    "Did urban buildings expand or decline between the two observations?",
                ],
            },
            {
                "id": "bitemporal_sample",
                "title": "Bi-Temporal Land Cover Change (2022 vs 2024)",
                "mode": "bitemporal",
                "files": [
                    {
                        "filename": "change_before_2022.tif",
                        "url": "/api/files/sample_images/change_before_2022.tif",
                        "label": "T1 • Baseline (2022)",
                        "modality": "optical",
                    },
                    {
                        "filename": "change_after_2024.tif",
                        "url": "/api/files/sample_images/change_after_2024.tif",
                        "label": "T2 • Follow-up (2024)",
                        "modality": "optical",
                    },
                ],
                "recommended_queries": [
                    "What changed between these two dates, and where did the change occur?",
                    "Has the built-up area increased, decreased, or remained unchanged between the baseline and follow-up scenes?",
                    "Measure the total area and hectares of new building construction.",
                ],
            },
            {
                "id": "optical_sar_sample",
                "title": "Optical-SAR Cross-Modal Fusion (Cartosat + RISAT-1A)",
                "mode": "optical_sar",
                "files": [
                    {
                        "filename": "cartosat_pune_urban.tif",
                        "url": "/api/files/sample_images/cartosat_pune_urban.tif",
                        "label": "Optical Multispectral (0.65m)",
                        "modality": "optical",
                    },
                    {
                        "filename": "risat_sar_radar.tif",
                        "url": "/api/files/sample_images/risat_sar_radar.tif",
                        "label": "SAR Microwave Radar (C-band)",
                        "modality": "sar",
                    },
                ],
                "recommended_queries": [
                    "Pierce optical cloud cover with SAR radar backscatter to uncover sub-cloud ground truth and water bodies.",
                    "Fuse optical spectral reflectance with SAR double-bounce geometry to map urban building envelopes and road networks.",
                    "Combine optical vegetation greenness with radar sub-canopy penetration to map wetlands and flooded agricultural fields.",
                    "Reconcile optical multispectral observations with microwave radar signatures for verified terrain intelligence.",
                ],
            },
            {
                "id": "single_optical_sample",
                "title": "High-Resolution Optical Scene (Forest & Built-up)",
                "mode": "single",
                "files": [
                    {
                        "filename": "forest_vegetation.tif",
                        "url": "/api/files/sample_images/forest_vegetation.tif",
                        "label": "Optical Multispectral Scene",
                        "modality": "optical",
                    },
                ],
                "recommended_queries": [
                    "Calculate NDVI vegetation health, assess chlorophyll vigor, and classify crop growth stages across agricultural parcels.",
                    "Describe the land-cover classes, natural vegetation density, and human settlements visible in this optical scene.",
                    "Identify built-up density, residential clusters, asphalt roads, and commercial building footprints.",
                    "Delineate lakes, river channels, and canals using NDWI spectral absorption and water boundary extraction.",
                ],
            },
        ]
    }
