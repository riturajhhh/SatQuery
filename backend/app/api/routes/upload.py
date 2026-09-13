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
            input_type = rec if rec else "bi_temporal"

    return UploadResponse(
        upload_id=upload_id,
        files=file_infos,
        input_type=input_type,
        compatibility=compatibility,
        timestamp=datetime.now(timezone.utc),
    )
