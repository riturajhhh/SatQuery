"""SatQuery AI — Analysis & Models API Routes.

Provides endpoints for submitting natural-language queries, polling analysis results,
inspecting auditable execution traces, retrieving evidence, and discovering registered models.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from app.api.dependencies import get_database, get_app_settings
from app.database import crud
from app.models import get_model_registry
from app.schemas.common import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisResult,
    AnswerInfo,
    ErrorResponse,
    ExecutionTraceResponse,
    ModelInfoResponse,
    ModelsListResponse,
    ModelUsedInfo,
    TraceStep,
)
from app.services.analysis_service import execute_analysis
from app.services.report_service import (
    generate_report_data,
    render_html_report,
    render_pdf_report,
)
from app.utils.config import Settings
from app.utils.logging import get_logger

logger = get_logger("routes.analysis")

router = APIRouter(tags=["analysis"])


@router.get(
    "/analysis/history",
    summary="Get analysis history",
)
def get_analysis_history_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_database),
):
    """Retrieve paginated list of past analyses with summary metadata."""
    analyses = crud.list_analyses(db, limit=limit, offset=offset)
    total = crud.count_analyses(db)
    items = []
    for a in analyses:
        files = a.uploaded_files or []
        thumb_url = None
        if files and files[0].thumbnail_path:
            thumb_url = f"/api/files/processed/thumbnails/{Path(files[0].thumbnail_path).name}"

        items.append({
            "analysis_id": a.id,
            "query": a.query,
            "task": a.task,
            "input_type": a.input_type,
            "status": a.status,
            "confidence": a.confidence,
            "confidence_level": a.confidence_level or "UNCERTAIN",
            "is_fallback": bool(a.is_fallback),
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "files_count": len(files),
            "modalities": list(set(f.modality for f in files if f.modality)),
            "thumbnail_url": thumb_url,
        })

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


@router.get(
    "/analysis/{analysis_id}/report",
    summary="Download or view audit report (PDF or HTML)",
)
def get_analysis_report_endpoint(
    analysis_id: str,
    format: str = Query("pdf", description="Report format: 'pdf' or 'html'"),
    db: Session = Depends(get_database),
):
    """Generate a comprehensive audit report in PDF or HTML format."""
    fmt = format.lower().strip()
    if fmt not in ("pdf", "html"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported report format '{format}'. Supported formats: 'pdf', 'html'."
        )

    try:
        report_data = generate_report_data(db, analysis_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    if fmt == "html":
        html_content = render_html_report(report_data)
        return HTMLResponse(content=html_content)

    # PDF format
    pdf_bytes = render_pdf_report(report_data)
    filename = f"satquery_report_{analysis_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query or options"},
        404: {"model": ErrorResponse, "description": "Upload ID not found"},
    },
    summary="Submit image query for analysis",
)
def submit_analysis_endpoint(
    request: AnalyzeRequest,
    db: Session = Depends(get_database),
):
    """Submit a query against an uploaded satellite image."""
    query = request.query.strip()
    if not query:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": "Query string must not be empty.",
            },
        )

    # Verify upload exists
    files = crud.get_files_by_upload_id(db, request.upload_id)
    if not files:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "not_found",
                "message": f"Upload session with id '{request.upload_id}' was not found. Please upload imagery first.",
            },
        )

    try:
        analysis = execute_analysis(
            db=db,
            upload_id=request.upload_id,
            query=query,
            options=request.options,
        )

        return AnalyzeResponse(
            analysis_id=analysis.id,
            status=analysis.status,
            message="Analysis completed successfully.",
            estimated_time_seconds=1,
        )
    except Exception as e:
        logger.error("analysis_submission_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline execution failed: {e}",
        )


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResult,
    responses={404: {"model": ErrorResponse, "description": "Analysis not found"}},
    summary="Get analysis results",
)
def get_analysis_endpoint(
    analysis_id: str,
    db: Session = Depends(get_database),
):
    """Retrieve completed analysis results including answer and confidence."""
    analysis = crud.get_analysis(db, analysis_id)
    if not analysis:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "not_found",
                "message": f"Analysis with id '{analysis_id}' was not found.",
            },
        )

    # Gather evidence items
    evidence_dict: Dict[str, Any] = {}
    evidence_items = crud.get_evidence_for_analysis(db, analysis_id)
    if evidence_items:
        evidence_dict = {
            item.evidence_type: item.metadata_json for item in evidence_items
        }

    answer_info = None
    if analysis.answer_text:
        calib_data = evidence_dict.get("confidence_calibration", {})
        answer_info = AnswerInfo(
            text=analysis.answer_text,
            confidence=analysis.confidence or 0.0,
            confidence_level=analysis.confidence_level or "UNCERTAIN",
            is_fallback=bool(analysis.is_fallback),
            calibrated_confidence=calib_data.get("calibrated_confidence") if calib_data else None,
            audit_findings=calib_data if calib_data else None,
        )

    models_used: List[ModelUsedInfo] = []
    for mr in analysis.model_runs:
        models_used.append(
            ModelUsedInfo(
                name=mr.model_name,
                version=mr.model_version,
                task=mr.task,
                is_fallback=bool(mr.is_fallback),
            )
        )

    return AnalysisResult(
        analysis_id=analysis.id,
        status=analysis.status,
        query=analysis.query,
        task=analysis.task,
        input_type=analysis.input_type,
        answer=answer_info,
        evidence=evidence_dict or None,
        models_used=models_used,
        timestamps={
            "created": analysis.created_at,
            "started": analysis.started_at,
            "completed": analysis.completed_at,
        },
        error={"message": analysis.error_message} if analysis.error_message else None,
        progress=None,
    )


@router.get(
    "/analysis/{analysis_id}/trace",
    response_model=ExecutionTraceResponse,
    summary="Get execution trace for an analysis",
)
def get_analysis_trace_endpoint(
    analysis_id: str,
    db: Session = Depends(get_database),
):
    """Retrieve full auditable execution trace steps for an analysis."""
    steps = crud.get_trace_steps(db, analysis_id)
    trace_steps = [
        TraceStep(
            step=step.step_number,
            action=step.action,
            status=step.status,
            duration_ms=step.duration_ms,
            details=step.details,
        )
        for step in steps
    ]

    return ExecutionTraceResponse(
        analysis_id=analysis_id,
        trace={
            "steps": [s.model_dump() for s in trace_steps],
            "total_steps": len(trace_steps),
        },
    )


@router.get(
    "/analysis/{analysis_id}/evidence",
    summary="Get evidence for an analysis",
)
def get_analysis_evidence_endpoint(
    analysis_id: str,
    db: Session = Depends(get_database),
):
    """Retrieve generated evidence items."""
    evidence_items = crud.get_evidence_for_analysis(db, analysis_id)
    return {
        "analysis_id": analysis_id,
        "evidence": [
            {
                "id": ev.id,
                "evidence_type": ev.evidence_type,
                "file_path": ev.file_path,
                "description": ev.description,
                "metadata": ev.metadata_json,
                "created_at": ev.created_at,
            }
            for ev in evidence_items
        ],
    }


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List all registered specialist models",
)
def list_models_endpoint(
    settings: Settings = Depends(get_app_settings),
):
    """List all models registered in the ModelRegistry."""
    registry = get_model_registry()
    model_infos = registry.list_models()

    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass

    results = [
        ModelInfoResponse(
            name=m.name,
            version=m.version,
            base_model=m.base_model,
            adapter=m.adapter,
            supported_tasks=[t.value for t in m.supported_tasks],
            supported_inputs=[i.value for i in m.supported_inputs],
            is_loaded=True,
            is_fallback=m.is_fallback,
            device=m.device,
        )
        for m in model_infos
    ]

    return ModelsListResponse(
        models=results,
        total=len(results),
        gpu_available=gpu_available,
    )
