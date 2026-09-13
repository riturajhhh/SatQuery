"""SatQuery AI — CRUD operations.

Database create/read/update operations for all ORM models.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.database.models import (
    Analysis,
    UploadedFile,
    ModelRun,
    ExecutionTrace,
    Evidence,
)
from app.utils.logging import get_logger

logger = get_logger("crud")


# ---- Analysis CRUD ----

def create_analysis(db: Session, query: str) -> Analysis:
    """Create a new analysis record."""
    analysis = Analysis(query=query)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    logger.info("analysis_created", analysis_id=analysis.id)
    return analysis


def get_analysis(db: Session, analysis_id: str) -> Optional[Analysis]:
    """Get an analysis by ID."""
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


def get_analyses(db: Session, skip: int = 0, limit: int = 50) -> List[Analysis]:
    """Get a list of analyses with pagination."""
    return (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_analysis_status(
    db: Session,
    analysis_id: str,
    status: str,
    **kwargs,
) -> Optional[Analysis]:
    """Update analysis status and optional fields."""
    analysis = get_analysis(db, analysis_id)
    if analysis is None:
        return None

    analysis.status = status

    if status == "processing" and analysis.started_at is None:
        analysis.started_at = datetime.now(timezone.utc)
    elif status in ("complete", "failed"):
        analysis.completed_at = datetime.now(timezone.utc)

    for key, value in kwargs.items():
        if hasattr(analysis, key):
            setattr(analysis, key, value)

    db.commit()
    db.refresh(analysis)
    logger.info("analysis_updated", analysis_id=analysis_id, status=status)
    return analysis


# ---- UploadedFile CRUD ----

def create_uploaded_file(db: Session, **kwargs) -> UploadedFile:
    """Create a new uploaded file record."""
    uploaded_file = UploadedFile(**kwargs)
    db.add(uploaded_file)
    db.commit()
    db.refresh(uploaded_file)
    logger.info("file_recorded", file_id=uploaded_file.id, name=uploaded_file.original_name)
    return uploaded_file


def get_files_by_upload_id(db: Session, upload_id: str) -> List[UploadedFile]:
    """Get all files from a specific upload."""
    return (
        db.query(UploadedFile)
        .filter(UploadedFile.upload_id == upload_id)
        .all()
    )


# ---- ModelRun CRUD ----

def create_model_run(db: Session, **kwargs) -> ModelRun:
    """Create a model run record."""
    model_run = ModelRun(**kwargs)
    db.add(model_run)
    db.commit()
    db.refresh(model_run)
    return model_run


# ---- ExecutionTrace CRUD ----

def create_trace_step(db: Session, **kwargs) -> ExecutionTrace:
    """Create an execution trace step."""
    trace = ExecutionTrace(**kwargs)
    db.add(trace)
    db.commit()
    db.refresh(trace)
    return trace


def get_trace_steps(db: Session, analysis_id: str) -> List[ExecutionTrace]:
    """Get all trace steps for an analysis, ordered by step number."""
    return (
        db.query(ExecutionTrace)
        .filter(ExecutionTrace.analysis_id == analysis_id)
        .order_by(ExecutionTrace.step_number)
        .all()
    )


# ---- Evidence CRUD ----

def create_evidence(db: Session, **kwargs) -> Evidence:
    """Create an evidence record."""
    evidence = Evidence(**kwargs)
    db.add(evidence)
    db.commit()
    db.refresh(evidence)
    return evidence


def get_evidence_for_analysis(db: Session, analysis_id: str) -> List[Evidence]:
    """Get all evidence items for an analysis."""
    return (
        db.query(Evidence)
        .filter(Evidence.analysis_id == analysis_id)
        .all()
    )


def list_analyses(db: Session, limit: int = 20, offset: int = 0) -> List[Analysis]:
    """Get paginated list of recent analyses ordered by creation time descending."""
    return (
        db.query(Analysis)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def count_analyses(db: Session) -> int:
    """Get total count of analyses."""
    return db.query(Analysis).count()

