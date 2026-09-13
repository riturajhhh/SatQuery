"""SatQuery AI — ORM models.

SQLAlchemy ORM models for analyses, uploaded files, model runs,
execution traces, and evidence. Designed to migrate cleanly to PostgreSQL.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from app.database.engine import Base


def _generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return uuid.uuid4().hex


class Analysis(Base):
    """Represents a single analysis session/query."""

    __tablename__ = "analyses"

    id = Column(String(32), primary_key=True, default=_generate_uuid)
    query = Column(Text, nullable=False)
    task = Column(String(50), nullable=True)  # vqa, captioning, grounding, etc.
    input_type = Column(String(50), nullable=True)  # single, bi_temporal, optical_sar
    status = Column(
        String(20), nullable=False, default="pending"
    )  # pending, processing, complete, failed
    confidence = Column(Float, nullable=True)
    confidence_level = Column(String(20), nullable=True)  # HIGH, MEDIUM, LOW, UNCERTAIN
    answer_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    is_fallback = Column(Integer, default=0)  # SQLite doesn't have native bool

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    uploaded_files = relationship("UploadedFile", back_populates="analysis", cascade="all, delete-orphan")
    model_runs = relationship("ModelRun", back_populates="analysis", cascade="all, delete-orphan")
    execution_traces = relationship("ExecutionTrace", back_populates="analysis", cascade="all, delete-orphan")
    evidence_items = relationship("Evidence", back_populates="analysis", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} task={self.task} status={self.status}>"


class UploadedFile(Base):
    """Represents a single uploaded image file with metadata."""

    __tablename__ = "uploaded_files"

    id = Column(String(32), primary_key=True, default=_generate_uuid)
    analysis_id = Column(String(32), ForeignKey("analyses.id"), nullable=True)
    upload_id = Column(String(32), nullable=True)  # Groups files from same upload

    # File info
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    stored_path = Column(Text, nullable=False)
    file_format = Column(String(20), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    # Image properties
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    bands = Column(Integer, nullable=True)
    dtype = Column(String(20), nullable=True)

    # Geospatial
    crs = Column(String(50), nullable=True)
    bounds = Column(JSON, nullable=True)  # {left, bottom, right, top}
    resolution = Column(JSON, nullable=True)  # {x, y}
    transform = Column(JSON, nullable=True)

    # Classification
    modality = Column(String(30), nullable=True)  # optical, sar, multispectral
    temporal_label = Column(String(10), nullable=True)  # t1, t2, or null

    # Preview
    preview_path = Column(Text, nullable=True)
    thumbnail_path = Column(Text, nullable=True)

    # Metadata
    metadata_json = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    analysis = relationship("Analysis", back_populates="uploaded_files")

    def __repr__(self) -> str:
        return f"<UploadedFile id={self.id} name={self.original_name} modality={self.modality}>"


class ModelRun(Base):
    """Records a single model execution within an analysis."""

    __tablename__ = "model_runs"

    id = Column(String(32), primary_key=True, default=_generate_uuid)
    analysis_id = Column(String(32), ForeignKey("analyses.id"), nullable=False)

    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=True)
    task = Column(String(50), nullable=False)
    is_fallback = Column(Integer, default=0)

    # Parameters & results
    parameters = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    device = Column(String(20), nullable=True)  # cpu, cuda:0, etc.

    # Status
    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    analysis = relationship("Analysis", back_populates="model_runs")

    def __repr__(self) -> str:
        return f"<ModelRun id={self.id} model={self.model_name} status={self.status}>"


class ExecutionTrace(Base):
    """Records a single step in the execution trace."""

    __tablename__ = "execution_traces"

    id = Column(String(32), primary_key=True, default=_generate_uuid)
    analysis_id = Column(String(32), ForeignKey("analyses.id"), nullable=False)

    step_number = Column(Integer, nullable=False)
    action = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # success, failed, skipped
    duration_ms = Column(Float, nullable=True)
    details = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    analysis = relationship("Analysis", back_populates="execution_traces")

    def __repr__(self) -> str:
        return f"<ExecutionTrace step={self.step_number} action={self.action}>"


class Evidence(Base):
    """Represents a piece of visual evidence generated for an analysis."""

    __tablename__ = "evidence"

    id = Column(String(32), primary_key=True, default=_generate_uuid)
    analysis_id = Column(String(32), ForeignKey("analyses.id"), nullable=False)

    evidence_type = Column(String(50), nullable=False)  # change_map, bbox, mask, overlay
    file_path = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    analysis = relationship("Analysis", back_populates="evidence_items")

    def __repr__(self) -> str:
        return f"<Evidence id={self.id} type={self.evidence_type}>"
