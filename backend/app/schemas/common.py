"""SatQuery AI — Common Pydantic schemas.

Shared request/response schemas used across the API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---- Health ----

class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="System health status", examples=["healthy"])
    version: str = Field(..., description="Application version")
    models_loaded: int = Field(0, description="Number of models currently loaded")
    gpu_available: bool = Field(False, description="Whether GPU is available")
    database: str = Field(..., description="Database connection status")
    timestamp: datetime = Field(..., description="Response timestamp")


# ---- Errors ----

class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# ---- File Upload ----

class FileInfo(BaseModel):
    """Information about an uploaded file."""

    file_id: str
    original_name: str
    stored_name: str
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    crs: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    resolution: Optional[Dict[str, float]] = None
    modality: Optional[str] = None
    file_size_bytes: Optional[int] = None
    preview_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    """Response from file upload endpoint."""

    upload_id: str
    files: List[FileInfo]
    input_type: str
    compatibility: Dict[str, Any]
    timestamp: datetime


# ---- Analysis ----

class AnalyzeRequest(BaseModel):
    """Request to start an analysis."""

    upload_id: str = Field(..., description="Upload ID from /api/upload")
    query: str = Field(..., min_length=1, description="Natural-language question")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "confidence_threshold": 0.5,
            "generate_evidence": True,
            "generate_report": True,
        }
    )


class AnalyzeResponse(BaseModel):
    """Response from analysis submission."""

    analysis_id: str
    status: str
    message: str
    estimated_time_seconds: Optional[int] = None


class AnswerInfo(BaseModel):
    """Analysis answer details."""

    text: str
    confidence: float
    confidence_level: str
    is_fallback: bool = False
    calibrated_confidence: Optional[Dict[str, Any]] = None
    audit_findings: Optional[Dict[str, Any]] = None


class ModelUsedInfo(BaseModel):
    """Information about a model used in analysis."""

    name: str
    version: Optional[str] = None
    task: str
    is_fallback: bool = False


class AnalysisResult(BaseModel):
    """Complete analysis result."""

    analysis_id: str
    status: str
    query: str
    task: Optional[str] = None
    input_type: Optional[str] = None
    answer: Optional[AnswerInfo] = None
    evidence: Optional[Dict[str, Any]] = None
    models_used: List[ModelUsedInfo] = []
    timestamps: Dict[str, Optional[datetime]] = {}
    error: Optional[Dict[str, str]] = None
    progress: Optional[Dict[str, Any]] = None


# ---- Execution Trace ----

class TraceStep(BaseModel):
    """A single step in the execution trace."""

    step: int
    action: str
    status: str
    duration_ms: Optional[float] = None
    details: Optional[str] = None


class ExecutionTraceResponse(BaseModel):
    """Full execution trace for an analysis."""

    analysis_id: str
    trace: Dict[str, Any]


# ---- Models ----

class ModelInfoResponse(BaseModel):
    """Information about a registered model."""

    name: str
    version: Optional[str] = None
    base_model: Optional[str] = None
    adapter: Optional[str] = None
    supported_tasks: List[str] = []
    supported_inputs: List[str] = []
    is_loaded: bool = False
    is_fallback: bool = False
    device: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response listing all registered models."""

    models: List[ModelInfoResponse]
    total: int
    gpu_available: bool = False
