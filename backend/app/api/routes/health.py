"""SatQuery AI — Health check endpoint.

Provides system health status including database, GPU, and model availability.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.dependencies import get_database
from app.utils.config import get_settings
from app.schemas.common import HealthResponse

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check system health status including database, GPU, and model availability.",
)
def health_check(db: Session = Depends(get_database)):
    """Return current system health status."""
    settings = get_settings()

    # Check database connectivity
    db_status = "disconnected"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"

    # Check GPU availability
    gpu_available = False
    if _TORCH_AVAILABLE:
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="0.1.0",
        models_loaded=0,  # Will be populated when model registry is built
        gpu_available=gpu_available,
        database=db_status,
        timestamp=datetime.now(timezone.utc),
    )
