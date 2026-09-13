"""SatQuery AI — Shared API dependencies.

FastAPI dependency injection providers for database sessions,
settings, and common utilities.
"""

from typing import Generator

from sqlalchemy.orm import Session

from app.database.engine import get_db
from app.utils.config import Settings, get_settings


def get_database() -> Generator[Session, None, None]:
    """Database session dependency. Use with Depends(get_database)."""
    yield from get_db()


def get_app_settings() -> Settings:
    """Settings dependency. Use with Depends(get_app_settings)."""
    return get_settings()
