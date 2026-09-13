"""SatQuery AI — SQLAlchemy database engine and session management.

Provides async-compatible database engine and session factory.
Designed for SQLite with a clear migration path to PostgreSQL.
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker, declarative_base

from app.utils.config import get_settings
from app.utils.logging import get_logger

logger = get_logger("database")

# SQLAlchemy declarative base for ORM models
Base = declarative_base()

# Module-level engine and session factory (initialized lazily)
_engine = None
_SessionLocal = None


def _get_engine():
    """Create or return the SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url

        # SQLite-specific settings
        connect_args = {}
        if "sqlite" in db_url:
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = 30.0

        _engine = create_engine(
            db_url,
            connect_args=connect_args,
            echo=settings.debug,
            pool_pre_ping=True,
        )

        # Enable WAL mode for SQLite (better concurrent read performance)
        if "sqlite" in db_url:
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info("database_engine_created", url=db_url)

    return _engine


def _get_session_factory():
    """Create or return the session factory singleton."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = _get_engine()
        _SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_db() -> Session:
    """Get a database session. Use as a FastAPI dependency.

    Yields:
        SQLAlchemy Session instance.
    """
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize the database — create all tables.

    Called once at application startup.
    """
    engine = _get_engine()

    # Import models to ensure they are registered with Base
    from app.database import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("database_initialized", tables=list(Base.metadata.tables.keys()))


def get_engine():
    """Public accessor for the engine (for testing/migrations)."""
    return _get_engine()
