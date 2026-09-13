"""SatQuery AI — FastAPI application entry point.

Creates and configures the FastAPI application with:
- CORS middleware
- API routers
- Database initialization
- Structured logging
- Static file serving for uploads/outputs
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.utils.config import get_settings
from app.utils.logging import setup_logging, get_logger
from app.database.engine import init_db

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.api.routes.analysis import router as analysis_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup and shutdown tasks."""
    settings = get_settings()
    logger = get_logger("main")

    # ---- Startup ----
    setup_logging(settings.log_level)
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        env=settings.app_env,
        debug=settings.debug,
    )

    # Create storage directories
    for dir_path in [
        settings.upload_dir,
        settings.processed_dir,
        settings.evidence_dir,
        settings.reports_dir,
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Initialize database
    init_db()
    logger.info("database_ready")

    logger.info("application_started", port=settings.backend_port)

    yield

    # ---- Shutdown ----
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Agentic Vision-Language Assistant for Remote-Sensing Imagery. "
            "Dynamically selects and combines specialist models across single-image, "
            "bi-temporal, and cross-modal optical-SAR workflows while producing evidence "
            "and an auditable execution trace."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- API Routers ----
    app.include_router(health_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")

    # ---- Static Files (uploads, processed previews, and outputs) ----
    uploads_path = Path(settings.upload_dir)
    processed_path = Path(settings.processed_dir)
    outputs_path = Path(settings.evidence_dir)

    for p in [uploads_path, processed_path, outputs_path]:
        p.mkdir(parents=True, exist_ok=True)

    app.mount(
        "/api/files/uploads",
        StaticFiles(directory=str(uploads_path)),
        name="uploads",
    )

    app.mount(
        "/api/files/processed",
        StaticFiles(directory=str(processed_path)),
        name="processed",
    )

    app.mount(
        "/api/files/evidence",
        StaticFiles(directory=str(outputs_path)),
        name="evidence",
    )

    return app


# Create the app instance
app = create_app()
