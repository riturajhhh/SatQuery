"""SatQuery AI — Configuration loader.

Loads configuration from .env and config.yaml with type-safe Pydantic settings.
"""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# Project root is two levels up from this file (backend/app/utils/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml_config() -> dict:
    """Load config.yaml from project root."""
    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ---- Application ----
    app_name: str = "SatQuery AI"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ---- Backend Server ----
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    allowed_origins: str = "http://localhost:5173,http://localhost:3000"

    # ---- Database ----
    database_url: str = "sqlite:///./satquery.db"

    # ---- Storage Paths ----
    upload_dir: str = "./uploads"
    processed_dir: str = "./outputs/processed"
    evidence_dir: str = "./outputs/evidence"
    reports_dir: str = "./reports"
    models_dir: str = "./models"

    # ---- Upload Limits ----
    max_upload_size_mb: int = 500
    allowed_extensions: str = ".tif,.tiff,.geotiff,.png,.jpg,.jpeg"

    # ---- GPU ----
    use_gpu: bool = True
    device_override: str = ""
    gpu_memory_fraction: float = 0.8

    # ---- Inference ----
    default_confidence_threshold: float = 0.5
    max_image_dimension: int = 4096
    tile_size: int = 512
    tile_overlap: int = 64
    batch_size: int = 1

    # ---- Agent ----
    agent_max_retries: int = 3
    agent_timeout_seconds: int = 300

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def cors_origins(self) -> List[str]:
        """Parse ALLOWED_ORIGINS into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse ALLOWED_EXTENSIONS into a list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return _PROJECT_ROOT

    @property
    def yaml_config(self) -> dict:
        """Return the parsed config.yaml contents."""
        if not hasattr(self, "_yaml_cache"):
            object.__setattr__(self, "_yaml_cache", _load_yaml_config())
        return self._yaml_cache


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
