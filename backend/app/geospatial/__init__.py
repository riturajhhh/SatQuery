"""SatQuery AI — Geospatial Processing Module.

Provides metadata extraction, modality classification, preview generation,
and paired image compatibility validation.
"""

from app.geospatial.metadata import extract_metadata
from app.geospatial.modality import detect_modality
from app.geospatial.preview import generate_preview_and_thumbnail
from app.geospatial.compatibility import validate_pair_compatibility

__all__ = [
    "extract_metadata",
    "detect_modality",
    "generate_preview_and_thumbnail",
    "validate_pair_compatibility",
]
