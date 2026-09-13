"""SatQuery AI — Specialist Models Package.

Exports base types, ModelRegistry, and initializes default specialist models.
"""

from app.models.base import (
    ConfidenceLevel,
    InputType,
    ModelInfo,
    ModelInput,
    ModelOutput,
    RemoteSensingModel,
    TaskType,
)
from app.models.registry import ModelNotFoundError, ModelRegistry, get_model_registry
from app.models.vqa import RSVQA_BLIP2, RSVQA_Fallback
from app.models.captioning import RSCaptioning_BLIP2, RSCaptioning_Fallback
from app.models.grounding import RSGrounding_GroundingDINO, RSGrounding_Fallback
from app.models.change_detection import RSChangeDetection_BIT, RSChangeDetection_Fallback
from app.models.change_vqa import RSChangeVQA_Model, RSChangeVQA_Fallback
from app.models.optical_sar import RSOpticalSAR_Model, RSOpticalSAR_Fallback


def init_default_models():
    """Register default models into the global registry singleton."""
    registry = get_model_registry()

    # 1. Register VQA adapters
    try:
        registry.register(RSVQA_BLIP2())
    except Exception:
        pass
    registry.register(RSVQA_Fallback())

    # 2. Register Captioning adapters
    try:
        registry.register(RSCaptioning_BLIP2())
    except Exception:
        pass
    registry.register(RSCaptioning_Fallback())

    # 3. Register Visual Grounding adapters
    try:
        registry.register(RSGrounding_GroundingDINO())
    except Exception:
        pass
    registry.register(RSGrounding_Fallback())

    # 4. Register Change Detection adapters
    try:
        registry.register(RSChangeDetection_BIT())
    except Exception:
        pass
    registry.register(RSChangeDetection_Fallback())

    # 5. Register Change VQA adapters
    try:
        registry.register(RSChangeVQA_Model())
    except Exception:
        pass
    registry.register(RSChangeVQA_Fallback())

    # 6. Register Optical-SAR Fusion adapters
    try:
        registry.register(RSOpticalSAR_Model())
    except Exception:
        pass
    registry.register(RSOpticalSAR_Fallback())


# Initialize default models on package import
init_default_models()

__all__ = [
    "TaskType",
    "InputType",
    "ConfidenceLevel",
    "ModelInfo",
    "ModelInput",
    "ModelOutput",
    "RemoteSensingModel",
    "ModelRegistry",
    "get_model_registry",
    "ModelNotFoundError",
    "RSVQA_BLIP2",
    "RSVQA_Fallback",
    "RSCaptioning_BLIP2",
    "RSCaptioning_Fallback",
    "RSGrounding_GroundingDINO",
    "RSGrounding_Fallback",
    "RSChangeDetection_BIT",
    "RSChangeDetection_Fallback",
    "RSChangeVQA_Model",
    "RSChangeVQA_Fallback",
    "RSOpticalSAR_Model",
    "RSOpticalSAR_Fallback",
]
