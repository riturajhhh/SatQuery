"""SatQuery AI — Base Model Architecture.

Defines the abstract base classes, dataclasses, and enums for all
specialist remote-sensing models in the registry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    """Supported remote-sensing analytical tasks."""
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR = "optical_sar_analysis"


class InputType(str, Enum):
    """Supported input image configurations."""
    SINGLE_OPTICAL = "single_optical"
    SINGLE_SAR = "single_sar"
    SINGLE_MULTISPECTRAL = "single_multispectral"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR_PAIR = "optical_sar_pair"


class ConfidenceLevel(str, Enum):
    """Calibrated confidence classifications."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ModelInfo:
    """Metadata describing a registered model."""
    name: str
    version: str
    base_model: str
    adapter: Optional[str] = None
    description: str = ""
    supported_tasks: List[TaskType] = field(default_factory=list)
    supported_inputs: List[InputType] = field(default_factory=list)
    is_fallback: bool = False
    device: str = "cpu"


@dataclass
class ModelInput:
    """Standardized input payload to any specialist model."""
    images: List[Any]  # PIL.Image instances or numpy arrays
    query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    file_paths: Optional[List[str]] = None


@dataclass
class ModelOutput:
    """Standardized output produced by any specialist model."""
    answer: Optional[str] = None
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN
    evidence: Optional[Dict[str, Any]] = None  # Masks, bounding boxes, attention maps
    raw_output: Optional[Any] = None
    model_info: Optional[ModelInfo] = None
    execution_time_ms: float = 0.0
    is_fallback: bool = False


class RemoteSensingModel(ABC):
    """Abstract base class for all remote-sensing specialist models."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model specification metadata."""
        ...

    @abstractmethod
    def validate_input(self, model_input: ModelInput) -> bool:
        """Validate that input data is compatible with this model."""
        ...

    @abstractmethod
    def predict(self, model_input: ModelInput) -> ModelOutput:
        """Execute inference and return answer with confidence estimation."""
        ...

    @abstractmethod
    def explain(self, model_input: ModelInput, output: ModelOutput) -> Dict[str, Any]:
        """Generate visual or spatial evidence (saliency, regions, bounding boxes)."""
        ...

    def load(self) -> None:
        """Load weights into memory. Default is no-op."""
        pass

    def unload(self) -> None:
        """Unload weights from memory. Default is no-op."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check whether model weights are currently memory-resident."""
        return True
