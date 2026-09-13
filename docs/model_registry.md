# SatQuery AI — Model Registry Design

## 1. Overview

The Model Registry is the central component that manages all specialist remote-sensing models. It enables:

- **Discovery** — the agent queries the registry to find models matching a task and input type
- **Swappability** — models can be replaced without changing application code
- **Fallback** — automatic fallback from production to demo models
- **Versioning** — track model versions and configurations

---

## 2. Base Interface

Every model in the registry must implement the `RemoteSensingModel` abstract base class:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class TaskType(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR = "optical_sar_analysis"


class InputType(str, Enum):
    SINGLE_OPTICAL = "single_optical"
    SINGLE_SAR = "single_sar"
    SINGLE_MULTISPECTRAL = "single_multispectral"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR_PAIR = "optical_sar_pair"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class ModelInfo:
    name: str
    version: str
    base_model: str
    adapter: Optional[str] = None
    description: str = ""
    supported_tasks: List[TaskType] = field(default_factory=list)
    supported_inputs: List[InputType] = field(default_factory=list)
    is_fallback: bool = False
    device: str = "auto"


@dataclass
class ModelInput:
    images: List[Any]          # PIL Images or numpy arrays
    query: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ModelOutput:
    answer: Optional[str] = None
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN
    evidence: Optional[Dict] = None      # masks, boxes, maps
    raw_output: Optional[Any] = None
    model_info: Optional[ModelInfo] = None
    execution_time_ms: float = 0.0


class RemoteSensingModel(ABC):
    """Abstract base class for all remote-sensing specialist models."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model metadata."""
        ...

    @abstractmethod
    def validate_input(self, model_input: ModelInput) -> bool:
        """Validate that the input is compatible with this model.

        Returns True if valid, raises ValueError with explanation if not.
        """
        ...

    @abstractmethod
    def predict(self, model_input: ModelInput) -> ModelOutput:
        """Run inference and return results.

        Must include confidence estimation.
        """
        ...

    @abstractmethod
    def explain(self, model_input: ModelInput, output: ModelOutput) -> Dict:
        """Generate explanations/evidence for the prediction.

        Returns evidence dict (attention maps, GradCAM, bounding boxes, etc.)
        """
        ...

    def load(self) -> None:
        """Load model weights into memory. Called on first use."""
        pass

    def unload(self) -> None:
        """Unload model weights from memory."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model weights are currently loaded."""
        return False
```

---

## 3. Registry Implementation

```python
class ModelRegistry:
    """Central registry for all remote-sensing models.

    Models self-register and are queried by the agentic controller
    based on task type and input configuration.
    """

    def __init__(self):
        self._models: Dict[str, RemoteSensingModel] = {}
        self._fallbacks: Dict[str, RemoteSensingModel] = {}

    def register(self, model: RemoteSensingModel) -> None:
        """Register a model in the registry."""
        info = model.info
        key = f"{info.name}:{info.version}"

        if info.is_fallback:
            self._fallbacks[key] = model
        else:
            self._models[key] = model

    def get_model(self, name: str, version: str = None) -> RemoteSensingModel:
        """Get a specific model by name and optional version."""
        ...

    def find_models(
        self,
        task: TaskType,
        input_type: InputType
    ) -> List[RemoteSensingModel]:
        """Find all models supporting the given task and input type.

        Returns models sorted by priority (production first, fallbacks last).
        """
        results = []

        # Search production models first
        for model in self._models.values():
            info = model.info
            if task in info.supported_tasks and input_type in info.supported_inputs:
                results.append(model)

        # Then fallback models
        for model in self._fallbacks.values():
            info = model.info
            if task in info.supported_tasks and input_type in info.supported_inputs:
                results.append(model)

        return results

    def select_best_model(
        self,
        task: TaskType,
        input_type: InputType
    ) -> RemoteSensingModel:
        """Select the best available model for the task.

        Tries production models first, falls back to demo models.
        Raises ModelNotFoundError if no compatible model exists.
        """
        candidates = self.find_models(task, input_type)

        if not candidates:
            raise ModelNotFoundError(
                f"No model available for task={task}, input={input_type}"
            )

        # Try to load the first available model
        for model in candidates:
            try:
                if not model.is_loaded:
                    model.load()
                return model
            except Exception:
                continue

        raise ModelNotFoundError("All compatible models failed to load")

    def list_models(self) -> List[ModelInfo]:
        """List all registered models with their metadata."""
        models = []
        for model in self._models.values():
            models.append(model.info)
        for model in self._fallbacks.values():
            models.append(model.info)
        return models
```

---

## 4. Model Adapters

Each specialist model is wrapped in an adapter that implements `RemoteSensingModel`:

### 4.1 VQA Model Adapter

```python
class RSVQA_BLIP2(RemoteSensingModel):
    """Remote-sensing VQA using BLIP-2 with LoRA adapter.

    Base: Salesforce/blip2-opt-2.7b
    Adapter: LoRA fine-tuned on RSVQA + BigEarthNet
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="blip2-rs-vqa",
            version="1.0.0",
            base_model="Salesforce/blip2-opt-2.7b",
            adapter="lora",
            description="BLIP-2 with LoRA adaptation for remote-sensing VQA",
            supported_tasks=[TaskType.VQA],
            supported_inputs=[
                InputType.SINGLE_OPTICAL,
                InputType.SINGLE_MULTISPECTRAL,
            ],
        )
```

### 4.2 Captioning Model Adapter

```python
class RSCaptioning_BLIP2(RemoteSensingModel):
    """Remote-sensing image captioning using BLIP-2 with LoRA.

    Generates scene descriptions with land-cover interpretation.
    """
    ...
```

### 4.3 Grounding Model Adapter

```python
class RSGrounding_DINO(RemoteSensingModel):
    """Visual grounding using GroundingDINO.

    Locates objects/regions mentioned in text queries.
    """
    ...
```

### 4.4 Change Detection Model Adapter

```python
class BIT_ChangeDetection(RemoteSensingModel):
    """Binary change detection using BIT (Binary Image Transformer).

    Pretrained on remote-sensing change detection datasets.
    """
    ...
```

### 4.5 Change VQA Model Adapter

```python
class CDVQA_Model(RemoteSensingModel):
    """Change-based VQA combining change detection + VLM.

    Answers questions about changes between bi-temporal image pairs.
    """
    ...
```

### 4.6 Optical-SAR Fusion Model Adapter

```python
class OpticalSAR_Fusion(RemoteSensingModel):
    """Optical-SAR fusion with dual-encoder feature integration.

    Combines complementary information from optical and SAR imagery.
    """
    ...
```

---

## 5. Model Directory Structure

```
models/
├── vqa/
│   ├── config.json              # Model configuration
│   ├── lora_adapter/            # LoRA weights
│   └── README.md                # Model documentation
├── captioning/
│   ├── config.json
│   ├── lora_adapter/
│   └── README.md
├── grounding/
│   ├── config.json
│   └── README.md
├── change_detection/
│   ├── config.json
│   ├── weights/                 # Pretrained weights
│   └── README.md
├── change_vqa/
│   ├── config.json
│   ├── lora_adapter/
│   └── README.md
└── optical_sar/
    ├── config.json
    ├── weights/
    └── README.md
```

---

## 6. Fallback Strategy

```
Production Model (GPU, full precision)
        ↓ load failure or GPU unavailable
Lightweight Model (GPU, quantized / smaller variant)
        ↓ still unavailable
Demo/Fallback Model (CPU, minimal)
        ↓ still unavailable
Error: "Model unavailable — please check configuration"
```

The UI must clearly indicate when a fallback/demo model is being used:

```
⚠️ Demo/Fallback inference — results may have reduced accuracy
```

---

## 7. Model Loading Strategy

- **Lazy Loading** — models loaded on first inference request, not at startup
- **Model Cache** — loaded models remain in memory (configurable cache size)
- **Device Management** — automatic GPU/CPU selection based on `CUDA_VISIBLE_DEVICES` and model config
- **Memory Guard** — check available GPU memory before loading; fall back to CPU if insufficient
- **Concurrent Access** — thread-safe model access with locking for single-GPU scenarios

---

## 8. Adding a New Model

To add a new specialist model:

1. Create a new adapter class extending `RemoteSensingModel`
2. Implement all required methods (`info`, `validate_input`, `predict`, `explain`)
3. Place model weights in the appropriate `models/` subdirectory
4. Register the model in the startup configuration
5. The agent will automatically discover and use it based on task/input matching

No changes to the agent, API, or frontend are required.
