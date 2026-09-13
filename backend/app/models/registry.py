"""SatQuery AI — Model Registry.

Central discovery and management registry for specialist remote-sensing models.
Supports dynamic discovery, priority matching, and seamless fallback.
"""

from typing import Dict, List, Optional
from app.models.base import (
    InputType,
    ModelInfo,
    RemoteSensingModel,
    TaskType,
)
from app.utils.logging import get_logger

logger = get_logger("models.registry")


class ModelNotFoundError(Exception):
    """Raised when no compatible model can be resolved."""
    pass


class ModelRegistry:
    """Central registry for specialist remote-sensing models."""

    def __init__(self):
        self._models: Dict[str, RemoteSensingModel] = {}
        self._fallbacks: Dict[str, RemoteSensingModel] = {}

    def register(self, model: RemoteSensingModel) -> None:
        """Register a model in the registry."""
        info = model.info
        key = f"{info.name}:{info.version}"

        if info.is_fallback:
            self._fallbacks[key] = model
            logger.info("fallback_model_registered", model=info.name, version=info.version)
        else:
            self._models[key] = model
            logger.info("production_model_registered", model=info.name, version=info.version)

    def get_model(self, name: str, version: Optional[str] = None) -> Optional[RemoteSensingModel]:
        """Get a specific model by name and optional version."""
        for collection in (self._models, self._fallbacks):
            for key, model in collection.items():
                if version:
                    if key == f"{name}:{version}":
                        return model
                else:
                    if model.info.name == name:
                        return model
        return None

    def find_models(
        self,
        task: TaskType,
        input_type: InputType,
    ) -> List[RemoteSensingModel]:
        """Find models supporting the given task and input configuration.

        Returns production models first, followed by fallbacks.
        """
        results: List[RemoteSensingModel] = []

        # Production models first
        for model in self._models.values():
            info = model.info
            if task in info.supported_tasks and input_type in info.supported_inputs:
                results.append(model)

        # Fallbacks second
        for model in self._fallbacks.values():
            info = model.info
            if task in info.supported_tasks and input_type in info.supported_inputs:
                results.append(model)

        return results

    def select_best_model(
        self,
        task: TaskType,
        input_type: InputType,
    ) -> RemoteSensingModel:
        """Select and verify the best available model for task and input type."""
        candidates = self.find_models(task, input_type)

        if not candidates:
            raise ModelNotFoundError(
                f"No model registered for task={task.value}, input_type={input_type.value}"
            )

        # Attempt to load first candidate, falling back on failure
        for model in candidates:
            try:
                if not model.is_loaded:
                    model.load()
                return model
            except Exception as e:
                logger.warning(
                    "model_load_failed_trying_next",
                    model=model.info.name,
                    error=str(e),
                )
                continue

        raise ModelNotFoundError(f"All candidates failed to load for {task.value}")

    def list_models(self) -> List[ModelInfo]:
        """List metadata for all registered models."""
        models: List[ModelInfo] = []
        for model in self._models.values():
            models.append(model.info)
        for model in self._fallbacks.values():
            models.append(model.info)
        return models


# Global singleton registry instance
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Return or initialize the global ModelRegistry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
