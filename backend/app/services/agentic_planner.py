"""SatQuery AI — Agentic Reasoning Planner & Intent Classifier.

Decomposes complex, multi-intent remote-sensing queries into ordered execution plans,
extracts execution parameters, and orchestrates chained tool execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import re

from app.models.base import InputType, TaskType
from app.utils.logging import get_logger

logger = get_logger("services.agentic_planner")


class PlanStrategy(str, Enum):
    """Execution strategy category for an analysis plan."""
    DIRECT_SINGLE = "direct_single"
    MULTI_STEP_CHAIN = "multi_step_chain"
    CROSS_MODAL_FUSION = "cross_modal_fusion"


@dataclass
class PlanStep:
    """Individual atomic sub-task within an agentic execution plan."""
    step_id: int
    task: TaskType
    description: str
    target_phrase: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Complete decomposed execution plan for a user query."""
    query: str
    strategy: PlanStrategy
    steps: List[PlanStep]
    synthesizer_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgenticPlanner:
    """Autonomous reasoning planner for satellite vision-language analysis."""

    GROUNDING_TRIGGERS = [
        "highlight",
        "locate",
        "find the",
        "find all",
        "where is",
        "where are",
        "bounding box",
        "detect the",
        "detect all",
        "grounding",
        "show me the",
        "box the",
    ]

    CAPTION_TRIGGERS = [
        "describe",
        "caption",
        "summarize",
        "overview",
        "scene description",
        "what does this scene",
        "what is in this picture",
        "what does this image show",
        "tell me about this image",
        "explain this scene",
        "describe scene",
    ]

    CHANGE_VQA_TRIGGERS = [
        "what happened",
        "did ",
        "has ",
        "is there",
        "are there",
        "was there",
        "how much",
        "how many",
        "expand or shrink",
        "shrink or expand",
        "increase or decrease",
        "decrease or increase",
        "which area",
        "why did",
        "any new",
        "has the",
        "did the",
    ]

    CHANGE_TRIGGERS = [
        "change",
        "difference",
        "compare",
        "what changed",
        "how changed",
        "changed",
        "modification",
        "temporal",
        "deforestation",
        "loss",
        "expansion",
        "growth",
        "diff",
        "bitemporal",
        "bi-temporal",
        "shift",
    ]

    OPTICAL_SAR_TRIGGERS = [
        "optical-sar",
        "optical sar",
        "sar fusion",
        "radar fusion",
        "cloud penetration",
        "pierce cloud",
        "all-weather",
        "radar backscatter",
        "under cloud",
        "under the cloud",
        "through cloud",
        "through the cloud",
    ]

    @classmethod
    def extract_grounding_target(cls, query: str) -> str:
        """Extract target entity/phrase to localize from a query."""
        q = query.lower().strip()
        # Common patterns: "highlight the [target]", "locate [target]", "find the [target]"
        patterns = [
            r"(?:highlight|locate|find|detect|box|show me)(?:\s+the|\s+all)?\s+([a-zA-Z\s/-]+?)(?:\s+and|\s+in|\s+visible|\s+between|\.|$)",
            r"where (?:is|are)(?:\s+the)?\s+([a-zA-Z\s/-]+?)(?:\s+and|\s+in|\s+visible|\.|\?|$)",
        ]
        for pat in patterns:
            match = re.search(pat, q)
            if match:
                target = match.group(1).strip()
                if target and target not in ["the", "a", "an", "all"]:
                    return target
        return "target feature"

    @classmethod
    def create_plan(
        cls,
        query: str,
        num_files: Any = 1,
        modalities: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Decompose a natural language query into an actionable execution plan."""
        if isinstance(num_files, (list, tuple, set)):
            num_files = len(num_files)
        q = query.lower().strip()
        modalities = modalities or []

        is_optical_sar_pair = (
            num_files >= 2
            and "sar" in modalities
            and ("optical" in modalities or "multispectral" in modalities)
        )

        has_grounding = any(t in q for t in cls.GROUNDING_TRIGGERS)
        has_caption = any(t in q for t in cls.CAPTION_TRIGGERS)
        has_change = any(t in q for t in cls.CHANGE_TRIGGERS) or any(t in q for t in cls.CHANGE_VQA_TRIGGERS)
        has_sar = is_optical_sar_pair or any(t in q for t in cls.OPTICAL_SAR_TRIGGERS)

        # 1. Multi-Step Composite Query: Grounding + Change Analysis
        if has_grounding and has_change and num_files >= 2:
            target = cls.extract_grounding_target(query)
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.GROUNDING,
                    description=f"Localize and generate spatial bounding box overlay for '{target}'",
                    target_phrase=target,
                    parameters={"target_expression": target},
                ),
                PlanStep(
                    step_id=2,
                    task=TaskType.CHANGE_DETECTION,
                    description="Compute bi-temporal radiometric differences, change mask, and ground area statistics",
                    depends_on=[1],
                ),
                PlanStep(
                    step_id=3,
                    task=TaskType.CHANGE_VQA,
                    description=f"Synthesize contextual answer to '{query}' using localized regions and change statistics",
                    depends_on=[1, 2],
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.MULTI_STEP_CHAIN,
                steps=steps,
                synthesizer_prompt="Combine spatial bounding coordinates with temporal change metrics into a unified answer.",
                metadata={"composite_type": "grounding_change", "target": target},
            )

        # 2. Multi-Step Composite Query: Grounding + Captioning / Scene Description
        if has_grounding and has_caption:
            target = cls.extract_grounding_target(query)
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.GROUNDING,
                    description=f"Localize and generate spatial bounding box overlay for '{target}'",
                    target_phrase=target,
                    parameters={"target_expression": target},
                ),
                PlanStep(
                    step_id=2,
                    task=TaskType.CAPTIONING,
                    description="Perform multi-spectral land-cover scene interpretation and spatial relationship description",
                    depends_on=[1],
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.MULTI_STEP_CHAIN,
                steps=steps,
                synthesizer_prompt="Integrate localized bounding box coordinates into the broader scene description.",
                metadata={"composite_type": "grounding_caption", "target": target},
            )

        # 3. Optical-SAR Multi-Sensor Cross-Modal Analysis
        if has_sar and num_files >= 2:
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.OPTICAL_SAR,
                    description="Execute Optical-SAR cross-modal fusion, optical cloud detection, and radar penetration analysis",
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.CROSS_MODAL_FUSION,
                steps=steps,
                metadata={"modality_synergy": "optical_sar"},
            )

        # 4. Bi-Temporal Temporal Question (Change VQA)
        if num_files >= 2 and (any(t in q for t in cls.CHANGE_VQA_TRIGGERS) or q.endswith("?")):
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.CHANGE_VQA,
                    description="Analyze bi-temporal changes and answer targeted natural-language temporal question",
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.DIRECT_SINGLE,
                steps=steps,
            )

        # 5. Bi-Temporal Change Detection
        if num_files >= 2:
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.CHANGE_DETECTION,
                    description="Compute bi-temporal difference map, adaptive change mask, and area statistics",
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.DIRECT_SINGLE,
                steps=steps,
            )

        # 6. Single-Image Grounding
        if has_grounding:
            target = cls.extract_grounding_target(query)
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.GROUNDING,
                    description=f"Identify and highlight spatial features matching '{target}'",
                    target_phrase=target,
                    parameters={"target_expression": target},
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.DIRECT_SINGLE,
                steps=steps,
                metadata={"target": target},
            )

        # 7. Single-Image Captioning
        if has_caption:
            steps = [
                PlanStep(
                    step_id=1,
                    task=TaskType.CAPTIONING,
                    description="Generate comprehensive remote-sensing scene description and land-cover breakdown",
                ),
            ]
            return ExecutionPlan(
                query=query,
                strategy=PlanStrategy.DIRECT_SINGLE,
                steps=steps,
            )

        # 8. Single-Image VQA
        steps = [
            PlanStep(
                step_id=1,
                task=TaskType.VQA,
                description=f"Perform visual question answering for query: '{query}'",
            ),
        ]
        return ExecutionPlan(
            query=query,
            strategy=PlanStrategy.DIRECT_SINGLE,
            steps=steps,
        )
