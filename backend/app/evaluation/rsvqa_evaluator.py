"""SatQuery AI — RSVQA Benchmark Evaluator.

Evaluates single-image Remote Sensing Visual Question Answering (RSVQA)
across presence, comparison, count, and land-cover categories.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from app.evaluation.metrics import compute_exact_match, compute_token_accuracy
from app.models.base import InputType, ModelInput, TaskType
from app.models.registry import get_model_registry
from app.utils.logging import get_logger

logger = get_logger("evaluation.rsvqa")


@dataclass
class RSVQAItem:
    """Individual question-answer sample in RSVQA benchmark."""
    sample_id: str
    image: Image.Image
    question: str
    ground_truth_answer: str
    category: str = "general"  # presence, comparison, count, land_cover


@dataclass
class RSVQAEvaluationReport:
    """Evaluation output metrics for RSVQA."""
    total_samples: int
    overall_exact_match: float
    overall_token_accuracy: float
    category_accuracies: Dict[str, Dict[str, float]]
    average_latency_ms: float
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": "RSVQA",
            "total_samples": self.total_samples,
            "overall_exact_match": round(self.overall_exact_match, 4),
            "overall_token_accuracy": round(self.overall_token_accuracy, 4),
            "category_accuracies": self.category_accuracies,
            "average_latency_ms": round(self.average_latency_ms, 2),
        }


class RSVQAEvaluator:
    """Automated evaluation harness for RSVQA dataset."""

    @classmethod
    def get_synthetic_benchmark_suite(cls) -> List[RSVQAItem]:
        """Generate high-fidelity representative RSVQA benchmark samples."""
        suite: List[RSVQAItem] = []

        # 1. Forest scene (presence)
        forest_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        forest_arr[:, :, 0] = 30
        forest_arr[:, :, 1] = 185
        forest_arr[:, :, 2] = 40
        suite.append(
            RSVQAItem(
                sample_id="rsvqa_001",
                image=Image.fromarray(forest_arr),
                question="Is there vegetation or forest canopy present?",
                ground_truth_answer="Yes, the area features dense vegetation and healthy forest canopy.",
                category="presence",
            )
        )

        # 2. Water reservoir (presence)
        water_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        water_arr[:, :, 0] = 20
        water_arr[:, :, 1] = 45
        water_arr[:, :, 2] = 190
        suite.append(
            RSVQAItem(
                sample_id="rsvqa_002",
                image=Image.fromarray(water_arr),
                question="What is the dominant surface water condition?",
                ground_truth_answer="Open water body detected with low surface reflectance and distinct shoreline boundaries.",
                category="land_cover",
            )
        )

        # 3. Urban built-up (comparison)
        urban_arr = np.full((64, 64, 3), 160, dtype=np.uint8)
        suite.append(
            RSVQAItem(
                sample_id="rsvqa_003",
                image=Image.fromarray(urban_arr),
                question="Is this an urban built-up area or undeveloped barren land?",
                ground_truth_answer="Urban built-up area characterized by impervious surfaces and structural scattering.",
                category="comparison",
            )
        )

        return suite

    @classmethod
    def evaluate(
        cls,
        samples: Optional[List[RSVQAItem]] = None,
    ) -> RSVQAEvaluationReport:
        """Run complete evaluation over RSVQA items."""
        items = samples or cls.get_synthetic_benchmark_suite()
        registry = get_model_registry()
        model = registry.select_best_model(TaskType.VQA, InputType.SINGLE_OPTICAL)

        exact_matches = []
        token_accs = []
        category_metrics: Dict[str, List[Dict[str, float]]] = {}
        durations = []
        detailed_records = []

        for item in items:
            t0 = time.perf_counter()
            model_input = ModelInput(
                images=[item.image],
                query=item.question,
            )
            output = model.predict(model_input)
            latency = (time.perf_counter() - t0) * 1000.0
            durations.append(latency)

            pred = output.answer
            ref = item.ground_truth_answer

            em = compute_exact_match(pred, ref)
            tok_acc = compute_token_accuracy(pred, ref)

            exact_matches.append(em)
            token_accs.append(tok_acc)

            cat = item.category
            if cat not in category_metrics:
                category_metrics[cat] = []
            category_metrics[cat].append({"em": em, "tok_acc": tok_acc})

            detailed_records.append({
                "sample_id": item.sample_id,
                "question": item.question,
                "prediction": pred,
                "ground_truth": ref,
                "category": cat,
                "exact_match": em,
                "token_accuracy": tok_acc,
                "latency_ms": round(latency, 2),
            })

        # Calculate category stratifications
        cat_summaries = {}
        for cat, scores in category_metrics.items():
            cat_summaries[cat] = {
                "exact_match": round(sum(s["em"] for s in scores) / len(scores), 4),
                "token_accuracy": round(sum(s["tok_acc"] for s in scores) / len(scores), 4),
                "count": len(scores),
            }

        return RSVQAEvaluationReport(
            total_samples=len(items),
            overall_exact_match=sum(exact_matches) / max(1, len(exact_matches)),
            overall_token_accuracy=sum(token_accs) / max(1, len(token_accs)),
            category_accuracies=cat_summaries,
            average_latency_ms=sum(durations) / max(1, len(durations)),
            detailed_results=detailed_records,
        )
