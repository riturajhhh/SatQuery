"""SatQuery AI — RSVQA Benchmark Evaluator.

Evaluates single-image Remote Sensing Visual Question Answering (RSVQA)
across presence, comparison, count, and land-cover categories.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
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
    def load_real_benchmark_suite(
        cls,
        dataset_dir: Union[str, Path] = "datasets/rsvqa/lr",
        split: str = "val",
        max_samples: int = 10,
    ) -> List[RSVQAItem]:
        """Load benchmark samples from downloaded RSVQA dataset."""
        root = Path(dataset_dir)
        if not root.exists():
            for alt in [Path("../") / dataset_dir, Path(__file__).resolve().parents[3] / dataset_dir]:
                if alt.exists():
                    root = alt
                    break
        images_dir = root / "images"
        splits_dir = root / "splits"
        questions_file = splits_dir / f"{split}_questions.json"
        answers_file = splits_dir / f"{split}_answers.json"

        # Fallback to base files if split-specific files are not present
        if not questions_file.exists():
            questions_file = root / "questions" / "questions.json"
        if not answers_file.exists():
            answers_file = root / "answers" / "answers.json"

        if not (questions_file.exists() and answers_file.exists() and images_dir.exists()):
            logger.warning("rsvqa_dataset_not_found", path=str(root))
            return []

        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                q_data = json.load(f).get("questions", [])
            with open(answers_file, "r", encoding="utf-8") as f:
                a_data = json.load(f).get("answers", [])
        except Exception as e:
            logger.error("failed_to_load_rsvqa_annotations", error=str(e))
            return []

        # Map answer_id -> answer text
        ans_map = {a["id"]: a.get("answer", "") for a in a_data if a.get("active", True)}

        items: List[RSVQAItem] = []
        for q in q_data:
            if not q.get("active", True):
                continue

            img_id = q.get("img_id")
            img_path = None
            for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                cand = images_dir / f"{img_id}{ext}"
                if cand.exists():
                    img_path = cand
                    break

            if not img_path or not img_path.exists():
                continue

            ans_ids = q.get("answers_ids", [])
            answer_text = ans_map.get(ans_ids[0]) if ans_ids else None
            if not answer_text:
                continue

            try:
                with Image.open(img_path) as pil_img:
                    image_obj = pil_img.convert("RGB")
            except Exception:
                continue

            items.append(
                RSVQAItem(
                    sample_id=f"rsvqa_{q['id']}",
                    image=image_obj,
                    question=q.get("question", ""),
                    ground_truth_answer=str(answer_text),
                    category=q.get("type", "general"),
                )
            )

            if len(items) >= max_samples:
                break

        return items

    @classmethod
    def evaluate(
        cls,
        samples: Optional[List[RSVQAItem]] = None,
        use_real_if_available: bool = True,
        max_samples: int = 10,
        dataset_dir: Union[str, Path] = "datasets/rsvqa/lr",
    ) -> RSVQAEvaluationReport:
        """Run complete evaluation over RSVQA items."""
        if samples is not None:
            items = samples
        elif use_real_if_available:
            real_items = cls.load_real_benchmark_suite(dataset_dir=dataset_dir, max_samples=max_samples)
            items = real_items if real_items else cls.get_synthetic_benchmark_suite()
        else:
            items = cls.get_synthetic_benchmark_suite()
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
