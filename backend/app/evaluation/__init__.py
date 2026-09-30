"""SatQuery AI Evaluation Package."""

from app.evaluation.cdvqa_evaluator import CDVQAEvaluationReport, CDVQAEvaluator, CDVQAItem
from app.evaluation.levir_cd_evaluator import LEVIRCDCEvaluator, LEVIRCDItem, LEVIRCDReport
from app.evaluation.metrics import (
    compute_bleu,
    compute_box_iou,
    compute_change_detection_metrics,
    compute_exact_match,
    compute_grounding_map,
    compute_mask_iou,
    compute_rouge_l,
    compute_routing_accuracy,
    compute_token_accuracy,
)
from app.evaluation.rsvqa_evaluator import RSVQAEvaluationReport, RSVQAEvaluator, RSVQAItem
from app.evaluation.runner import run_benchmarks
from app.evaluation.vrsbench_evaluator import (
    VRSBenchCaptionSample,
    VRSBenchEvaluationReport,
    VRSBenchEvaluator,
    VRSBenchGroundingSample,
)

__all__ = [
    "CDVQAEvaluationReport",
    "CDVQAEvaluator",
    "CDVQAItem",
    "LEVIRCDCEvaluator",
    "LEVIRCDItem",
    "LEVIRCDReport",
    "RSVQAEvaluationReport",
    "RSVQAEvaluator",
    "RSVQAItem",
    "VRSBenchCaptionSample",
    "VRSBenchEvaluationReport",
    "VRSBenchEvaluator",
    "VRSBenchGroundingSample",
    "compute_bleu",
    "compute_box_iou",
    "compute_change_detection_metrics",
    "compute_exact_match",
    "compute_grounding_map",
    "compute_mask_iou",
    "compute_rouge_l",
    "compute_routing_accuracy",
    "compute_token_accuracy",
    "run_benchmarks",
]
