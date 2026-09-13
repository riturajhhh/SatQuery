import pytest

from app.evaluation.rsvqa_evaluator import RSVQAEvaluator, RSVQAEvaluationReport
from app.evaluation.vrsbench_evaluator import VRSBenchEvaluator, VRSBenchEvaluationReport
from app.evaluation.cdvqa_evaluator import CDVQAEvaluator, CDVQAEvaluationReport


def test_rsvqa_evaluator():
    report = RSVQAEvaluator.evaluate()
    assert isinstance(report, RSVQAEvaluationReport)
    assert report.total_samples > 0
    assert report.overall_token_accuracy > 0.0
    assert report.average_latency_ms >= 0.0
    report_dict = report.to_dict()
    assert report_dict["benchmark"] == "RSVQA"
    assert "presence" in report.category_accuracies or "land_cover" in report.category_accuracies


def test_vrsbench_evaluator():
    report = VRSBenchEvaluator.evaluate()
    assert isinstance(report, VRSBenchEvaluationReport)
    assert report.grounding_samples > 0
    assert report.captioning_samples > 0
    assert report.mean_iou > 0.0
    assert report.map_at_50 > 0.0
    assert report.bleu_1 > 0.0
    assert report.rouge_l_f1 > 0.0
    report_dict = report.to_dict()
    assert report_dict["benchmark"] == "VRSBench"


def test_cdvqa_evaluator():
    report = CDVQAEvaluator.evaluate()
    assert isinstance(report, CDVQAEvaluationReport)
    assert report.total_samples > 0
    assert report.overall_token_accuracy > 0.0
    report_dict = report.to_dict()
    assert report_dict["benchmark"] == "CDVQA"
    assert "vegetation_loss" in report.change_type_accuracies
