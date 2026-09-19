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


def test_rsvqa_real_dataset_loader():
    real_samples = RSVQAEvaluator.load_real_benchmark_suite(max_samples=3)
    if real_samples:
        assert len(real_samples) == 3
        for item in real_samples:
            assert item.sample_id.startswith("rsvqa_")
            assert item.question
            assert item.ground_truth_answer
            assert item.image.size == (256, 256)



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
    assert "vegetation_loss" in report.change_type_accuracies or "change_or_not" in report.change_type_accuracies or len(report.change_type_accuracies) > 0


def test_cdvqa_real_dataset_loader():
    real_samples = CDVQAEvaluator.load_real_benchmark_suite(max_samples=3)
    if real_samples:
        assert len(real_samples) == 3
        for item in real_samples:
            assert item.sample_id.startswith("cdvqa_")
            assert item.question
            assert item.ground_truth_answer
            assert item.image_t1 is not None
            assert item.image_t2 is not None

