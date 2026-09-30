import numpy as np
import pytest
from PIL import Image

from app.evaluation.levir_cd_evaluator import LEVIRCDCEvaluator, LEVIRCDItem, LEVIRCDReport
from app.evaluation.metrics import compute_change_detection_metrics, compute_mask_iou
from app.evaluation.runner import run_benchmarks


def test_change_detection_metrics_perfect():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1

    metrics = compute_change_detection_metrics(pred_mask=mask, target_mask=mask)
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["oa"] == 1.0
    assert metrics["kappa"] == 1.0


def test_change_detection_metrics_partial():
    target = np.zeros((100, 100), dtype=np.uint8)
    target[20:60, 20:60] = 1  # 40x40 = 1600 pixels

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[20:60, 20:40] = 1   # 40x20 = 800 pixels (half of target)

    metrics = compute_change_detection_metrics(pred_mask=pred, target_mask=target)
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert 0.65 < metrics["f1"] < 0.68
    assert metrics["iou"] == 0.5


def test_levir_cd_synthetic_evaluator():
    report = LEVIRCDCEvaluator.evaluate(use_real_if_available=False)
    assert isinstance(report, LEVIRCDReport)
    assert report.total_samples == 3
    assert report.mean_overall_accuracy > 0.70
    assert report.average_latency_ms >= 0.0

    d = report.to_dict()
    assert d["benchmark"] == "LEVIR-CD"
    assert d["task"] == "building_change_detection"
    assert "mean_change_iou" in d
    assert "mean_f1" in d
    assert "mean_precision" in d


def test_levir_cd_real_dataset_loader():
    items = LEVIRCDCEvaluator.load_real_benchmark_suite(dataset_dir="datasets/levir_cd", split="test", max_samples=3)
    if items:
        assert len(items) <= 3
        for item in items:
            assert isinstance(item, LEVIRCDItem)
            assert item.image_t1.size == (512, 512) or item.image_t1.size == (1024, 1024)
            assert item.ground_truth_mask is not None
            assert item.ground_truth_mask.shape == (item.image_t1.height, item.image_t1.width)


def test_levir_cd_runner_cli(tmp_path):
    summary = run_benchmarks(benchmarks=["levir_cd"], output_dir=tmp_path)
    assert "levir_cd" in summary["benchmarks"]
    assert "agentic_routing" in summary["benchmarks"]
    levir_res = summary["benchmarks"]["levir_cd"]
    assert levir_res["benchmark"] == "LEVIR-CD"
    assert "mean_change_iou" in levir_res

    # Check generated files
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "evaluation_report.md").exists()
    report_content = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    assert "LEVIR-CD" in report_content
    assert "Building Change Detection" in report_content
