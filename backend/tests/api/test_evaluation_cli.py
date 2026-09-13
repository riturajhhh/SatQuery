import json
from pathlib import Path
import pytest

from app.evaluation.runner import run_benchmarks


def test_evaluation_runner_pipeline(tmp_path: Path):
    """Verify evaluation runner generates benchmark_summary.json and evaluation_report.md."""
    output_dir = tmp_path / "eval_results"

    summary = run_benchmarks(benchmarks=["all"], output_dir=output_dir)

    assert "benchmarks" in summary
    assert "rsvqa" in summary["benchmarks"]
    assert "vrsbench" in summary["benchmarks"]
    assert "cdvqa" in summary["benchmarks"]
    assert "agentic_routing" in summary["benchmarks"]

    # Verify files created
    json_path = output_dir / "benchmark_summary.json"
    md_path = output_dir / "evaluation_report.md"

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert "timestamp" in loaded
        assert "benchmarks" in loaded

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        assert "# SatQuery AI — Benchmark Evaluation Report" in md_content
        assert "| **RSVQA** |" in md_content
        assert "| **VRSBench (Grounding)** |" in md_content
        assert "| **CDVQA** |" in md_content
        assert "| **Agentic Planner** |" in md_content
