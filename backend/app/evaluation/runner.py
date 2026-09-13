"""SatQuery AI — Benchmark Evaluation Runner & CLI.

Executes reproducible evaluation pipelines across RSVQA, VRSBench, and CDVQA benchmarks,
computes quantitative metrics, profiles agentic routing, and exports structured reports.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.evaluation.cdvqa_evaluator import CDVQAEvaluator
from app.evaluation.metrics import compute_routing_accuracy
from app.evaluation.rsvqa_evaluator import RSVQAEvaluator
from app.evaluation.vrsbench_evaluator import VRSBenchEvaluator
from app.services.agentic_planner import AgenticPlanner
from app.utils.logging import get_logger

logger = get_logger("evaluation.runner")


def evaluate_agentic_routing() -> Dict[str, Any]:
    """Evaluate agentic planner intent decomposition and task routing accuracy."""
    test_cases = [
        ("What is the water surface condition?", 1, "vqa"),
        ("Describe this satellite scene in detail", 1, "captioning"),
        ("Locate the runway in this image", 1, "grounding"),
        ("Show difference between these two images", 2, "change_detection"),
        ("Did urban buildings expand between 2020 and 2023?", 2, "change_vqa"),
        ("Pierce cloud cover with SAR radar backscatter", 2, "optical_sar_analysis"),
        ("Locate the forest and analyze temporal changes over time", 2, "agentic_multi_task"),
    ]

    predicted = []
    ground_truth = []

    for query, num_files, expected_task in test_cases:
        modalities = ["optical", "sar"] if "sar" in query.lower() else []
        plan = AgenticPlanner.create_plan(query=query, num_files=num_files, modalities=modalities)
        if len(plan.steps) > 1:
            actual_task = "agentic_multi_task"
        else:
            actual_task = plan.steps[0].task.value
        predicted.append(actual_task)
        ground_truth.append(expected_task)

    accuracy = compute_routing_accuracy(predicted, ground_truth)
    return {
        "total_queries": len(test_cases),
        "routing_accuracy": round(accuracy, 4),
        "test_cases": [
            {"query": q, "expected": g, "predicted": p, "passed": p == g}
            for (q, _, _), g, p in zip(test_cases, ground_truth, predicted)
        ],
    }


def run_benchmarks(
    benchmarks: Sequence[str] = ("all",),
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute evaluation across selected benchmarks and export reports."""
    out_dir = Path(output_dir or "outputs/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    do_all = "all" in benchmarks
    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
    }

    # 1. RSVQA
    if do_all or "rsvqa" in benchmarks:
        logger.info("evaluating_benchmark", name="RSVQA")
        rsvqa_report = RSVQAEvaluator.evaluate()
        summary["benchmarks"]["rsvqa"] = rsvqa_report.to_dict()

    # 2. VRSBench
    if do_all or "vrsbench" in benchmarks:
        logger.info("evaluating_benchmark", name="VRSBench")
        vrs_report = VRSBenchEvaluator.evaluate()
        summary["benchmarks"]["vrsbench"] = vrs_report.to_dict()

    # 3. CDVQA
    if do_all or "cdvqa" in benchmarks:
        logger.info("evaluating_benchmark", name="CDVQA")
        cdvqa_report = CDVQAEvaluator.evaluate()
        summary["benchmarks"]["cdvqa"] = cdvqa_report.to_dict()

    # 4. Agentic Routing Evaluation
    logger.info("evaluating_benchmark", name="AgenticRouting")
    routing_results = evaluate_agentic_routing()
    summary["benchmarks"]["agentic_routing"] = routing_results

    # Save JSON summary
    json_path = out_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Report
    md_path = out_dir / "evaluation_report.md"
    generate_markdown_report(summary, md_path)

    logger.info("benchmarks_completed", output_dir=str(out_dir))
    return summary


def generate_markdown_report(summary: Dict[str, Any], output_path: Path):
    """Generate a clean markdown table report from benchmark results."""
    lines = [
        "# SatQuery AI — Benchmark Evaluation Report",
        "",
        f"**Generated:** {summary['timestamp']}",
        "",
        "## Summary of Results",
        "",
        "| Benchmark | Task Focus | Key Metric | Score | Latency (ms) |",
        "|:----------|:-----------|:-----------|:------|:-------------|",
    ]

    bms = summary["benchmarks"]
    if "rsvqa" in bms:
        r = bms["rsvqa"]
        lines.append(
            f"| **RSVQA** | Single-Image VQA | Token Accuracy | {r['overall_token_accuracy'] * 100:.1f}% | {r['average_latency_ms']:.1f}ms |"
        )
    if "vrsbench" in bms:
        v = bms["vrsbench"]
        lines.append(
            f"| **VRSBench (Grounding)** | Spatial Localization | Mean IoU | {v['grounding']['mean_iou']:.3f} (mAP: {v['grounding']['map_at_50'] * 100:.1f}%) | {v['average_latency_ms']:.1f}ms |"
        )
        lines.append(
            f"| **VRSBench (Captioning)** | Scene Description | ROUGE-L F1 | {v['captioning']['rouge_l_f1']:.3f} (BLEU-1: {v['captioning']['bleu_1']:.3f}) | {v['average_latency_ms']:.1f}ms |"
        )
    if "cdvqa" in bms:
        c = bms["cdvqa"]
        lines.append(
            f"| **CDVQA** | Bi-Temporal Change Q&A | Token Accuracy | {c['overall_token_accuracy'] * 100:.1f}% | {c['average_latency_ms']:.1f}ms |"
        )
    if "agentic_routing" in bms:
        ar = bms["agentic_routing"]
        lines.append(
            f"| **Agentic Planner** | Intent Routing | Accuracy | {ar['routing_accuracy'] * 100:.1f}% | < 1.0ms |"
        )

    lines.extend([
        "",
        "## Detailed Benchmark Stratifications",
        "",
    ])

    if "rsvqa" in bms:
        lines.append("### RSVQA Category Breakdown")
        lines.append("| Category | Samples | Token Accuracy | Exact Match |")
        lines.append("|:---------|:--------|:---------------|:------------|")
        for cat, scores in bms["rsvqa"].get("category_accuracies", {}).items():
            lines.append(
                f"| {cat.title()} | {scores['count']} | {scores['token_accuracy'] * 100:.1f}% | {scores['exact_match'] * 100:.1f}% |"
            )
        lines.append("")

    if "vrsbench" in bms:
        lines.append("### VRSBench Metric Details")
        lines.append("- **Visual Grounding Mean IoU:** " + str(bms["vrsbench"]["grounding"]["mean_iou"]))
        lines.append("- **Visual Grounding mAP@0.5:** " + str(bms["vrsbench"]["grounding"]["map_at_50"]))
        lines.append("- **Captioning BLEU-1 to BLEU-4:** " + f"{bms['vrsbench']['captioning']['bleu_1']} / {bms['vrsbench']['captioning']['bleu_2']} / {bms['vrsbench']['captioning']['bleu_3']} / {bms['vrsbench']['captioning']['bleu_4']}")
        lines.append("- **Captioning ROUGE-L F1:** " + str(bms["vrsbench"]["captioning"]["rouge_l_f1"]))
        lines.append("")

    if "cdvqa" in bms:
        lines.append("### CDVQA Change Type Breakdown")
        lines.append("| Change Scenario | Samples | Token Accuracy |")
        lines.append("|:----------------|:--------|:---------------|")
        for ctype, scores in bms["cdvqa"].get("change_type_accuracies", {}).items():
            lines.append(f"| {ctype.replace('_', ' ').title()} | {scores['count']} | {scores['token_accuracy'] * 100:.1f}% |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — Benchmark Evaluation Runner")
    parser.add_argument(
        "--benchmark",
        "-b",
        nargs="+",
        default=["all"],
        choices=["all", "rsvqa", "vrsbench", "cdvqa"],
        help="Benchmarks to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs/eval_results"),
        help="Directory to save evaluation artifacts",
    )
    args = parser.parse_args()
    res = run_benchmarks(benchmarks=args.benchmark, output_dir=args.output_dir)
    print(f"\n[+] Benchmark Evaluation Completed! Output saved to: {args.output_dir}")
    print(f"    - RSVQA Token Acc: {res['benchmarks'].get('rsvqa', {}).get('overall_token_accuracy', 'N/A')}")
    print(f"    - VRSBench Mean IoU: {res['benchmarks'].get('vrsbench', {}).get('grounding', {}).get('mean_iou', 'N/A')}")
    print(f"    - CDVQA Token Acc: {res['benchmarks'].get('cdvqa', {}).get('overall_token_accuracy', 'N/A')}")
    print(f"    - Agent Routing Acc: {res['benchmarks'].get('agentic_routing', {}).get('routing_accuracy', 'N/A')}\n")


if __name__ == "__main__":
    main()
