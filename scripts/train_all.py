"""SatQuery AI — Unified Model Training Orchestrator.

Orchestrates training across single-image VQA (RSVQA) and bi-temporal Change-VQA (CDVQA)
models, profiles training loss and validation convergence, and produces a consolidated
training benchmark report.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time


def run_training_command(cmd: list[str], task_name: str) -> bool:
    print("\n" + "=" * 65)
    print(f"Starting Training: {task_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 65)
    t0 = time.perf_counter()
    res = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if res.returncode == 0:
        print(f"[SUCCESS] {task_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"[FAILURE] {task_name} failed with exit code {res.returncode}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — Unified Model Training Orchestrator")
    parser.add_argument("--task", choices=["all", "vqa", "vrsbench", "cdvqa"], default="all", help="Models to train")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--limit-samples", type=int, default=1000, help="Max training samples per task")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto/cpu/cuda)")

    args = parser.parse_args()
    python_exe = sys.executable

    tasks = ["vqa", "vrsbench", "cdvqa"] if args.task == "all" else [args.task]
    results = {}

    if "vqa" in tasks:
        cmd = [
            python_exe, "scripts/train_vqa.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--limit-samples", str(args.limit_samples),
            "--device", args.device,
        ]
        results["RSVQA"] = run_training_command(cmd, "RSVQA Single-Image VQA Model")

    if "vrsbench" in tasks:
        cmd = [
            python_exe, "scripts/train_vrsbench.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--limit-samples", str(args.limit_samples),
            "--device", args.device,
        ]
        results["VRSBench"] = run_training_command(cmd, "VRSBench Specialist Models (Grounding & Captioning)")

    if "cdvqa" in tasks:
        cmd = [
            python_exe, "scripts/train_cdvqa.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--limit-samples", str(args.limit_samples),
            "--device", args.device,
        ]
        results["CDVQA"] = run_training_command(cmd, "CDVQA Bi-Temporal Change Model")

    print("\n" + "=" * 65)
    print("SatQuery AI — Training Summary")
    print("=" * 65)
    all_ok = True
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  - {name}: {status}")
        if not success:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
