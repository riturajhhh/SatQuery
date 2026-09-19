"""SatQuery AI — CDVQA Dataset Downloader & Preprocessor.

Downloads and organizes the Change Detection Visual Question Answering (CDVQA) dataset
from the official GitHub repository (https://github.com/YZHJessica/CDVQA) into the
SatQuery AI dataset hierarchy.

References:
- Paper: Yuan et al., "Change Detection Meets Visual Question Answering", IEEE TGRS 2022
- Repository: https://github.com/YZHJessica/CDVQA
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm


GITHUB_REPO_API = "https://api.github.com/repos/YZHJessica/CDVQA/contents"
RAW_BASE_URL = "https://raw.githubusercontent.com/YZHJessica/CDVQA/main"

EXPECTED_FILES = [
    "Train_questions.json",
    "Train_answers.json",
    "Train_images.json",
    "Val_questions.json",
    "Val_answers.json",
    "Val_images.json",
    "Test_questions.json",
    "Test_answers.json",
    "Test_images.json",
    "Test2_questions.json",
    "Test2_answers.json",
    "Test2_images.json",
]


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 10,
) -> None:
    """Download a file with progress tracking and retry support."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        print(f"  [Already Downloaded] {destination.name} ({expected_size / (1024 * 1024):.2f} MB)")
        return

    temp_file = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "SatQueryAI-DatasetDownloader/1.0"}

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0)) or expected_size or 0

                    with open(temp_file, "wb") as f:
                        with tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=destination.name[:25],
                            ncols=80,
                        ) as pbar:
                            for chunk in resp.iter_bytes(chunk_size=65536):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
            temp_file.replace(destination)
            print(f"  [Complete] Saved {destination.name}")
            return
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {destination.name}: {exc}") from exc
            print(f"  [Retry {attempt}/{max_retries}] {destination.name}: {exc}. Waiting 3s...")
            time.sleep(3.0)


def organize_cdvqa_files(download_dir: Path, output_base: Path) -> Dict[str, Any]:
    """Organize downloaded CDVQA JSON files into standard SatQuery structure:
    
    cdvqa/
    ├── images/
    │   ├── A/
    │   └── B/
    ├── annotations/
    │   └── qa_pairs.json
    └── splits/
        ├── train_questions.json
        ├── train_answers.json
        ├── train_images.json
        ├── val_questions.json
        ├── val_answers.json
        ├── val_images.json
        ├── test_questions.json
        ├── test_answers.json
        └── test_images.json
    """
    images_dir = output_base / "images"
    (images_dir / "A").mkdir(parents=True, exist_ok=True)
    (images_dir / "B").mkdir(parents=True, exist_ok=True)
    
    annotations_dir = output_base / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dir = output_base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to splits with normalized lowercase names
    split_files = []
    for f in download_dir.glob("*.json"):
        if f.name.endswith(".json"):
            norm_name = f.name.lower()
            dest = splits_dir / norm_name
            shutil.copy2(f, dest)
            split_files.append(norm_name)

    # Build consolidated sample qa_pairs.json from val split for quick inspection
    val_q_file = splits_dir / "val_questions.json"
    val_a_file = splits_dir / "val_answers.json"
    val_img_file = splits_dir / "val_images.json"
    
    consolidated_pairs = []
    if val_q_file.exists() and val_a_file.exists() and val_img_file.exists():
        try:
            with open(val_q_file, "r", encoding="utf-8") as fq:
                q_list = json.load(fq).get("questions", [])
            with open(val_a_file, "r", encoding="utf-8") as fa:
                a_list = json.load(fa).get("answers", [])
            with open(val_img_file, "r", encoding="utf-8") as fi:
                img_list = json.load(fi).get("images", [])

            ans_map = {a["id"]: a.get("answer") for a in a_list if a.get("active", True)}
            img_map = {im["id"]: im.get("file_name") for im in img_list if im.get("active", True)}

            for q in q_list[:1000]:  # index first 1000 pairs
                if not q.get("active", True):
                    continue
                ans_ids = q.get("answers_ids", [])
                ans = ans_map.get(ans_ids[0]) if ans_ids else None
                img_name = img_map.get(q.get("img_id"))
                if ans and img_name:
                    consolidated_pairs.append({
                        "question_id": q["id"],
                        "image_id": q.get("img_id"),
                        "file_name": img_name,
                        "image_A": f"A/{img_name}",
                        "image_B": f"B/{img_name}",
                        "question": q.get("question"),
                        "answer": str(ans),
                        "question_type": q.get("type", "change_or_not"),
                    })

            qa_pairs_path = annotations_dir / "qa_pairs.json"
            with open(qa_pairs_path, "w", encoding="utf-8") as fo:
                json.dump(consolidated_pairs, fo, indent=2)
        except Exception as e:
            print(f"  [Warning] Failed to generate consolidated qa_pairs.json: {e}")

    return {
        "output_dir": str(output_base.resolve()),
        "splits_count": len(split_files),
        "split_files": sorted(split_files),
        "consolidated_pairs": len(consolidated_pairs),
    }


def verify_cdvqa(output_base: Path) -> bool:
    """Verify integrity of CDVQA dataset."""
    splits_dir = output_base / "splits"
    annotations_dir = output_base / "annotations"
    images_dir = output_base / "images"

    print(f"\nVerifying CDVQA dataset integrity at {output_base}...")
    ok = True

    split_files = list(splits_dir.glob("*.json"))
    print(f"  - Split files: {len(split_files)} files in {splits_dir}")
    if len(split_files) < len(EXPECTED_FILES):
        print(f"    [WARNING] Expected {len(EXPECTED_FILES)} files, found {len(split_files)}")
        ok = False

    qa_file = annotations_dir / "qa_pairs.json"
    if qa_file.exists():
        print(f"  - Annotations: OK ({qa_file.name}, {qa_file.stat().st_size / (1024 * 1024):.2f} MB)")
    else:
        print(f"    [WARNING] Missing {qa_file}")
        ok = False

    print(f"  - Image directories: A/ ({images_dir / 'A'}) and B/ ({images_dir / 'B'}) ready")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI CDVQA Dataset Downloader")
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/cdvqa",
        help="Target output directory for CDVQA (default: ./datasets/cdvqa)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing dataset without downloading.",
    )

    args = parser.parse_args()
    output_base = Path(args.output).resolve()

    if args.verify_only:
        valid = verify_cdvqa(output_base)
        sys.exit(0 if valid else 1)

    print("=" * 65)
    print("SatQuery AI — CDVQA Dataset Downloader")
    print("Source: https://github.com/YZHJessica/CDVQA")
    print(f"Target Directory: {output_base}")
    print("=" * 65)

    cache_dir = output_base / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(EXPECTED_FILES)} CDVQA annotation files from GitHub...")
    for fname in EXPECTED_FILES:
        url = f"{RAW_BASE_URL}/{fname}"
        dest = cache_dir / fname
        download_file(url, dest)

    print("\n--- Organizing Dataset Structure ---")
    stats = organize_cdvqa_files(cache_dir, output_base)

    print("\n" + "=" * 65)
    print("CDVQA Dataset Organization Complete!")
    print(f"Directory: {stats['output_dir']}")
    print(f"Splits Configured: {stats['splits_count']} files")
    print(f"Consolidated Sample Q&A Pairs: {stats['consolidated_pairs']}")
    print("=" * 65)

    verify_cdvqa(output_base)


if __name__ == "__main__":
    main()
