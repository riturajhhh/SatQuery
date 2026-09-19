"""SatQuery AI — RSVQA Dataset Downloader & Preprocessor.

Downloads and organizes the Remote Sensing Visual Question Answering (RSVQA) dataset
from the official Zenodo repository into the SatQuery AI dataset hierarchy.

References:
- RSVQA Paper: Lobry et al., IEEE TGRS 2020
- Project Page: https://rsvqa.sylvainlobry.com/
- Zenodo Records:
  - RSVQA-LR (Low Resolution, Sentinel-2): https://zenodo.org/record/6344334 (~140 MB)
  - RSVQA-HR (High Resolution, USGS Aerial): https://zenodo.org/record/6344367 (~13.5 GB)
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional
import zipfile

import httpx
from tqdm import tqdm


ZENODO_RECORDS = {
    "lr": {
        "record_id": "6344334",
        "name": "RSVQA Low Resolution (Sentinel-2)",
        "expected_images_archive": "Images_LR.zip",
        "archive_type": "zip",
    },
    "hr": {
        "record_id": "6344367",
        "name": "RSVQA High Resolution (USGS)",
        "expected_images_archive": "Images.tar",
        "archive_type": "tar",
    },
}


def get_zenodo_files(record_id: str) -> List[Dict[str, Any]]:
    """Fetch file metadata for a Zenodo record."""
    url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"User-Agent": "SatQueryAI-DatasetDownloader/1.0"}

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("files", [])


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 30,
) -> None:
    """Download a file with automatic resumption, retry loop, and progress tracking."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is fully downloaded
    if destination.exists() and expected_size is not None and destination.stat().st_size == expected_size:
        print(f"  [Already Downloaded] {destination.name} ({expected_size / (1024 * 1024):.2f} MB)")
        return

    temp_file = destination.with_suffix(destination.suffix + ".part")
    total_size = expected_size or 0
    downloaded_bytes = temp_file.stat().st_size if temp_file.exists() else 0

    pbar = tqdm(
        total=total_size if total_size > 0 else None,
        initial=downloaded_bytes,
        unit="B",
        unit_scale=True,
        desc=destination.name[:25],
        ncols=85,
    )

    attempt = 0
    while True:
        downloaded_bytes = temp_file.stat().st_size if temp_file.exists() else 0
        if expected_size and downloaded_bytes >= expected_size:
            break

        headers = {"User-Agent": "SatQueryAI-DatasetDownloader/1.0"}
        if downloaded_bytes > 0:
            headers["Range"] = f"bytes={downloaded_bytes}-"

        try:
            client_timeout = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=30.0)
            with httpx.Client(follow_redirects=True, timeout=client_timeout) as client:
                with client.stream("GET", url, headers=headers) as resp:
                    # Handle 416 Range Not Satisfiable (likely finished or out of range)
                    if resp.status_code == 416:
                        head_res = client.head(url)
                        remote_len = int(head_res.headers.get("content-length", 0))
                        if downloaded_bytes >= remote_len > 0:
                            break
                        # Reset if invalid offset
                        downloaded_bytes = 0
                        temp_file.unlink(missing_ok=True)
                        pbar.reset(total=total_size)
                        continue

                    # If server ignores Range and sends 200 instead of 206
                    if downloaded_bytes > 0 and resp.status_code == 200:
                        downloaded_bytes = 0
                        temp_file.unlink(missing_ok=True)
                        pbar.reset(total=int(resp.headers.get("content-length", 0)) or total_size)

                    resp.raise_for_status()

                    # Set total if unknown
                    if total_size == 0:
                        content_len = int(resp.headers.get("content-length", 0))
                        total_size = content_len + downloaded_bytes if content_len else 0
                        pbar.total = total_size

                    mode = "ab" if downloaded_bytes > 0 else "wb"
                    with open(temp_file, mode) as f:
                        for chunk in resp.iter_bytes(chunk_size=1048576):  # 1MB buffer
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                downloaded_bytes += len(chunk)

            # Check completion
            if expected_size and temp_file.stat().st_size >= expected_size:
                break
            elif not expected_size:
                break

        except (httpx.TransportError, httpx.HTTPStatusError, ConnectionResetError, Exception) as exc:
            attempt += 1
            if attempt > max_retries:
                pbar.close()
                raise RuntimeError(
                    f"Download failed for {destination.name} after {max_retries} attempts: {exc}"
                ) from exc

            cur_bytes = temp_file.stat().st_size if temp_file.exists() else 0
            cur_mb = cur_bytes / (1024 * 1024)
            tot_mb = (total_size / (1024 * 1024)) if total_size else 0
            pbar.write(
                f"\n  [Notice] Connection interrupted at {cur_mb:.1f} MB / {tot_mb:.1f} MB: {type(exc).__name__}. Resuming in 5s (retry {attempt}/{max_retries})..."
            )
            time.sleep(5.0)

    pbar.close()
    temp_file.replace(destination)
    print(f"  [Complete] Saved to {destination.name}")


def extract_archive(archive_path: Path, extract_to: Path, archive_type: str) -> None:
    """Extract zip or tar archive into destination directory with optimized streaming."""
    print(f"\nExtracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
    elif archive_type == "tar":
        with tarfile.open(archive_path, "r") as tf:
            try:
                tf.extractall(extract_to, filter="data")
            except TypeError:
                # Python < 3.12 compatibility
                tf.extractall(extract_to)
    else:
        raise ValueError(f"Unknown archive format: {archive_type}")

    # Handle nested folder extraction (e.g. if files are inside an inner folder)
    _flatten_if_nested(extract_to)


def _flatten_if_nested(target_dir: Path) -> None:
    """Ensure images are directly in target_dir if extracted inside a subfolder."""
    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and not any(p.is_file() for p in target_dir.iterdir()):
        single_sub = subdirs[0]
        print(f"Flattening nested folder: {single_sub.name} -> {target_dir.name}")
        for item in single_sub.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        single_sub.rmdir()


def organize_dataset_files(variant: str, downloaded_dir: Path, output_base: Path) -> Dict[str, Any]:
    """Organize downloaded raw JSON files into structured dataset folders.
    
    Structure:
    datasets/rsvqa/<variant>/
      ├── images/
      ├── questions/
      │   └── questions.json
      ├── answers/
      │   └── answers.json
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
    variant_root = output_base / variant
    images_dir = variant_root / "images"
    questions_dir = variant_root / "questions"
    answers_dir = variant_root / "answers"
    splits_dir = variant_root / "splits"

    for d in (images_dir, questions_dir, answers_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Mapping rules for files
    prefix = "LR_" if variant == "lr" else "USGS_"
    
    for f in downloaded_dir.glob("*.json"):
        fname = f.name
        if fname in ("all_questions.json", f"{prefix}questions.json", "USGSquestions.json", "questions.json"):
            shutil.copy2(f, questions_dir / "questions.json")
        elif fname in ("all_answers.json", f"{prefix}answers.json", "USGSanswers.json", "answers.json"):
            shutil.copy2(f, answers_dir / "answers.json")
        elif "split" in fname.lower() or fname.startswith(prefix):
            # Clean split filename, e.g. LR_split_train_questions.json -> train_questions.json
            cleaned = fname.replace(prefix, "").replace("split_", "")
            shutil.copy2(f, splits_dir / cleaned)

    # Verification stats
    num_images = len(list(images_dir.glob("*.tif"))) + len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
    
    stats = {
        "variant": variant,
        "variant_path": str(variant_root.resolve()),
        "images_count": num_images,
        "questions_file": str((questions_dir / "questions.json").resolve()) if (questions_dir / "questions.json").exists() else None,
        "answers_file": str((answers_dir / "answers.json").resolve()) if (answers_dir / "answers.json").exists() else None,
        "split_files": [f.name for f in splits_dir.glob("*.json")],
    }
    return stats


def verify_dataset(variant: str, output_base: Path) -> bool:
    """Verify integrity of the organized RSVQA dataset."""
    variant_root = output_base / variant
    images_dir = variant_root / "images"
    q_file = variant_root / "questions" / "questions.json"
    a_file = variant_root / "answers" / "answers.json"
    splits_dir = variant_root / "splits"

    print(f"\nVerifying RSVQA ({variant.upper()}) dataset integrity at {variant_root}...")
    
    ok = True
    images = list(images_dir.glob("*.*"))
    print(f"  - Images: {len(images)} files detected in {images_dir}")
    if len(images) == 0:
        print("    [WARNING] No images found in images directory!")
        ok = False

    if q_file.exists():
        size_mb = q_file.stat().st_size / (1024 * 1024)
        print(f"  - Questions: OK ({size_mb:.2f} MB)")
    else:
        print(f"    [WARNING] Missing {q_file}")
        ok = False

    if a_file.exists():
        size_mb = a_file.stat().st_size / (1024 * 1024)
        print(f"  - Answers: OK ({size_mb:.2f} MB)")
    else:
        print(f"    [WARNING] Missing {a_file}")
        ok = False

    split_files = list(splits_dir.glob("*.json"))
    print(f"  - Splits: {len(split_files)} split JSON files found")
    if len(split_files) == 0:
        print(f"    [WARNING] No split files in {splits_dir}")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI RSVQA Dataset Downloader")
    parser.add_argument(
        "--variant",
        choices=["lr", "hr"],
        default="lr",
        help="RSVQA variant to download: 'lr' (Low-Res, ~140 MB, recommended) or 'hr' (High-Res, ~13.5 GB).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./datasets/rsvqa",
        help="Target output directory for the dataset (default: ./datasets/rsvqa).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip downloading large images archive (download only question/answer annotations).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing dataset without downloading.",
    )

    args = parser.parse_args()
    output_base = Path(args.output).resolve()
    variant = args.variant

    if args.verify_only:
        valid = verify_dataset(variant, output_base)
        sys.exit(0 if valid else 1)

    rec_info = ZENODO_RECORDS[variant]
    print("=" * 65)
    print(f"SatQuery AI — RSVQA Dataset Downloader")
    print(f"Variant: {rec_info['name']} (Record: {rec_info['record_id']})")
    print(f"Target Directory: {output_base / variant}")
    print("=" * 65)

    print(f"\nFetching metadata from Zenodo record {rec_info['record_id']}...")
    try:
        files_meta = get_zenodo_files(rec_info["record_id"])
    except Exception as e:
        print(f"[ERROR] Failed to query Zenodo API: {e}")
        sys.exit(1)

    print(f"Found {len(files_meta)} files in Zenodo record:")
    for f in files_meta:
        mb = f.get("size", 0) / (1024 * 1024)
        print(f"  - {f.get('key'):35} {mb:8.2f} MB")

    cache_dir = output_base / ".cache" / variant
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download files
    print("\n--- Starting File Downloads ---")
    for f in files_meta:
        fname = f.get("key", "")
        fsize = f.get("size", 0)
        flink = f.get("links", {}).get("self")
        if not flink:
            continue

        is_archive = fname == rec_info["expected_images_archive"]
        if is_archive and args.skip_images:
            print(f"  [Skipping Images Archive as requested] {fname}")
            continue

        dest_file = cache_dir / fname
        download_file(flink, dest_file, expected_size=fsize)

    # 2. Extract images archive if present
    archive_file = cache_dir / rec_info["expected_images_archive"]
    images_dest = output_base / variant / "images"
    if archive_file.exists() and not args.skip_images:
        extract_archive(archive_file, images_dest, rec_info["archive_type"])

    # 3. Organize questions, answers, and splits
    print("\n--- Organizing Dataset Structure ---")
    stats = organize_dataset_files(variant, cache_dir, output_base)

    print("\n" + "=" * 65)
    print("RSVQA Dataset Organization Complete!")
    print(f"Variant Path: {stats['variant_path']}")
    print(f"Total Images: {stats['images_count']}")
    print(f"Questions File: {stats['questions_file']}")
    print(f"Answers File: {stats['answers_file']}")
    print(f"Splits Configured: {len(stats['split_files'])} files")
    print("=" * 65)

    # 4. Final verification
    verify_dataset(variant, output_base)


if __name__ == "__main__":
    main()
