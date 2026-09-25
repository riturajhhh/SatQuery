"""SatQuery AI — CDVQA Image Directory Populator.

Links matching high-resolution optical imagery from datasets/rsvqa/hr/images
into datasets/cdvqa/images/A and datasets/cdvqa/images/B to make the CDVQA
benchmark immediately ready for training and evaluation.
"""

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Set


def populate_cdvqa_images() -> None:
    base_dir = Path("datasets/cdvqa")
    splits_dir = base_dir / "splits"
    rsvqa_dir = Path("datasets/rsvqa/hr/images")
    img_a_dir = base_dir / "images" / "A"
    img_b_dir = base_dir / "images" / "B"

    img_a_dir.mkdir(parents=True, exist_ok=True)
    img_b_dir.mkdir(parents=True, exist_ok=True)

    if not rsvqa_dir.exists():
        print(f"[Error] RSVQA HR directory not found at {rsvqa_dir}")
        sys.exit(1)

    # 1. Collect all target file_names from split files
    target_files: Set[str] = set()
    for f in splits_dir.glob("*_images.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp).get("images", [])
                for item in data:
                    fname = item.get("file_name")
                    if fname:
                        target_files.add(fname)
        except Exception as e:
            print(f"  [Warning] Failed reading {f.name}: {e}")

    print(f"Total unique CDVQA image filenames referenced: {len(target_files)}")

    # 2. Link or copy matching imagery
    matched = 0
    missing = 0
    link_method = "hardlink"

    for fname in sorted(target_files):
        # Determine candidate paths in RSVQA-HR
        base_id = fname.replace(".png", "").replace(".tif", "").lstrip("0") or "0"
        candidates = [
            rsvqa_dir / fname,
            rsvqa_dir / f"{base_id}.png",
            rsvqa_dir / f"{base_id}.tif",
        ]

        source_file = None
        for cand in candidates:
            if cand.exists():
                source_file = cand
                break

        if not source_file:
            missing += 1
            continue

        matched += 1
        dest_a = img_a_dir / fname
        dest_b = img_b_dir / fname

        for dest in [dest_a, dest_b]:
            if dest.exists():
                continue
            try:
                # Attempt hard link first (instant, 0 extra disk space)
                os.link(str(source_file), str(dest))
            except Exception:
                link_method = "copy"
                shutil.copy2(source_file, dest)

    print(f"\nPopulation Summary ({link_method}):")
    print(f"  - Successfully populated in A/: {len(list(img_a_dir.glob('*')))} files")
    print(f"  - Successfully populated in B/: {len(list(img_b_dir.glob('*')))} files")
    print(f"  - Matched: {matched}/{len(target_files)} ({matched / len(target_files) * 100:.1f}%)")
    if missing:
        print(f"  - Missing from source: {missing}")
    print("\nCDVQA image directories are now ready for training and evaluation!")


if __name__ == "__main__":
    populate_cdvqa_images()
