"""
Extract ImageNet-1K from HuggingFace parquet format into PyTorch ImageFolder structure.

Output layout:
  <output_dir>/train/<synset_id>/<image_filename>.JPEG
  <output_dir>/val/<synset_id>/<image_filename>.JPEG

The repo expects this via torchvision.datasets.ImageFolder.
"""

import os
import sys
import glob
import pandas as pd
from collections import OrderedDict

# Add the downloaded dataset's classes.py
sys.path.insert(0, "/data/home/mislambhuian/data/imagenet-1k")
from classes import IMAGENET2012_CLASSES

PARQUET_DIR = "/data/home/mislambhuian/data/imagenet-1k/data"
OUTPUT_DIR = "/data/home/mislambhuian/data/imagenet-1k"

# Build label_id -> synset mapping (sorted synset order = label order)
synsets = list(IMAGENET2012_CLASSES.keys())  # already sorted in classes.py
assert len(synsets) == 1000, f"Expected 1000 classes, got {len(synsets)}"


def extract_split(split_name, parquet_glob, out_subdir):
    """Extract one split from parquet files into ImageFolder structure."""
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, parquet_glob)))
    print(f"\n=== Extracting {split_name}: {len(parquet_files)} parquet files -> {out_subdir}/ ===")

    if not parquet_files:
        print(f"  No parquet files found for pattern: {parquet_glob}")
        return

    out_path = os.path.join(OUTPUT_DIR, out_subdir)

    # Pre-create all synset directories
    for s in synsets:
        os.makedirs(os.path.join(out_path, s), exist_ok=True)

    total = 0
    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            label_idx = row["label"]
            synset = synsets[label_idx]
            img_data = row["image"]
            img_bytes = img_data["bytes"]
            img_name = img_data.get("path", f"{total:08d}.JPEG")
            if not img_name:
                img_name = f"{total:08d}.JPEG"

            dest = os.path.join(out_path, synset, img_name)
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(img_bytes)
            total += 1

        print(f"  [{i+1}/{len(parquet_files)}] Processed {os.path.basename(pf)} (running total: {total})")

    print(f"  Done: {total} images extracted to {out_path}")


if __name__ == "__main__":
    extract_split("train", "train-*.parquet", "train")
    # extract_split("val", "validation-*.parquet", "val")
    print("\nExtraction complete!")
    print(f"Set IMAGENET1K_PATH={OUTPUT_DIR} or update paths_config.py")
