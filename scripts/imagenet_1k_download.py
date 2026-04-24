import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_LOCAL_DIR = "/data/home/mislambhuian/data/imagenet-o"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the gated Hugging Face ImageNet-1K dataset locally."
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help="Directory where the dataset snapshot will be stored.",
    )
    parser.add_argument(
        "--repo-id",
        default="cais/imagenet-o",
        help="Hugging Face dataset repo to download.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HF token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN if set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {local_dir}")
    if args.token:
        print("Using token from CLI or environment.")
    else:
        print("No token provided; using any existing Hugging Face login state.")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=args.token,
        resume_download=True,
    )

    print(f"Download completed at {local_dir}")


if __name__ == "__main__":
    main()
