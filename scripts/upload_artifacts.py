"""
Upload model checkpoints and dataset artifacts to AWS S3.

Usage:
    # Upload a trained model
    python scripts/upload_artifacts.py \
        --type model \
        --local_path ./checkpoints/best_model \
        --config configs/pipeline_config.yaml

    # Upload a labeled dataset
    python scripts/upload_artifacts.py \
        --type dataset \
        --local_path ./data/auto_labeled \
        --config configs/pipeline_config.yaml
"""

import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.tracking.s3_handler import S3Handler


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["model", "dataset"])
    parser.add_argument("--local_path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--s3_key", default=None, help="Override S3 key prefix")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s3_cfg = cfg.get("s3", {})

    bucket = s3_cfg.get("bucket_name")
    if not bucket or bucket == "your-bucket-name":
        raise ValueError(
            "Set a valid bucket_name in configs/pipeline_config.yaml "
            "or via the S3_BUCKET_NAME environment variable."
        )

    handler = S3Handler(bucket_name=bucket, region=s3_cfg.get("region", "us-east-1"))

    local_path = Path(args.local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    if args.s3_key:
        s3_prefix = args.s3_key
    elif args.type == "model":
        s3_prefix = f"{s3_cfg.get('model_prefix', 'models')}/{local_path.name}"
    else:
        s3_prefix = f"{s3_cfg.get('dataset_prefix', 'datasets')}/{local_path.name}"

    if local_path.is_dir():
        count = handler.upload_directory(str(local_path), s3_prefix)
        logger.info(f"Uploaded {count} files to s3://{bucket}/{s3_prefix}")
    else:
        success = handler.upload_file(str(local_path), s3_prefix)
        if success:
            logger.info(f"Uploaded to s3://{bucket}/{s3_prefix}")
        else:
            logger.error("Upload failed")


if __name__ == "__main__":
    main()
