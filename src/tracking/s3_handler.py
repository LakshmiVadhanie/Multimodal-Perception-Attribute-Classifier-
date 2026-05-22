"""
AWS S3 artifact management for datasets and model checkpoints.
"""

import os
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
from tqdm import tqdm


class S3Handler:
    """
    Handles uploading and downloading datasets and model artifacts to/from S3.

    Credentials are read from environment variables:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        AWS_DEFAULT_REGION
    """

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = boto3.client("s3", region_name=self.region)
            except NoCredentialsError:
                raise RuntimeError(
                    "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                    "AWS_SECRET_ACCESS_KEY environment variables."
                )
        return self._client

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        try:
            file_size = Path(local_path).stat().st_size
            logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{s3_key}")
            self.client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                Callback=_ProgressCallback(file_size, local_path),
            )
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            response = self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            file_size = response["ContentLength"]
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} -> {local_path}")
            self.client.download_file(
                self.bucket_name,
                s3_key,
                local_path,
                Callback=_ProgressCallback(file_size, s3_key),
            )
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            return False

    def upload_directory(self, local_dir: str, s3_prefix: str) -> int:
        """Upload all files in a directory. Returns number of files uploaded."""
        local_dir = Path(local_dir)
        files = list(local_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        uploaded = 0
        for file_path in files:
            relative = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")
            if self.upload_file(str(file_path), s3_key):
                uploaded += 1

        logger.info(f"Uploaded {uploaded}/{len(files)} files to s3://{self.bucket_name}/{s3_prefix}")
        return uploaded

    def download_directory(self, s3_prefix: str, local_dir: str) -> int:
        """Download all objects under a prefix. Returns number of files downloaded."""
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

        keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        downloaded = 0
        for key in keys:
            relative = key[len(s3_prefix):].lstrip("/")
            local_path = str(Path(local_dir) / relative)
            if self.download_file(key, local_path):
                downloaded += 1

        logger.info(f"Downloaded {downloaded}/{len(keys)} files from s3://{self.bucket_name}/{s3_prefix}")
        return downloaded

    def list_objects(self, prefix: str) -> List[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        return [obj["Key"] for page in pages for obj in page.get("Contents", [])]

    def object_exists(self, s3_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False


class _ProgressCallback:
    def __init__(self, total_size: int, filename: str):
        self._pbar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=Path(filename).name,
            leave=False,
        )

    def __call__(self, bytes_transferred: int):
        self._pbar.update(bytes_transferred)
