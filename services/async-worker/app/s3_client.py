"""S3/MinIO client factory for the async worker.

Mirrors ``services/storage-worker/app/s3_client.py``. The outcome grader
(``app.outcome_grader``, via ``shared.traces.fetch_trace``) reads the SAME
trace objects storage-worker wrote, so this points at the same generic
``S3_*`` endpoint/credentials rather than a grader-specific bucket — the
bucket itself comes from the ``transcript_ref`` stored on the
``session_traces`` row, not from configuration here.
"""

import os

import boto3
from botocore.config import Config


def get_s3_client():
    """Create and return a configured boto3 S3 client with retry logic."""
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=10,
    )
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "agentguard"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "agentguard_dev"),
        config=config,
    )
