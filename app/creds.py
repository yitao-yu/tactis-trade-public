"""Secrets & credential resolution for the live runner (AWS Batch or local dev).

Resolution order per key:
  1. environment variable TACTIS_SECRET_<KEY>
  2. local file ~/.secret/<key>      (chmod 600; local dev only)
  3. SSM Parameter Store  /tactis/<key>   (SecureString; Batch jobs)

Secret values are NEVER logged or printed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SSM_PREFIX = "/tactis/"
LOCAL_SECRET_DIR = Path("~/.secret").expanduser()


def _env_name(key: str) -> str:
    return f"TACTIS_SECRET_{key.upper()}"


def _ssm_name(key: str) -> str:
    return f"{SSM_PREFIX}{key}"


def get_secret(key: str) -> Optional[str]:
    """Resolve a named secret without ever logging its value."""
    val = os.environ.get(_env_name(key))
    if val:
        return val
    local = LOCAL_SECRET_DIR / key
    if local.is_file():
        return local.read_text().strip()
    try:
        import boto3

        client = boto3.client("ssm")
        resp = client.get_parameter(Name=_ssm_name(key), WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception:
        return None


def has_credential(key: str) -> bool:
    """True if the secret is resolvable from env or local file (no network call)."""
    return bool(os.environ.get(_env_name(key))) or (LOCAL_SECRET_DIR / key).exists()
