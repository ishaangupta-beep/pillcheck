"""Storage backends, selected by PILLCHECK_STORAGE (local | aws)."""
from __future__ import annotations

from functools import lru_cache

from ..config import settings
from .base import Store, new_id, utcnow_iso

__all__ = ["Store", "get_store", "new_id", "utcnow_iso"]


@lru_cache(maxsize=1)
def get_store() -> Store:
    cfg = settings.storage
    if cfg.backend == "aws":
        from .aws import AwsStore
        return AwsStore(cfg.ddb_table, cfg.s3_bucket, settings.agent.aws_region)
    from .local import LocalStore
    return LocalStore(cfg.data_dir)
