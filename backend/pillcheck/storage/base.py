"""Storage interface: a small document store plus a blob store.

Two implementations share it: LocalStore (SQLite + files, for development and
the offline demo) and AwsStore (DynamoDB + S3, for the deployed service).
Documents are plain JSON dicts with at least `id`, and usually `patient_id`
and `created_at`.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

COLLECTIONS = ("patients", "medications", "references", "checks", "doses", "actions", "notifications", "runs",
               "counters", "demo_templates")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store(ABC):
    @abstractmethod
    def put(self, collection: str, doc: dict) -> dict: ...

    @abstractmethod
    def get(self, collection: str, doc_id: str) -> dict | None: ...

    @abstractmethod
    def list(self, collection: str, patient_id: str | None = None) -> list[dict]:
        """All docs in a collection (optionally one patient's), oldest first."""

    @abstractmethod
    def put_blob(self, key: str, data: bytes, content_type: str) -> str: ...

    @abstractmethod
    def get_blob(self, key: str) -> bytes | None: ...

    def update(self, collection: str, doc_id: str, **fields) -> dict:
        doc = self.get(collection, doc_id)
        if doc is None:
            raise KeyError(f"{collection}/{doc_id} not found")
        doc.update(fields)
        return self.put(collection, doc)

    def incr(self, key: str, ttl_seconds: int) -> int:
        """Increment a named counter and return the new value (rate limits, budgets).

        The default is read-modify-write, fine for the single-process local store;
        AwsStore overrides it with an atomic DynamoDB update and a TTL.
        """
        doc = self.get("counters", key) or {"id": key, "value": 0}
        doc["value"] = int(doc["value"]) + 1
        self.put("counters", doc)
        return doc["value"]

    def find(self, collection: str, patient_id: str | None = None, **equals) -> list[dict]:
        return [d for d in self.list(collection, patient_id) if all(d.get(k) == v for k, v in equals.items())]
