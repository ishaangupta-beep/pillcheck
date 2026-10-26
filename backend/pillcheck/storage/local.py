"""SQLite + filesystem store for development and the offline demo."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .base import Store, utcnow_iso


class LocalStore(Store):
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.blob_dir = self.data_dir / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / "pillcheck.db", check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS docs (collection TEXT, id TEXT, patient_id TEXT, created_at TEXT, body TEXT,"
            " PRIMARY KEY (collection, id))")
        self._db.execute("CREATE INDEX IF NOT EXISTS by_patient ON docs (collection, patient_id, created_at)")
        self._db.commit()

    def put(self, collection: str, doc: dict) -> dict:
        doc.setdefault("created_at", utcnow_iso())
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?, ?)",
                (collection, doc["id"], doc.get("patient_id"), doc["created_at"], json.dumps(doc)))
            self._db.commit()
        return doc

    def get(self, collection: str, doc_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT body FROM docs WHERE collection=? AND id=?", (collection, doc_id)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, collection: str, patient_id: str | None = None) -> list[dict]:
        with self._lock:
            if patient_id is None:
                rows = self._db.execute("SELECT body FROM docs WHERE collection=? ORDER BY created_at, rowid",
                                        (collection,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT body FROM docs WHERE collection=? AND patient_id=? ORDER BY created_at, rowid",
                    (collection, patient_id)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _path(self, key: str) -> Path:
        p = (self.blob_dir / key).resolve()
        if not str(p).startswith(str(self.blob_dir.resolve())):
            raise ValueError("invalid blob key")
        return p

    def put_blob(self, key: str, data: bytes, content_type: str) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def get_blob(self, key: str) -> bytes | None:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None
