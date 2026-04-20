"""
Ember's Diaries — EmberDB
The main connection interface. This is what users import and use.

Usage:
    from embers import EmberDB

    db = EmberDB.connect("./my_store")
    record_id = db.write(EmberRecord(namespace="memories", data={"content": "hello"}))
    record = db.get(record_id)
"""

from pathlib import Path
from datetime import datetime

from .core.record import EmberRecord
from .core.annotation import Annotation
from .core.types import RecordType, DeprecationReason, AccessLevel
from .core.edge import EdgeRef
from .storage.store import PhysicalStore
from .engine.writer import WriteEngine
from .engine.reader import ReadEngine


class EmberDB:
    """
    Main interface for Ember's Diaries.
    One EmberDB instance = one store (a directory on disk).
    Thread-safe. Multiple agents can write concurrently.
    """

    def __init__(self, store_path: str | Path):
        self._path  = Path(store_path)
        self._store  = PhysicalStore(self._path)
        self._writer = WriteEngine(self._store)
        self._reader = ReadEngine(self._store, self._writer)

        print(f"🔥 Ember's Diaries connected → {self._path}")
        stats = self._store.stats()
        print(f"   Records: {stats['record_count']} | WAL: {stats['wal_size_bytes']} bytes")

    @classmethod
    def connect(cls, store_path: str | Path) -> "EmberDB":
        """Connect to (or create) an Ember's Diaries store."""
        return cls(store_path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(self, record: EmberRecord) -> str:
        """
        Write a new record. Returns the record ID.
        The record is permanent — it can never be deleted.
        """
        return self._writer.write(record)

    def update(self, record_id: str, new_data: dict,
               written_by: str = "system") -> tuple[str, str]:
        """
        Create a new version of an existing record.
        Old record is preserved (superseded, never deleted).
        Returns (new_id, old_id).
        """
        return self._writer.update(record_id, new_data, written_by)

    def annotate(self, record_id: str, annotation: Annotation) -> str:
        """
        Add an annotation to a record without modifying it.
        Returns the annotation ID.
        """
        return self._writer.annotate(record_id, annotation)

    def deprecate(self, record_id: str,
                  reason: DeprecationReason = DeprecationReason.MANUAL,
                  note: str = "",
                  written_by: str = "system") -> bool:
        """
        Mark a record as deprecated. It remains in the store forever.
        Returns True if successful.
        """
        return self._writer.deprecate(record_id, reason, note, written_by)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, record_id: str,
            include_deprecated: bool = False,
            include_superseded: bool = False) -> EmberRecord | None:
        """Get a record by ID. Returns None if not found or filtered."""
        return self._reader.get(record_id, include_deprecated, include_superseded)

    def get_current(self, record_id: str) -> EmberRecord | None:
        """Follow supersession chain to get the latest version."""
        return self._reader.get_current(record_id)

    def get_history(self, record_id: str) -> list[EmberRecord]:
        """Full supersession chain, oldest to newest."""
        return self._reader.get_history(record_id)

    def get_at(self, record_id: str, timestamp: datetime) -> EmberRecord | None:
        """State of a record at a specific point in time."""
        return self._reader.get_at(record_id, timestamp)

    def get_namespace(self, namespace: str,
                      include_deprecated: bool = False,
                      limit: int | None = None) -> list[EmberRecord]:
        """All records in a namespace."""
        return self._reader.get_namespace(namespace, include_deprecated, limit=limit)

    def exists(self, record_id: str) -> bool:
        return self._reader.exists(record_id)

    # ── Query (Phase 3 — stubs that work now via scan) ────────────────────────

    def query(self, namespace: str, filters: dict | None = None,
              limit: int = 100,
              include_deprecated: bool = False,
              include_superseded: bool = False) -> list[EmberRecord]:
        """
        Document query. Filters by field values.
        Phase 1: linear scan. Phase 3: index-accelerated.
        """
        records = self._reader.get_namespace(
            namespace, include_deprecated, limit=None
        )
        if not filters:
            return records[:limit]

        results = []
        for r in records:
            if self._matches(r, filters):
                results.append(r)
                if len(results) >= limit:
                    break
        return results

    def _matches(self, record: EmberRecord, filters: dict) -> bool:
        """Check if a record matches all filter conditions."""
        for key, value in filters.items():
            if key == "tags":
                if isinstance(value, list):
                    if not any(t in record.tags for t in value):
                        return False
                elif value not in record.tags:
                    return False
            elif key == "record_type":
                if record.record_type.value != value and record.record_type != value:
                    return False
            elif key == "written_by":
                if record.written_by != value:
                    return False
            elif key == "confidence_min":
                if record.confidence < value:
                    return False
            elif hasattr(record, key):
                if getattr(record, key) != value:
                    return False
            elif isinstance(record.data, dict):
                if record.data.get(key) != value:
                    return False
            else:
                return False
        return True

    # ── Graph (Phase 2 stubs) ─────────────────────────────────────────────────

    def connect(self, from_id: str, to_id: str,
                edge_ref: EdgeRef | None = None) -> bool:
        """
        Connect two records. Stores connection metadata.
        Full graph traversal in Phase 2.
        """
        # For now: store as annotations on both records
        import uuid
        from .core.types import EdgeType
        edge = edge_ref or EdgeRef(
            edge_id   = str(uuid.uuid4()),
            target_id = to_id,
            edge_type = EdgeType.RELATES_TO,
        )
        ann = Annotation(
            content    = f"connected_to:{to_id}",
            context    = "graph_connection",
            written_by = "system",
            tags       = ["graph_edge"],
        )
        self.annotate(from_id, ann)
        return True

    def neighbors(self, record_id: str, depth: int = 1) -> list[EmberRecord]:
        """Graph traversal. Full implementation in Phase 2."""
        # Stub: return empty — Phase 2 builds the graph index
        return []

    # ── Time ──────────────────────────────────────────────────────────────────

    def timeline(self, namespace: str,
                 start: datetime | None = None,
                 end: datetime | None = None) -> list[EmberRecord]:
        """Records in a namespace ordered by creation time, optionally filtered."""
        records = self.get_namespace(namespace, include_deprecated=True)
        if start:
            records = [r for r in records if r.created_at >= start]
        if end:
            records = [r for r in records if r.created_at <= end]
        return sorted(records, key=lambda r: r.created_at)

    # ── Annotations ───────────────────────────────────────────────────────────

    def get_annotations(self, record_id: str) -> list[Annotation]:
        return self._writer.get_annotations(record_id)

    # ── Store info ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return self._store.stats()

    def checkpoint(self):
        """Compact the WAL. Safe to call periodically."""
        self._store.checkpoint_wal()

    def __repr__(self) -> str:
        return f"EmberDB(path={self._path}, records={self._store.record_count()})"