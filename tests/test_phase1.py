"""
Ember's Diaries — Phase 1 Tests
Tests for: record schema, write engine, read engine, WAL, storage, annotations.
Run: python -m pytest tests/ -v
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from embers import (
    EmberDB, EmberRecord, Annotation, ReflectiveAnnotation,
    RecordType, DeprecationReason, EdgeType, EdgeRef,
)


@pytest.fixture
def db(tmp_path):
    """Fresh DB for each test."""
    return EmberDB.connect(tmp_path / "test_store")


@pytest.fixture
def sample_record():
    return EmberRecord(
        namespace   = "test",
        record_type = RecordType.DOCUMENT,
        data        = {"content": "hello world", "value": 42},
        tags        = ["test", "sample"],
        written_by  = "pytest",
    )


# ── Record schema ─────────────────────────────────────────────────────────────

class TestEmberRecord:
    def test_default_id_generated(self):
        r = EmberRecord()
        assert len(r.id) == 36  # UUID format

    def test_is_active_new_record(self):
        r = EmberRecord()
        assert r.is_active is True

    def test_is_current_new_record(self):
        r = EmberRecord()
        assert r.is_current is True

    def test_is_head_new_record(self):
        r = EmberRecord()
        assert r.is_head is True

    def test_serialization_roundtrip(self, sample_record):
        d = sample_record.to_dict()
        restored = EmberRecord.from_dict(d)
        assert restored.id == sample_record.id
        assert restored.namespace == sample_record.namespace
        assert restored.data == sample_record.data
        assert restored.tags == sample_record.tags
        assert restored.record_type == sample_record.record_type

    def test_created_at_is_datetime(self, sample_record):
        assert isinstance(sample_record.created_at, datetime)

    def test_valid_time_window(self):
        r = EmberRecord(
            valid_from  = datetime.utcnow() + timedelta(hours=1),
            valid_until = datetime.utcnow() + timedelta(hours=2),
        )
        assert r.is_active is False  # Not yet valid

    def test_expired_record(self):
        r = EmberRecord(
            valid_until = datetime.utcnow() - timedelta(seconds=1)
        )
        assert r.is_active is False


# ── Write engine ──────────────────────────────────────────────────────────────

class TestWrite:
    def test_write_returns_id(self, db, sample_record):
        record_id = db.write(sample_record)
        assert record_id == sample_record.id

    def test_write_duplicate_raises(self, db, sample_record):
        db.write(sample_record)
        with pytest.raises(ValueError, match="already exists"):
            db.write(sample_record)

    def test_write_multiple_records(self, db):
        ids = []
        for i in range(10):
            r = EmberRecord(namespace="bulk", data={"i": i})
            ids.append(db.write(r))
        assert len(set(ids)) == 10  # All unique

    def test_write_preserves_data(self, db, sample_record):
        db.write(sample_record)
        retrieved = db.get(sample_record.id)
        assert retrieved.data == sample_record.data


# ── Read engine ───────────────────────────────────────────────────────────────

class TestRead:
    def test_get_existing_record(self, db, sample_record):
        db.write(sample_record)
        r = db.get(sample_record.id)
        assert r is not None
        assert r.id == sample_record.id

    def test_get_nonexistent_returns_none(self, db):
        assert db.get("nonexistent-id") is None

    def test_exists(self, db, sample_record):
        assert db.exists(sample_record.id) is False
        db.write(sample_record)
        assert db.exists(sample_record.id) is True

    def test_get_namespace(self, db):
        for i in range(5):
            db.write(EmberRecord(namespace="ns_test", data={"i": i}))
        for i in range(3):
            db.write(EmberRecord(namespace="other_ns", data={"i": i}))

        ns_records = db.get_namespace("ns_test")
        assert len(ns_records) == 5
        assert all(r.namespace == "ns_test" for r in ns_records)

    def test_timeline_ordering(self, db):
        import time
        for i in range(3):
            db.write(EmberRecord(namespace="timeline_test", data={"i": i}))
            time.sleep(0.01)
        records = db.timeline("timeline_test")
        times = [r.created_at for r in records]
        assert times == sorted(times)


# ── Update (supersession) ─────────────────────────────────────────────────────

class TestUpdate:
    def test_update_creates_new_record(self, db, sample_record):
        old_id = db.write(sample_record)
        new_id, returned_old = db.update(old_id, {"content": "updated"})
        assert new_id != old_id
        assert returned_old == old_id

    def test_old_record_still_readable_after_update(self, db, sample_record):
        old_id = db.write(sample_record)
        db.update(old_id, {"content": "updated"})
        old = db.get(old_id, include_superseded=True)
        assert old is not None
        assert old.data == sample_record.data

    def test_superseded_record_hidden_by_default(self, db, sample_record):
        old_id = db.write(sample_record)
        db.update(old_id, {"content": "updated"})
        old = db.get(old_id)  # No include_superseded
        assert old is None

    def test_get_current_follows_chain(self, db, sample_record):
        id_v1 = db.write(sample_record)
        id_v2, _ = db.update(id_v1, {"version": 2})
        id_v3, _ = db.update(id_v2, {"version": 3})

        current = db.get_current(id_v1)
        assert current is not None
        assert current.id == id_v3

    def test_history_returns_full_chain(self, db, sample_record):
        id_v1 = db.write(sample_record)
        id_v2, _ = db.update(id_v1, {"version": 2})
        id_v3, _ = db.update(id_v2, {"version": 3})

        history = db.get_history(id_v1)
        assert len(history) == 3
        assert history[0].id == id_v1
        assert history[-1].id == id_v3

    def test_get_at_timestamp(self, db, sample_record):
        import time
        id_v1 = db.write(sample_record)
        time.sleep(0.05)
        checkpoint = datetime.utcnow()
        time.sleep(0.05)
        id_v2, _ = db.update(id_v1, {"version": 2})

        at_checkpoint = db.get_at(id_v1, checkpoint)
        assert at_checkpoint is not None
        assert at_checkpoint.id == id_v1


# ── Deprecation ───────────────────────────────────────────────────────────────

class TestDeprecation:
    def test_deprecate_record(self, db, sample_record):
        db.write(sample_record)
        result = db.deprecate(sample_record.id, DeprecationReason.MANUAL, "test")
        assert result is True

    def test_deprecated_hidden_by_default(self, db, sample_record):
        db.write(sample_record)
        db.deprecate(sample_record.id)
        r = db.get(sample_record.id)
        assert r is None

    def test_deprecated_visible_with_flag(self, db, sample_record):
        db.write(sample_record)
        db.deprecate(sample_record.id)
        r = db.get(sample_record.id, include_deprecated=True)
        assert r is not None

    def test_deprecate_nonexistent_raises(self, db):
        with pytest.raises(KeyError):
            db.deprecate("nonexistent-id")

    def test_deprecate_idempotent(self, db, sample_record):
        db.write(sample_record)
        db.deprecate(sample_record.id)
        result = db.deprecate(sample_record.id)
        assert result is False  # Already deprecated


# ── Annotations ───────────────────────────────────────────────────────────────

class TestAnnotations:
    def test_annotate_record(self, db, sample_record):
        db.write(sample_record)
        ann = Annotation(content="This is a note", written_by="pytest")
        ann_id = db.annotate(sample_record.id, ann)
        assert ann_id == ann.id

    def test_get_annotations(self, db, sample_record):
        db.write(sample_record)
        db.annotate(sample_record.id, Annotation(content="Note 1"))
        db.annotate(sample_record.id, Annotation(content="Note 2"))
        annotations = db.get_annotations(sample_record.id)
        assert len(annotations) == 2

    def test_annotate_nonexistent_raises(self, db):
        with pytest.raises(KeyError):
            db.annotate("nonexistent", Annotation(content="test"))

    def test_reflective_annotation(self, db, sample_record):
        db.write(sample_record)
        ann = ReflectiveAnnotation(
            content="New interpretation",
            triggered_by="emergence_engine",
            insight_score=0.8,
        )
        ann_id = db.annotate(sample_record.id, ann)
        assert ann_id == ann.id


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_by_tag(self, db):
        db.write(EmberRecord(namespace="q", data={}, tags=["alpha"]))
        db.write(EmberRecord(namespace="q", data={}, tags=["beta"]))
        db.write(EmberRecord(namespace="q", data={}, tags=["alpha", "beta"]))

        results = db.query("q", filters={"tags": "alpha"})
        assert len(results) == 2

    def test_query_empty_filters(self, db):
        for _ in range(5):
            db.write(EmberRecord(namespace="q2", data={}))
        results = db.query("q2")
        assert len(results) == 5

    def test_query_limit(self, db):
        for _ in range(10):
            db.write(EmberRecord(namespace="q3", data={}))
        results = db.query("q3", limit=3)
        assert len(results) == 3


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_record_count(self, db):
        for _ in range(5):
            db.write(EmberRecord(namespace="stats_test", data={}))
        stats = db.stats()
        assert stats["record_count"] == 5

    def test_checkpoint_runs(self, db, sample_record):
        db.write(sample_record)
        db.checkpoint()  # Should not raise


# ── WAL recovery ─────────────────────────────────────────────────────────────

class TestWALRecovery:
    def test_data_survives_reconnect(self, tmp_path):
        store_path = tmp_path / "recovery_test"
        db1 = EmberDB.connect(store_path)
        record = EmberRecord(namespace="recovery", data={"value": "persisted"})
        db1.write(record)

        # Reconnect
        db2 = EmberDB.connect(store_path)
        r = db2.get(record.id)
        assert r is not None
        assert r.data["value"] == "persisted"
