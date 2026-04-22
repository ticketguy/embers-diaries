"""
Ember's Diaries — Tests for Index Layer (Phase 2) and Query Engine (Phase 3)
"""

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path

from embers import EmberDB, EmberRecord, RecordType, EdgeType


@pytest.fixture
def db(tmp_path):
    return EmberDB.connect(tmp_path / "test_store")


@pytest.fixture
def populated_db(tmp_path):
    """DB with several records for query tests."""
    db = EmberDB.connect(tmp_path / "pop_store")
    for i in range(10):
        db.write(EmberRecord(
            namespace="memories",
            data={"content": f"Memory {i}", "value": i},
            tags=["test", f"group_{i % 3}"],
            written_by="pytest",
        ))
        time.sleep(0.005)
    # Add some in another namespace
    for i in range(5):
        db.write(EmberRecord(
            namespace="facts",
            data={"fact": f"Fact {i}", "topic": "science"},
            tags=["fact", "science"],
        ))
    return db


# ── Full-text search ──────────────────────────────────────────────────────────

class TestFullTextSearch:
    def test_search_finds_records(self, populated_db):
        results = populated_db.search("Memory 5", namespace="memories")
        assert len(results) > 0
        # First result should be the best match
        record, score = results[0]
        assert "Memory" in str(record.data)

    def test_search_namespace_filter(self, populated_db):
        results = populated_db.search("science", namespace="facts")
        assert len(results) > 0
        for record, _ in results:
            assert record.namespace == "facts"

    def test_search_returns_scores(self, populated_db):
        results = populated_db.search("Memory", namespace="memories")
        for record, score in results:
            assert isinstance(score, float)
            assert score > 0

    def test_search_empty_query(self, populated_db):
        results = populated_db.search("", namespace="memories")
        assert len(results) == 0


# ── Graph index ───────────────────────────────────────────────────────────────

class TestGraphIndex:
    def test_link_records(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={"name": "A"}))
        id2 = db.write(EmberRecord(namespace="g", data={"name": "B"}))
        result = db.link(id1, id2, edge_type="relates_to")
        assert result is True

    def test_neighbors(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={"name": "A"}))
        id2 = db.write(EmberRecord(namespace="g", data={"name": "B"}))
        id3 = db.write(EmberRecord(namespace="g", data={"name": "C"}))

        db.link(id1, id2, edge_type="relates_to")
        db.link(id1, id3, edge_type="caused_by")

        neighbors = db.neighbors(id1, depth=1)
        neighbor_ids = [r.id for r in neighbors]
        assert id2 in neighbor_ids
        assert id3 in neighbor_ids

    def test_path_finding(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={"name": "A"}))
        id2 = db.write(EmberRecord(namespace="g", data={"name": "B"}))
        id3 = db.write(EmberRecord(namespace="g", data={"name": "C"}))

        db.link(id1, id2)
        db.link(id2, id3)

        path = db.path(id1, id3)
        assert path is not None
        assert len(path) == 3

    def test_no_path(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={"name": "A"}))
        id2 = db.write(EmberRecord(namespace="g", data={"name": "B"}))
        # No link between them
        path = db.path(id1, id2)
        assert path is None

    def test_subgraph(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={"name": "center"}))
        id2 = db.write(EmberRecord(namespace="g", data={"name": "leaf1"}))
        id3 = db.write(EmberRecord(namespace="g", data={"name": "leaf2"}))

        db.link(id1, id2)
        db.link(id1, id3)

        sg = db.subgraph(id1, depth=1)
        assert id1 in sg["nodes"]
        assert id2 in sg["nodes"]
        assert id3 in sg["nodes"]

    def test_link_nonexistent_fails(self, db):
        id1 = db.write(EmberRecord(namespace="g", data={}))
        result = db.link(id1, "nonexistent")
        assert result is False


# ── Timeline index ────────────────────────────────────────────────────────────

class TestTimelineIndex:
    def test_timeline_ordered(self, populated_db):
        records = populated_db.timeline("memories")
        times = [r.created_at for r in records]
        assert times == sorted(times)

    def test_latest(self, populated_db):
        latest = populated_db.latest("memories", limit=3)
        assert len(latest) == 3

    def test_timeline_range(self, db):
        now = datetime.utcnow()
        for i in range(5):
            db.write(EmberRecord(namespace="t", data={"i": i}))
            time.sleep(0.01)

        # Get all
        all_records = db.timeline("t")
        assert len(all_records) == 5


# ── Vector search ─────────────────────────────────────────────────────────────

class TestVectorSearch:
    def test_similar_finds_records(self, db):
        # Write records with embeddings
        for i in range(5):
            embedding = [0.0] * 10
            embedding[i] = 1.0
            db.write(EmberRecord(
                namespace="v",
                data={"content": f"Vector {i}"},
                embedding=embedding,
            ))

        # Search for something similar to the first
        query = [0.0] * 10
        query[0] = 1.0
        results = db.similar(query, namespace="v", top_k=3)
        assert len(results) > 0
        record, score = results[0]
        assert score > 0.9  # Should be very similar

    def test_similar_empty(self, db):
        results = db.similar([1.0, 0.0, 0.0], namespace="empty")
        assert len(results) == 0


# ── Checkpoint persistence ────────────────────────────────────────────────────

class TestCheckpoint:
    def test_indexes_persist_across_reconnect(self, tmp_path):
        store_path = tmp_path / "persist_test"

        db1 = EmberDB.connect(store_path)
        id1 = db1.write(EmberRecord(
            namespace="persist", data={"content": "hello"}, tags=["test"]))
        id2 = db1.write(EmberRecord(
            namespace="persist", data={"content": "world"}, tags=["test"]))
        db1.link(id1, id2)
        db1.checkpoint()

        # Reconnect
        db2 = EmberDB.connect(store_path)
        r = db2.get(id1)
        assert r is not None

        # Search still works
        results = db2.search("hello", namespace="persist")
        assert len(results) > 0

    def test_stats_include_indexes(self, populated_db):
        stats = populated_db.stats()
        assert "indexes" in stats
        assert "master" in stats["indexes"]
        assert "graph" in stats["indexes"]
        assert "fulltext" in stats["indexes"]
