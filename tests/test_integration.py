"""
Ember's Diaries — Tests for LLM Integration Layer (Phase 5)
MemoryProtocol, ContextBuilder, EmbeddingPipeline.
"""

import pytest
from datetime import datetime, timedelta, timezone

from embers import EmberDB, EmberRecord, RecordType
from embers.integration.memory_protocol import MemoryProtocol
from embers.integration.context import ContextBuilder
from embers.integration.embeddings import EmbeddingPipeline
from embers.cognitive.decay import DecayEngine


@pytest.fixture
def db(tmp_path):
    return EmberDB.connect(tmp_path / "test_store")


@pytest.fixture
def protocol(db):
    return MemoryProtocol(db)


# ── Memory Protocol ──────────────────────────────────────────────────────────

class TestMemoryProtocol:
    def test_remember(self, protocol):
        rid = protocol.remember("The user's name is Alice", tags=["personal"])
        assert rid is not None
        assert len(rid) == 36  # UUID

    def test_remember_and_recall(self, protocol):
        protocol.remember("The user likes dark mode", tags=["preference"])
        protocol.remember("The user works at Acme Corp", tags=["personal"])
        protocol.remember("The user prefers Python", tags=["preference", "tech"])

        context = protocol.recall("What does the user prefer?", format="text")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_recall_structured(self, protocol):
        protocol.remember("Important fact: water is wet", tags=["fact"])
        result = protocol.recall("water", format="structured")
        assert isinstance(result, list)
        if result:
            assert "id" in result[0]
            assert "data" in result[0]

    def test_recall_messages(self, protocol):
        protocol.remember("The sky is blue", tags=["fact"])
        result = protocol.recall("sky color", format="messages")
        assert isinstance(result, list)

    def test_verify(self, protocol):
        rid = protocol.remember("Hypothesis: X causes Y", tags=["hypothesis"])
        result = protocol.verify(rid, status="verified", note="Confirmed by experiment")
        assert result is True

    def test_update(self, protocol):
        rid = protocol.remember("User age: 25", tags=["personal"])
        new_id, old_id = protocol.update(rid, {"content": "User age: 26"})
        assert new_id != old_id

    def test_forget(self, protocol):
        rid = protocol.remember("Temporary note", tags=["temp"])
        result = protocol.forget(rid, reason="No longer needed")
        assert result is True
        # Should still be retrievable with include_deprecated
        record = protocol.db.get(rid, include_deprecated=True)
        assert record is not None

    def test_reflect(self, protocol):
        # Write some memories with decay
        for i in range(5):
            protocol.remember(
                f"Old memory {i}", tags=["old"],
                decay_rate=0.5, confidence=1.0)
        annotations = protocol.reflect()
        # May or may not have annotations depending on time elapsed
        assert isinstance(annotations, list)

    def test_consolidate(self, protocol):
        for i in range(5):
            protocol.remember(f"Related fact {i}", tags=["topic_x"])
        new_ids = protocol.consolidate()
        assert isinstance(new_ids, list)

    def test_segment_episodes(self, protocol):
        for i in range(5):
            protocol.remember(f"Event {i}", tags=["narrative"])
        episodes = protocol.segment_episodes()
        assert isinstance(episodes, list)
        if episodes:
            assert "id" in episodes[0]
            assert "record_ids" in episodes[0]

    def test_stats(self, protocol):
        protocol.remember("test", tags=["t"])
        stats = protocol.stats()
        assert stats["memories_written"] >= 1
        assert "conflicts" in stats
        assert "episodes" in stats


# ── Context Builder ──────────────────────────────────────────────────────────

class TestContextBuilder:
    def test_build_text_context(self):
        builder = ContextBuilder(max_tokens=1000)
        records = [
            EmberRecord(data={"content": "Hello world"}, tags=["test"]),
            EmberRecord(data={"content": "Second memory"}, tags=["test"]),
        ]
        text = builder.build_text_context(records)
        assert "Hello world" in text
        assert "Memory Context" in text

    def test_token_budget(self):
        builder = ContextBuilder(max_tokens=50)  # Very small budget
        records = [
            EmberRecord(data={"content": "x" * 1000}),
        ]
        text = builder.build_text_context(records)
        # Should either include the record (if it fits) or be empty
        assert isinstance(text, str)

    def test_build_message_context(self):
        builder = ContextBuilder()
        records = [EmberRecord(data={"content": "Test memory"})]
        messages = builder.build_message_context(records)
        assert isinstance(messages, list)
        if messages:
            assert messages[0]["role"] == "system"

    def test_build_structured_context(self):
        builder = ContextBuilder()
        records = [EmberRecord(data={"content": "Structured"})]
        result = builder.build_structured_context(records)
        assert isinstance(result, list)
        if result:
            assert "id" in result[0]
            assert "confidence" in result[0]

    def test_get_last_injected(self):
        builder = ContextBuilder()
        records = [EmberRecord(data={"content": "test"})]
        builder.build_text_context(records)
        injected = builder.get_last_injected()
        assert len(injected) >= 1


# ── Embedding Pipeline ────────────────────────────────────────────────────────

class TestEmbeddingPipeline:
    def test_builtin_embeddings(self):
        pipeline = EmbeddingPipeline(dimension=64)
        records = [
            EmberRecord(data={"content": "The cat sat on the mat"}, tags=["animal"]),
            EmberRecord(data={"content": "Dogs are loyal companions"}, tags=["animal"]),
            EmberRecord(data={"content": "Python is a programming language"}, tags=["tech"]),
        ]
        results = pipeline.embed_batch(records)
        assert len(results) == 3
        for rid, embedding in results:
            assert len(embedding) == 64

    def test_custom_embed_fn(self):
        custom_fn = lambda text: [1.0, 0.0, 0.0]
        pipeline = EmbeddingPipeline(embed_fn=custom_fn, dimension=3)
        r = EmberRecord(data={"content": "test"})
        embedding = pipeline.embed_record(r)
        assert embedding == [1.0, 0.0, 0.0]

    def test_embed_text(self):
        pipeline = EmbeddingPipeline(dimension=32)
        # Need to build vocab first
        records = [EmberRecord(data={"content": "hello world test"}) for _ in range(3)]
        pipeline.embed_batch(records)
        emb = pipeline.embed_text("hello world")
        assert len(emb) == 32


# ── End-to-end integration ────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_memory_lifecycle(self, db):
        protocol = MemoryProtocol(db, default_namespace="life")

        # 1. Remember
        id1 = protocol.remember("User was born in 1990", tags=["personal", "birth"])
        id2 = protocol.remember("User graduated in 2012", tags=["personal", "education"])
        id3 = protocol.remember("User started job in 2013", tags=["personal", "career"])

        # 2. Recall
        context = protocol.recall("When was the user born?")
        assert isinstance(context, str)
        assert len(context) > 0

        # 3. Update
        new_id, _ = protocol.update(id1, {"content": "User was born on March 15, 1990"})

        # 4. History preserved
        history = db.get_history(id1)
        assert len(history) == 2

        # 5. Verify
        protocol.verify(new_id, status="verified")

        # 6. Stats
        stats = protocol.stats()
        assert stats["memories_written"] >= 3
