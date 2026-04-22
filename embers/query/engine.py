"""
Ember's Diaries — Query Engine
Routes queries to the appropriate index and assembles results.
Supports compound queries across multiple indexes.
"""

from datetime import datetime
from typing import Any

from ..core.record import EmberRecord
from ..index.master import MasterIndex
from ..index.graph import GraphIndex
from ..index.timeline import TimelineIndex
from ..index.vector import VectorIndex
from ..index.fulltext import FullTextIndex


class QueryEngine:
    """
    Unified query interface across all indexes.
    Every query method returns EmberRecord objects (not just IDs).
    """

    def __init__(self, master: MasterIndex, graph: GraphIndex,
                 timeline: TimelineIndex, vector: VectorIndex,
                 fulltext: FullTextIndex, reader):
        self._master = master
        self._graph = graph
        self._timeline = timeline
        self._vector = vector
        self._fulltext = fulltext
        self._reader = reader  # ReadEngine for resolving IDs to records

    def _resolve(self, record_ids: list[str],
                 include_deprecated: bool = False,
                 include_superseded: bool = False) -> list[EmberRecord]:
        """Resolve a list of IDs to full records."""
        results = []
        for rid in record_ids:
            r = self._reader.get(rid, include_deprecated, include_superseded)
            if r is not None:
                results.append(r)
        return results

    # ── Document Query ────────────────────────────────────────────────────────

    def query(self, namespace: str,
              filters: dict | None = None,
              tags: list[str] | None = None,
              match_all_tags: bool = False,
              written_by: str | None = None,
              limit: int = 100,
              include_deprecated: bool = False,
              include_superseded: bool = False) -> list[EmberRecord]:
        """
        Document-style query with field filters.
        Uses master index for fast namespace + tag lookups.
        """
        # Start with namespace
        candidate_ids = set(self._master.get_namespace_ids(
            namespace, include_deprecated, include_superseded))

        # Filter by tags
        if tags:
            tag_ids = self._master.get_by_tags(tags, match_all=match_all_tags)
            candidate_ids &= tag_ids

        # Filter by author
        if written_by:
            author_ids = self._master.get_by_author(written_by)
            candidate_ids &= author_ids

        # Resolve to records
        records = self._resolve(
            list(candidate_ids), include_deprecated, include_superseded)

        # Apply additional field filters
        if filters:
            records = [r for r in records if self._matches(r, filters)]

        return records[:limit]

    def _matches(self, record: EmberRecord, filters: dict) -> bool:
        """Check if a record matches filter conditions."""
        for key, value in filters.items():
            if key == "record_type":
                if record.record_type.value != value and record.record_type != value:
                    return False
            elif key == "confidence_min":
                if record.confidence < value:
                    return False
            elif key == "confidence_max":
                if record.confidence > value:
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

    # ── Graph Traversal ───────────────────────────────────────────────────────

    def neighbors(self, record_id: str, depth: int = 1,
                  edge_type: str | None = None,
                  direction: str = "outgoing",
                  include_deprecated: bool = False) -> list[EmberRecord]:
        """Graph neighbors up to depth."""
        ids = self._graph.neighbors(record_id, depth, edge_type, direction)
        return self._resolve(ids, include_deprecated)

    def path(self, from_id: str, to_id: str,
             max_depth: int = 10) -> list[EmberRecord] | None:
        """Shortest path between two records."""
        ids = self._graph.path(from_id, to_id, max_depth)
        if ids is None:
            return None
        return self._resolve(ids, include_deprecated=True, include_superseded=True)

    def subgraph(self, root_id: str, depth: int = 2) -> dict:
        """Extract subgraph with full records."""
        sg = self._graph.subgraph(root_id, depth)
        sg["records"] = {r.id: r for r in self._resolve(sg["nodes"])}
        return sg

    # ── Time-Range Query ──────────────────────────────────────────────────────

    def timeline(self, namespace: str,
                 start: datetime | None = None,
                 end: datetime | None = None,
                 include_deprecated: bool = False) -> list[EmberRecord]:
        """Records in a time range, ordered by timestamp."""
        ids = self._timeline.range_query(namespace, start, end)
        return self._resolve(ids, include_deprecated)

    def latest(self, namespace: str, limit: int = 10,
               include_deprecated: bool = False) -> list[EmberRecord]:
        """Most recent N records in a namespace."""
        ids = self._timeline.latest(namespace, limit)
        return self._resolve(ids, include_deprecated)

    def history(self, record_id: str) -> list[EmberRecord]:
        """Full supersession chain, oldest to newest."""
        chain = self._master.get_supersession_chain(record_id)
        return self._resolve(chain, include_deprecated=True, include_superseded=True)

    # ── Vector Similarity ─────────────────────────────────────────────────────

    def similar(self, embedding: list[float],
                namespace: str | None = None,
                top_k: int = 10,
                threshold: float = 0.0,
                exclude_ids: set[str] | None = None,
                include_deprecated: bool = False) -> list[tuple[EmberRecord, float]]:
        """
        Semantic similarity search.
        Returns list of (record, score) tuples.
        """
        results = self._vector.similar(
            embedding, namespace, top_k, threshold, exclude_ids)
        scored = []
        for rid, score in results:
            r = self._reader.get(rid, include_deprecated)
            if r is not None:
                scored.append((r, score))
        return scored

    # ── Full-Text Search ──────────────────────────────────────────────────────

    def search(self, query_text: str,
               namespace: str | None = None,
               top_k: int = 10,
               include_deprecated: bool = False) -> list[tuple[EmberRecord, float]]:
        """
        BM25 full-text search.
        Returns list of (record, relevance_score) tuples.
        """
        results = self._fulltext.search(query_text, namespace, top_k)
        scored = []
        for rid, score in results:
            r = self._reader.get(rid, include_deprecated)
            if r is not None:
                scored.append((r, score))
        return scored

    # ── Compound Queries ──────────────────────────────────────────────────────

    def hybrid_search(self, query_text: str,
                      query_embedding: list[float] | None = None,
                      namespace: str | None = None,
                      top_k: int = 10,
                      text_weight: float = 0.5,
                      vector_weight: float = 0.5) -> list[tuple[EmberRecord, float]]:
        """
        Combined text + vector search with weighted scoring.
        """
        text_results = {}
        if query_text:
            for rid, score in self._fulltext.search(query_text, namespace, top_k * 2):
                text_results[rid] = score

        vector_results = {}
        if query_embedding:
            for rid, score in self._vector.similar(
                    query_embedding, namespace, top_k * 2):
                vector_results[rid] = score

        # Normalize scores
        max_text = max(text_results.values()) if text_results else 1.0
        max_vec = max(vector_results.values()) if vector_results else 1.0

        all_ids = set(text_results.keys()) | set(vector_results.keys())
        combined = {}
        for rid in all_ids:
            t_score = text_results.get(rid, 0) / max_text if max_text > 0 else 0
            v_score = vector_results.get(rid, 0) / max_vec if max_vec > 0 else 0
            combined[rid] = text_weight * t_score + vector_weight * v_score

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for rid, score in ranked:
            r = self._reader.get(rid)
            if r is not None:
                results.append((r, score))
        return results
