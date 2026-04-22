"""
Ember's Diaries — Graph Index
Native graph traversal over records.
Adjacency list stored in-memory, persisted to disk.
"""

import threading
from collections import defaultdict
from pathlib import Path

from ..storage.format import encode_index, decode_index
from ..core.types import EdgeType


class GraphIndex:
    """
    Bidirectional adjacency index for graph traversal.
    Supports: neighbors, path finding, subgraph extraction, edge-type filtering.
    """

    def __init__(self, store_path: Path):
        self._path = store_path / "indexes" / "graph"
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # Adjacency: node_id → list of (target_id, edge_type, weight, edge_id)
        self._outgoing: dict[str, list[dict]] = defaultdict(list)
        self._incoming: dict[str, list[dict]] = defaultdict(list)

        self._load()

    def _load(self):
        index_file = self._path / "adjacency.json"
        if not index_file.exists():
            return
        try:
            data = decode_index(index_file.read_bytes())
            for nid, edges in data.get("outgoing", {}).items():
                self._outgoing[nid] = edges
            for nid, edges in data.get("incoming", {}).items():
                self._incoming[nid] = edges
        except Exception as e:
            print(f"[GraphIndex] Failed to load: {e}")

    def persist(self):
        with self._lock:
            data = {
                "outgoing": dict(self._outgoing),
                "incoming": dict(self._incoming),
            }
            index_file = self._path / "adjacency.json"
            index_file.write_bytes(encode_index(data))

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str,
                 edge_type: str = "relates_to",
                 weight: float = 1.0,
                 edge_id: str = "",
                 label: str = "",
                 metadata: dict | None = None):
        """Add a directed edge from_id → to_id."""
        with self._lock:
            edge_data = {
                "target": to_id,
                "edge_type": edge_type,
                "weight": weight,
                "edge_id": edge_id,
                "label": label,
                "metadata": metadata or {},
            }
            self._outgoing[from_id].append(edge_data)

            reverse_data = {
                "target": from_id,
                "edge_type": edge_type,
                "weight": weight,
                "edge_id": edge_id,
                "label": label,
                "metadata": metadata or {},
            }
            self._incoming[to_id].append(reverse_data)

    def remove_edge(self, from_id: str, to_id: str, edge_type: str | None = None):
        """Remove edge(s) between two nodes. If edge_type=None, removes all."""
        with self._lock:
            self._outgoing[from_id] = [
                e for e in self._outgoing[from_id]
                if not (e["target"] == to_id and
                        (edge_type is None or e["edge_type"] == edge_type))
            ]
            self._incoming[to_id] = [
                e for e in self._incoming[to_id]
                if not (e["target"] == from_id and
                        (edge_type is None or e["edge_type"] == edge_type))
            ]

    # ── Queries ───────────────────────────────────────────────────────────────

    def neighbors(self, node_id: str, depth: int = 1,
                  edge_type: str | None = None,
                  direction: str = "outgoing") -> list[str]:
        """
        BFS traversal from node_id up to depth.
        direction: 'outgoing', 'incoming', or 'both'
        """
        visited = set()
        frontier = {node_id}
        result = []

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)

                edges = []
                if direction in ("outgoing", "both"):
                    edges.extend(self._outgoing.get(nid, []))
                if direction in ("incoming", "both"):
                    edges.extend(self._incoming.get(nid, []))

                for edge in edges:
                    target = edge["target"]
                    if edge_type and edge["edge_type"] != edge_type:
                        continue
                    if target not in visited:
                        next_frontier.add(target)
                        result.append(target)

            frontier = next_frontier

        return result

    def path(self, from_id: str, to_id: str,
             max_depth: int = 10) -> list[str] | None:
        """
        BFS shortest path from from_id to to_id.
        Returns list of node IDs or None if no path.
        """
        if from_id == to_id:
            return [from_id]

        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, current_path = queue.pop(0)
            if len(current_path) > max_depth:
                return None

            for edge in self._outgoing.get(current, []):
                target = edge["target"]
                if target == to_id:
                    return current_path + [target]
                if target not in visited:
                    visited.add(target)
                    queue.append((target, current_path + [target]))

        return None

    def subgraph(self, root_id: str, depth: int = 2) -> dict:
        """
        Extract subgraph around root_id up to depth.
        Returns {nodes: [ids], edges: [edge_dicts]}.
        """
        nodes = set()
        edges = []
        frontier = {root_id}

        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                if nid in nodes:
                    continue
                nodes.add(nid)
                for edge in self._outgoing.get(nid, []):
                    edges.append({"from": nid, **edge})
                    if edge["target"] not in nodes:
                        next_frontier.add(edge["target"])
            frontier = next_frontier

        # Add remaining frontier nodes
        nodes.update(frontier)

        return {"nodes": list(nodes), "edges": edges}

    def connected(self, node_id: str, edge_type: str) -> list[str]:
        """Get all nodes connected by a specific edge type."""
        return [
            e["target"] for e in self._outgoing.get(node_id, [])
            if e["edge_type"] == edge_type
        ]

    def get_edges(self, node_id: str, direction: str = "outgoing") -> list[dict]:
        """Get all edges for a node."""
        if direction == "outgoing":
            return list(self._outgoing.get(node_id, []))
        elif direction == "incoming":
            return list(self._incoming.get(node_id, []))
        else:
            return (list(self._outgoing.get(node_id, [])) +
                    list(self._incoming.get(node_id, [])))

    def degree(self, node_id: str, direction: str = "both") -> int:
        out_d = len(self._outgoing.get(node_id, []))
        in_d = len(self._incoming.get(node_id, []))
        if direction == "outgoing":
            return out_d
        elif direction == "incoming":
            return in_d
        return out_d + in_d

    def node_count(self) -> int:
        return len(set(self._outgoing.keys()) | set(self._incoming.keys()))

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._outgoing.values())

    def stats(self) -> dict:
        return {
            "nodes": self.node_count(),
            "edges": self.edge_count(),
        }
