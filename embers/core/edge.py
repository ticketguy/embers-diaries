"""
Ember's Diaries — Edge Reference
Graph connection schema between records.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from .types import EdgeType


@dataclass
class EdgeRef:
    """
    A directed connection between two records.
    Edges are themselves stored as EDGE-type records,
    but referenced inline on the source record for fast traversal.
    """
    # Identity
    edge_id: str              # UUID of the EDGE record in the store
    target_id: str            # Record this edge points to
    edge_type: EdgeType       # Semantic type of the connection

    # Weight and confidence
    weight: float = 1.0       # Strength of the connection (0.0 - 1.0)
    confidence: float = 1.0   # How certain this connection is

    # Context
    label: str = ""           # Human-readable label for the edge
    metadata: dict = field(default_factory=dict)  # Extra context

    # Time
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def is_active(self) -> bool:
        """Check if this edge is currently valid."""
        now = datetime.now(timezone.utc)
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "edge_id":    self.edge_id,
            "target_id":  self.target_id,
            "edge_type":  self.edge_type.value,
            "weight":     self.weight,
            "confidence": self.confidence,
            "label":      self.label,
            "metadata":   self.metadata,
            "created_at": self.created_at.isoformat(),
            "valid_from":  self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeRef":
        return cls(
            edge_id    = d["edge_id"],
            target_id  = d["target_id"],
            edge_type  = EdgeType(d["edge_type"]),
            weight     = d.get("weight", 1.0),
            confidence = d.get("confidence", 1.0),
            label      = d.get("label", ""),
            metadata   = d.get("metadata", {}),
            created_at = datetime.fromisoformat(d["created_at"]),
            valid_from  = datetime.fromisoformat(d["valid_from"]) if d.get("valid_from") else None,
            valid_until = datetime.fromisoformat(d["valid_until"]) if d.get("valid_until") else None,
        )