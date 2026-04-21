"""
Ember's Diaries — Annotation
The only form of "update" — layers new understanding on existing records.
Original records are NEVER touched.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


@dataclass
class Annotation:
    """
    A new interpretation or observation added to an existing record.
    Annotations accumulate over time. They never modify the original.
    Think of them as margin notes in a book — the book stays unchanged.
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_record_id: str = ""     # Which record this annotates

    # Content
    content: str = ""              # The annotation itself
    context: str = ""              # What prompted this annotation
    annotation_type: str = "note"  # note|correction|insight|reflection|validation

    # Authorship
    written_by: str = "system"     # system | agent_id | "sammie" | "lila_emergence"

    # Confidence
    confidence: float = 1.0        # 0.0 - 1.0

    # Time
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Links
    related_record_ids: list = field(default_factory=list)  # Other records referenced
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "target_record_id": self.target_record_id,
            "content":          self.content,
            "context":          self.context,
            "annotation_type":  self.annotation_type,
            "written_by":       self.written_by,
            "confidence":       self.confidence,
            "timestamp":        self.timestamp.isoformat(),
            "related_record_ids": self.related_record_ids,
            "tags":             self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            id               = d["id"],
            target_record_id = d["target_record_id"],
            content          = d.get("content", ""),
            context          = d.get("context", ""),
            annotation_type  = d.get("annotation_type", "note"),
            written_by       = d.get("written_by", "system"),
            confidence       = d.get("confidence", 1.0),
            timestamp        = datetime.fromisoformat(d["timestamp"]),
            related_record_ids = d.get("related_record_ids", []),
            tags             = d.get("tags", []),
        )


@dataclass
class ReflectiveAnnotation(Annotation):
    """
    Annotation written by the Emergence Engine during reflection cycles.
    Records Lila's reinterpretation of old memories with new understanding.
    """
    annotation_type: str = "reflection"
    triggered_by: str = ""            # What prompted this reflection
    context_at_reflection: str = ""   # World state when reflection happened
    insight_score: float = 0.0        # How significant this reinterpretation is

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "triggered_by":           self.triggered_by,
            "context_at_reflection":  self.context_at_reflection,
            "insight_score":          self.insight_score,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReflectiveAnnotation":
        base = Annotation.from_dict(d)
        return cls(
            **{k: v for k, v in base.__dict__.items()},
            triggered_by          = d.get("triggered_by", ""),
            context_at_reflection = d.get("context_at_reflection", ""),
            insight_score         = d.get("insight_score", 0.0),
        )