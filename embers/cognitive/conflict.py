"""
Ember's Diaries — Conflict Detection Engine
Detects contradictions between memory records.
Implements epistemic status tracking (verified, hypothesis, contested).

Key insight from HaluMem (2024): hallucinations accumulate during memory
updating. By detecting conflicts at write time and marking contested
memories, we prevent hallucination propagation.
"""

import uuid
from datetime import datetime, timezone
from collections import defaultdict

from ..core.record import EmberRecord
from ..core.annotation import Annotation
from ..core.types import VerifyStatus, EdgeType


class Conflict:
    """Represents a detected conflict between two or more records."""

    def __init__(self, record_ids: list[str],
                 conflict_type: str,
                 description: str,
                 severity: float = 0.5):
        self.id = str(uuid.uuid4())
        self.record_ids = record_ids
        self.conflict_type = conflict_type  # "contradiction" | "temporal" | "value_mismatch" | "semantic"
        self.description = description
        self.severity = severity  # 0.0 - 1.0
        self.detected_at = datetime.now(timezone.utc)
        self.resolved = False
        self.resolution: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "record_ids": self.record_ids,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


class ConflictDetector:
    """
    Detects and tracks conflicts between memory records.
    
    Detection strategies:
    1. Value conflict: same key/field has different values across records
    2. Temporal conflict: timeline inconsistencies
    3. Semantic conflict: embedding similarity but contradictory content
    4. Supersession conflict: conflicting update chains
    
    When a conflict is detected:
    - Both records get Annotations marking the conflict
    - Records' VerifyStatus can be set to CONTESTED
    - A Conflict object is stored for resolution tracking
    """

    def __init__(self):
        self._conflicts: dict[str, Conflict] = {}
        self._record_conflicts: dict[str, list[str]] = defaultdict(list)  # rid → [conflict_ids]

    def detect_value_conflict(self, new_record: EmberRecord,
                               existing_records: list[EmberRecord],
                               fields: list[str] | None = None) -> list[Conflict]:
        """
        Check if a new record contradicts existing records on specific fields.
        """
        conflicts = []

        if not isinstance(new_record.data, dict):
            return conflicts

        check_fields = fields or list(new_record.data.keys())

        for existing in existing_records:
            if not isinstance(existing.data, dict):
                continue
            if existing.id == new_record.id:
                continue
            if existing.namespace != new_record.namespace:
                continue

            for field in check_fields:
                new_val = new_record.data.get(field)
                old_val = existing.data.get(field)

                if new_val is not None and old_val is not None and new_val != old_val:
                    conflict = Conflict(
                        record_ids=[existing.id, new_record.id],
                        conflict_type="value_mismatch",
                        description=(
                            f"Field '{field}' differs: "
                            f"'{old_val}' (record {existing.id[:8]}) vs "
                            f"'{new_val}' (record {new_record.id[:8]})"
                        ),
                        severity=0.7,
                    )
                    conflicts.append(conflict)
                    self._register_conflict(conflict)

        return conflicts

    def detect_temporal_conflict(self, records: list[EmberRecord]) -> list[Conflict]:
        """
        Detect temporal inconsistencies (e.g., events out of causal order).
        """
        conflicts = []
        sorted_records = sorted(records, key=lambda r: r.created_at)

        for i in range(len(sorted_records) - 1):
            curr = sorted_records[i]
            nxt = sorted_records[i + 1]

            # Check if valid_from/valid_until overlap in conflicting ways
            if (curr.valid_until and nxt.valid_from and
                    curr.valid_until < nxt.valid_from):
                # Gap is fine — no conflict
                continue

            if (curr.valid_from and nxt.valid_until and
                    curr.valid_from > nxt.valid_until):
                conflict = Conflict(
                    record_ids=[curr.id, nxt.id],
                    conflict_type="temporal",
                    description=(
                        f"Temporal ordering conflict between "
                        f"{curr.id[:8]} and {nxt.id[:8]}"
                    ),
                    severity=0.5,
                )
                conflicts.append(conflict)
                self._register_conflict(conflict)

        return conflicts

    def detect_semantic_conflict(self, record_a: EmberRecord,
                                  record_b: EmberRecord,
                                  similarity_score: float,
                                  similarity_threshold: float = 0.85) -> Conflict | None:
        """
        If two records are semantically very similar but have different data,
        they may be conflicting versions of the same information.
        """
        if similarity_score < similarity_threshold:
            return None

        # Check if data differs
        if record_a.data == record_b.data:
            return None

        conflict = Conflict(
            record_ids=[record_a.id, record_b.id],
            conflict_type="semantic",
            description=(
                f"High similarity ({similarity_score:.2f}) but different data "
                f"between {record_a.id[:8]} and {record_b.id[:8]}"
            ),
            severity=similarity_score,
        )
        self._register_conflict(conflict)
        return conflict

    def _register_conflict(self, conflict: Conflict):
        self._conflicts[conflict.id] = conflict
        for rid in conflict.record_ids:
            self._record_conflicts[rid].append(conflict.id)

    def create_conflict_annotations(self, conflict: Conflict) -> list[Annotation]:
        """
        Create annotations for each record involved in a conflict.
        These annotations make the conflict visible in the record's history.
        """
        annotations = []
        for rid in conflict.record_ids:
            ann = Annotation(
                content=f"CONFLICT DETECTED: {conflict.description}",
                context=f"conflict_id:{conflict.id}",
                annotation_type="conflict",
                written_by="conflict_detector",
                confidence=1.0 - conflict.severity,
                tags=["conflict", conflict.conflict_type],
                related_record_ids=[
                    r for r in conflict.record_ids if r != rid
                ],
            )
            ann.target_record_id = rid
            annotations.append(ann)
        return annotations

    def resolve_conflict(self, conflict_id: str, resolution: str,
                          winner_id: str | None = None):
        """
        Mark a conflict as resolved.
        The resolution is recorded but the original conflict remains queryable.
        """
        conflict = self._conflicts.get(conflict_id)
        if conflict:
            conflict.resolved = True
            conflict.resolution = resolution

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_conflicts(self, record_id: str) -> list[Conflict]:
        """Get all conflicts involving a record."""
        conflict_ids = self._record_conflicts.get(record_id, [])
        return [self._conflicts[cid] for cid in conflict_ids
                if cid in self._conflicts]

    def get_unresolved(self) -> list[Conflict]:
        """Get all unresolved conflicts."""
        return [c for c in self._conflicts.values() if not c.resolved]

    def has_conflicts(self, record_id: str) -> bool:
        return bool(self._record_conflicts.get(record_id))

    def conflict_count(self) -> int:
        return len(self._conflicts)

    def stats(self) -> dict:
        unresolved = sum(1 for c in self._conflicts.values() if not c.resolved)
        return {
            "total_conflicts": len(self._conflicts),
            "unresolved": unresolved,
            "records_with_conflicts": len(self._record_conflicts),
        }
