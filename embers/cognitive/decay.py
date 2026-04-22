"""
Ember's Diaries — Confidence Decay Engine
Implements Ebbinghaus-inspired forgetting curves at read time.
Records are NEVER mutated — decay is computed on access.

Model: effective_confidence = base_confidence * exp(-decay_rate * hours_since_last_access)
Reinforcement: accessing a record resets the decay clock (via annotation).
"""

import math
from datetime import datetime, timezone
from ..core.record import EmberRecord


class DecayEngine:
    """
    Computes effective confidence for records at read time.
    The immutable record stores base confidence and decay_rate.
    Effective confidence degrades over time unless reinforced.
    
    Based on the Ebbinghaus forgetting curve:
      R = e^(-t/S)
    where R is retention, t is time, S is stability (inverse of decay_rate).
    
    Reinforcement (accessing a record) is tracked via access_count
    and last_accessed, which increases effective stability.
    """

    def __init__(self, min_confidence: float = 0.01,
                 reinforcement_bonus: float = 0.1):
        self.min_confidence = min_confidence
        self.reinforcement_bonus = reinforcement_bonus

    def effective_confidence(self, record: EmberRecord,
                             now: datetime | None = None) -> float:
        """
        Compute the effective confidence of a record right now.
        Never modifies the record — pure computation.
        
        Returns a float in [min_confidence, 1.0].
        """
        if record.decay_rate <= 0:
            return record.confidence

        now = now or datetime.now(timezone.utc)

        # Time since last access or creation
        reference_time = record.last_accessed or record.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - reference_time).total_seconds() / 3600.0
        if elapsed_hours <= 0:
            return record.confidence

        # Stability increases with access count (spaced repetition effect)
        # More accesses = slower decay
        stability_multiplier = 1.0 + (record.access_count * self.reinforcement_bonus)
        effective_decay = record.decay_rate / stability_multiplier

        # Ebbinghaus curve
        retention = math.exp(-effective_decay * elapsed_hours)
        effective = record.confidence * retention

        return max(effective, self.min_confidence)

    def should_reinforce(self, record: EmberRecord,
                         threshold: float = 0.5,
                         now: datetime | None = None) -> bool:
        """
        Check if a record's confidence has decayed below threshold.
        Suggests that this memory should be reinforced (re-accessed, re-validated).
        """
        return self.effective_confidence(record, now) < threshold

    def decay_summary(self, record: EmberRecord,
                      now: datetime | None = None) -> dict:
        """
        Full decay status report for a record.
        """
        now = now or datetime.now(timezone.utc)
        eff = self.effective_confidence(record, now)
        reference_time = record.last_accessed or record.created_at
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        elapsed_hours = max(0, (now - reference_time).total_seconds() / 3600.0)

        return {
            "record_id": record.id,
            "base_confidence": record.confidence,
            "effective_confidence": round(eff, 4),
            "decay_rate": record.decay_rate,
            "access_count": record.access_count,
            "hours_since_access": round(elapsed_hours, 2),
            "needs_reinforcement": eff < 0.5,
            "retention_percentage": round(eff / record.confidence * 100, 1) if record.confidence > 0 else 0,
        }

    def rank_by_urgency(self, records: list[EmberRecord],
                        now: datetime | None = None) -> list[tuple[EmberRecord, float]]:
        """
        Rank records by how urgently they need reinforcement.
        Records with lowest effective confidence come first.
        """
        now = now or datetime.now(timezone.utc)
        scored = [(r, self.effective_confidence(r, now)) for r in records
                  if r.decay_rate > 0]
        scored.sort(key=lambda x: x[1])
        return scored
