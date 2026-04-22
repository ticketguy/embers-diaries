"""
Ember's Diaries — Cognitive Engine
Human-inspired memory processing: episodic segmentation, consolidation,
conflict detection, confidence decay, reflective triggers.
"""

from .decay import DecayEngine
from .consolidation import ConsolidationEngine
from .conflict import ConflictDetector
from .episodic import EpisodicSegmenter
from .reflection import ReflectionEngine

__all__ = [
    "DecayEngine", "ConsolidationEngine", "ConflictDetector",
    "EpisodicSegmenter", "ReflectionEngine",
]
