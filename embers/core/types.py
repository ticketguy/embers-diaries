"""
Ember's Diaries — Core Types
All enums used across the system.
"""

from enum import Enum, auto


class RecordType(str, Enum):
    """The six record types Ember's Diaries natively supports."""
    DOCUMENT   = "document"    # Structured or semi-structured data
    NODE       = "node"        # Graph vertex — entity, concept, memory
    EDGE       = "edge"        # Graph connection between nodes
    TIMESERIES = "timeseries"  # Sequential data indexed by time
    VECTOR     = "vector"      # Embedding record for semantic search
    RAW        = "raw"         # Binary, blobs, unstructured


class MemoryType(str, Enum):
    """Memory types for AI cognitive systems (e.g. A Thousand Pearls)."""
    RAW        = "raw"
    SKILL      = "skill"
    FAILURE    = "failure"
    EPISODIC   = "episodic"
    CONNECTIVE = "connective"
    REFLECTIVE = "reflective"


class MemoryScope(str, Enum):
    """Scope levels in the memory tree."""
    TASK      = "task"
    AGENT     = "agent"
    LILACORE  = "lilacore"


class AccessLevel(str, Enum):
    """Namespace access control levels."""
    PUBLIC   = "public"
    PRIVATE  = "private"
    INTERNAL = "internal"


class VerifyStatus(str, Enum):
    """Knowledge entry verification states."""
    VERIFIED   = "verified"
    HYPOTHESIS = "hypothesis"
    CONTESTED  = "contested"
    DEPRECATED = "deprecated"


class DeprecationReason(str, Enum):
    """Why a record was deprecated."""
    SUPERSEDED   = "superseded"    # Replaced by newer version
    INVALID      = "invalid"       # Found to be incorrect
    EXPIRED      = "expired"       # Time-sensitive data past valid_until
    MERGED       = "merged"        # Combined into another record
    MANUAL       = "manual"        # Manually deprecated by operator


class EdgeType(str, Enum):
    """Semantic types for graph edges."""
    RELATES_TO    = "relates_to"
    CAUSED_BY     = "caused_by"
    LED_TO        = "led_to"
    CONTRADICTS   = "contradicts"
    SUPPORTS      = "supports"
    PART_OF       = "part_of"
    INSTANCE_OF   = "instance_of"
    SUPERSEDES    = "supersedes"
    REFLECTS_ON   = "reflects_on"
    DERIVED_FROM  = "derived_from"
    SIMILAR_TO    = "similar_to"
    CUSTOM        = "custom"
