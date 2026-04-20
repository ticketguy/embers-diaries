"""
Ember's Diaries
A cognitive database engine for AI memory systems.

Nothing is ever deleted. Nothing is ever overwritten.
Every state that ever existed is preserved. The past is first-class.

Quick start:
    from embers import EmberDB, EmberRecord, RecordType

    db = EmberDB.connect("./my_store")
    record_id = db.write(EmberRecord(
        namespace="memories",
        data={"content": "First memory"},
        tags=["test"],
    ))
    record = db.get(record_id)
"""

from .db import EmberDB
from .core.record import EmberRecord
from .core.annotation import Annotation, ReflectiveAnnotation
from .core.edge import EdgeRef
from .core.types import (
    RecordType, MemoryType, MemoryScope,
    AccessLevel, VerifyStatus, DeprecationReason, EdgeType,
)

__version__ = "0.1.0"
__author__  = "Sammie — ticketguy"

__all__ = [
    "EmberDB", "EmberRecord", "Annotation", "ReflectiveAnnotation",
    "EdgeRef", "RecordType", "MemoryType", "MemoryScope",
    "AccessLevel", "VerifyStatus", "DeprecationReason", "EdgeType",
]