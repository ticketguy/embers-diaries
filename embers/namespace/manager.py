"""
Ember's Diaries — Namespace Manager
Logical partitions within a single store.
Namespace metadata, access control, schema hints.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from ..core.types import AccessLevel
from ..storage.format import encode_index, decode_index


class NamespaceInfo:
    def __init__(self, name: str, description: str = "",
                 access_level: AccessLevel = AccessLevel.PRIVATE,
                 schema_hint: dict | None = None,
                 created_at: datetime | None = None):
        self.name = name
        self.description = description
        self.access_level = access_level
        self.schema_hint = schema_hint
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "access_level": self.access_level.value,
            "schema_hint": self.schema_hint,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NamespaceInfo":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            access_level=AccessLevel(d.get("access_level", "private")),
            schema_hint=d.get("schema_hint"),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else None,
        )


class NamespaceManager:
    """
    Manages namespace creation, listing, access control.
    Multiple AI systems can share one store via separate namespaces.
    """

    def __init__(self, store_path: Path):
        self._path = store_path / "namespaces"
        self._path.mkdir(parents=True, exist_ok=True)
        self._namespaces: dict[str, NamespaceInfo] = {}
        self._load()

    def _load(self):
        index_file = self._path / "registry.json"
        if not index_file.exists():
            return
        try:
            data = decode_index(index_file.read_bytes())
            for name, info in data.get("namespaces", {}).items():
                self._namespaces[name] = NamespaceInfo.from_dict(info)
        except Exception:
            pass

    def persist(self):
        data = {
            "namespaces": {
                name: info.to_dict() for name, info in self._namespaces.items()
            }
        }
        index_file = self._path / "registry.json"
        index_file.write_bytes(encode_index(data))

    def create(self, name: str, description: str = "",
               access_level: AccessLevel = AccessLevel.PRIVATE,
               schema_hint: dict | None = None) -> NamespaceInfo:
        if name in self._namespaces:
            raise ValueError(f"Namespace '{name}' already exists")
        ns = NamespaceInfo(name, description, access_level, schema_hint)
        self._namespaces[name] = ns
        self.persist()
        return ns

    def get(self, name: str) -> NamespaceInfo | None:
        return self._namespaces.get(name)

    def list_all(self) -> list[NamespaceInfo]:
        return list(self._namespaces.values())

    def exists(self, name: str) -> bool:
        return name in self._namespaces

    def delete(self, name: str) -> bool:
        """Remove namespace metadata (records remain in store)."""
        if name in self._namespaces:
            del self._namespaces[name]
            self.persist()
            return True
        return False

    def count(self) -> int:
        return len(self._namespaces)
