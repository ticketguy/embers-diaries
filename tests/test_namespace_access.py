"""
Ember's Diaries — Tests for Namespace Access Control (Phase 4 gap fix)
"""

import pytest
from embers import EmberDB, EmberRecord, AccessLevel
from embers.namespace.manager import NamespaceManager, AccessDeniedError


@pytest.fixture
def db(tmp_path):
    return EmberDB.connect(tmp_path / "test_store")


# ── Namespace CRUD ────────────────────────────────────────────────────────────

class TestNamespaceCRUD:
    def test_create_namespace(self, db):
        ns = db.create_namespace("memories", description="AI memories",
                                  owner="lila")
        assert ns.name == "memories"
        assert ns.owner == "lila"

    def test_create_duplicate_raises(self, db):
        db.create_namespace("memories")
        with pytest.raises(ValueError, match="already exists"):
            db.create_namespace("memories")

    def test_list_namespaces(self, db):
        db.create_namespace("ns1")
        db.create_namespace("ns2")
        ns_list = db.list_namespaces()
        names = [ns.name for ns in ns_list]
        assert "ns1" in names
        assert "ns2" in names


# ── Access Control ────────────────────────────────────────────────────────────

class TestAccessControl:
    def test_public_namespace_open_to_all(self, db):
        db.create_namespace("public_ns", access_level=AccessLevel.PUBLIC,
                             owner="admin")
        assert db.check_namespace_access("public_ns", "anyone", "read") is True
        assert db.check_namespace_access("public_ns", "anyone", "write") is True

    def test_private_namespace_owner_only(self, db):
        db.create_namespace("secret", access_level=AccessLevel.PRIVATE,
                             owner="lila")
        assert db.check_namespace_access("secret", "lila", "read") is True
        assert db.check_namespace_access("secret", "lila", "write") is True
        assert db.check_namespace_access("secret", "stranger", "read") is False
        assert db.check_namespace_access("secret", "stranger", "write") is False

    def test_internal_namespace_read_open_write_restricted(self, db):
        db.create_namespace("internal_ns", access_level=AccessLevel.INTERNAL,
                             owner="system")
        assert db.check_namespace_access("internal_ns", "anyone", "read") is True
        assert db.check_namespace_access("internal_ns", "anyone", "write") is False
        assert db.check_namespace_access("internal_ns", "system", "write") is True

    def test_unregistered_namespace_open(self, db):
        # Namespace not created via create_namespace — should be open
        assert db.check_namespace_access("unknown_ns", "anyone", "read") is True
        assert db.check_namespace_access("unknown_ns", "anyone", "write") is True

    def test_grant_read_access(self, db):
        db.create_namespace("private_ns", access_level=AccessLevel.PRIVATE,
                             owner="lila")
        assert db.check_namespace_access("private_ns", "sammie", "read") is False

        db.grant_namespace_access("private_ns", "sammie", "read")
        assert db.check_namespace_access("private_ns", "sammie", "read") is True

    def test_grant_write_access(self, db):
        db.create_namespace("private_ns", access_level=AccessLevel.PRIVATE,
                             owner="lila")
        assert db.check_namespace_access("private_ns", "agent_7", "write") is False

        db.grant_namespace_access("private_ns", "agent_7", "write")
        assert db.check_namespace_access("private_ns", "agent_7", "write") is True

    def test_revoke_access(self, db):
        db.create_namespace("ns", access_level=AccessLevel.PRIVATE, owner="lila")
        db.grant_namespace_access("ns", "sammie", "read")
        assert db.check_namespace_access("ns", "sammie", "read") is True

        db.revoke_namespace_access("ns", "sammie", "read")
        assert db.check_namespace_access("ns", "sammie", "read") is False

    def test_require_read_raises(self, db):
        db.create_namespace("locked", access_level=AccessLevel.PRIVATE,
                             owner="lila")
        with pytest.raises(AccessDeniedError):
            db._ns_manager.require_read("locked", "intruder")

    def test_require_write_raises(self, db):
        db.create_namespace("locked", access_level=AccessLevel.INTERNAL,
                             owner="lila")
        with pytest.raises(AccessDeniedError):
            db._ns_manager.require_write("locked", "intruder")

    def test_access_persists_across_reconnect(self, tmp_path):
        store_path = tmp_path / "access_persist"
        db1 = EmberDB.connect(store_path)
        db1.create_namespace("secure", access_level=AccessLevel.PRIVATE,
                              owner="lila")
        db1.grant_namespace_access("secure", "sammie", "write")
        db1.checkpoint()

        db2 = EmberDB.connect(store_path)
        assert db2.check_namespace_access("secure", "sammie", "write") is True
        assert db2.check_namespace_access("secure", "stranger", "write") is False
