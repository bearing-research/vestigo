"""Adding a column to a store that already holds data, on SQLite.

The Postgres half lives in ``test_artifact_store_postgres.py``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3

import pytest

from vestigo.build_store import BuildStore
from vestigo.provenance import TransformSpec
from vestigo.sql_backend import SqliteDialect
from vestigo.store import (
    _BASELINE_SCHEMA_VERSION,
    _LATEST_SCHEMA_VERSION,
    _MIGRATIONS,
    ArtifactStore,
    StoreSchemaMismatch,
)
from vestigo.tables import LEDGER_TABLES, TRANSIENT_TABLES


def _columns(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(artifact_versions)")}
    finally:
        conn.close()


def _schema(db_path) -> set[tuple[str, str]]:
    """Every table and index with its DDL."""
    conn = sqlite3.connect(db_path)
    try:
        return set(conn.execute("SELECT name, sql FROM sqlite_master").fetchall())
    finally:
        conn.close()


def _version(db_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    finally:
        conn.close()


class TestMigrationList:
    def test_versions_are_unique_and_ordered(self):
        """Applying by ``version`` means a duplicate would silently skip one."""
        versions = [m.version for m in _MIGRATIONS]
        assert versions == sorted(versions)
        assert len(versions) == len(set(versions))

    def test_every_version_is_above_the_baseline(self):
        """A migration at or below the baseline never runs on an old database."""
        assert all(m.version > _BASELINE_SCHEMA_VERSION for m in _MIGRATIONS)


class TestFreshDatabase:
    def test_is_stamped_at_the_latest_version(self, tmp_path):
        store = ArtifactStore(tmp_path / "fresh")

        assert _version(store.db_path) == _LATEST_SCHEMA_VERSION

    def test_has_every_migrated_column(self, tmp_path):
        store = ArtifactStore(tmp_path / "fresh")

        assert "content_sha256" in _columns(store.db_path)


class TestExistingDatabase:
    def test_a_store_predating_the_version_table_is_migrated(self, tmp_path):
        """The case the mechanism exists for."""
        store = ArtifactStore(tmp_path / "old")
        db_path = store.db_path

        # Rewind it to what a pre-migration database looked like.
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE schema_version")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
        conn.commit()
        conn.close()
        assert "content_sha256" not in _columns(db_path)

        ArtifactStore(tmp_path / "old")  # reopening applies what is pending

        assert "content_sha256" in _columns(db_path)
        assert _version(db_path) == _LATEST_SCHEMA_VERSION

    def test_reopening_a_current_database_applies_nothing(self, tmp_path):
        """Idempotent, and the common path: every construction runs this."""
        db_path = ArtifactStore(tmp_path / "s").db_path
        before = _version(db_path)

        for _ in range(3):
            ArtifactStore(tmp_path / "s")

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        finally:
            conn.close()
        assert _version(db_path) == before
        assert rows == 1, "a no-op reopen must not stamp another row"

    def test_data_survives_a_migration(self, tmp_path):
        """The rows are why the store could not be recreated."""
        store = ArtifactStore(tmp_path / "old")
        version = store.create_artifact("keeper", hashlib.sha256(b"x").hexdigest())
        with store.open_blob_writer("keeper", version) as writer:
            writer.write(b"x")
        store.finalize_artifact("keeper", version, schema_json="", row_count=0, byte_size=1)

        conn = sqlite3.connect(store.db_path)
        conn.execute("DROP TABLE schema_version")
        conn.execute("ALTER TABLE artifact_versions DROP COLUMN content_sha256")
        conn.commit()
        conn.close()

        reopened = ArtifactStore(tmp_path / "old")

        artifact = reopened.get_artifact("keeper", version)
        assert artifact is not None
        assert artifact.state == "ready"

    def test_a_store_a_newer_vestigo_migrated_is_refused(self, tmp_path):
        """An older vestigo on a migrated store would write rows the newer one does not expect."""
        db_path = ArtifactStore(tmp_path / "s").db_path
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, 0)",
            (_LATEST_SCHEMA_VERSION + 1,),
        )
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="newer than this vestigo supports"):
            ArtifactStore(tmp_path / "s")


class TestTwoFirstBoots:
    """Two processes creating one SQLite store together both come out at the latest version."""

    def test_a_second_boot_between_the_tables_and_the_stamp_finds_it_current(
        self, tmp_path, monkeypatch, caplog
    ):
        """The tables and the stamp go in one locked script: a second boot that found
        the tables without the stamp would take the store for the baseline and replay
        every migration on one born at the latest shape."""
        target = tmp_path / "artifacts"
        real = SqliteDialect.begin_write
        second: list[ArtifactStore] = []

        def then_the_second_boots(self, conn, key):
            # The first boot's first lock after its tables were made.
            if not second:
                monkeypatch.setattr(SqliteDialect, "begin_write", real)
                second.append(ArtifactStore(target))
            real(self, conn, key)

        monkeypatch.setattr(SqliteDialect, "begin_write", then_the_second_boots)
        with caplog.at_level(logging.INFO, logger="vestigo.store"):
            first = ArtifactStore(target)

        assert second
        assert [r.message for r in caplog.records if "migration" in r.message] == []
        assert _version(first.db_path) == _LATEST_SCHEMA_VERSION
        conn = sqlite3.connect(first.db_path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        finally:
            conn.close()


class TestAReadOnlyOpen:
    """For commands that only inspect: the store stays exactly as it was."""

    def test_an_older_database_is_refused_untouched(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version >= 9")
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (8, 0)")
        conn.commit()
        conn.close()

        before = _schema(db_path)

        with pytest.raises(StoreSchemaMismatch, match="version 8, older"):
            ArtifactStore(tmp_path / "old", read_only=True)

        assert _version(db_path) == 8
        assert _schema(db_path) == before

    def test_a_store_one_migration_behind_is_refused_as_older_not_part_made(self, tmp_path):
        """The newest migration's index is missing because the migration has not run,
        not because a first boot died: the refusal says the store is older."""
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DROP INDEX idx_versions_superseded_by")
        conn.execute("UPDATE schema_version SET version = ?", (_LATEST_SCHEMA_VERSION - 1,))
        conn.commit()
        conn.close()

        with pytest.raises(
            StoreSchemaMismatch, match=f"version {_LATEST_SCHEMA_VERSION - 1}, older"
        ):
            ArtifactStore(tmp_path / "old", read_only=True)

        ArtifactStore(tmp_path / "old").close()
        ArtifactStore(tmp_path / "old", read_only=True).close()
        assert ("idx_versions_superseded_by",) in {(n,) for n, _ in _schema(db_path)}

    def test_a_missing_store_is_refused_before_anything_is_created(self, tmp_path):
        target = tmp_path / "nope" / "artifacts"

        with pytest.raises(StoreSchemaMismatch, match="No artifact store"):
            ArtifactStore(target, read_only=True)

        assert not target.exists()
        assert not target.parent.exists()

    def test_a_database_from_before_versioning_is_refused(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "old").db_path
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE schema_version")
        conn.commit()
        conn.close()

        with pytest.raises(StoreSchemaMismatch, match="older"):
            ArtifactStore(tmp_path / "old", read_only=True)

    def test_a_newer_database_is_refused_as_before(self, tmp_path):
        db_path = ArtifactStore(tmp_path / "s").db_path
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, 0)",
            (_LATEST_SCHEMA_VERSION + 1,),
        )
        conn.commit()
        conn.close()

        with pytest.raises(StoreSchemaMismatch, match="newer than this vestigo supports"):
            ArtifactStore(tmp_path / "s", read_only=True)

    def test_a_current_database_opens(self, tmp_path):
        ArtifactStore(tmp_path / "s")

        assert ArtifactStore(tmp_path / "s", read_only=True).stats()["total_versions"] == 0

    def test_a_dialect_without_the_blob_directory_is_refused_creating_nothing(self, tmp_path):
        """The database is found elsewhere; the default blob store would be created here."""
        real = ArtifactStore(tmp_path / "real")
        wrong = tmp_path / "wrong" / "artifacts"

        with pytest.raises(StoreSchemaMismatch, match="No artifact store"):
            ArtifactStore(wrong, dialect=SqliteDialect(real.db_path), read_only=True)

        assert not wrong.parent.exists()

    def test_both_backends_given_open_creating_nothing(self, tmp_path):
        real = ArtifactStore(tmp_path / "real")
        wrong = tmp_path / "wrong" / "artifacts"

        opened = ArtifactStore(
            wrong, blob_store=real.blob_store, dialect=SqliteDialect(real.db_path), read_only=True
        )

        assert opened.stats()["total_versions"] == 0
        assert not wrong.parent.exists()


# What each migration added: stripped, a database is back at the baseline the chain starts
# from. A migration that adds a column, an index or a table lists it here.
_MIGRATED_COLUMNS = {
    "artifact_versions": [
        "content_sha256",
        "blob_attempt",
        "last_used_at",
        "minted",
        "superseded_by",
        "definition_id",
        "release_id",
        "computation_key",
    ],
    "artifact_publications": ["authors", "external_ids"],
}
_MIGRATED_INDEXES = [
    "idx_versions_definition",
    "idx_versions_release",
    "idx_versions_computation",
    "idx_publications_active",
    "idx_versions_superseded_by",
    "idx_names_version",
    "idx_aliases_version",
    "idx_pending_version",
]
_MIGRATED_TABLES = [
    "schema_version",
    "import_staging",
    "retention_roots",
    "retention_edges",
    "store_upgrades",
    "nodes",
    "node_inputs",
    "node_links",
    "build_releases",
    "release_references",
    "release_authorities",
]


def _shape(db_path) -> dict:
    """Every table's columns and its explicit indexes, with their keys and predicates."""
    conn = sqlite3.connect(db_path)
    try:
        tables = sorted(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        shape = {}
        for table in tables:
            columns = sorted(
                (r[1], r[2].upper(), r[3], r[4], r[5])
                for r in conn.execute(f"PRAGMA table_info({table})")
            )
            indexes = {}
            for index in conn.execute(f"PRAGMA index_list({table})"):
                name, unique, origin = index[1], index[2], index[3]
                if origin != "c":
                    continue
                keyed = tuple(r[2] for r in conn.execute(f"PRAGMA index_info({name})"))
                sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (name,))
                indexes[name] = (keyed, bool(unique), "WHERE" in (sql.fetchone()[0] or "").upper())
            shape[table] = {"columns": columns, "indexes": indexes}
        return shape
    finally:
        conn.close()


def _strip_to_baseline(db_path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for name in _MIGRATED_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {name}")
        for table, columns in _MIGRATED_COLUMNS.items():
            for column in columns:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        for table in _MIGRATED_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    finally:
        conn.close()


def _differences(expected: dict, actual: dict) -> dict:
    return {
        table: (expected.get(table), actual.get(table))
        for table in sorted(set(expected) | set(actual))
        if expected.get(table) != actual.get(table)
    }


class TestMigrationsProduceTheConstantsShape:
    """A change to the schema constants without its migration breaks only old stores,
    which every other test, starting fresh, never opens."""

    def test_the_migration_chain_alone_matches_a_fresh_store(self, tmp_path):
        """What Postgres gets: the migrations, and nothing from the constants."""
        fresh = ArtifactStore(tmp_path / "fresh")
        aged = ArtifactStore(tmp_path / "aged")
        _strip_to_baseline(aged.db_path)
        conn = aged._get_connection()
        try:
            aged._apply_schema_migrations(conn)
        finally:
            conn.close()

        assert _differences(_shape(fresh.db_path), _shape(aged.db_path)) == {}

    def test_a_reopen_matches_a_fresh_store(self, tmp_path):
        """What SQLite gets: the constants' CREATE IF NOT EXISTS, then the migrations."""
        fresh = ArtifactStore(tmp_path / "fresh")
        aged = ArtifactStore(tmp_path / "aged")
        _strip_to_baseline(aged.db_path)

        ArtifactStore(tmp_path / "aged")

        assert _differences(_shape(fresh.db_path), _shape(aged.db_path)) == {}


def _primary_key(db_path, table: str) -> tuple[str, ...]:
    conn = sqlite3.connect(db_path)
    try:
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return tuple(r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5]))


class TestRootsKeyedByTenant:
    def test_the_migration_rekeys_an_old_table_and_keeps_its_rows(self, tmp_path):
        """(audit SR-D) The Postgres half is in ``test_artifact_store_postgres.py``."""
        store = ArtifactStore(tmp_path / "s")
        version = store.create_artifact("shared", "prov-1")
        store.retain("shared", version, kind="pin", holder="review", tenant="a")
        conn = sqlite3.connect(store.db_path)
        conn.executescript(
            """
            CREATE TABLE old_roots (
                tenant TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL,
                holder TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at REAL NOT NULL,
                created_by TEXT,
                PRIMARY KEY (kind, holder, artifact_id, version)
            );
            INSERT INTO old_roots SELECT * FROM retention_roots;
            DROP TABLE retention_roots;
            ALTER TABLE old_roots RENAME TO retention_roots;
            DELETE FROM schema_version;
            INSERT INTO schema_version (version, applied_at) VALUES (16, 0);
            """
        )
        conn.close()
        assert _primary_key(store.db_path, "retention_roots") == (
            "kind",
            "holder",
            "artifact_id",
            "version",
        )

        reopened = ArtifactStore(tmp_path / "s")

        assert _primary_key(store.db_path, "retention_roots") == (
            "tenant",
            "kind",
            "holder",
            "artifact_id",
            "version",
        )
        assert [r["tenant"] for r in reopened.retention_roots("shared", version)] == ["a"]
        reopened.retain("shared", version, kind="pin", holder="review", tenant="b")
        tenants = sorted(r["tenant"] for r in reopened.retention_roots("shared", version))
        assert tenants == ["a", "b"]
        indexes = {name for name, _ in _schema(store.db_path)}
        assert {"idx_retention_roots_version", "idx_retention_roots_tenant"} <= indexes


class TestTablesListing:
    def test_the_listing_names_every_table_with_its_key(self, tmp_path):
        """A copy keyed by the listing is idempotent only if each key is the table's."""
        store = ArtifactStore(tmp_path / "artifacts")
        BuildStore(tmp_path / "builds.sqlite", dialect=SqliteDialect(store.db_path))
        tables = set(_shape(store.db_path))
        keys = {name: _primary_key(store.db_path, name) for name, _ in LEDGER_TABLES}

        assert {name for name, _ in LEDGER_TABLES} | set(TRANSIENT_TABLES) == tables
        assert keys == dict(LEDGER_TABLES)


def _ready(store, artifact_id, executor, inputs, input_versions=None) -> None:
    spec = TransformSpec(executor=executor, params={"sql": "SELECT 1"}, inputs=inputs)
    version = store.create_artifact(
        artifact_id, f"hash-{artifact_id}", transform_spec=spec, input_versions=input_versions
    )
    store.finalize_artifact(artifact_id, version, "{}", 1, 1)


def test_column_exists_reports_truthfully(tmp_path):
    """The primitive every migration is built from."""
    store = ArtifactStore(tmp_path / "s")
    dialect = SqliteDialect(store.db_path)
    conn = dialect.connect()
    try:
        assert dialect.column_exists(conn, "artifact_versions", "content_sha256")
        assert not dialect.column_exists(conn, "artifact_versions", "no_such_column")
    finally:
        conn.close()
