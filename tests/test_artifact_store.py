"""Tests for the artifact store: lifecycle, provenance dedup, names, blob I/O, cleanup."""

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import fields, replace

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from vestigo import uris
from vestigo.blob_store import BLOB_STREAM_CHUNK_BYTES, S3BlobStore
from vestigo.build_store import BuildStore
from vestigo.graph import NodeSpec
from vestigo.provenance import (
    TransformSpec,
    checkpoint_output_key,
    compute_provenance_hash,
    parse_transform_spec,
)
from vestigo.releases import ReleaseStore
from vestigo.sql_backend import SqlDialect, SqliteDialect
from vestigo.store import (
    _VERSION_FIELDS,
    IMPORT_STAGING_TTL_SECONDS,
    ArtifactIdTaken,
    ArtifactImportConflict,
    ArtifactImportInProgress,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactVersion,
    StagedVersion,
    get_artifact_store,
    reset_artifact_store,
)


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    yield artifact_dir
    reset_artifact_store()


@pytest.fixture
def store(artifact_dir):
    """An artifact store."""
    store = ArtifactStore(artifact_dir)
    yield store
    store.close()


class TestTransformSpec:
    def test_canonical_json(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT * FROM input"},
            inputs=["vestigo://table/db.events"],
        )
        json_str = spec.canonical_json
        data = json.loads(json_str)
        assert data["executor"] == "local://duckdb_sql@v1"
        assert data["params"] == {"sql": "SELECT * FROM input"}
        assert data["inputs"] == ["vestigo://table/db.events"]

    def test_parse(self):
        json_str = json.dumps(
            {
                "executor": "local://duckdb_sql@v1",
                "params": {"sql": "SELECT 1"},
                "inputs": [],
            }
        )
        spec = parse_transform_spec(json_str)
        assert spec == TransformSpec("local://duckdb_sql@v1", {"sql": "SELECT 1"}, [])

    def test_roundtrip(self):
        original = TransformSpec(
            executor="local://polars_expr@v1",
            params={"expr": "col('a') + 1"},
            inputs=["input1", "input2"],
        )
        assert parse_transform_spec(original.canonical_json) == original

    @pytest.mark.parametrize("text", [None, "", "{not json", "[1, 2]", "null"])
    def test_an_absent_or_unreadable_spec_parses_to_none(self, text):
        """Rows arrive from other servers too, so an unreadable spec reads as unrecorded."""
        assert parse_transform_spec(text) is None

    def test_missing_or_mistyped_fields_read_as_empty(self):
        spec = parse_transform_spec(json.dumps({"params": "oops", "inputs": {"a": 1}}))
        assert spec == TransformSpec(executor="", params={}, inputs=[])

    def test_an_artifact_version_reads_its_spec_and_params(self):
        stored = TransformSpec("notebook/cell@v1", {"content_type": "json/object"}, ["u"])
        version = ArtifactVersion(
            id="a",
            version=1,
            state="ready",
            provenance_hash="p",
            transform_spec=stored.canonical_json,
        )
        assert version.spec == stored
        assert version.params == {"content_type": "json/object"}

    @pytest.mark.parametrize("transform_spec", [None, "{not json"])
    def test_an_artifact_version_without_a_readable_spec_has_empty_params(self, transform_spec):
        version = ArtifactVersion(
            id="a", version=1, state="ready", provenance_hash="p", transform_spec=transform_spec
        )
        assert version.spec is None
        assert version.params == {}


class TestVersionRowMapping:
    def test_the_select_list_is_the_dataclass(self):
        # Every SELECT of a version and its constructor read `_VERSION_FIELDS`; a
        # column added to the dataclass alone would come back as its default.

        assert _VERSION_FIELDS == tuple(field.name for field in fields(ArtifactVersion))


class TestProvenanceHash:
    def test_deterministic(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["abc", "def"], spec)
        assert hash1 == hash2

    def test_input_order_independent(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["def", "abc"], spec)
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc"], spec)
        hash2 = compute_provenance_hash(["xyz"], spec)
        assert hash1 != hash2

    def test_different_transform_different_hash(self):
        spec1 = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        spec2 = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 2"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc"], spec1)
        hash2 = compute_provenance_hash(["abc"], spec2)
        assert hash1 != hash2

    def test_spec_bytes_and_hash_are_pinned(self):
        """Every stored provenance hash depends on these bytes: a format change orphans them all."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT * FROM input0", "b": [1, 2], "a": {"z": 1, "y": None}},
            # Inputs are opaque bytes to the hash: the scheme is whatever the consumer wrote.
            inputs=["strata://artifact/b@v=2", "file:///wh#db.events"],
        )
        assert spec.canonical_json == (
            '{"executor": "local://duckdb_sql@v1", '
            '"inputs": ["strata://artifact/b@v=2", "file:///wh#db.events"], '
            '"params": {"a": {"y": null, "z": 1}, "b": [1, 2], "sql": "SELECT * FROM input0"}}'
        )
        assert (
            compute_provenance_hash(["h2", "h1"], spec)
            == "c96cecb9766d0728d2fc1ac223f0026613001bfbbd6b475b80933c5bc62eb6ff"
        )


class TestArtifactCRUD:
    def test_create_artifact(self, store):
        """A created artifact starts in building state."""
        version = store.create_artifact(
            artifact_id="test-id",
            provenance_hash="hash123",
        )
        assert version == 1

        artifact = store.get_artifact("test-id", version)
        assert artifact is not None
        assert artifact.id == "test-id"
        assert artifact.version == 1
        assert artifact.state == "building"
        assert artifact.provenance_hash == "hash123"

    def test_create_increments_version(self, store):
        v1 = store.create_artifact("test-id", "hash1")
        v2 = store.create_artifact("test-id", "hash2")
        v3 = store.create_artifact("test-id", "hash3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_create_with_transform_spec(self, store):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        version = store.create_artifact(
            artifact_id="test-id",
            provenance_hash="hash123",
            transform_spec=spec,
        )
        artifact = store.get_artifact("test-id", version)
        assert artifact.transform_spec == spec.canonical_json

    def test_finalize_artifact(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact(
            artifact_id="test-id",
            version=version,
            schema_json='{"fields": []}',
            row_count=100,
            byte_size=1024,
        )

        artifact = store.get_artifact("test-id", version)
        assert artifact.state == "ready"
        assert artifact.schema_json == '{"fields": []}'
        assert artifact.row_count == 100
        assert artifact.byte_size == 1024

    def test_finalize_nonexistent_raises(self, store):
        with pytest.raises(ValueError) as exc_info:
            store.finalize_artifact("nonexistent", 1, "{}", 0, 0)
        assert "not found" in str(exc_info.value)

    def test_finalize_already_ready_is_idempotent(self, store):
        """Finalizing an already-ready artifact returns the existing one."""
        version = store.create_artifact("test-id", "hash123")
        first_result = store.finalize_artifact("test-id", version, "{}", 0, 0)

        # Finalize is idempotent.
        second_result = store.finalize_artifact("test-id", version, "{}", 0, 0)
        assert second_result is not None
        assert second_result.id == first_result.id
        assert second_result.version == first_result.version
        assert second_result.state == "ready"

    def test_fail_artifact(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        artifact = store.get_artifact("test-id", version)
        assert artifact.state == "failed"

    def test_get_nonexistent(self, store):
        result = store.get_artifact("nonexistent", 1)
        assert result is None

    def test_get_latest_version(self, store):
        """Returns the latest ready version."""
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)

        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)

        # v3 is still building.
        store.create_artifact("test-id", "hash3")

        # Latest is v2, the highest ready version.
        latest = store.get_latest_version("test-id")
        assert latest is not None
        assert latest.version == 2
        assert latest.row_count == 200


class TestProvenanceLookup:
    def test_find_by_provenance(self, store):
        version = store.create_artifact("test-id", "unique-hash")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        found = store.find_by_provenance("unique-hash")
        assert found is not None
        assert found.id == "test-id"
        assert found.version == version

    def test_find_by_provenance_not_found(self, store):
        found = store.find_by_provenance("unknown-hash")
        assert found is None

    def test_find_by_provenance_ignores_building(self, store):
        store.create_artifact("test-id", "hash123")
        # Not finalized, so not found.
        found = store.find_by_provenance("hash123")
        assert found is None

    def test_find_by_provenance_ignores_failed(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        found = store.find_by_provenance("hash123")
        assert found is None

    def test_a_lookup_does_not_wait_for_a_writer_to_note_its_use(self, store):
        """The use note is advisory: a hit returns while a sweep holds the write lock,
        rather than waiting the busy timeout behind it. The lock is held throughout,
        so the lookup can only come back by giving the note up."""
        version = store.create_artifact("test-id", "unique-hash")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        conn = store._get_connection()
        try:
            conn.execute("UPDATE artifact_versions SET last_used_at = 0")
            conn.commit()
        finally:
            conn.close()
        writer = sqlite3.connect(str(store.db_path), check_same_thread=False)
        writer.execute("BEGIN IMMEDIATE")
        found = []

        def look_up():
            found.append(store.find_by_provenance("unique-hash"))

        lookup = threading.Thread(target=look_up)
        try:
            lookup.start()
            lookup.join(timeout=10)
            assert not lookup.is_alive(), "the lookup is waiting on the writer's lock"
        finally:
            writer.rollback()
            writer.close()
            lookup.join()

        assert found[0] is not None and found[0].id == "test-id"
        assert store.get_artifact("test-id", version) is not None


class TestBlobIO:
    def test_write_and_read_blob(self, store):
        version = store.create_artifact("test-id", "hash123")
        data = b"test arrow data"

        store.write_blob("test-id", version, data)
        result = store.read_blob("test-id", version)

        assert result == data

    def test_read_nonexistent_blob(self, store):
        result = store.read_blob("nonexistent", 1)
        assert result is None

    def test_blob_exists(self, store):
        version = store.create_artifact("test-id", "hash123")

        assert store.blob_exists("test-id", version) is False

        store.write_blob("test-id", version, b"data")

        assert store.blob_exists("test-id", version) is True

    def test_write_blob_atomic(self, store, artifact_dir):
        """Blob writes are atomic (no partial writes)."""
        version = store.create_artifact("test-id", "hash123")
        data = b"x" * 10000

        store.write_blob("test-id", version, data)

        temp_files = list(artifact_dir.glob("**/*.tmp"))
        assert len(temp_files) == 0

        assert store.read_blob("test-id", version) == data


class TestNamePointers:
    def test_set_and_resolve_name(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        store.set_name("my-artifact", "test-id", version)

        resolved = store.resolve_name("my-artifact")
        assert resolved is not None
        assert resolved.id == "test-id"
        assert resolved.version == version

    def test_resolve_nonexistent_name(self, store):
        resolved = store.resolve_name("nonexistent")
        assert resolved is None

    def test_set_name_requires_ready(self, store):
        version = store.create_artifact("test-id", "hash123")
        # Not finalized

        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "test-id", version)
        assert "not readable" in str(exc_info.value)

    def test_set_name_accepts_a_superseded_current_value(self, store):
        """An id overtaken by another id's run keeps a current value, which can be named."""
        _make_ready_artifact(store, "theirs", "prov-x")
        _make_ready_artifact(store, "mine", "prov-x")
        assert store.get_artifact("mine", 1).state == "superseded"

        store.set_name("my-artifact", "mine", 1)

        resolved = store.resolve_name("my-artifact")
        assert (resolved.id, resolved.version) == ("mine", 1)

    def test_set_name_requires_exists(self, store):
        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "nonexistent", 1)
        assert "not found" in str(exc_info.value)

    def test_update_name(self, store):
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", v1)

        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)
        store.set_name("my-artifact", "test-id", v2)

        resolved = store.resolve_name("my-artifact")
        assert resolved.version == v2

    def test_get_name(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", version)

        name_info = store.get_name("my-artifact")
        assert name_info is not None
        assert name_info.name == "my-artifact"
        assert name_info.artifact_id == "test-id"
        assert name_info.version == version
        assert name_info.updated_at > 0

    def test_delete_name(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", version)

        assert store.delete_name("my-artifact") is True
        assert store.resolve_name("my-artifact") is None

    def test_delete_nonexistent_name(self, store):
        assert store.delete_name("nonexistent") is False

    def test_list_names(self, store):
        for i in range(3):
            v = store.create_artifact(f"id-{i}", f"hash-{i}")
            store.finalize_artifact(f"id-{i}", v, "{}", i * 100, i * 1000)
            store.set_name(f"name-{i}", f"id-{i}", v)

        names = store.list_names()
        assert len(names) == 3
        assert [n.name for n in names] == ["name-0", "name-1", "name-2"]


class TestCleanup:
    def test_cleanup_failed(self, store, artifact_dir):
        """Cleanup removes failed artifacts older than max age."""
        version = store.create_artifact("test-id", "hash123")
        store.write_blob("test-id", version, b"data")
        store.fail_artifact("test-id", version)

        # Too recent to clean up yet.
        count = store.cleanup_failed(max_age_seconds=3600)
        assert count == 0

        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 1

        assert store.get_artifact("test-id", version) is None
        assert store.blob_exists("test-id", version) is False

    def test_cleanup_failed_takes_a_failed_build_row_with_it(self, store):
        """A server build that failed leaves a build row, whose foreign key refused the delete."""
        builds = BuildStore(store.db_path, dialect=store.dialect)
        version = store.create_artifact("built", "hash-built")
        builds.create_build("build-1", "built", version, "exec@v1")
        builds.fail_build("build-1", "boom")
        store.fail_artifact("built", version)

        assert store.cleanup_failed(max_age_seconds=-10) == 1
        assert store.get_artifact("built", version) is None
        assert builds.get_build("build-1") is None

    def test_cleanup_failed_leaves_the_top_version_while_a_lower_one_stays(self, store):
        """Otherwise the next create reuses the number, as the sweep's gap rule prevents."""
        store.create_artifact("model", "prov-1")
        store.finalize_artifact("model", 1, "{}", 1, 10)
        failed = store.create_artifact("model", "prov-2")
        store.fail_artifact("model", failed)
        assert failed == 2

        assert store.cleanup_failed(max_age_seconds=-10) == 0
        assert store.get_artifact("model", 2).state == "failed"
        assert store.create_artifact("model", "prov-3") == 3

        store.fail_artifact("model", 3)
        assert store.cleanup_failed(max_age_seconds=-10) == 1
        assert store.get_artifact("model", 2) is None
        assert store.get_artifact("model", 3).state == "failed"

    def test_cleanup_preserves_ready(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        # Ready artifacts survive even with 0 age.
        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 0
        assert store.get_artifact("test-id", version) is not None


class TestStats:
    def test_stats_empty(self, store):
        stats = store.stats()
        assert stats["total_versions"] == 0
        assert stats["ready_versions"] == 0
        assert stats["building_versions"] == 0
        assert stats["failed_versions"] == 0
        assert stats["total_bytes"] == 0
        assert stats["total_rows"] == 0
        assert stats["name_count"] == 0

    def test_stats_with_data(self, store):
        v1 = store.create_artifact("id-1", "hash1")
        store.finalize_artifact("id-1", v1, "{}", 100, 1000)

        store.create_artifact("id-2", "hash2")

        v3 = store.create_artifact("id-3", "hash3")
        store.fail_artifact("id-3", v3)

        store.set_name("my-name", "id-1", v1)

        stats = store.stats()
        assert stats["total_versions"] == 3
        assert stats["ready_versions"] == 1
        assert stats["building_versions"] == 1
        assert stats["failed_versions"] == 1
        assert stats["total_bytes"] == 1000
        assert stats["total_rows"] == 100
        assert stats["name_count"] == 1

    def test_byte_totals_count_what_versions_still_hold_as_the_sweep_does(self, store):
        """An older version a refresh superseded keeps its bytes until a sweep takes them."""
        for size in (1000, 300):
            version = store.create_artifact("id-1", "hash1")
            store.finalize_artifact("id-1", version, "{}", 1, size)
        store.create_artifact("id-2", "hash1")
        store.finalize_artifact("id-2", 1, "{}", 1, 300)  # reads id-1@v=2's bytes

        swept = store.garbage_collect(dry_run=True)["store_bytes"]
        assert store.stats()["total_bytes"] == swept == 1300
        assert store.get_usage()["total_bytes"] == 1300

    def test_superseded_versions_are_counted_so_the_states_add_up(self, store):
        """A refresh and a dedup each leave a superseded version, which no other count holds."""
        for _ in range(2):
            version = store.create_artifact("id-1", "hash1")
            store.finalize_artifact("id-1", version, "{}", 1, 10)
        store.create_artifact("id-2", "hash1")
        store.finalize_artifact("id-2", 1, "{}", 1, 10)
        store.create_artifact("id-3", "hash3")

        for counts in (store.stats(), store.get_usage()):
            assert counts["superseded_versions"] == 2
            assert counts["total_versions"] == sum(
                counts[f"{state}_versions"]
                for state in ("ready", "building", "superseded", "failed")
            )

    def test_name_prefix_treats_an_underscore_literally(self, store):
        for name, artifact_id in (("taxi_model", "a"), ("taxi/model", "b")):
            version = store.create_artifact(artifact_id, f"hash-{artifact_id}")
            store.finalize_artifact(artifact_id, version, "{}", 1, 1)
            store.set_name(name, artifact_id, version)

        assert [a.id for a in store.list_artifacts(name_prefix="taxi_")] == ["a"]


def _ipc_bytes(num_rows: int) -> bytes:
    """A single valid Arrow IPC stream with ``num_rows`` rows."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


class TestForeignKeyEnforcement:
    """Declared foreign keys are enforced, and deletes clean up so enforcement does not break
    collection.

    SQLite ignores FOREIGN KEY clauses unless ``PRAGMA foreign_keys=ON`` is set per connection;
    Postgres always enforces them.
    """

    def _ready(self, store, artifact_id, provenance):
        version = store.create_artifact(artifact_id, provenance)
        store.blob_store.write_blob(artifact_id, version, b"x")
        store.finalize_artifact(artifact_id, version, "{}", 1, 1)
        return version

    def test_a_name_cannot_point_at_a_version_that_does_not_exist(self, store):
        conn = store._get_connection()
        try:
            with pytest.raises(store.dialect.integrity_error):
                conn.execute(
                    "INSERT INTO artifact_names (name, artifact_id, version, updated_at, tenant) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("ghost", "art-missing", 1, 0.0, ""),
                )
                conn.commit()
        finally:
            conn.close()

    def test_a_version_cannot_be_deleted_out_from_under_its_name(self, store):
        self._ready(store, "art-1", "prov-a")
        store.set_name("team/model", "art-1", 1)

        conn = store._get_connection()
        try:
            with pytest.raises(store.dialect.integrity_error):
                conn.execute(
                    "DELETE FROM artifact_versions WHERE id = ? AND version = ?", ("art-1", 1)
                )
                conn.commit()
        finally:
            conn.close()

        assert store.get_artifact("art-1", 1) is not None

    def test_delete_artifact_takes_the_build_row_with_it(self, tmp_path, store):
        builds = BuildStore(store.db_path)
        self._ready(store, "art-1", "prov-a")
        builds.create_build("build-1", "art-1", 1, "exec@v1")

        # Without the cascade this raises: every server-built artifact has a
        # build row, so the foreign key would make it undeletable.
        assert store.delete_artifact("art-1", 1) is True
        assert builds.get_build("build-1") is None

    def test_garbage_collect_takes_build_rows_with_it(self, tmp_path, store):
        builds = BuildStore(store.db_path)
        self._ready(store, "art-1", "prov-a")
        self._ready(store, "art-1", "prov-b")
        builds.create_build("build-1", "art-1", 1, "exec@v1")
        # Age v1 past the cutoff; v2 stays as the artifact's current value.
        conn = store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_versions SET created_at = 0, last_used_at = 0 "
                "WHERE id = ? AND version = 1",
                ("art-1",),
            )
            conn.commit()
        finally:
            conn.close()

        stats = store.garbage_collect(max_idle_days=1.0)

        assert stats["deleted_count"] == 1
        assert store.get_artifact("art-1", 1) is None
        assert builds.get_build("build-1") is None


class TestVersionPromotion:
    """Re-pointing ``latest`` at a result the artifact already holds.

    Consumers read through ``get_latest_version``, so returning to a previous provenance is a cache
    hit only if promotion re-records the older version.
    """

    def _ready(self, store, artifact_id, provenance, blob):
        version = store.create_artifact(artifact_id, provenance)
        store.blob_store.write_blob(artifact_id, version, blob)
        store.finalize_artifact(artifact_id, version, "{}", 1, len(blob))
        return version

    def test_finds_a_superseded_version_of_the_same_artifact(self, store):
        # v1 is superseded by the time we look for it: an intervening rebuild
        # under the same provenance demotes it. That row is the one worth
        # finding, so the search cannot be restricted to ready.
        self._ready(store, "art-1", "prov-a", b"a1")
        self._ready(store, "art-1", "prov-a", b"a2")
        assert store.get_artifact("art-1", 1).state == "superseded"

        found = store.find_version_by_provenance("art-1", "prov-a")
        assert found is not None
        assert found.version == 2

    def test_does_not_reach_into_another_artifact(self, store):
        # find_by_provenance answers "does anyone hold this"; this answers
        # "does *this* artifact hold it", and must not confuse the two.
        self._ready(store, "art-other", "prov-a", b"a1")

        assert store.find_version_by_provenance("art-1", "prov-a") is None

    def test_promotion_makes_an_older_version_current(self, store):
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        assert store.get_latest_version("art-1").provenance_hash == "prov-b"

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert promoted.id == "art-1"
        assert promoted.version == 3
        assert promoted.provenance_hash == "prov-a"
        # What a consumer sees, which is the whole point.
        latest = store.get_latest_version("art-1")
        assert latest.version == 3
        assert latest.provenance_hash == "prov-a"
        assert store.read_blob("art-1", 3) == b"first"
        # And the row it was copied from steps aside, exactly as a rebuild
        # under the same provenance would leave it.
        assert store.get_artifact("art-1", 1).state == "superseded"

    def test_promotion_reports_failure_when_the_blob_is_gone(self, store):
        # A GC pass can take the blob while the row survives. The caller falls
        # back to recomputing, so this must say so rather than register a
        # version pointing at nothing.
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        store.blob_store.delete_blob("art-1", 1)

        assert store.promote_version("art-1", 1) is None
        assert store.get_latest_version("art-1").provenance_hash == "prov-b"
        # And nothing was registered on the way out: opening the source before
        # inserting the row is what keeps a failed promotion from stranding a
        # building version that points at no blob.
        assert store.get_artifact("art-1", 3) is None

    def test_promotion_refuses_a_version_that_was_never_ready(self, store):
        # A building row may be a partial or abandoned write and a failed one
        # was rejected on purpose. Neither is a result to make current.
        self._ready(store, "art-1", "prov-a", b"first")
        building = store.create_artifact("art-1", "prov-c")
        store.blob_store.write_blob("art-1", building, b"partial")

        assert store.promote_version("art-1", building) is None
        assert store.get_latest_version("art-1").provenance_hash == "prov-a"

    def test_lookup_does_not_cross_tenants(self, store):
        # An artifact id is not an isolation boundary, so a tenantless lookup must not
        # select a row another tenant wrote (why find_by_provenance is tenant-scoped).
        version = store.create_artifact("art-1", "prov-a", tenant="acme")
        store.blob_store.write_blob("art-1", version, b"theirs")
        store.finalize_artifact("art-1", version, "{}", 1, 6)

        assert store.find_version_by_provenance("art-1", "prov-a") is None
        assert store.find_version_by_provenance("art-1", "prov-a", tenant="acme") is not None

    def test_promotion_copies_a_blob_larger_than_one_stream_chunk(self, store):
        # The copy is streamed rather than read into memory; a payload spanning
        # several chunks is what tells a chunk-boundary bug from a working one.
        payload = bytes(range(256)) * ((BLOB_STREAM_CHUNK_BYTES * 2) // 256 + 3)
        self._ready(store, "art-1", "prov-a", payload)
        self._ready(store, "art-1", "prov-b", b"small")

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert store.read_blob("art-1", promoted.version) == payload
        assert promoted.byte_size == len(payload)

    def test_promotion_carries_requirements_minted_and_computation(self, store):
        """A promoted folder manifest requires its files, or a sweep takes them from
        under the current value; the copy stays minted and keeps its computation, but
        names no release: the promote is not part of one (audit SL-3)."""
        store.create_artifact("file-1", "prov-f", minted=True)
        store.blob_store.write_blob("file-1", 1, b"file")
        store.finalize_artifact("file-1", 1, "{}", 1, 4)
        manifest = store.create_artifact(
            "manifest",
            "prov-m",
            requires=[("file-1", 1)],
            release_id="rel-1",
            computation_key="comp-1",
        )
        store.blob_store.write_blob("manifest", manifest, b"m")
        store.finalize_artifact("manifest", manifest, "{}", 1, 1)
        self._ready(store, "manifest", "prov-m2", b"m2")

        promoted = store.promote_version("manifest", manifest)

        assert (promoted.version, promoted.release_id, promoted.computation_key) == (
            3,
            None,
            "comp-1",
        )
        conn = store._get_connection()
        try:
            edges = conn.execute(
                "SELECT from_version, to_artifact_id, to_version FROM retention_edges "
                "WHERE from_artifact_id = 'manifest' ORDER BY from_version"
            ).fetchall()
        finally:
            conn.close()
        assert [tuple(e) for e in edges] == [(1, "file-1", 1), (3, "file-1", 1)]
        assert store.requirements_met(promoted)
        # A minted id stays minted through a promotion, as through a create.
        assert store.promote_version("file-1", 1).version == 2
        conn = store._get_connection()
        try:
            minted = conn.execute(
                "SELECT minted FROM artifact_versions WHERE id = 'file-1' AND version = 2"
            ).fetchone()["minted"]
        finally:
            conn.close()
        assert minted == 1
        _idle_everything(store)
        store.garbage_collect(max_idle_days=0, keep_superseded=0)
        assert store.get_latest_version("manifest").version == 3
        assert store.get_artifact("file-1", 1) is not None

    def test_promotion_refuses_an_id_another_tenant_holds(self, store):
        version = store.create_artifact("art-1", "prov-a", tenant="acme")
        store.blob_store.write_blob("art-1", version, b"theirs")
        store.finalize_artifact("art-1", version, "{}", 1, 6)

        with pytest.raises(ArtifactIdTaken, match="another tenant"):
            store.promote_version("art-1", version, tenant="globex")
        with pytest.raises(ArtifactIdTaken, match="another tenant"):
            store.promote_version("art-1", version)
        assert store.get_artifact("art-1", 2) is None
        assert store.promote_version("art-1", version, tenant="acme").version == 2

    def test_a_refused_promotion_closes_the_source_reader(self, store, monkeypatch):
        version = store.create_artifact("art-1", "prov-a", tenant="acme")
        store.blob_store.write_blob("art-1", version, b"theirs")
        store.finalize_artifact("art-1", version, "{}", 1, 6)
        readers = []
        open_reader = store.blob_store.open_blob_reader

        def tracked(artifact_id, version):
            reader = open_reader(artifact_id, version)
            readers.append(reader)
            return reader

        monkeypatch.setattr(store.blob_store, "open_blob_reader", tracked)

        with pytest.raises(ArtifactIdTaken):
            store.promote_version("art-1", version, tenant="globex")

        # On an object store the reader is a response body; left open it holds a connection.
        assert len(readers) == 1 and readers[0].closed

    def test_promotion_keeps_the_canonical_id_when_dedup_diverts(self, store):
        # Another artifact already holds this provenance in ready state, so
        # finalize_artifact would hand back *its* id and fail ours. Callers
        # resolve by id, so the promotion has to end with our id current.
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        self._ready(store, "art-twin", "prov-a", b"first")

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert promoted.id == "art-1"
        assert store.get_latest_version("art-1").provenance_hash == "prov-a"

    def test_promotion_keeps_its_own_bytes_when_another_id_holds_the_provenance(self, store):
        """A non-deterministic cell's twin holds other bytes; the reverted cell gets its own."""
        self._ready(store, "art-1", "prov-a", b"mine-bytes")
        self._ready(store, "art-1", "prov-b", b"second")
        twin = store.create_artifact("art-twin", "prov-a")
        store.write_blob("art-twin", twin, b"twin-bytes-differ")
        store.finalize_canonical_together([StagedVersion("art-twin", twin, "{}", 1, 17, "digest")])

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert (promoted.id, promoted.state) == ("art-1", "ready")
        assert store.read_blob("art-1", promoted.version) == b"mine-bytes"
        assert promoted.byte_size == len(b"mine-bytes")
        assert store.read_blob("art-twin", twin) == b"twin-bytes-differ"


class TestRefreshSupersede:
    """Refresh rebuilds become new versions of the same artifact."""

    def test_finalize_supersedes_older_ready_version(self, store):
        """Finalizing v2 with v1's provenance demotes v1 to superseded."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 1, "{}", 10, 100)

        version = store.create_artifact("art-1", "prov-x")
        assert version == 2
        finalized = store.finalize_artifact("art-1", 2, "{}", 12, 120)

        assert finalized.version == 2
        assert finalized.state == "ready"
        assert store.get_artifact("art-1", 1).state == "superseded"

    def test_provenance_lookup_returns_rebuild(self, store):
        """After supersede, dedup lookups resolve the new version."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 2, "{}", 12, 120)

        found = store.find_by_provenance("prov-x")
        assert found is not None
        assert (found.id, found.version) == ("art-1", 2)

    def test_different_id_dedup_supersedes_the_duplicate(self, store):
        """The cross-id duplicate race keeps one ready row; the loser is overtaken, not failed.

        It finished, so it keeps its metadata and stays readable by its own id and version,
        and usage counts no failure. Its own bytes go at once: it reads the canonical's, the
        store holds them once, and retention collects it without touching them.
        """
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)

        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        result = store.finalize_artifact("art-2", 1, "{}", 10, 100)

        assert (result.id, result.version) == ("art-1", 1)
        assert store.get_artifact("art-1", 1).state == "ready"
        overtaken = store.get_artifact("art-2", 1)
        assert overtaken.state == "superseded"
        assert (overtaken.row_count, overtaken.byte_size) == (10, 100)
        assert overtaken.content_sha256 == store.get_artifact("art-1", 1).content_sha256
        assert store.read_blob("art-2", 1) == _ipc_bytes(10)
        assert not store.blob_store.blob_exists("art-2", 1)
        assert store.get_usage()["failed_versions"] == 0
        assert store.get_usage()["total_bytes"] == 100
        assert store.garbage_collect(dry_run=True)["store_bytes"] == 100
        assert store.verify_artifacts() == []

        store.set_name("canonical", "art-1", 1)
        collected = store.garbage_collect(max_idle_days=0, collect_latest=True)
        assert collected["deleted_count"] == 1
        assert collected["deleted_bytes"] == 0
        assert store.get_artifact("art-2", 1) is None
        assert store.read_blob("art-1", 1) == _ipc_bytes(10)
        assert store.verify_artifacts() == []

    @pytest.mark.parametrize("hold", ["none", "alias", "current"])
    def test_deleting_a_canonical_hands_its_bytes_to_every_version_reading_them(self, store, hold):
        """An aliased pointer, or a cell's current value, keeps resolving after the delete,
        so it needs bytes of its own; a sweep never parts a pointer from its bytes either."""
        reader = "nb_x_cell_c_var_df" if hold == "current" else "art-2"
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact(reader, "prov-x", minted=hold != "current")
        store.write_blob(reader, 1, _ipc_bytes(10))
        store.finalize_artifact(reader, 1, "{}", 10, 100)
        if hold == "alias":
            store.set_alias("model", "champion", reader, 1)

        assert store.delete_artifact("art-1", 1)

        assert store.read_blob(reader, 1) == _ipc_bytes(10)
        assert store.blob_store.blob_exists(reader, 1)
        assert store.verify_artifacts() == []
        if hold == "alias":
            assert store.resolve_alias("model", "champion").version == 1
        if hold == "current":
            assert store.get_latest_version(reader).version == 1

    def test_an_overtaken_version_whose_canonical_lost_its_blob_is_reported(self, store):
        """verify names the pointer's canonical rather than reporting an orphan blob."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)

        store.blob_store.delete_blob("art-1", 1)

        assert store.read_blob("art-2", 1) is None
        findings = {f["artifact_id"]: f for f in store.verify_artifacts()}
        assert {k: f["problem"] for k, f in findings.items()} == {
            "art-1": "missing_blob",
            "art-2": "missing_blob",
        }
        assert "art-1@v=1" in findings["art-2"]["detail"]

    def test_a_retry_of_an_overtaken_finalize_returns_the_canonical(self, store):
        """A caller that lost the first call's response gets what it returned, not an error."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        first = store.finalize_artifact("art-2", 1, "{}", 10, 100)
        assert (first.id, first.version) == ("art-1", 1)

        again = store.finalize_and_set_name("art-2", 1, "{}", 10, 100, name="model")

        assert (again.id, again.version, again.state) == ("art-1", 1, "ready")
        assert store.get_artifact("art-2", 1).state == "superseded"
        resolved = store.resolve_name("model")
        assert (resolved.id, resolved.version) == ("art-1", 1)
        # A version a refresh of its own id superseded was finalized too: it comes back
        # as it is, readable (audit SL-5).
        store.create_artifact("art-3", "prov-y")
        store.write_blob("art-3", 1, _ipc_bytes(10))
        store.finalize_artifact("art-3", 1, "{}", 10, 100)
        store.create_artifact("art-3", "prov-y")
        store.write_blob("art-3", 2, _ipc_bytes(10))
        store.finalize_artifact("art-3", 2, "{}", 10, 100)
        retried = store.finalize_artifact("art-3", 1, "{}", 10, 100)
        assert (retried.id, retried.version, retried.state) == ("art-3", 1, "superseded")
        assert store.read_blob("art-3", 1) == _ipc_bytes(10)
        assert store.get_artifact("art-3", 2).state == "ready"
        # Only a failed version is refused.
        store.create_artifact("art-4", "prov-z")
        store.fail_artifact("art-4", 1)
        with pytest.raises(ValueError, match="not in building state"):
            store.finalize_artifact("art-4", 1, "{}", 10, 100)

    def test_the_integrity_branch_returns_the_winner(self, store, monkeypatch):
        """The unique index rejects a second ready row the lookup missed: the loser is
        superseded, reads the winner's bytes, and its own blob goes."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        find = store.find_by_provenance
        calls = []

        def missing_once(*args, **kwargs):
            calls.append(True)
            return None if len(calls) == 1 else find(*args, **kwargs)

        monkeypatch.setattr(store, "find_by_provenance", missing_once)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))

        result = store.finalize_artifact("art-2", 1, "{}", 10, 100)

        assert len(calls) == 2
        assert (result.id, result.version) == ("art-1", 1)
        assert store.get_artifact("art-2", 1).state == "superseded"
        assert store.read_blob("art-2", 1) == _ipc_bytes(10)
        assert not store.blob_store.blob_exists("art-2", 1)

    @pytest.mark.parametrize("hold", ["publish", "pin"])
    def test_deleting_a_canonical_hands_its_bytes_to_a_held_version_reading_them(self, store, hold):
        """A published or pinned version is a link somebody holds; the delete must not empty it."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)
        if hold == "publish":
            store.publish_artifact("art-2", 1)
        else:
            store.retain("art-2", 1, kind="pin", holder="review")

        assert store.delete_artifact("art-1", 1)

        assert store.read_blob("art-2", 1) == _ipc_bytes(10)
        assert store.blob_store.blob_exists("art-2", 1)
        assert store.verify_artifacts() == []

    @pytest.mark.parametrize("hold", ["publish", "pin"])
    def test_deleting_a_canonical_whose_blob_is_gone_under_a_held_reader_refuses(self, store, hold):
        """With no bytes to hand over, the delete would leave the held link empty: it stops."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)
        if hold == "publish":
            store.publish_artifact("art-2", 1)
        else:
            store.retain("art-2", 1, kind="pin", holder="review")
        store.blob_store.delete_blob("art-1", 1)

        with pytest.raises(ValueError, match="whose blob is gone"):
            store.delete_artifact("art-1", 1)

        assert store.get_artifact("art-1", 1) is not None
        conn = store._get_connection()
        try:
            held = conn.execute(
                "SELECT state, superseded_by FROM artifact_versions WHERE id = ? AND version = 1",
                ("art-2",),
            ).fetchone()
        finally:
            conn.close()
        assert (held["state"], held["superseded_by"]) == ("superseded", "art-1@v=1")

    def test_finalize_and_set_name_supersedes(self, store):
        """The atomic finalize+name path supersedes the same way."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_and_set_name("art-1", 1, "{}", 10, 100, name="model")
        store.create_artifact("art-1", "prov-x")
        finalized = store.finalize_and_set_name("art-1", 2, "{}", 12, 120, name="model")

        assert finalized.version == 2
        assert store.get_artifact("art-1", 1).state == "superseded"
        resolved = store.resolve_name("model")
        assert (resolved.id, resolved.version) == ("art-1", 2)


class TestZombieSweep:
    """Stale building artifacts are demoted to failed."""

    def test_old_building_demoted(self, store):
        store.create_artifact("zombie", "prov-z")
        swept = store.sweep_zombie_builds(max_age_seconds=0)
        assert swept == 1
        assert store.get_artifact("zombie", 1).state == "failed"

    def test_recent_building_kept(self, store):
        store.create_artifact("fresh", "prov-f")
        swept = store.sweep_zombie_builds(max_age_seconds=3600)
        assert swept == 0
        assert store.get_artifact("fresh", 1).state == "building"

    def test_ready_untouched(self, store):
        store.create_artifact("done", "prov-d")
        store.finalize_artifact("done", 1, "{}", 1, 10)
        swept = store.sweep_zombie_builds(max_age_seconds=0)
        assert swept == 0
        assert store.get_artifact("done", 1).state == "ready"


class TestVerifyArtifacts:
    """Store-wide blob/metadata consistency check."""

    def _make_ready(self, store, artifact_id: str, provenance: str, rows: int) -> None:
        store.create_artifact(artifact_id, provenance)
        store.write_blob(artifact_id, 1, _ipc_bytes(rows))
        store.finalize_artifact(artifact_id, 1, "{}", rows, 100)

    def test_consistent_store_is_clean(self, store):
        self._make_ready(store, "good", "prov-g", 5)
        assert store.verify_artifacts() == []

    def test_row_count_mismatch_detected(self, store):
        store.create_artifact("short", "prov-s")
        store.write_blob("short", 1, _ipc_bytes(3))
        store.finalize_artifact("short", 1, "{}", 99, 100)  # lies about rows

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "row_count_mismatch"
        assert findings[0]["artifact_id"] == "short"

    def test_concatenated_streams_detected(self, store):
        """Concatenated IPC streams are flagged as invalid_stream."""
        store.create_artifact("concat", "prov-c")
        store.write_blob("concat", 1, _ipc_bytes(3) + _ipc_bytes(3))
        store.finalize_artifact("concat", 1, "{}", 6, 100)

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "invalid_stream"

    def test_missing_blob_detected(self, store):
        store.create_artifact("ghost", "prov-gh")
        store.finalize_artifact("ghost", 1, "{}", 1, 10)  # never wrote a blob

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "missing_blob"


class TestLegacyDefaultTenantNames:
    """Artifacts stamped with legacy '_default' stay nameable."""

    def test_tenantless_request_can_name_default_artifact(self, store):
        store.create_artifact("legacy", "prov-l", tenant="_default")
        store.finalize_artifact("legacy", 1, "{}", 1, 10)

        # Older PUT uploads stamped "_default" while the names routes resolve single-tenant
        # requests to None; that combination must not leave the artifact unnameable.
        store.set_name("legacy-name", "legacy", 1, tenant=None)  # must not raise
        resolved = store.resolve_name("legacy-name")
        assert (resolved.id, resolved.version) == ("legacy", 1)

    def test_real_tenant_mismatch_still_rejected(self, store):
        store.create_artifact("owned", "prov-o", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)

        # Answered as a missing artifact: the error must not confirm it exists, or name its tenant.
        with pytest.raises(ArtifactNotFoundError, match="not found") as excinfo:
            store.set_name("steal", "owned", 1, tenant="globex")
        assert "acme" not in str(excinfo.value)


def _idle_everything(store) -> None:
    """Every version last used long ago, so a sweep by age may take it."""
    conn = store._get_connection()
    try:
        conn.execute("UPDATE artifact_versions SET created_at = 0, last_used_at = 0")
        conn.commit()
    finally:
        conn.close()


def _make_ready_artifact(store, artifact_id: str, provenance: str) -> None:
    store.create_artifact(artifact_id, provenance)
    store.write_blob(artifact_id, 1, _ipc_bytes(1))
    store.finalize_artifact(artifact_id, 1, "{}", 1, 10)


class TestAliases:
    """Registry aliases: intent pointers on a name."""

    def test_set_and_resolve(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_alias("demo/model", "champion", "model-a", 1)
        resolved = store.resolve_alias("demo/model", "champion")
        assert (resolved.id, resolved.version) == ("model-a", 1)

    def test_many_aliases_per_name(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        _make_ready_artifact(store, "model-b", "prov-b")
        store.set_alias("demo/model", "champion", "model-a", 1)
        store.set_alias("demo/model", "candidate", "model-b", 1)

        aliases = store.list_aliases("demo/model")
        assert [(a.alias, a.artifact_id) for a in aliases] == [
            ("candidate", "model-b"),
            ("champion", "model-a"),
        ]

    def test_alias_move_keeps_old_version_reachable(self, store):
        """Promotion must not lose the old champion."""
        _make_ready_artifact(store, "model-a", "prov-a")
        _make_ready_artifact(store, "model-b", "prov-b")
        store.set_alias("demo/model", "champion", "model-a", 1)
        store.set_alias("demo/model", "champion", "model-b", 1)  # promote

        # Champion moved...
        assert store.resolve_alias("demo/model", "champion").id == "model-b"
        # ...and the audit answers "what was champion before?"
        moves = [e for e in store.read_audit(name="demo/model") if e["action"] == "alias_set"]
        assert len(moves) == 2
        latest = moves[0]
        assert latest["artifact_id"] == "model-b"
        assert latest["from_version"] == 1  # pointed at model-a v1 before

    def test_unknown_artifact_rejected(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.set_alias("demo/model", "champion", "ghost", 1)

    def test_superseded_artifact_allowed(self, store):
        _make_ready_artifact(store, "model-a", "prov-x")
        store.create_artifact("model-a", "prov-x")
        store.write_blob("model-a", 2, _ipc_bytes(1))
        store.finalize_artifact("model-a", 2, "{}", 1, 10)  # supersedes v1

        # An alias may pin the superseded version (still immutable + readable)
        store.set_alias("demo/model", "baseline", "model-a", 1)
        assert store.resolve_alias("demo/model", "baseline").version == 1

    def test_delete_alias(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_alias("demo/model", "champion", "model-a", 1)
        assert store.delete_alias("demo/model", "champion") is True
        assert store.resolve_alias("demo/model", "champion") is None
        assert store.delete_alias("demo/model", "champion") is False


class TestTags:
    """Version tags: facts about one artifact build."""

    def test_set_get_tags(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        store.set_tag("model-a", 1, "validated_by", "fangchen")
        assert store.get_tags("model-a", 1) == {"auc": "0.91", "validated_by": "fangchen"}

    def test_tag_overwrite(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        store.set_tag("model-a", 1, "auc", "0.93")
        assert store.get_tags("model-a", 1) == {"auc": "0.93"}

    def test_tag_unknown_artifact_rejected(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.set_tag("ghost", 1, "k", "v")

    def test_delete_tag(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        assert store.delete_tag("model-a", 1, "auc") is True
        assert store.get_tags("model-a", 1) == {}


class TestRegistryAudit:
    """Append-only audit of name/alias/tag mutations."""

    def test_name_moves_audited_with_history(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.set_name("demo/model", "m1", 1)
        store.set_name("demo/model", "m2", 1)  # the silent swap, recorded

        entries = store.read_audit(name="demo/model")
        assert [e["action"] for e in entries] == ["name_set", "name_set"]
        assert entries[0]["artifact_id"] == "m2"
        assert entries[0]["from_version"] == 1
        assert entries[1]["from_version"] is None  # first set had no previous

    def test_name_delete_audited(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_name("demo/model", "m1", 1)
        store.delete_name("demo/model")
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[0] == "name_delete"

    def test_finalize_promotion_audited(self, store):
        """Names set through finalize_and_set_name land in the audit too."""
        store.create_artifact("m1", "prov-f")
        store.write_blob("m1", 1, _ipc_bytes(1))
        store.finalize_and_set_name("m1", 1, "{}", 1, 10, name="auto/model")
        entries = store.read_audit(name="auto/model")
        assert entries and entries[0]["action"] == "name_set"

    def test_tag_audit_carries_key_value(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_tag("m1", 1, "auc", "0.91", actor="ci-bot")
        entry = store.read_audit(artifact_id="m1")[0]
        assert entry["action"] == "tag_set"
        assert (entry["key"], entry["value"]) == ("auc", "0.91")
        assert entry["actor"] == "ci-bot"

    def test_audit_filters_and_limit(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        for i in range(5):
            store.set_tag("m1", 1, f"k{i}", "v")
        assert len(store.read_audit(artifact_id="m1", limit=3)) == 3
        assert store.read_audit(name="unrelated") == []


class TestPendingAliasChanges:
    """Approval-gate mechanics: request, approve, reject."""

    def test_request_and_approve_set(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor="requester"
        )

        # Not applied yet
        assert store.resolve_alias("demo/model", "champion") is None
        pending = store.list_pending_changes()
        assert len(pending) == 1 and pending[0]["action"] == "set"

        applied = store.approve_alias_change("demo/model", "champion", actor="approver")
        assert applied["artifact_id"] == "m1"
        assert store.resolve_alias("demo/model", "champion").id == "m1"
        assert store.list_pending_changes() == []

        # Audit trail: request -> approved -> the applied alias_set
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[:3] == ["alias_set", "alias_approved", "alias_request_set"]
        approved = next(
            e for e in store.read_audit(name="demo/model") if e["action"] == "alias_approved"
        )
        assert approved["actor"] == "approver"

    def test_reject_discards(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        rejected = store.reject_alias_change("demo/model", "champion", actor="reviewer")
        assert rejected["artifact_id"] == "m1"
        assert store.resolve_alias("demo/model", "champion") is None
        assert store.list_pending_changes() == []
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[0] == "alias_rejected"

    def test_request_delete_then_approve(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_alias("demo/model", "champion", "m1", 1)
        store.request_alias_change("demo/model", "champion", "delete")
        # still resolvable until approved
        assert store.resolve_alias("demo/model", "champion") is not None
        store.approve_alias_change("demo/model", "champion")
        assert store.resolve_alias("demo/model", "champion") is None

    def test_new_request_replaces_previous(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m2", version=1)
        pending = store.list_pending_changes()
        assert len(pending) == 1
        assert pending[0]["artifact_id"] == "m2"

    def test_approve_without_pending_raises(self, store):
        with pytest.raises(ValueError, match="No pending change"):
            store.approve_alias_change("demo/model", "champion")

    def test_request_set_validates_artifact(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.request_alias_change(
                "demo/model", "champion", "set", artifact_id="ghost", version=1
            )


class TestAliasedArtifactProtection:
    """GC and delete respect alias pointers."""

    def test_gc_spares_aliased_artifact(self, store):
        _make_ready_artifact(store, "pinned", "prov-p")
        store.set_alias("demo/model", "champion", "pinned", 1)

        result = store.garbage_collect(max_idle_days=0)
        assert store.get_artifact("pinned", 1) is not None, result
        assert store.resolve_alias("demo/model", "champion") is not None

    def test_gc_spares_aliased_superseded_version(self, store):
        """An alias pinning a superseded version keeps it from GC."""
        _make_ready_artifact(store, "model", "prov-s")
        store.create_artifact("model", "prov-s")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)  # supersedes v1
        store.set_alias("demo/model", "champion", "model", 1)
        store.set_name("demo/model", "model", 2)  # name guards v2

        store.garbage_collect(max_idle_days=0)

        champion = store.resolve_alias("demo/model", "champion")
        assert champion is not None and champion.version == 1

    def test_gc_still_collects_unreferenced(self, store):
        """A lone unnamed artifact is a current value; reclaiming it takes the explicit opt-in."""
        _make_ready_artifact(store, "loose", "prov-l")
        result = store.garbage_collect(max_idle_days=0, collect_latest=True)
        assert result["deleted_count"] == 1
        assert store.get_artifact("loose", 1) is None

    def test_delete_artifact_cleans_aliases_and_tags(self, store):
        _make_ready_artifact(store, "doomed", "prov-d")
        store.set_alias("demo/model", "candidate", "doomed", 1)
        store.set_tag("doomed", 1, "auc", "0.5")

        assert store.delete_artifact("doomed", 1) is True
        assert store.resolve_alias("demo/model", "candidate") is None
        assert store.get_tags("doomed", 1) == {}
        # The forced alias removal is auditable
        entries = store.read_audit(name="demo/model")
        assert entries[0]["action"] == "alias_delete"


class TestApproveAtomicity:
    """approve_alias_change is one transaction."""

    def test_dead_target_keeps_pending_intact(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        store.delete_artifact("m1", 1)  # target dies between request and approve

        with pytest.raises(ValueError, match="no longer available"):
            store.approve_alias_change("demo/model", "champion")

        # The pending entry survives for an explicit reject
        assert len(store.list_pending_changes()) == 1
        # And no phantom approval landed in the audit
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert "alias_approved" not in actions

    def test_approve_applies_and_audits_in_order(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor="req"
        )
        store.approve_alias_change("demo/model", "champion", actor="approver")

        assert store.resolve_alias("demo/model", "champion").id == "m1"
        entries = store.read_audit(name="demo/model")
        assert [e["action"] for e in entries] == [
            "alias_set",
            "alias_approved",
            "alias_request_set",
        ]
        assert entries[0]["actor"] == "approver"


def _no_wait_store(store: ArtifactStore) -> ArtifactStore:
    """A second store on the same database, as another process opens one, that never waits."""
    dialect = store.dialect
    if dialect.name == "sqlite":
        other: SqlDialect = SqliteDialect(store.db_path)
        no_wait = "PRAGMA busy_timeout = 0"
    else:
        other = type(dialect)(dialect.dsn)  # type: ignore[attr-defined]
        no_wait = "SET lock_timeout = '1ms'"
    competitor = ArtifactStore(store.artifact_dir, dialect=other)
    connect = competitor._get_connection

    def connect_without_waiting():
        conn = connect()
        conn.execute(no_wait)
        return conn

    competitor._get_connection = connect_without_waiting
    return competitor


def race_after_pending_read(store, competitor, first, second) -> tuple[object, object]:
    """Run ``first(store)``, landing ``second(competitor)`` right after it reads the pending row.

    ``competitor`` is a second store whose connections do not wait for a lock. One that finds it
    held runs ``second`` again once ``first`` is done, as a second process waiting on it would.
    Returns each call's result, or the ``ValueError`` it raised.
    """
    outcomes: dict[str, object] = {}
    state = {"read": False, "locked": False}

    def call(key, fn, target):
        try:
            outcomes[key] = fn(target)
        except ValueError as exc:
            outcomes[key] = exc

    class Interleaved:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, attr):
            return getattr(self._inner, attr)

        def execute(self, sql, params=()):
            if state["read"] and not state["locked"] and "second" not in outcomes:
                try:
                    call("second", second, competitor)
                except competitor.dialect.operational_error:
                    state["locked"] = True
            cursor = self._inner.execute(sql, params)
            if "FROM registry_pending" in sql:
                state["read"] = True
            return cursor

    connect = store._get_connection
    store._get_connection = lambda: Interleaved(connect())
    try:
        call("first", first, store)
    finally:
        store._get_connection = connect
    if state["locked"]:
        call("second", second, competitor)
    return outcomes["first"], outcomes["second"]


_DECISIONS = {
    "approve": lambda s: s.approve_alias_change("demo/model", "champion"),
    "reject": lambda s: s.reject_alias_change("demo/model", "champion"),
}


class TestPendingChangeRaces:
    """Reading a pending change and consuming it is one step, across connections and nodes."""

    @pytest.fixture
    def store(self, backend_store):
        return backend_store

    @pytest.fixture
    def competitor(self, store):
        competitor = _no_wait_store(store)
        yield competitor
        competitor.dialect.close()

    @pytest.mark.parametrize(("first", "second"), [("approve", "reject"), ("reject", "approve")])
    def test_an_approve_and_a_reject_of_one_change_have_one_winner(
        self, store, competitor, first, second
    ):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)

        won, lost = race_after_pending_read(
            store, competitor, _DECISIONS[first], _DECISIONS[second]
        )

        assert won["artifact_id"] == "m1"
        assert isinstance(lost, ValueError) and "No pending change" in str(lost)
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert ("alias_approved" in actions, "alias_rejected" in actions) == (
            first == "approve",
            first == "reject",
        )
        assert (store.resolve_alias("demo/model", "champion") is not None) == (first == "approve")

    def test_an_approval_leaves_a_newer_request_pending(self, store, competitor):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)

        applied, requested = race_after_pending_read(
            store,
            competitor,
            _DECISIONS["approve"],
            lambda s: s.request_alias_change(
                "demo/model", "champion", "set", artifact_id="m2", version=1
            ),
        )

        assert applied["artifact_id"] == "m1" and requested is True
        assert store.resolve_alias("demo/model", "champion").id == "m1"
        audit = store.read_audit(name="demo/model")
        assert [e["artifact_id"] for e in audit if e["action"] == "alias_approved"] == ["m1"]
        assert [p["artifact_id"] for p in store.list_pending_changes()] == ["m2"]


class TestCreateArtifactVersionRace:
    """Concurrent creates for one artifact id allocate distinct versions."""

    def test_threaded_creates_never_collide(self, store):
        _make_ready_artifact(store, "contended", "prov-base")
        errors: list[Exception] = []
        versions: list[int] = []
        barrier = threading.Barrier(4)

        def create(n):
            try:
                barrier.wait()
                versions.append(store.create_artifact("contended", f"prov-{n}"))
            except Exception as e:  # noqa: BLE001 (collecting for assertion)
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []
        assert sorted(versions) == [2, 3, 4, 5]


class TestAliasIdempotence:
    """Identical-target alias writes are no-ops."""

    def test_set_alias_same_target_is_noop(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        assert store.set_alias("demo/model", "champion", "m1", 1) is True
        before = len(store.read_audit(name="demo/model"))
        assert store.set_alias("demo/model", "champion", "m1", 1) is False
        assert len(store.read_audit(name="demo/model")) == before  # no audit spam

    def test_request_for_live_pointer_is_noop(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_alias("demo/model", "champion", "m1", 1)
        assert (
            store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
            is False
        )
        assert store.list_pending_changes() == []

    def test_request_for_new_target_still_queues(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.set_alias("demo/model", "champion", "m1", 1)
        assert (
            store.request_alias_change("demo/model", "champion", "set", artifact_id="m2", version=1)
            is True
        )
        assert len(store.list_pending_changes()) == 1


class TestAuditTenantScoping:
    """read_audit isolates by tenant for request-serving callers."""

    def _seed(self, store, tenant, name):
        store.create_artifact(f"{tenant}-a", f"prov-{tenant}", tenant=tenant)
        store.write_blob(f"{tenant}-a", 1, _ipc_bytes(1))
        store.finalize_artifact(f"{tenant}-a", 1, "{}", 1, 10)
        store.set_name(name, f"{tenant}-a", 1, tenant=tenant)

    def test_tenant_filter_isolates_history(self, store):
        self._seed(store, "acme", "acme/model")
        self._seed(store, "globex", "globex/model")

        acme = store.read_audit(tenant="acme")
        assert acme, "expected acme audit rows"
        assert all(e["tenant"] == "acme" for e in acme)
        assert not any(e["name"] == "globex/model" for e in acme)

        globex = store.read_audit(tenant="globex")
        assert all(e["tenant"] == "globex" for e in globex)

    def test_default_sentinel_returns_all_tenants(self, store):
        """No tenant arg gives the whole-store view (CLI / admin)."""
        self._seed(store, "acme", "acme/model")
        self._seed(store, "globex", "globex/model")
        tenants = {e["tenant"] for e in store.read_audit()}
        assert {"acme", "globex"} <= tenants

    def test_none_tenant_filters_to_default(self, store):
        """Explicit tenant=None scopes to the '' default tenant, not all."""
        _make_ready_artifact(store, "m1", "prov-1")  # tenant None -> ''
        store.set_name("default/model", "m1", 1)
        self._seed(store, "acme", "acme/model")

        default_rows = store.read_audit(tenant=None)
        assert default_rows
        assert all(e["tenant"] in (None, "") for e in default_rows)
        assert not any(e["name"] == "acme/model" for e in default_rows)


class TestApproveSeparationOfDuty:
    """approve_alias_change can forbid self-approval."""

    def _pending(self, store, requester):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor=requester
        )

    def test_self_approve_blocked_when_required(self, store):
        self._pending(store, "alice")
        with pytest.raises(ValueError, match="Separation of duty"):
            store.approve_alias_change(
                "demo/model", "champion", actor="alice", require_distinct_approver=True
            )
        # Pending survives the rejected self-approval
        assert len(store.list_pending_changes()) == 1
        assert store.resolve_alias("demo/model", "champion") is None

    def test_distinct_approver_allowed(self, store):
        self._pending(store, "alice")
        store.approve_alias_change(
            "demo/model", "champion", actor="bob", require_distinct_approver=True
        )
        assert store.resolve_alias("demo/model", "champion").id == "m1"

    def test_self_approve_allowed_when_not_required(self, store):
        """Break-glass / personal mode: separation of duty off."""
        self._pending(store, "alice")
        store.approve_alias_change(
            "demo/model", "champion", actor="alice", require_distinct_approver=False
        )
        assert store.resolve_alias("demo/model", "champion").id == "m1"


class TestTagAndNameLookups:
    """Reverse lookups for the per-cell published-artifacts route: artifacts a cell published
    (``nb_cell`` tag) and the names pointing at a version.
    """

    def test_list_artifacts_by_tag(self, store):
        for aid, h in (("a1", "h1"), ("a2", "h2"), ("a3", "h3")):
            store.create_artifact(aid, h)
            store.finalize_artifact(aid, 1, "{}", 0, 0)
        store.set_tag("a1", 1, "nb_cell", "cellX")
        store.set_tag("a3", 1, "nb_cell", "cellX")
        store.set_tag("a2", 1, "nb_cell", "cellY")

        assert store.list_artifacts_by_tag("nb_cell", "cellX") == [("a1", 1), ("a3", 1)]
        assert store.list_artifacts_by_tag("nb_cell", "cellY") == [("a2", 1)]
        assert store.list_artifacts_by_tag("nb_cell", "absent") == []


class TestListArtifactsSortAndSince:
    """The since / sort / order filters behind the artifacts dashboard."""

    @staticmethod
    def _add(store, aid, byte_size):
        store.create_artifact(aid, f"{aid}-hash")
        store.finalize_artifact(aid, 1, "{}", 0, byte_size)

    def test_sort_by_byte_size(self, store):
        self._add(store, "small", 100)
        self._add(store, "big", 300)
        self._add(store, "mid", 200)
        asc = [a.id for a in store.list_artifacts(sort="byte_size", order="asc")]
        assert asc == ["small", "mid", "big"]
        desc = [a.id for a in store.list_artifacts(sort="byte_size", order="desc")]
        assert desc == ["big", "mid", "small"]

    def test_unknown_sort_falls_back_safely(self, store):
        # A non-whitelisted sort (or an injection attempt) falls back to
        # created_at rather than reaching the SQL.
        self._add(store, "a", 1)
        result = store.list_artifacts(sort="; DROP TABLE artifact_versions;--")
        assert [a.id for a in result] == ["a"]

    def test_since_filter(self, store):
        self._add(store, "a", 1)
        assert len(store.list_artifacts(since=0)) == 1
        assert store.list_artifacts(since=time.time() + 3600) == []

    def test_names_for_artifact(self, store):
        store.create_artifact("a1", "h1")
        store.finalize_artifact("a1", 1, "{}", 0, 0)
        store.set_name("team/model", "a1", 1)

        assert store.names_for_artifact("a1", 1) == ["team/model"]
        assert store.names_for_artifact("a1", 2) == []
        assert store.names_for_artifact("nope", 1) == []


def _as_legacy_nullable_tenant_table(conn) -> None:
    """Rebuild ``artifact_versions`` in the pre-tenant shape: a nullable ``tenant`` and no
    uniqueness index, as stores written before tenants were normalized have."""
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_tenant_provenance_unique;
        ALTER TABLE artifact_versions RENAME TO _av_old;
        CREATE TABLE artifact_versions (
            id TEXT NOT NULL, version INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'building', provenance_hash TEXT NOT NULL,
            schema_json TEXT, row_count INTEGER, byte_size INTEGER,
            created_at REAL NOT NULL, transform_spec TEXT, input_versions TEXT,
            tenant TEXT, principal TEXT, PRIMARY KEY (id, version)
        );
        -- Columns named, not ``SELECT *``: this table is deliberately the
        -- legacy shape, so every column added to the current schema would
        -- otherwise break the copy and this test would fail on each new
        -- migration for a reason that has nothing to do with tenants.
        INSERT INTO artifact_versions
            (id, version, state, provenance_hash, schema_json, row_count,
             byte_size, created_at, transform_spec, input_versions,
             tenant, principal)
        SELECT id, version, state, provenance_hash, schema_json, row_count,
               byte_size, created_at, transform_spec, input_versions,
               tenant, principal
        FROM _av_old;
        DROP TABLE _av_old;
        """
    )


class TestTenantNormalization:
    """Tenant is stored as '' (never NULL) for tenantless rows, and the isolation and uniqueness
    guarantees that depend on it.
    """

    def test_tenantless_provenance_uniqueness_enforced(self, store):
        """Two tenantless artifacts with one provenance cannot both be ready; NULL would be
        distinct.
        """
        store.create_artifact("art-1", "dup-prov")
        store.finalize_artifact("art-1", 1, "{}", 1, 10)
        store.create_artifact("art-2", "dup-prov")
        result = store.finalize_artifact("art-2", 1, "{}", 1, 10)
        assert result.id == "art-1"  # deduped to the first

        conn = sqlite3.connect(str(store.db_path))
        ready = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions "
            "WHERE provenance_hash = 'dup-prov' AND state = 'ready'"
        ).fetchone()[0]
        conn.close()
        assert ready == 1

    def test_tenantless_finalize_does_not_dedup_cross_tenant(self, store):
        """A tenantless finalize must not dedup against a tenant's artifact and name it."""
        store.create_artifact("team-art", "shared-prov", tenant="team-a")
        store.finalize_artifact("team-art", 1, "{}", 1, 10)

        store.create_artifact("none-art", "shared-prov")  # tenantless
        result = store.finalize_and_set_name(
            "none-art", 1, "{}", 1, 10, name="my-name", tenant=None
        )
        # It finalizes as its own tenantless artifact, NOT team-a's.
        assert result.id == "none-art"
        resolved = store.resolve_name("my-name")
        assert resolved.id == "none-art"

    @staticmethod
    def _staged_over_a_ready_equal(store, provenance):
        """``canonical`` staged with the provenance ``equiv`` already holds ready, in team-a."""
        store.create_artifact("equiv", provenance, tenant="team-a")
        store.write_blob("equiv", 1, b"x" * 10)
        store.finalize_artifact("equiv", 1, "{}", 1, 10)
        store.create_artifact("canonical", provenance, tenant="team-a")
        store.write_blob("canonical", 1, b"x" * 10)
        return StagedVersion("canonical", 1, "{}", 1, 10, "digest")

    def test_canonical_finalize_retries_past_a_losing_race(self, store):
        """A competing writer can commit our provenance between our supersede and promote; the retry
        must still promote.

        SQLite's BEGIN IMMEDIATE prevents this in production, so the conflict is injected; the
        retry loop is dialect-agnostic.
        """
        staged = self._staged_over_a_ready_equal(store, "race-prov")

        real_get_connection = store._get_connection
        # Counting commits, not connections: the trailing get_artifact opens
        # one more connection but never commits.
        commits = []

        class _FailsCommitOnce:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def commit(self):
                commits.append(1)
                if len(commits) == 1:
                    raise sqlite3.IntegrityError("injected UNIQUE constraint failure")
                return self._inner.commit()

        store._get_connection = lambda: _FailsCommitOnce(real_get_connection())
        try:
            [promoted] = store.finalize_canonical_together([staged])
        finally:
            store._get_connection = real_get_connection

        assert len(commits) == 2, "expected exactly one retry after the conflict"
        assert promoted is not None
        assert promoted.id == "canonical", "must never hand back the foreign winner"
        assert promoted.state == "ready"

    def test_canonical_finalize_surfaces_a_persistent_conflict(self, store):
        """Sustained contention must surface, not report a promotion that did not happen."""
        staged = self._staged_over_a_ready_equal(store, "race-prov")

        real_get_connection = store._get_connection

        class _AlwaysFailsCommit:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def commit(self):
                raise sqlite3.IntegrityError("injected UNIQUE constraint failure")

        store._get_connection = lambda: _AlwaysFailsCommit(real_get_connection())
        try:
            with pytest.raises(sqlite3.IntegrityError):
                store.finalize_canonical_together([staged])
        finally:
            store._get_connection = real_get_connection

    def test_canonical_finalize_supersedes_conflicting_ready_row(self, store):
        """One ready row per (tenant, provenance) afterwards, for tenant-scoped rows too."""
        staged = self._staged_over_a_ready_equal(store, "canon-prov")

        [promoted] = store.finalize_canonical_together([staged])
        assert promoted is not None and promoted.state == "ready"

        conn = sqlite3.connect(str(store.db_path))
        ready_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM artifact_versions "
                "WHERE provenance_hash = 'canon-prov' AND state = 'ready'"
            ).fetchall()
        ]
        conn.close()
        assert ready_ids == ["canonical"]  # exactly one ready, the canonical
        # The equivalent is still fetchable by id+version.
        assert store.get_artifact("equiv", 1).state == "superseded"

    def test_set_tag_rejects_cross_tenant(self, store):
        """set_tag enforces ownership like set_alias."""
        store.create_artifact("owned", "p", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)
        with pytest.raises(ArtifactNotFoundError, match="not found") as excinfo:
            store.set_tag("owned", 1, "k", "v", tenant="globex")
        assert "acme" not in str(excinfo.value)

    def test_set_tag_rejects_non_readable(self, store):
        """Tags are only allowed on readable (ready/superseded) artifacts."""
        store.create_artifact("building", "p")  # still 'building'
        with pytest.raises(ValueError, match="not readable"):
            store.set_tag("building", 1, "k", "v")

    def test_delete_artifact_refuses_other_tenant(self, store):
        """delete_artifact refuses to delete another tenant's artifact."""
        store.create_artifact("owned", "p", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)
        assert store.delete_artifact("owned", 1, tenant="globex") is False
        assert store.get_artifact("owned", 1) is not None  # not deleted
        assert store.delete_artifact("owned", 1, tenant="acme") is True

    def test_migration_normalizes_null_tenant_and_dedups(self, artifact_dir):
        """A legacy store with NULL tenants and a duplicate ready row normalizes to '' and collapses
        the duplicate so the uniqueness index holds.
        """
        store = ArtifactStore(artifact_dir)
        store.create_artifact("a1", "h")
        store.finalize_artifact("a1", 1, "{}", 1, 10)

        # Rebuild artifact_versions as the legacy nullable-tenant table, then plant NULL
        # tenants and a second ready row with the same provenance (the duplicate the old
        # NULL-distinct index permitted).
        conn = sqlite3.connect(str(store.db_path))
        _as_legacy_nullable_tenant_table(conn)
        conn.execute("UPDATE artifact_versions SET tenant = NULL")
        conn.execute(
            "INSERT INTO artifact_versions "
            "(id, version, state, provenance_hash, created_at, tenant) "
            "VALUES ('a2', 1, 'ready', 'h', 1.0, NULL)"
        )
        conn.commit()
        conn.close()

        # Reopen -> _init_schema migration runs.
        ArtifactStore(artifact_dir)

        conn = sqlite3.connect(str(store.db_path))
        n_null = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions WHERE tenant IS NULL"
        ).fetchone()[0]
        n_ready = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions WHERE provenance_hash = 'h' AND state = 'ready'"
        ).fetchone()[0]
        conn.close()
        assert n_null == 0  # all normalized to ''
        assert n_ready == 1  # duplicate collapsed

    def test_a_tenant_migration_that_fails_part_way_keeps_the_old_index(self, artifact_dir):
        """DROP INDEX used to commit on its own, so a failure later in the rebuild left the
        store with no uniqueness index at all (and two upgrading processes could interleave)."""
        store = ArtifactStore(artifact_dir)
        store.create_artifact("a1", "h")
        store.finalize_artifact("a1", 1, "{}", 1, 10)
        conn = sqlite3.connect(str(store.db_path))
        _as_legacy_nullable_tenant_table(conn)
        conn.executescript(
            """
            CREATE UNIQUE INDEX idx_tenant_provenance_unique
                ON artifact_versions(tenant, provenance_hash) WHERE state = 'ready';
            UPDATE artifact_versions SET tenant = NULL;
            INSERT INTO artifact_versions (id, version, state, provenance_hash, created_at, tenant)
                VALUES ('a2', 1, 'ready', 'h', 1.0, NULL);
            -- The rebuild supersedes the duplicate ready row; fail it there.
            CREATE TRIGGER fail_the_rebuild BEFORE UPDATE OF state ON artifact_versions
                WHEN NEW.state = 'superseded'
                BEGIN SELECT RAISE(ABORT, 'rebuild failed'); END;
            """
        )
        conn.commit()
        conn.close()

        with pytest.raises(sqlite3.DatabaseError, match="rebuild failed"):
            ArtifactStore(artifact_dir)

        conn = sqlite3.connect(str(store.db_path))
        index = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_tenant_provenance_unique'"
        ).fetchone()
        nulls = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions WHERE tenant IS NULL"
        ).fetchone()[0]
        conn.close()
        assert index is not None
        assert nulls == 2  # nothing of the half-done rebuild was kept


class TestGcSparesCurrentValues:
    """GC must not delete the current value of an artifact id.

    Notebook cell outputs (``nb_{notebook}_cell_{cell}_var_{name}``) are never named, so
    ``get_latest_version`` is their only handle; treating unnamed as garbage would delete every
    notebook's state after the GC cutoff.
    """

    def test_gc_spares_an_unnamed_notebook_cell_artifact(self, store):
        _make_ready_artifact(store, "nb_abc_cell_def_var_model", "prov-nb")

        result = store.garbage_collect(max_idle_days=0)

        assert result["deleted_count"] == 0
        assert store.get_latest_version("nb_abc_cell_def_var_model") is not None

    def test_gc_still_collects_superseded_versions(self, store):
        """Older versions are still collected."""
        _make_ready_artifact(store, "model", "prov-v1")
        store.create_artifact("model", "prov-v2")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)  # supersedes v1

        result = store.garbage_collect(max_idle_days=0)

        assert result["deleted_count"] == 1
        assert store.get_artifact("model", 1) is None
        assert store.get_artifact("model", 2) is not None  # current value kept

    def test_gc_spares_the_current_value_while_a_rebuild_is_in_flight(self, store):
        """A rebuild's ``building`` row is the highest version, so "spare MAX(version)" would leave
        the readable value unprotected during the build and after it fails.
        """
        artifact_id = "nb_x_cell_c1_var_df"
        _make_ready_artifact(store, artifact_id, "prov-v1")
        rebuild = store.create_artifact(artifact_id, "prov-v2")

        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.get_latest_version(artifact_id).version == 1

        store.fail_artifact(artifact_id, rebuild)
        store.garbage_collect(max_idle_days=0)
        assert store.get_latest_version(artifact_id).version == 1

    def test_collect_latest_opt_in_still_reclaims(self, store):
        _make_ready_artifact(store, "loose", "prov-loose")
        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.garbage_collect(max_idle_days=0, collect_latest=True)["deleted_count"] == 1


class TestGcDeletesMetadataBeforeBlobs:
    """Metadata is deleted and committed before best-effort blob removal, so a mid-GC crash leaves
    orphaned bytes, not ready rows without blobs.
    """

    def test_blob_failure_does_not_abort_gc_or_strand_ready_rows(self, store, monkeypatch):
        _make_ready_artifact(store, "model", "prov-v1")
        store.create_artifact("model", "prov-v2")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)

        def boom(artifact_id, version):
            raise OSError("backend unavailable")

        monkeypatch.setattr(store.blob_store, "delete_blob", boom)

        result = store.garbage_collect(max_idle_days=0)

        # The run completed and the metadata is gone: no ready row survives pointing at a
        # blob we may or may not have removed.
        assert result["deleted_count"] == 1
        assert store.get_artifact("model", 1) is None


class TestTenantlessArtifactsAreVisibleToScopedQueries:
    """Tenantless rows are stored as ``''``, so tenant filters must match ``''``, not only NULL.

    Otherwise dedup still resolves these rows while listing, usage, stats, verify and GC cannot see
    them: a permanent leak.
    """

    def _seed(self, store):
        _make_ready_artifact(store, "legacy", "prov-legacy")  # tenantless ('')
        version = store.create_artifact("scoped", "prov-scoped", tenant="team-a")
        store.write_blob("scoped", version, _ipc_bytes(1))
        store.finalize_artifact("scoped", version, "{}", 1, 10)

    def test_tenantless_rows_are_stored_as_empty_string_not_null(self, store):
        self._seed(store)
        artifact = store.get_artifact("legacy", 1)
        assert artifact.tenant == "", "schema stores '' for tenantless rows"

    def test_list_artifacts_includes_legacy_rows(self, store):
        self._seed(store)
        ids = {a.id for a in store.list_artifacts(tenant="team-a")}
        assert "scoped" in ids
        assert "legacy" in ids, "legacy tenantless artifact was invisible to the tenant"

    def test_usage_and_stats_count_legacy_rows(self, store):
        self._seed(store)
        assert store.get_usage(tenant="team-a")["total_versions"] >= 2
        assert store.stats(tenant="team-a")["total_versions"] >= 2

    def test_verify_sees_legacy_rows(self, store):
        self._seed(store)
        store.blob_store.write_blob("legacy", 1, b"not an arrow stream")

        # verify walks the same filter; it must not skip legacy rows.
        [finding] = store.verify_artifacts(tenant="team-a")

        assert (finding["artifact_id"], finding["problem"]) == ("legacy", "invalid_stream")

    def test_gc_can_finally_reach_legacy_rows(self, store):
        """Asserts reachability rather than an exact count, so the GC policy is not encoded here."""
        self._seed(store)
        store.create_artifact("legacy", "prov-legacy-v2")  # supersede v1
        store.write_blob("legacy", 2, _ipc_bytes(1))
        store.finalize_artifact("legacy", 2, "{}", 1, 10)

        store.garbage_collect(max_idle_days=0, tenant="team-a", include_tenantless=True)

        assert store.get_artifact("legacy", 1) is None, (
            "tenant-scoped GC still cannot reach a tenantless artifact"
        )


class TestAnIdTwoComputationsClaim:
    """``import_artifact`` keeps the given id, and notebook ids derive from notebook and cell ids,
    so two people from one repo can send the same id for different cells.
    """

    @staticmethod
    def _record(provenance: str):
        return ArtifactVersion(
            id="nb_shared_cell_c1_var_rows",
            version=1,
            state="ready",
            provenance_hash=provenance,
            schema_json="",
            row_count=0,
            byte_size=5,
            created_at=1.0,
            transform_spec=json.dumps({"executor": "notebook/cell@v1", "params": {}}),
            input_versions="{}",
            principal=None,
            tenant=None,
            content_sha256=None,
        )

    def test_a_repeated_import_of_the_same_computation_is_a_no_op(self, store):
        store.import_artifact(self._record("a" * 64), b"ALICE")

        again = store.import_artifact(self._record("a" * 64), b"ALICE")

        assert again.written is False

    def test_a_different_computation_under_that_id_is_refused(self, store):
        """Keeping the existing row would leave the caller's name, tags and descendants on someone
        else's bytes.
        """
        store.import_artifact(self._record("a" * 64), b"ALICE")

        with pytest.raises(ArtifactImportConflict):
            store.import_artifact(self._record("b" * 64), b"BOBBY")

        assert store.read_blob("nb_shared_cell_c1_var_rows", 1) == b"ALICE"


class TestAnIdIsHeldByOneTenant:
    """A version under another tenant's id becomes that id's latest, which its notebook reads.

    Refused inside the write, since a route's earlier look can go stale before the insert.
    """

    @pytest.mark.parametrize("holder,caller", [("team-b", "team-a"), ("team-b", None), (None, "a")])
    def test_create_refuses_an_id_another_tenant_holds(self, store, holder, caller):
        store.create_artifact("nb_x", "a" * 64, tenant=holder)

        with pytest.raises(ArtifactIdTaken):
            store.create_artifact("nb_x", "b" * 64, tenant=caller)

        assert store.get_artifact("nb_x", 2) is None

    @pytest.mark.parametrize("tenant,again", [("team-a", "team-a"), (None, ""), ("", None)])
    def test_create_appends_to_its_own_tenants_id(self, store, tenant, again):
        store.create_artifact("nb_x", "a" * 64, tenant=tenant)

        assert store.create_artifact("nb_x", "b" * 64, tenant=again) == 2

    @pytest.mark.parametrize("version", [1, 2])
    def test_import_refuses_an_id_another_tenant_holds(self, store, version):
        """Version 1 is the same computation, which used to resolve onto the other tenant's row."""
        held = store.create_artifact("nb_x", "a" * 64, tenant="team-b")
        store.write_blob("nb_x", held, b"TEAMB")
        store.finalize_artifact("nb_x", held, "", row_count=None, byte_size=5)
        record = ArtifactVersion(
            id="nb_x",
            version=version,
            state="ready",
            provenance_hash="a" * 64,
            schema_json="",
            row_count=None,
            byte_size=5,
            created_at=1.0,
            transform_spec=None,
            input_versions=json.dumps({"vestigo://artifact/up@v=1": "up@v=1"}),
            principal=None,
            tenant="team-a",
            content_sha256=None,
        )

        with pytest.raises(ArtifactImportConflict):
            store.import_artifact(record, b"TEAMA")

        assert store.get_artifact("nb_x", 2) is None
        assert store.read_blob("nb_x", 1) == b"TEAMB"
        assert store.get_artifact("nb_x", 1).input_versions is None


class TestTwoIdsOneComputation:
    """A duplicated notebook yields one provenance under two ids.

    Finalizing the second id's run supersedes the first id's row, which keeps its bytes and stays
    that id's current value for its downstream cells.
    """

    A = "nb_A_cell_c1_var_df"
    B = "nb_B_cell_c1_var_df"

    def _both_finalized(self, store):

        _make_ready_artifact(store, self.A, "same-prov")
        vb = store.create_artifact(self.B, "same-prov")
        data = _ipc_bytes(1)
        store.write_blob(self.B, vb, data)
        digest = hashlib.sha256(data).hexdigest()
        # What finalize_cell_outputs does with a cell run's outputs.
        store.finalize_canonical_together([StagedVersion(self.B, vb, "{}", 1, len(data), digest)])
        assert store.get_artifact(self.A, 1).state == "superseded"

    def test_promoting_one_id_leaves_the_other_its_value(self, store):
        self._both_finalized(store)

        a = store.get_latest_version(self.A)
        assert a is not None and a.version == 1
        assert store.read_blob(self.A, a.version) is not None
        assert store.get_latest_version(self.B) is not None
        assert [v.id for v in store.list_latest_by_id_prefix("nb_A_")] == [self.A]

    def test_the_promoted_id_owns_its_bytes(self, store):
        """A is now the superseded one and may be collected; B must not read through it."""
        self._both_finalized(store)

        assert store.blob_store.blob_exists(self.B, 1)
        assert store.delete_artifact(self.A, 1)
        assert store.read_blob(self.B, 1) == _ipc_bytes(1)
        assert store.verify_artifacts() == []

    def test_gc_spares_a_superseded_current_value_during_a_rebuild(self, store):
        self._both_finalized(store)
        store.create_artifact(self.A, "prov-a-edited")  # A reruns after an edit

        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.get_latest_version(self.A).version == 1


class TestFinalizeTogether:
    """``finalize_canonical_together``: one cell run's outputs become current all at once."""

    @staticmethod
    def _staged(store, artifact_id: str, provenance: str):
        version = store.create_artifact(artifact_id, provenance)
        store.write_blob(artifact_id, version, b"x")
        return StagedVersion(artifact_id, version, "{}", 1, 1, "digest")

    def test_each_becomes_the_canonical_row_under_its_own_id(self, store):
        _make_ready_artifact(store, "nb_A_cell_c_var_x", "prov-x")  # another id, same result
        _make_ready_artifact(store, "nb_B_cell_c_var_y", "prov-y")  # this id's earlier run
        staged = [
            self._staged(store, "nb_B_cell_c_var_x", "prov-x"),
            self._staged(store, "nb_B_cell_c_var_y", "prov-y"),
        ]

        finalized = store.finalize_canonical_together(staged)

        assert [(v.id, v.version, v.state) for v in finalized] == [
            ("nb_B_cell_c_var_x", 1, "ready"),
            ("nb_B_cell_c_var_y", 2, "ready"),
        ]
        assert store.get_artifact("nb_A_cell_c_var_x", 1).state == "superseded"
        assert store.get_artifact("nb_B_cell_c_var_y", 1).state == "superseded"

    def test_one_that_cannot_be_finalized_finalizes_none(self, store):
        first = self._staged(store, "nb_B_cell_c_var_x", "prov-x")
        second = self._staged(store, "nb_B_cell_c_var_y", "prov-y")
        store.fail_artifact(second.artifact_id, second.version)

        with pytest.raises(ValueError, match="building"):
            store.finalize_canonical_together([first, second])

        assert store.get_artifact(first.artifact_id, first.version).state == "building"


def _rebuild_after_first_lookup(store, monkeypatch, artifact_id: str, provenance: str) -> None:
    """Land a rebuild of ``artifact_id``, superseding what the first provenance lookup returns."""
    find = store.find_by_provenance
    landed = []

    def find_then_rebuild(*args, **kwargs):
        found = find(*args, **kwargs)
        if not landed:
            landed.append(True)
            version = store.create_artifact(artifact_id, provenance)
            store.finalize_artifact(artifact_id, version, "{}", 1, 10, content_sha256="e" * 64)
        return found

    monkeypatch.setattr(store, "find_by_provenance", find_then_rebuild)


class TestFindReadyAndSetName:
    def test_it_names_the_ready_version(self, store):
        _make_ready_artifact(store, "a1", "prov-1")

        found = store.find_ready_and_set_name("prov-1", "the-name")

        assert (found.id, found.version) == ("a1", 1)
        name = store.get_name("the-name")
        assert (name.artifact_id, name.version) == ("a1", 1)

    def test_nothing_ready_names_nothing(self, store):
        store.create_artifact("a1", "prov-1")

        assert store.find_ready_and_set_name("prov-1", "the-name") is None
        assert store.get_name("the-name") is None

    def test_a_rebuild_between_the_lookup_and_the_name_gets_the_name(self, store, monkeypatch):
        _make_ready_artifact(store, "a1", "prov-1")
        _rebuild_after_first_lookup(store, monkeypatch, "a1", "prov-1")

        found = store.find_ready_and_set_name("prov-1", "the-name")

        assert (found.id, found.version, found.state) == ("a1", 2, "ready")
        name = store.get_name("the-name")
        assert (name.artifact_id, name.version) == ("a1", 2)
        assert store.get_artifact("a1", 1).state == "superseded"


class TestWholeStoreReads:
    """``tenant`` means the default tenant when ``None``; every tenant only when left out."""

    def _seed(self, store):
        _make_ready_artifact(store, "up", "prov-up")
        for tenant, artifact_id in ((None, "mine"), ("acme", "theirs")):
            version = store.create_artifact(
                artifact_id,
                f"prov-{artifact_id}",
                input_versions={"vestigo://name/model@champion": "up@v=1"},
                tenant=tenant,
            )
            store.write_blob(artifact_id, version, _ipc_bytes(1))
            store.finalize_artifact(artifact_id, version, "{}", 1, 10)

    def test_find_dependents(self, store):
        self._seed(store)

        assert {d.id for d, _ in store.find_dependents("up", 1)} == {"mine", "theirs"}
        assert [d.id for d, _ in store.find_dependents("up", 1, tenant=None)] == ["mine"]
        assert [d.id for d, _ in store.find_dependents("up", 1, tenant="acme")] == ["theirs"]

    def test_list_name_reads(self, store):
        self._seed(store)

        assert {r[1] for r in store.list_name_reads()} == {"mine", "theirs"}
        assert store.list_name_reads(tenant=None) == [(None, "mine", "model@champion")]
        assert store.list_name_reads(tenant="acme") == [("acme", "theirs", "model@champion")]


class TestVersionsOfComputation:
    """Every output of one bound computation can be found by its ``computation_key``."""

    def test_the_outputs_of_a_computation_are_found_per_tenant(self, store):
        for artifact_id, tenant in (("model", None), ("metrics", None), ("theirs", "acme")):
            version = store.create_artifact(
                artifact_id, f"prov-{artifact_id}", tenant=tenant, computation_key="comp-1"
            )
            store.write_blob(artifact_id, version, _ipc_bytes(1))
            store.finalize_artifact(artifact_id, version, "{}", 1, 10)
        _make_ready_artifact(store, "other", "prov-other")
        store.create_artifact("building", "prov-b", computation_key="comp-1")
        store.create_artifact("twin", "prov-model")
        store.write_blob("twin", 1, _ipc_bytes(1))
        store.finalize_canonical_together([StagedVersion("twin", 1, "{}", 1, 10, "d")])

        found = store.versions_of_computation("comp-1")

        assert [(v.id, v.state) for v in found] == [("model", "superseded"), ("metrics", "ready")]
        assert [v.id for v in store.versions_of_computation("comp-1", tenant="acme")] == ["theirs"]
        assert store.versions_of_computation("comp-2") == []


class TestSetDefinitions:
    """A version is attributed only to a node stored in its tenant's scope."""

    def _node(self, store, tenant=None):
        return store.graph.put_batch([NodeSpec("test/cell", b"x")], tenant=tenant)[0]

    def test_a_definition_is_recorded_in_the_versions_tenant(self, store):
        node = self._node(store, tenant="t1")
        store.create_artifact("out", "prov-1", tenant="t1")
        store.create_artifact("out2", "prov-2")

        store.set_definitions({("out", 1): node, ("out2", 1): node}, tenant="t1")

        assert store.definition_of("out", 1, tenant="t1") == node
        assert store.definition_of("out2", 1) is None

    def test_a_node_from_another_scope_or_none_at_all_is_refused(self, store):
        theirs = self._node(store, tenant="t2")
        store.create_artifact("out", "prov-1", tenant="t1")
        store.create_artifact("out2", "prov-2", tenant="t1")
        mine = self._node(store, tenant="t1")

        with pytest.raises(ValueError, match="not in the store"):
            store.set_definitions({("out", 1): mine, ("out2", 1): theirs}, tenant="t1")
        with pytest.raises(ValueError, match="not in the store"):
            store.set_definitions({("out", 1): "f" * 64}, tenant="t1")
        with pytest.raises(ValueError, match="not a node id"):
            store.set_definitions({("out", 1): "zz"}, tenant="t1")
        assert store.definition_of("out", 1, tenant="t1") is None


class TestIdPrefixListing:
    def test_a_prefix_with_like_wildcards_matches_itself_only(self, store):
        _make_ready_artifact(store, "a_b", "prov-1")
        _make_ready_artifact(store, "axb", "prov-2")
        _make_ready_artifact(store, "a%b", "prov-3")

        assert [v.id for v in store.list_latest_by_id_prefix("a_")] == ["a_b"]
        assert [v.id for v in store.list_versions_by_id_prefix("a%")] == ["a%b"]


class TestUriScheme:
    """The consumer's scheme reaches the rows: what the store writes and what it matches."""

    @pytest.fixture(autouse=True)
    def _default_scheme_after(self):
        yield
        uris.set_uri_scheme(uris.DEFAULT_URI_SCHEME)

    def test_name_reads_and_lineage_follow_the_consumers_scheme(self, store):
        uris.set_uri_scheme("strata")
        spec = TransformSpec(executor="cell@v1", params={}, inputs=[])
        store.create_artifact("scan", "p-scan", spec)
        store.finalize_artifact("scan", 1, "{}", row_count=0, byte_size=0)
        edges = {"strata://artifact/scan@v=1": "scan@v=1", "strata://name/taxi/model": "m@v=1"}
        store.create_artifact("top", "p-top", spec, input_versions=edges)
        store.finalize_artifact("top", 1, "{}", row_count=0, byte_size=0)

        assert store.list_name_reads() == [(None, "top", "taxi/model")]
        ancestors = store.ancestor_transform_specs(store.get_artifact("top", 1), max_depth=10)
        assert ancestors == [spec.canonical_json]
        # Under the default scheme the same rows are foreign URIs: no name read, no edge.
        uris.set_uri_scheme(uris.DEFAULT_URI_SCHEME)
        assert store.list_name_reads() == []
        assert store.ancestor_transform_specs(store.get_artifact("top", 1), max_depth=10) == []


# --- Audit 0.2.0, store lane ---------------------------------------------------------

_OCTETS = TransformSpec(
    executor="t@v1", params={"content_type": "application/octet-stream"}, inputs=[]
)


def _finished(store, artifact_id: str, provenance: str, data: bytes | None = None, **kwargs) -> int:
    """A ready version holding *data* (one Arrow row by default)."""
    version = store.create_artifact(artifact_id, provenance, **kwargs)
    store.write_blob(artifact_id, version, _ipc_bytes(1) if data is None else data)
    store.finalize_artifact(artifact_id, version, "{}", None, 10)
    return version


def _row(store, artifact_id: str, version: int):
    conn = store._get_connection()
    try:
        return conn.execute(
            "SELECT state, superseded_by, last_used_at FROM artifact_versions "
            "WHERE id = ? AND version = ?",
            (artifact_id, version),
        ).fetchone()
    finally:
        conn.close()


def _record(artifact_id: str, version: int, provenance: str, state: str = "ready", **kwargs):
    return ArtifactVersion(
        id=artifact_id,
        version=version,
        state=state,
        provenance_hash=provenance,
        transform_spec=_OCTETS.canonical_json,
        created_at=1.0,
        **kwargs,
    )


class _Racing:
    """A store on each backend, and a competitor on its database that never waits."""

    @pytest.fixture
    def store(self, backend_store):
        return backend_store

    @pytest.fixture
    def competitor(self, store):
        competitor = _no_wait_store(store)
        yield competitor
        competitor.dialect.close()


class TestDeleteAgainstADuplicateFinalize(_Racing):
    """A duplicate's finalize and a delete of its canonical, in every order, leave the
    duplicate reading bytes (audit SR-B)."""

    def test_a_delete_waits_while_the_duplicate_points_at_the_canonical(
        self, store, competitor, monkeypatch
    ):
        _finished(store, "c1", "dup")
        version = store.create_artifact("p1", "dup")
        store.write_blob("p1", version, _ipc_bytes(1))
        refused = []
        supersede = ArtifactStore._supersede_duplicate

        def delete_meanwhile(self, conn, *args):
            try:
                competitor.delete_artifact("c1", 1)
            except competitor.dialect.operational_error as exc:
                refused.append(exc)
            return supersede(self, conn, *args)

        monkeypatch.setattr(ArtifactStore, "_supersede_duplicate", delete_meanwhile)
        overtaken = store.finalize_artifact("p1", version, "{}", None, 10)
        monkeypatch.undo()

        assert (overtaken.id, overtaken.version) == ("c1", 1)
        assert len(refused) == 1, "the canonical was not locked"
        assert store.delete_artifact("c1", 1)
        assert store.read_blob("p1", version) == _ipc_bytes(1)
        assert store.verify_artifacts() == []

    def test_a_canonical_deleted_after_the_lookup_leaves_the_duplicate_ready(
        self, store, competitor, monkeypatch
    ):
        _finished(store, "c1", "dup")
        version = store.create_artifact("p1", "dup")
        store.write_blob("p1", version, _ipc_bytes(1))
        find = store.find_by_provenance

        def found_then_deleted(*args, **kwargs):
            found = find(*args, **kwargs)
            if found is not None and found.id == "c1":
                competitor.delete_artifact("c1", 1)
            return found

        monkeypatch.setattr(store, "find_by_provenance", found_then_deleted)
        result = store.finalize_artifact("p1", version, "{}", None, 10)
        monkeypatch.undo()

        assert (result.id, result.version, result.state) == ("p1", version, "ready")
        assert store.read_blob("p1", version) == _ipc_bytes(1)
        assert store.verify_artifacts() == []

    def test_a_pointer_landing_during_the_copies_gets_a_copy_too(
        self, store, competitor, monkeypatch
    ):
        _finished(store, "c1", "dup")
        _finished(store, "p1", "dup")
        assert _row(store, "p1", 1)["superseded_by"] == "c1@v=1"
        version = store.create_artifact("p2", "dup")
        store.write_blob("p2", version, _ipc_bytes(1))
        reclaim = ArtifactStore._reclaim_blob
        landed = []

        def land_another(self, *args):
            if not landed:
                landed.append(competitor.finalize_artifact("p2", version, "{}", None, 10))
            return reclaim(self, *args)

        monkeypatch.setattr(ArtifactStore, "_reclaim_blob", land_another)
        assert store.delete_artifact("c1", 1)
        monkeypatch.undo()

        assert landed[0].id == "c1"
        assert store.read_blob("p1", 1) == _ipc_bytes(1)
        assert store.read_blob("p2", version) == _ipc_bytes(1)
        assert store.verify_artifacts() == []


class TestTheGapRuleMidSweep(_Racing):
    """A name or a use landing on a lower version during a sweep keeps the id's top
    version too, so no version number is reused (audit SR-A)."""

    @pytest.mark.parametrize("landing", ["name", "read"])
    def test_the_top_version_stays_with_the_lower_one(
        self, store, competitor, monkeypatch, landing
    ):
        _finished(store, "x", "prov-1")
        _finished(store, "x", "prov-2")
        _idle_everything(store)
        gaps = ArtifactStore._without_version_gaps
        landed = []

        def land(conn, chosen):
            if not landed:
                landed.append(True)
                if landing == "name":
                    competitor.set_name("keep", "x", 1)
                else:
                    competitor.read_blob("x", 1)
            return gaps(conn, chosen)

        monkeypatch.setattr(ArtifactStore, "_without_version_gaps", staticmethod(land))
        result = store.garbage_collect(max_idle_days=30, collect_latest=True)
        monkeypatch.undo()

        assert landed and result["deleted_count"] == 0
        assert store.get_artifact("x", 2) is not None
        assert store.create_artifact("x", "prov-3") == 3


class TestAHoldAfterTheClaim(_Racing):
    """A pin or a publication reaching a version the sweep claimed is refused, not
    deleted with it (audit SR-E)."""

    @pytest.mark.parametrize("hold", ["retain", "publish"])
    def test_the_hold_is_refused_while_the_claim_holds_the_row(self, store, competitor, hold):
        _finished(store, "gone", "prov-gone")
        _idle_everything(store)
        children = store._delete_version_children
        refused = []

        def hold_meanwhile(conn, artifact_id, version):
            try:
                if hold == "retain":
                    competitor.retain(artifact_id, version, kind="pin", holder="review")
                else:
                    competitor.publish_artifact(artifact_id, version)
            except competitor.dialect.operational_error as exc:
                refused.append(exc)
            children(conn, artifact_id, version)

        store._delete_version_children = hold_meanwhile
        result = store.garbage_collect(max_idle_days=0, collect_latest=True)

        assert result["deleted_count"] == 1
        assert len(refused) == 1, "the hold landed on a claimed version"
        assert store.retention_roots() == []
        assert store.list_publications(include_revoked=True) == []


_POINTER_WRITERS = {
    "alias": lambda s: s.set_alias("m", "champion", "d", 1),
    "tag": lambda s: s.set_tag("d", 1, "stage", "prod"),
    "request": lambda s: s.request_alias_change("m", "champion", "set", artifact_id="d", version=1),
    "name": lambda s: s.set_name("n", "d", 1),
}


class TestPointersLockTheirTarget(_Racing):
    """A delete between a pointer's state check and its write waits, so no pointer
    dangles (audit SL-4)."""

    @pytest.mark.parametrize("writer", sorted(_POINTER_WRITERS))
    def test_a_delete_in_between_is_refused(self, store, competitor, monkeypatch, writer):
        _finished(store, "d", "prov-d")
        audit = ArtifactStore._audit_in_connection
        outcome = []

        def delete_meanwhile(self, conn, **kwargs):
            if not outcome:
                try:
                    outcome.append(competitor.delete_artifact("d", 1))
                except competitor.dialect.operational_error as exc:
                    outcome.append(exc)
            return audit(self, conn, **kwargs)

        monkeypatch.setattr(ArtifactStore, "_audit_in_connection", delete_meanwhile)
        _POINTER_WRITERS[writer](store)
        monkeypatch.undo()

        assert isinstance(outcome[0], competitor.dialect.operational_error), outcome
        assert store.get_artifact("d", 1) is not None

    def test_a_name_on_a_version_gone_after_the_lock_is_not_found(self, store, monkeypatch):
        """The names foreign key has the last word; it reads as the writers' error."""
        monkeypatch.setattr(
            ArtifactStore,
            "_lock_target",
            lambda self, conn, *args, **kwargs: {"state": "ready", "tenant": ""},
        )

        with pytest.raises(ArtifactNotFoundError):
            store.set_name("n", "nowhere", 1)


class TestASameIdRaceIsARefresh:
    """Two builds of one id and computation racing: the newer wins, as a refresh does
    (audit SL-6)."""

    def test_the_newer_version_wins_the_integrity_branch(self, backend_store, monkeypatch):
        store = backend_store
        v1 = store.create_artifact("a", "prov-4")
        v2 = store.create_artifact("a", "prov-4")
        store.write_blob("a", v1, _ipc_bytes(1))
        store.write_blob("a", v2, _ipc_bytes(2))
        store.finalize_artifact("a", v1, "{}", None, 10)
        find = store.find_by_provenance
        calls = []

        def stale_once(*args, **kwargs):
            calls.append(True)
            return None if len(calls) == 1 else find(*args, **kwargs)

        monkeypatch.setattr(store, "find_by_provenance", stale_once)
        result = store.finalize_artifact("a", v2, "{}", None, 10)
        monkeypatch.undo()

        assert len(calls) == 2
        assert (result.version, result.state) == (v2, "ready")
        older = _row(store, "a", v1)
        assert (older["state"], older["superseded_by"]) == ("superseded", None)
        assert store.read_blob("a", v2) == _ipc_bytes(2)
        assert store.find_by_provenance("prov-4").version == v2


class TestCleanupFailedSparesAHold:
    def test_a_held_failed_version_stays_until_released(self, backend_store):
        """A hold is a promise someone was given (audit SR-C)."""
        store = backend_store
        version = store.create_artifact("b", "prov-b")
        store.retain("b", version, kind="pin", holder="review")
        store.fail_artifact("b", version)

        assert store.cleanup_failed(max_age_seconds=-10) == 0
        assert store.get_artifact("b", version).state == "failed"
        assert len(store.retention_roots("b", version)) == 1
        assert store.release("b", version, kind="pin", holder="review")
        assert store.cleanup_failed(max_age_seconds=-10) == 1


class TestRootsAreEachTenants:
    def test_two_tenants_hold_one_shared_version_under_one_holder(self, backend_store):
        """Each tenant lists and releases its own root (audit SR-D)."""
        store = backend_store
        version = _finished(store, "shared", "prov-shared")

        store.retain("shared", version, kind="pin", holder="review", tenant="a")
        claimed = store.retain("shared", version, kind="pin", holder="review", tenant="b")

        assert claimed["tenant"] == "b"
        assert [r["tenant"] for r in store.retention_roots("shared", version, tenant="b")] == ["b"]
        assert store.release("shared", version, kind="pin", holder="review", tenant="b")
        assert [r["tenant"] for r in store.retention_roots("shared", version)] == ["a"]


class TestImportOverALocalFailure:
    """A result arriving for a version that failed here fills it in (audit SL-1)."""

    def test_a_result_fills_a_failed_row_in(self, backend_store):
        store = backend_store
        version = store.create_artifact("cell-out", "prov-1", _OCTETS)
        store.write_blob("cell-out", version, b"partial")
        store.fail_artifact("cell-out", version)

        landed = store.import_artifact(_record("cell-out", version, "prov-1"), b"good bytes")

        assert (landed.version, landed.written) == (version, True)
        assert store.get_artifact("cell-out", version).state == "ready"
        assert store.find_by_provenance("prov-1").version == version
        assert store.read_blob("cell-out", version) == b"good bytes"
        assert not store.blob_store.blob_exists("cell-out", version), "the failed run's bytes"
        assert store.cleanup_failed(max_age_seconds=-10) == 0
        assert store.read_blob("cell-out", version) == b"good bytes"
        assert store.verify_artifacts() == []

    def test_a_build_in_progress_is_a_conflict_to_retry(self, backend_store):
        store = backend_store
        version = store.create_artifact("cell-out", "prov-1", _OCTETS)

        with pytest.raises(ArtifactImportInProgress):
            store.import_artifact(_record("cell-out", version, "prov-1"), b"good bytes")

        assert store.get_artifact("cell-out", version).state == "building"
        assert not store.blob_exists("cell-out", version)

    def test_a_failed_record_leaves_a_failed_row_as_it_is(self, backend_store):
        store = backend_store
        version = store.create_artifact("cell-out", "prov-1", _OCTETS)
        store.fail_artifact("cell-out", version)

        landed = store.import_artifact(_record("cell-out", version, "prov-1", "failed"), None)

        assert not landed.written
        assert store.get_artifact("cell-out", version).state == "failed"


class TestImportWithoutBytes:
    """A result without bytes would be served as a cache hit (audit SL-2)."""

    def test_a_result_without_bytes_is_refused(self, backend_store):
        store = backend_store
        with pytest.raises(ValueError, match="carries no bytes"):
            store.import_artifact(_record("ghost", 1, "prov-g"), None)

        assert store.get_artifact("ghost", 1) is None
        assert store.find_by_provenance("prov-g") is None

    def test_a_result_already_here_needs_none(self, backend_store):
        store = backend_store
        version = _finished(store, "here", "prov-h", b"x", transform_spec=_OCTETS)

        landed = store.import_artifact(_record("here", version, "prov-h"), None)

        assert (landed.id, landed.version, landed.written) == ("here", version, False)

    def test_a_row_here_without_bytes_is_refused_too(self, backend_store):
        store = backend_store
        version = store.create_artifact("hollow", "prov-x", _OCTETS)
        store.finalize_artifact("hollow", version, "{}", None, 0)

        with pytest.raises(ValueError, match="has none either"):
            store.import_artifact(_record("hollow", version, "prov-x"), None)

    def test_a_failed_record_needs_none(self, backend_store):
        store = backend_store

        landed = store.import_artifact(_record("bad", 1, "prov-b", "failed"), None)

        assert landed.written
        assert store.get_artifact("bad", 1).state == "failed"


class TestImportChecksWhatItStores:
    """Only the import path writes a state and ``input_versions`` as given (audit SL-8)."""

    @pytest.mark.parametrize("state", ["bogus", "building"])
    def test_a_state_no_stored_version_arrives_in_is_refused(self, store, state):
        with pytest.raises(ValueError, match="cannot import a version in state"):
            store.import_artifact(_record("s", 1, "prov-s", state), b"b")

        assert store.get_artifact("s", 1) is None

    @pytest.mark.parametrize("text", ["not json", "[1, 2]", "NaN"])
    def test_input_versions_must_be_a_json_object(self, store, text):
        with pytest.raises(ValueError, match="input_versions"):
            store.import_artifact(_record("bad", 1, "prov-b", input_versions=text), b"b")

        assert store.list_name_reads() == []

    def test_input_versions_are_stored_as_the_store_writes_them(self, store):
        store.import_artifact(_record("ok", 1, "prov-o", input_versions='{ "a" : "b@v=1" }'), b"b")

        assert store.get_artifact("ok", 1).input_versions == json.dumps({"a": "b@v=1"})


class TestPromotionIsNotARelease:
    def test_a_promoted_checkpoint_output_is_neither_output_nor_collectable(self, store):
        """The promote is no part of the release its source came from (audit SL-3)."""
        builds = BuildStore(store.db_path, dialect=store.dialect)
        releases = ReleaseStore(store, builds)
        parent = hashlib.sha256(b"train").hexdigest()
        target = store.create_artifact(
            "final-b1", f"{parent}-final", _OCTETS, computation_key=parent, minted=True
        )
        builds.create_build("b1", "final-b1", target, executor_ref="train@v1")
        assert builds.open_attempt("b1", "a1", writable_until=4e9, lease_owner="r1")
        bundles = []
        for _ in range(2):
            release = releases.prepare("b1", "a1", "checkpoint")
            artifact_id = f"model-{release.sequence}"
            data = b"weights" + bytes([release.sequence])
            version = store.create_artifact(
                artifact_id,
                checkpoint_output_key(parent, "b1", release.sequence, "model"),
                _OCTETS,
                computation_key=parent,
                minted=True,
                release_id=release.release_id,
            )
            store.write_blob(artifact_id, version, data)
            staged = StagedVersion(
                artifact_id, version, "{}", None, len(data), hashlib.sha256(data).hexdigest()
            )
            bundles.append(releases.accept(release.release_id, [staged], []))

        promoted = store.promote_version("model-1", 1)

        assert (promoted.version, promoted.release_id) == (2, None)
        first = bundles[0].release.release_id
        assert [(o.id, o.version) for o in releases.bundle(first).outputs] == [("model-1", 1)]
        assert ("model-1", 2) not in releases.collectable_checkpoints(keep=1)


class TestAReadThroughAPointer:
    def test_both_uses_are_noted_after_every_read(self, store):
        """The note turns SQLite's busy handler off for what follows (audit SL-9)."""
        _finished(store, "c1", "dup")
        _finished(store, "p1", "dup")
        _idle_everything(store)
        statements: list[str] = []
        connect = store._get_connection

        def traced():
            conn = connect()
            conn.set_trace_callback(statements.append)
            return conn

        store._get_connection = traced
        assert store.read_blob("p1", 1) == _ipc_bytes(1)

        after = statements[statements.index("PRAGMA busy_timeout = 0") + 1 :]
        assert after and not any(s.lstrip().upper().startswith("SELECT") for s in after), after
        assert _row(store, "p1", 1)["last_used_at"] > 0
        assert _row(store, "c1", 1)["last_used_at"] > 0


class TestAnObjectStoreStagesBesideTheLedger:
    """A remote store's streamed writes stage under the artifact dir, where the sweep looks."""

    def test_an_unset_staging_dir_is_the_artifact_dirs(self, tmp_path):
        blobs = S3BlobStore(bucket="b", anonymous=True)
        ArtifactStore(tmp_path, blob_store=blobs)
        assert blobs.staging_dir == tmp_path / "staging"

    def test_a_given_staging_dir_is_kept(self, tmp_path):
        blobs = S3BlobStore(bucket="b", anonymous=True, staging_dir=tmp_path / "elsewhere")
        ArtifactStore(tmp_path, blob_store=blobs)
        assert blobs.staging_dir == tmp_path / "elsewhere"


class TestTheStoreSingleton:
    """One store per process, and a reset releases it (audit SL-10)."""

    def test_another_directory_is_refused(self, tmp_path):
        first = get_artifact_store(tmp_path / "a")

        assert get_artifact_store(tmp_path / "a") is first
        with pytest.raises(ValueError, match="singleton is open at"):
            get_artifact_store(tmp_path / "b")
        assert get_artifact_store() is first

    def test_a_reset_closes_the_store(self, tmp_path, monkeypatch):
        held = get_artifact_store(tmp_path / "a")
        closed = []
        monkeypatch.setattr(held, "close", lambda: closed.append(True))

        reset_artifact_store()

        assert closed == [True]
        assert get_artifact_store() is None


class TestTenantScopedLifecycleWrites:
    """Another tenant's version reads as missing to these writers (audit SL-11)."""

    @pytest.mark.parametrize("call", ["fail_artifact", "retire_ready", "record_use"])
    def test_another_tenants_version_is_not_found(self, store, call):
        if call == "fail_artifact":
            version = store.create_artifact("theirs", "prov-t", tenant="b")
        else:
            version = _finished(store, "theirs", "prov-t", tenant="b")
        _idle_everything(store)
        before = tuple(_row(store, "theirs", version))
        act = getattr(store, call)

        with pytest.raises(ArtifactNotFoundError):
            act("theirs", version, tenant="a")
        with pytest.raises(ArtifactNotFoundError):
            act("missing", 1, tenant="b")

        assert tuple(_row(store, "theirs", version)) == before
        act("theirs", version, tenant="b")
        assert tuple(_row(store, "theirs", version)) != before


class TestVerifyStreams:
    def test_a_blob_no_validator_reads_is_hashed_as_a_stream(self, store, monkeypatch):
        """Read whole, a large blob is a large allocation per version (audit SR-G)."""
        version = _finished(store, "plain", "prov-p", b"plain" * 10_000, transform_spec=_OCTETS)

        def whole(*args, **kwargs):
            raise AssertionError("read whole")

        monkeypatch.setattr(store.blob_store, "read_blob", whole)

        assert store.verify_artifacts() == []
        store.blob_store.write_blob("plain", version, b"changed")
        assert [f["problem"] for f in store.verify_artifacts()] == ["digest_mismatch"]

    def test_a_spec_that_does_not_parse_does_not_end_the_check(self, store):
        """An imported spec is stored verbatim, and NaN is not JSON (audit G-1)."""
        nan_spec = '{"executor": "x@v1", "inputs": [], "params": {"w": NaN}}'
        odd = replace(_record("odd", 1, "prov-odd"), transform_spec=nan_spec)
        store.import_artifact(odd, _ipc_bytes(2))
        store.create_artifact("ghost", "prov-ghost")
        store.finalize_artifact("ghost", 1, "{}", None, 10)

        findings = store.verify_artifacts()

        assert [(f["artifact_id"], f["problem"]) for f in findings] == [("ghost", "missing_blob")]


class TestUnaddressableIds:
    """An id no artifact URI can name has no lineage (audit G-5)."""

    @pytest.mark.parametrize("artifact_id", ["", "a@b"])
    def test_create_refuses_it(self, store, artifact_id):
        with pytest.raises(ValueError):
            store.create_artifact(artifact_id, "prov")

        assert store.stats()["total_versions"] == 0

    @pytest.mark.parametrize("artifact_id", ["", "a@b"])
    def test_import_refuses_it(self, store, artifact_id):
        with pytest.raises(ValueError):
            store.import_artifact(_record(artifact_id, 1, "prov"), b"b")

        assert store.stats()["total_versions"] == 0


class TestSetDefinitionsCount:
    def test_the_count_is_of_the_versions_recorded(self, store):
        """A pair skipped (missing, or already set) is not counted (audit G-9)."""
        node = store.graph.put_batch([NodeSpec("test/cell", b"x")])[0]
        store.create_artifact("out", "prov-1")

        assert store.set_definitions({("out", 1): node, ("missing", 1): node}) == 1
        assert store.set_definitions({("out", 1): node}) == 0


class TestADeletedBuildsChildren:
    def test_its_console_goes_and_its_attempts_stay_for_the_attempt_sweep(
        self, backend_store, tmp_path
    ):
        """An attempt row is the only record of bytes an unpromoted attempt wrote (BR-5)."""
        store = backend_store
        builds = BuildStore(tmp_path / "builds.sqlite", dialect=store.dialect)
        version = store.create_artifact("built", "prov-built")
        builds.create_build("build-1", "built", version, "exec@v1")
        assert builds.start_build("build-1")
        assert builds.append_console_chunk("build-1", "stdout", 0, "hello")
        builds.record_attempt("build-1", "built", version, "att-1", writable_until=0)

        assert store.delete_artifact("built", version)

        assert builds.get_build("build-1") is None
        conn = store._get_connection()
        try:
            chunks = conn.execute(
                "SELECT COUNT(*) AS n FROM build_console_chunks WHERE build_id = 'build-1'"
            ).fetchone()["n"]
        finally:
            conn.close()
        assert chunks == 0
        assert [a[:4] for a in builds.settled_attempts()] == [
            ("build-1", "built", version, "att-1")
        ]


class TestPublicCallablesTheSuiteMissed:
    """Callables nothing else in the suite runs (audit TP-3)."""

    def test_staging_drops_this_tenants_stale_stagings_only(self, store, tmp_path):
        def stage(tenant, data, now):
            source = tmp_path / f"{tenant}-{data.decode()}"
            source.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            store.stage_import_blob(tenant, digest, source, len(data), now=now)
            return digest

        old = stage("t", b"old", 1000.0)
        theirs = stage("u", b"theirs", 1000.0)
        new = stage("t", b"new", 1000.0 + IMPORT_STAGING_TTL_SECONDS + 1)

        assert store.open_staged_import("t", old) is None
        with store.open_staged_import("t", new) as reader:
            assert reader.read() == b"new"
        with store.open_staged_import("u", theirs) as reader:
            assert reader.read() == b"theirs"
        assert store.open_staged_import("t", theirs) is None

    def test_a_prepared_file_is_published_as_the_blob(self, store, tmp_path):
        version = store.create_artifact("filed", "prov-f")
        source = tmp_path / "prepared"
        source.write_bytes(_ipc_bytes(3))

        store.publish_blob_from_path("filed", version, source)

        assert store.read_blob("filed", version) == _ipc_bytes(3)

    def test_a_revoked_publication_lets_the_version_go(self, store):
        version = _finished(store, "pub", "prov-pub")
        publication = store.publish_artifact("pub", version)
        with pytest.raises(ValueError, match="is published"):
            store.delete_artifact("pub", version)

        assert store.revoke_publication(publication.token)
        assert not store.revoke_publication(publication.token)
        assert store.list_publications() == []
        assert store.delete_artifact("pub", version)

    def test_tag_and_name_lookups_by_version(self, store):
        version = _finished(store, "tagged", "prov-tag")
        store.set_tag("tagged", version, "stage", "prod")
        store.set_name("best", "tagged", version)

        assert store.list_artifacts_with_tag_key("stage") == [("tagged", version, "prod")]
        assert store.list_artifacts_with_tag_key("other") == []
        assert store.get_name_for_artifact("tagged", version) == "best"
        assert store.get_name_for_artifact("tagged", version + 1) is None
