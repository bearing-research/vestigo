"""Tests for the SQL dialect seam."""

from __future__ import annotations

import gc
import logging
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from vestigo import sql_backend
from vestigo.build_store import BuildStore
from vestigo.sql_backend import (
    SERVER_THREAD_LIMIT,
    PostgresDialect,
    SqliteDialect,
    _Lease,
    _PostgresConnection,
    _Row,
    advisory_lock_id,
    create_tables,
    one,
    translate_placeholders,
)
from vestigo.store import _FIRST_BOOT_OBJECTS, ArtifactStore, StoreSchemaMismatch


class TestTranslatePlaceholders:
    """Qmark to pyformat, without corrupting data that looks like syntax."""

    def test_bare_placeholders_convert(self):
        assert (
            translate_placeholders("SELECT * FROM t WHERE id = ? AND v = ?")
            == "SELECT * FROM t WHERE id = %s AND v = %s"
        )

    def test_statement_without_placeholders_is_unchanged(self):
        sql = "SELECT COUNT(*) FROM artifact_versions"
        assert translate_placeholders(sql) == sql

    def test_question_mark_inside_a_string_literal_is_data(self):
        assert (
            translate_placeholders("SELECT * FROM t WHERE name = 'why?' AND id = ?")
            == "SELECT * FROM t WHERE name = 'why?' AND id = %s"
        )

    def test_question_mark_inside_a_quoted_identifier_is_data(self):
        assert (
            translate_placeholders('SELECT "odd?col" FROM t WHERE id = ?')
            == 'SELECT "odd?col" FROM t WHERE id = %s'
        )

    def test_doubled_quote_does_not_end_the_literal(self):
        # 'it''s ?' is one literal containing an apostrophe, so its question mark stays
        # data. Treating '' as a close would convert it.
        assert (
            translate_placeholders("SELECT * FROM t WHERE s = 'it''s ?' AND id = ?")
            == "SELECT * FROM t WHERE s = 'it''s ?' AND id = %s"
        )

    def test_percent_is_escaped_outside_literals(self):
        assert translate_placeholders("SELECT 5 % 2") == "SELECT 5 %% 2"

    def test_percent_is_escaped_inside_literals(self):
        # The driver scans the whole statement, so a percent is a substitution marker even
        # inside quotes.
        assert (
            translate_placeholders("SELECT * FROM t WHERE n LIKE 'nb_%'")
            == "SELECT * FROM t WHERE n LIKE 'nb_%%'"
        )

    def test_line_comment_contents_are_not_translated(self):
        sql = "SELECT 1 -- is this a ? placeholder\nWHERE id = ?"
        assert translate_placeholders(sql) == "SELECT 1 -- is this a ? placeholder\nWHERE id = %s"

    def test_block_comment_contents_are_not_translated(self):
        sql = "SELECT /* a ? here */ 1 WHERE id = ?"
        assert translate_placeholders(sql) == "SELECT /* a ? here */ 1 WHERE id = %s"

    def test_unterminated_literal_does_not_raise(self):
        # The driver gives a far better syntax error; the translator just must not crash.
        assert translate_placeholders("SELECT 'oops") == "SELECT 'oops"

    def test_real_statement_from_the_store(self):
        # The one place the store mixes a placeholder with a quoted literal in a statement.
        sql = "WHERE id LIKE ? ESCAPE '\\' AND state = 'ready'"
        assert translate_placeholders(sql) == "WHERE id LIKE %s ESCAPE '\\' AND state = 'ready'"

    def test_multiline_ddl(self):
        sql = """
            INSERT INTO artifact_versions
                (id, version, state, provenance_hash)
            VALUES (?, ?, 'building', ?)
        """
        translated = translate_placeholders(sql)
        assert translated.count("%s") == 3
        # The inline state literal is data and must survive intact.
        assert "'building'" in translated


class TestSqliteDialectPreservesTodaysBehavior:
    """Personal mode must not notice the seam exists."""

    def test_adapt_ddl_is_identity(self):
        # The schema is already written in this dialect.
        ddl = "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL)"
        assert SqliteDialect(Path("x.sqlite")).adapt_ddl(ddl) == ddl

    def test_connect_sets_wal_and_a_name_indexable_row_factory(self, tmp_path):
        dialect = SqliteDialect(tmp_path / "artifacts.sqlite")
        conn = dialect.connect()
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            conn.execute("CREATE TABLE t (tenant TEXT)")
            conn.execute("INSERT INTO t VALUES ('acme')")
            # The store reads columns by name throughout.
            assert conn.execute("SELECT tenant FROM t").fetchone()["tenant"] == "acme"
        finally:
            conn.close()

    def test_begin_write_serializes_writers(self, tmp_path):
        dialect = SqliteDialect(tmp_path / "artifacts.sqlite")
        conn = dialect.connect()
        try:
            # SQLite locks the whole file, so the key is accepted and ignored.
            dialect.begin_write(conn, "a1")
            assert conn.in_transaction
        finally:
            conn.rollback()
            conn.close()

    def test_legacy_migration_is_supported(self):
        # SQLite is the only backend with deployed history to upgrade.
        assert SqliteDialect(Path("x.sqlite")).supports_legacy_migration is True

    def test_a_connection_acquired_on_one_thread_is_used_and_closed_on_another(self, tmp_path):
        # The same consumer code runs on Postgres, whose wrappers release from any thread.
        dialect = SqliteDialect(tmp_path / "artifacts.sqlite")
        box: dict[str, object] = {}
        thread = threading.Thread(target=lambda: box.__setitem__("conn", dialect.connect()))
        thread.start()
        thread.join()
        conn = box["conn"]
        assert isinstance(conn, sqlite3.Connection)

        assert conn.execute("SELECT 1").fetchone()[0] == 1
        conn.close()


class _FakeInner:
    """Records what the wrapper would send to psycopg, so no server is needed."""

    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []
        self.statusmessage = None
        # The server's tag where the statement's verb is not it (a CTE is tagged SELECT).
        self.tag: str | None = None

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))
        # The command tag psycopg exposes on the cursor, from the statement's verb.
        self.statusmessage = self.tag or sql.split(None, 1)[0].upper()
        return self

    def commit(self):
        return None

    def rollback(self):
        return None


class TestPostgresConnectionAdapter:
    """The wrapper keeps the store's call sites unchanged."""

    def _wrap(self):
        inner = _FakeInner()
        released = []
        lease = _Lease(inner)
        return _PostgresConnection(lease, lambda lease: released.append(1)), inner, released

    def test_execute_translates_placeholders(self):
        conn, inner, _ = self._wrap()
        conn.execute("SELECT * FROM t WHERE id = ?", ("a1",))
        assert inner.executed == [("SELECT * FROM t WHERE id = %s", ("a1",))]

    def test_execute_accepts_a_list_of_params(self):
        # The store builds params as a list where filters are optional.
        conn, inner, _ = self._wrap()
        conn.execute("SELECT ? , ?", ["a", "b"])
        assert inner.executed[0][1] == ("a", "b")

    def test_executescript_sends_ddl_unchanged(self):
        # Multi-statement schema DDL, no parameters, no percent signs.
        conn, inner, _ = self._wrap()
        conn.executescript("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);")
        assert inner.executed == [("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);", ())]

    def test_begin_write_takes_an_advisory_lock_on_the_key(self):
        conn, inner, _ = self._wrap()
        with conn:
            PostgresDialect("postgresql:///x").begin_write(conn, "a1")
        sql, params = inner.executed[0]
        assert "pg_advisory_xact_lock" in sql
        assert params == (advisory_lock_id("a1"),)

    def test_close_returns_the_connection_instead_of_closing_it(self):
        # The store calls close() to mean "done with this"; under a pool that must mean
        # "give it back", or the pool drains to nothing.
        conn, inner, released = self._wrap()
        conn.close()
        assert released == [1]
        assert not hasattr(inner, "closed")

    def test_double_close_releases_once(self):
        # A second release would hand the same connection to two threads.
        conn, _, released = self._wrap()
        conn.close()
        conn.close()
        assert released == [1]

    def test_the_context_manager_releases_on_exit(self):
        conn, _, released = self._wrap()
        with conn as entered:
            assert entered is conn
            assert released == []
        assert released == [1]

    def test_a_dropped_wrapper_releases_itself_with_a_warning(self, caplog):
        # The ledger closes in a finally everywhere; this is for a consumer that does not.
        conn, _, released = self._wrap()
        with caplog.at_level(logging.WARNING, logger="vestigo.sql_backend"):
            del conn
            gc.collect()
        assert released == [1]
        assert any("without close()" in r.getMessage() for r in caplog.records)

    def test_writes_mark_the_lease_dirty_until_commit_or_rollback(self):
        # Decided by the server's command tag: a plain read is not a write.
        conn, _, _ = self._wrap()
        with conn:
            conn.execute("SELECT x FROM t WHERE x = ?", (1,))
            assert conn._lease.dirty is False
            conn.execute("INSERT INTO t VALUES (?)", (1,))
            assert conn._lease.dirty is True
            conn.commit()
            assert conn._lease.dirty is False
            conn.executescript("CREATE TABLE a (x TEXT);")
            assert conn._lease.dirty is True
            conn.rollback()
            assert conn._lease.dirty is False

    @pytest.mark.parametrize(
        "sql",
        [
            "WITH w AS (INSERT INTO t VALUES (1) RETURNING x) SELECT x FROM w",
            "with w as (delete from t returning x) select count(*) from w",
            "SELECT x FROM t FOR UPDATE",
            "SELECT x FROM t FOR NO KEY UPDATE",
            "SELECT x FROM t FOR KEY SHARE",
            "SELECT pg_advisory_xact_lock(?)",
            "SELECT pg_try_advisory_xact_lock_shared(?)",
        ],
    )
    def test_a_select_that_writes_or_locks_marks_the_lease_dirty(self, sql):
        # Postgres tags each of these SELECT; a nested commit or rollback would end what it holds.
        conn, inner, _ = self._wrap()
        inner.tag = "SELECT 1"
        with conn:
            conn.execute(sql, (1,) if "?" in sql else ())
            assert conn._lease.dirty is True

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 'DELETE FROM t' AS s",
            'SELECT 1 AS "update"',
            "SELECT 1 -- FOR UPDATE",
            "SELECT 1 /* INSERT */",
            "SELECT updated_at, shared, deleted FROM t",
        ],
    )
    def test_those_words_as_data_or_inside_a_name_mark_nothing(self, sql):
        conn, inner, _ = self._wrap()
        inner.tag = "SELECT 1"
        with conn:
            conn.execute(sql)
            assert conn._lease.dirty is False

    def test_a_closed_wrapper_refuses_use(self):
        # Its raw connection is back in the pool and may be another thread's.
        conn, inner, _ = self._wrap()
        conn.close()
        for use in (
            lambda: conn.execute("SELECT 1"),
            lambda: conn.executescript("SELECT 1"),
            conn.commit,
            conn.rollback,
        ):
            with pytest.raises(RuntimeError, match="closed"):
                use()
        assert inner.executed == []


class _FakePool:
    """Hands out idle raw connections and counts how many are out."""

    def __init__(self):
        self.out = 0

    def getconn(self):
        self.out += 1
        return SimpleNamespace(info=SimpleNamespace(transaction_status=TransactionStatus.IDLE))

    def putconn(self, raw):
        self.out -= 1

    def close(self):
        return None


class _LockWatchingDepth(int):
    """A lease depth that records, at each change, whether the lease's lock is held."""

    def __new__(cls, value: int, lease: _Lease, seen: list[bool]):
        depth = super().__new__(cls, value)
        depth.lease = lease
        depth.seen = seen
        return depth

    def _changed(self, value: int) -> _LockWatchingDepth:
        self.seen.append(self.lease.lock.locked())
        return _LockWatchingDepth(value, self.lease, self.seen)

    def __add__(self, other):
        return self._changed(int(self) + other)

    def __sub__(self, other):
        return self._changed(int(self) - other)


class TestLeaseDepth:
    def test_depth_changes_under_the_lease_lock(self):
        # Unlocked, a close on another thread racing the owner's loses a decrement and the
        # connection never returns to the pool.
        dialect = PostgresDialect("postgresql:///x")
        pool = _FakePool()
        dialect._pool = pool
        outer = dialect.connect()
        lease = dialect._local.lease
        seen: list[bool] = []
        lease.depth = _LockWatchingDepth(lease.depth, lease, seen)

        inner = dialect.connect()
        inner.close()
        outer.close()

        assert seen == [True, True, True]
        assert pool.out == 0
        dialect.close()


class TestRowBehavesLikeSqliteRow:
    """The store reads results by name, by position, and via dict()."""

    def _row(self):
        return _Row(("id", "tenant"), ("a1", "acme"))

    def test_indexable_by_name(self):
        assert self._row()["tenant"] == "acme"

    def test_indexable_by_position(self):
        # create_artifact reads `cursor.fetchone()[0]`.
        assert self._row()[0] == "a1"

    def test_sliceable_by_position(self):
        # sqlite3.Row slices; a reader unpacking `row[1:]` must work on both.
        assert self._row()[1:] == ("acme",)
        assert self._row()[:2] == ("a1", "acme")

    def test_convertible_to_dict(self):
        # read_audit and list_pending_changes do `dict(row)`.
        assert dict(self._row()) == {"id": "a1", "tenant": "acme"}

    def test_get_returns_the_default_for_a_missing_column(self):
        # Mapping's mixin get() catches KeyError only, so a raw ValueError out
        # of list.index would propagate instead of returning the default.
        assert self._row().get("principal", "fallback") == "fallback"

    def test_membership_test_does_not_raise(self):
        assert "principal" not in self._row()
        assert "tenant" in self._row()

    def test_hashes_and_compares_by_columns_and_values(self):
        # A consumer's set of rows or dict keyed by row works on SQLite.
        assert self._row() == self._row()
        assert len({self._row(), self._row()}) == 1
        assert self._row() != _Row(("id", "tenant"), ("a1", "other"))
        assert self._row() != _Row(("id", "owner"), ("a1", "acme"))

    def test_never_equals_a_dict_as_sqlite3_row_does_not(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            sqlite_row = conn.execute("SELECT 'a1' AS id, 'acme' AS tenant").fetchone()
        finally:
            conn.close()
        as_dict = {"id": "a1", "tenant": "acme"}

        assert (sqlite_row == as_dict) is False
        assert (self._row() == as_dict) is False


class TestPostgresDialectRendering:
    """What SQL to emit."""

    def test_adapt_ddl_rewrites_the_types_that_differ(self):
        ddl = (
            "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "at REAL NOT NULL, byte_size INTEGER)"
        )
        adapted = PostgresDialect("postgresql:///x").adapt_ddl(ddl)
        assert "BIGSERIAL PRIMARY KEY" in adapted
        assert "at DOUBLE PRECISION NOT NULL" in adapted
        assert "byte_size BIGINT" in adapted
        assert "AUTOINCREMENT" not in adapted

    def test_autoincrement_is_rewritten_before_bare_integer(self):
        # Order matters: the autoincrement rule must consume its own INTEGER first, or the
        # bare-INTEGER rule makes "BIGINT PRIMARY KEY AUTOINCREMENT" and the rule never matches.
        adapted = PostgresDialect("postgresql:///x").adapt_ddl(
            "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        assert adapted == "CREATE TABLE t (seq BIGSERIAL PRIMARY KEY)"

    def test_adapt_ddl_leaves_portable_types_alone(self):
        ddl = "CREATE TABLE t (id TEXT NOT NULL, tenant TEXT DEFAULT '')"
        assert PostgresDialect("postgresql:///x").adapt_ddl(ddl) == ddl

    def test_integer_widens_to_bigint(self):
        # Postgres INTEGER is int4 (max 2147483647), smaller than byte_size for any artifact
        # of 2 GiB or more.
        assert PostgresDialect("postgresql:///x").integer_type == "BIGINT"
        assert SqliteDialect(Path("x")).integer_type == "INTEGER"

    def test_autoincrement_differs_from_sqlite(self):
        assert PostgresDialect("postgresql:///x").autoincrement_pk == "BIGSERIAL PRIMARY KEY"
        assert SqliteDialect(Path("x")).autoincrement_pk == "INTEGER PRIMARY KEY AUTOINCREMENT"

    def test_float_type_is_double_precision(self):
        # Not REAL, which is single precision and drops a timestamp's sub-second part;
        # TestTimestampPrecision in the Postgres suite checks that against a live server.
        assert PostgresDialect("postgresql:///x").float_type == "DOUBLE PRECISION"

    def test_legacy_migration_is_not_supported(self):
        # No deployed Postgres history exists, so the sqlite_master / PRAGMA /
        # rowid migration path never runs there.
        assert PostgresDialect("postgresql:///x").supports_legacy_migration is False

    def test_lock_ids_are_stable_and_distinct(self):
        # Recomputed at every call site, so drift would silently stop serializing the
        # writers the lock exists for.
        assert advisory_lock_id("a1") == advisory_lock_id("a1")
        assert advisory_lock_id("a1") != advisory_lock_id("a2")
        assert -(2**63) <= advisory_lock_id("a1") < 2**63


class TestThreadLimitAndPoolAreOneNumber:
    """The pool holds a connection for every thread that may call the store at once."""

    def test_the_pool_covers_every_thread_token_and_the_event_loop(self):
        assert PostgresDialect("postgresql:///x").max_size == SERVER_THREAD_LIMIT + 1


@pytest.fixture
def dialect(postgres_dsn):
    dialect = PostgresDialect(postgres_dsn)
    with dialect.connect() as conn:
        conn.executescript("DROP TABLE IF EXISTS t; CREATE TABLE t (x INT)")
        conn.commit()
    yield dialect
    dialect.close()


def _count(dialect) -> int:
    with dialect.connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]


def _lock_is_free(dsn: str, key: str) -> bool:
    """Whether another session could take ``begin_write``'s lock on ``key`` now."""
    with psycopg.connect(dsn) as other:
        free = other.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (advisory_lock_id(key),)
        ).fetchone()[0]
        other.rollback()
        return free


def test_create_tables_runs_its_ddl_once_per_database(backend_store):
    # No IF NOT EXISTS, and a seed row: a second run would fail or seed twice.
    dialect = backend_store._dialect
    ddl = (
        "CREATE TABLE consumer_things (x INTEGER); "
        "CREATE INDEX idx_consumer_things ON consumer_things (x); "
        "INSERT INTO consumer_things VALUES (1);"
    )
    try:
        create_tables(dialect, "consumer_things", ddl)
        create_tables(dialect, "consumer_things", ddl)
        conn = dialect.connect()
        try:
            assert conn.execute("SELECT COUNT(*) FROM consumer_things").fetchone()[0] == 1
        finally:
            conn.close()
    finally:
        conn = dialect.connect()
        try:
            conn.executescript("DROP TABLE IF EXISTS consumer_things")
            conn.commit()
        finally:
            conn.close()


def _audit_seqs(conn) -> list[int]:
    return [row["seq"] for row in conn.execute("SELECT seq FROM registry_audit ORDER BY seq")]


def test_after_a_copy_with_explicit_keys_the_next_insert_follows_the_largest(backend_store):
    # A migration copies rows with their own keys; a Postgres sequence does not see them.
    dialect = backend_store._dialect
    conn = dialect.connect()
    try:
        for seq in (1, 7):
            conn.execute(
                "INSERT INTO registry_audit (seq, at, action) VALUES (?, 0, 'copied')", (seq,)
            )
        # Twice: a resumed migration resyncs again, and must land on the same value.
        dialect.resync_autoincrement(conn, "registry_audit", "seq")
        dialect.resync_autoincrement(conn, "registry_audit", "seq")
        conn.commit()

        conn.execute("INSERT INTO registry_audit (at, action) VALUES (0, 'new')")
        conn.commit()

        assert _audit_seqs(conn) == [1, 7, 8]
    finally:
        conn.close()


def test_a_duplicate_key_is_rejectable(backend_store):
    dialect = backend_store._dialect
    insert = "INSERT INTO registry_audit (seq, at, action) VALUES (1, 0, 'copied')"
    conn = dialect.connect()
    try:
        conn.execute(insert)
        with pytest.raises(dialect.rejectable_errors):
            conn.execute(insert)
        conn.rollback()
    finally:
        conn.close()


def test_an_error_that_is_not_the_row_s_is_not_rejectable(backend_store):
    # A missing table fails every row; a copy that swallowed it would report each as rejected.
    dialect = backend_store._dialect
    conn = dialect.connect()
    try:
        missing_table = (
            sqlite3.OperationalError if dialect.name == "sqlite" else psycopg.errors.UndefinedTable
        )
        with pytest.raises(missing_table) as excinfo:
            conn.execute("INSERT INTO no_such_table (x) VALUES (1)")
        conn.rollback()
    finally:
        conn.close()

    assert not isinstance(excinfo.value, dialect.rejectable_errors)


@pytest.mark.integration
@pytest.mark.slow
def test_a_nul_byte_postgres_refuses_in_text_is_rejectable(postgres_store):
    # SQLite stores the byte; a copy into Postgres meets it in executor logs or tag values.
    dialect = postgres_store._dialect
    conn = dialect.connect()
    try:
        with pytest.raises(dialect.rejectable_errors):
            conn.execute("INSERT INTO registry_audit (at, action) VALUES (0, ?)", ("a\x00b",))
        conn.rollback()
    finally:
        conn.close()


@pytest.mark.integration
@pytest.mark.slow
class TestPooledConnectionLifecycle:
    """What a thread's pooled connection does across wrappers, threads and failures."""

    def test_a_wrapper_dropped_after_a_failed_statement_does_not_poison_the_thread(
        self, dialect, caplog
    ):
        def forgets_to_close() -> None:
            conn = dialect.connect()
            try:
                conn.execute("SELECT * FROM no_such_table")
            except psycopg.Error:
                return

        with caplog.at_level(logging.WARNING, logger="vestigo.sql_backend"):
            forgets_to_close()
            gc.collect()

        assert any("without close()" in r.getMessage() for r in caplog.records)
        # The next acquisition on this thread is a clean connection, not the failed one.
        with dialect.connect() as again:
            assert again.execute("SELECT 1").fetchone()[0] == 1

    def test_a_nested_connect_on_a_failed_transaction_is_refused(self, dialect):
        outer = dialect.connect()
        try:
            with pytest.raises(psycopg.Error):
                outer.execute("SELECT * FROM no_such_table")
            with pytest.raises(RuntimeError, match="failed transaction"):
                dialect.connect()
            outer.rollback()
            with dialect.connect() as inner:
                assert inner.execute("SELECT 1").fetchone()[0] == 1
        finally:
            outer.close()

    def test_a_nested_connect_over_uncommitted_writes_is_refused(self, dialect):
        # Joined, the inner's rollback would discard the outer's INSERT and its commit would
        # publish it; refused, the outer's own decision stands.
        outer = dialect.connect()
        try:
            outer.execute("INSERT INTO t VALUES (1)")
            with pytest.raises(RuntimeError, match="uncommitted writes"):
                dialect.connect()
            outer.commit()
            # The store nests reads after its commit; those stay allowed.
            with dialect.connect() as inner:
                inner.execute("SELECT 1").fetchone()
                inner.rollback()
            outer.execute("INSERT INTO t VALUES (2)")
            outer.rollback()
        finally:
            outer.close()
        assert _count(dialect) == 1

    def test_a_nested_connect_over_a_held_advisory_lock_is_refused(self, dialect, postgres_dsn):
        # Joined, the inner's commit or rollback would release the lock while the outer goes
        # on as if it still serialized writers on the key.
        outer = dialect.connect()
        try:
            dialect.begin_write(outer, "a1")
            with pytest.raises(RuntimeError, match="transaction-scoped lock"):
                dialect.connect()
            assert _lock_is_free(postgres_dsn, "a1") is False
            outer.rollback()
            assert _lock_is_free(postgres_dsn, "a1") is True
            with dialect.connect() as inner:
                assert inner.execute("SELECT 1").fetchone()[0] == 1
        finally:
            outer.close()

    def test_a_nested_connect_over_a_row_lock_is_refused(self, dialect):
        outer = dialect.connect()
        try:
            outer.execute("SELECT x FROM t FOR UPDATE").fetchall()
            with pytest.raises(RuntimeError, match="transaction-scoped lock"):
                dialect.connect()
        finally:
            outer.rollback()
            outer.close()

    def test_a_nested_connect_over_a_write_in_a_cte_is_refused(self, dialect):
        # Postgres tags the statement SELECT; joined, the inner's rollback would discard the row.
        outer = dialect.connect()
        try:
            cursor = outer.execute(
                "WITH w AS (INSERT INTO t VALUES (1) RETURNING x) SELECT x FROM w"
            )
            assert cursor.statusmessage.startswith("SELECT")
            with pytest.raises(RuntimeError, match="uncommitted writes"):
                dialect.connect()
            outer.commit()
        finally:
            outer.close()
        assert _count(dialect) == 1

    def test_a_closed_wrapper_on_a_live_connection_refuses_use(self, dialect):
        conn = dialect.connect()
        conn.close()
        with pytest.raises(RuntimeError, match="closed"):
            conn.execute("SELECT 1")

    def test_schema_questions_ask_the_current_schema_only(self, dialect):
        # A second copy of a table on the search path must not make a migration skip its
        # ALTER, or stamp a version the current schema's table does not have.
        with dialect.connect() as conn:
            conn.executescript(
                "DROP SCHEMA IF EXISTS other_ledger CASCADE; CREATE SCHEMA other_ledger; "
                "CREATE TABLE other_ledger.elsewhere (only_there INT); "
                "CREATE TABLE other_ledger.t (x INT, only_there INT)"
            )
            conn.commit()
            try:
                conn.execute("SET LOCAL search_path TO public, other_ledger")
                assert conn.execute("SELECT current_schema()").fetchone()[0] == "public"
                assert dialect.schema_exists(conn, "elsewhere") is False
                assert dialect.column_exists(conn, "elsewhere", "only_there") is False
                assert dialect.column_exists(conn, "t", "only_there") is False
                assert dialect.schema_exists(conn, "t") is True
                assert dialect.column_exists(conn, "t", "x") is True
            finally:
                conn.rollback()
                conn.executescript("DROP SCHEMA other_ledger CASCADE")
                conn.commit()

    def test_a_nested_connect_after_reads_shares_the_connection(self, dialect):
        outer = dialect.connect()
        try:
            outer.execute("SELECT 1").fetchone()
            inner = dialect.connect()
            try:
                assert inner._inner is outer._inner
            finally:
                inner.close()
            # The inner's close keeps the outer's connection checked out.
            assert dialect._local.lease.depth == 1
        finally:
            outer.close()
        assert dialect._local.lease is None

    def test_a_close_from_another_thread_returns_the_connection(self, postgres_dsn, monkeypatch):
        # Released through the record captured at connect, not the closing thread's state.
        monkeypatch.setattr(sql_backend, "_POOL_TIMEOUT_SECONDS", 2.0)
        dialect = PostgresDialect(postgres_dsn, max_size=1)
        try:
            box: dict[str, object] = {}
            thread = threading.Thread(target=lambda: box.__setitem__("conn", dialect.connect()))
            thread.start()
            thread.join()
            box["conn"].close()
            with dialect.connect() as conn:
                assert conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            dialect.close()

    def test_a_connection_the_server_dropped_is_not_handed_out(self, dialect, postgres_dsn):
        # After a restart or failover every idle pooled connection is dead; the pool checks
        # before handing one out, so no store call fails for it.
        with dialect.connect() as conn:
            pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        with psycopg.connect(postgres_dsn, autocommit=True) as killer:
            assert killer.execute("SELECT pg_terminate_backend(%s)", (pid,)).fetchone()[0]
        with dialect.connect() as conn:
            assert conn.execute("SELECT pg_backend_pid()").fetchone()[0] != pid


def _unlocked_ddl(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Each CREATE or ALTER a SQLite connection runs outside a ``BEGIN IMMEDIATE``."""
    unlocked: list[str] = []
    real = SqliteDialect.connect

    def traced(self: SqliteDialect):
        conn = real(self)
        locked = [False]

        def seen(statement: str) -> None:
            # Past a leading comment, which a script's statement keeps (a header's).
            lines = statement.strip().splitlines()
            head = "\n".join(line for line in lines if not line.lstrip().startswith("--"))
            head = head.lstrip().upper()
            if head.startswith("BEGIN IMMEDIATE"):
                locked[0] = True
            elif head.startswith(("COMMIT", "ROLLBACK")):
                locked[0] = False
            elif head.startswith(("CREATE", "ALTER")) and not locked[0]:
                unlocked.append(" ".join(head.split()[:6]))

        conn.set_trace_callback(seen)
        return conn

    monkeypatch.setattr(SqliteDialect, "connect", traced)
    return unlocked


def test_a_new_sqlite_ledger_s_schema_is_made_under_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two processes opening a new ledger at once: DDL that runs unlocked can have the
    other's land between its prepare and its step, and fail with "database schema has
    changed" (seen on a two-rank torch run)."""
    unlocked = _unlocked_ddl(monkeypatch)
    for _ in range(2):  # the first open creates, the second finds
        store = ArtifactStore(tmp_path / "ledger")
        BuildStore(store.db_path, dialect=store.dialect)
        create_tables(
            store.dialect, "consumer_rows", "CREATE TABLE IF NOT EXISTS consumer_rows (id TEXT);"
        )
    assert unlocked == []


def _rolled_back_to_the_oldest_shape(db_path: Path) -> None:
    """A ledger as the oldest vestigo left it: no tenant columns or key, no lease, pull or
    log columns on builds, no audit source column, no version stamped."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            DROP INDEX idx_tenant_provenance_unique;
            DROP INDEX idx_versions_tenant;
            DROP INDEX idx_build_lease_expires;
            ALTER TABLE artifact_versions DROP COLUMN tenant;
            ALTER TABLE artifact_versions DROP COLUMN principal;
            ALTER TABLE artifact_builds DROP COLUMN lease_owner;
            ALTER TABLE artifact_builds DROP COLUMN lease_expires_at;
            ALTER TABLE artifact_builds DROP COLUMN input_uris;
            ALTER TABLE artifact_builds DROP COLUMN params;
            ALTER TABLE artifact_builds DROP COLUMN name;
            ALTER TABLE artifact_builds DROP COLUMN logs;
            ALTER TABLE registry_audit DROP COLUMN from_artifact_id;
            DROP TABLE artifact_names;
            CREATE TABLE artifact_names (
                name TEXT PRIMARY KEY, artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL, updated_at REAL NOT NULL
            );
            DELETE FROM schema_version;
        """)
    finally:
        conn.close()


def test_an_old_sqlite_ledger_is_upgraded_under_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Two ranks opening a ledger an older vestigo wrote both find a column missing: one
    that adds it unlocked fails the other ("duplicate column name")."""
    store = ArtifactStore(tmp_path / "ledger")
    BuildStore(store.db_path, dialect=store.dialect)
    _rolled_back_to_the_oldest_shape(store.db_path)
    unlocked = _unlocked_ddl(monkeypatch)

    store = ArtifactStore(tmp_path / "ledger")
    BuildStore(store.db_path, dialect=store.dialect)

    assert unlocked == []
    conn = store.dialect.connect()
    try:
        assert store.dialect.column_exists(conn, "artifact_versions", "tenant")
        assert store.dialect.column_exists(conn, "artifact_names", "tenant")
        assert store.dialect.column_exists(conn, "artifact_builds", "logs")
    finally:
        conn.close()


def test_sqlite_ddl_runs_in_the_open_transaction_statement_by_statement(tmp_path: Path):
    """Cut where SQLite says a statement ends, not at every ``;``, and never committed:
    a rollback takes it all back."""
    dialect = SqliteDialect(tmp_path / "ddl.sqlite")
    conn = dialect.connect()
    try:
        dialect.begin_write(conn, "__schema__")
        dialect.execute_ddl(
            conn,
            """
            -- a comment's ; is not an end
            CREATE TABLE notes (body TEXT DEFAULT 'a;b');
            CREATE TRIGGER stamp AFTER INSERT ON notes BEGIN
                UPDATE notes SET body = body || ';' WHERE rowid = new.rowid;
            END;
            CREATE TABLE last (x) -- the last statement, its ; left off
            """,
        )
        # A comment after the last ``;``, as ``executescript`` took it.
        dialect.execute_ddl(conn, "CREATE TABLE more (x); -- the end")
        conn.execute("INSERT INTO notes DEFAULT VALUES")
        assert conn.execute("SELECT body FROM notes").fetchone()[0] == "a;b;"
        assert dialect.schema_exists(conn, "last")
        assert dialect.schema_exists(conn, "more")
        conn.rollback()
        assert not dialect.schema_exists(conn, "notes")
        # A trigger never ended would otherwise be dropped without a word.
        with pytest.raises(ValueError, match="incomplete"):
            dialect.execute_ddl(conn, "CREATE TRIGGER half AFTER INSERT ON x BEGIN SELECT 1;")
    finally:
        conn.close()


def test_consumer_tables_made_meanwhile_are_found_under_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Another process creates the table while this one waits for the lock: its DDL, with
    no ``IF NOT EXISTS``, must not run again."""
    dialect = SqliteDialect(tmp_path / "consumer.sqlite")
    ddl = "CREATE TABLE consumer_rows (id TEXT);"
    real = SqliteDialect.schema_exists
    first_looks: list[bool] = []

    def the_other_creates_after_the_first_look(self: SqliteDialect, conn, table: str) -> bool:
        first_looks.append(real(self, conn, table))
        monkeypatch.setattr(SqliteDialect, "schema_exists", real)
        other = sqlite3.connect(self.db_path)
        try:
            other.executescript(ddl)
        finally:
            other.close()
        return first_looks[0]

    statements: list[str] = []
    connect = SqliteDialect.connect

    def traced(self: SqliteDialect):
        conn = connect(self)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(SqliteDialect, "schema_exists", the_other_creates_after_the_first_look)
    monkeypatch.setattr(SqliteDialect, "connect", traced)
    create_tables(dialect, "consumer_rows", ddl)

    assert first_looks == [False]
    assert "BEGIN IMMEDIATE" in statements
    assert not any(s.lstrip().startswith("CREATE") for s in statements)


def test_a_first_boot_that_dies_part_way_leaves_nothing_half_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A boot killed before its last schema commit: the next finds no store and makes the
    whole of it, rather than one stamped current that lacks the rest (the unique index on
    active publications, which only migration 10 adds to an existing store)."""
    real = SqliteDialect.execute_ddl

    def dies_at_the_publication_index(self: SqliteDialect, conn, script: str) -> None:
        if "idx_publications_active" in script:
            raise sqlite3.OperationalError("killed")
        real(self, conn, script)

    monkeypatch.setattr(SqliteDialect, "execute_ddl", dies_at_the_publication_index)
    with pytest.raises(sqlite3.OperationalError, match="killed"):
        ArtifactStore(tmp_path / "ledger")
    monkeypatch.setattr(SqliteDialect, "execute_ddl", real)

    store = ArtifactStore(tmp_path / "ledger")
    conn = store.dialect.connect()
    try:
        index = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'idx_publications_active'"
        ).fetchone()
    finally:
        conn.close()
    assert index is not None


def test_a_current_sqlite_build_store_opens_under_one_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Each lock an open takes waits behind any live writer, a rank releasing a checkpoint;
    a new store gets its lineage indexes with the attempts table, not on every open."""
    store = ArtifactStore(tmp_path / "ledger")
    BuildStore(store.db_path, dialect=store.dialect)
    conn = sqlite3.connect(store.db_path)
    try:
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert {"idx_build_executor", "idx_attempt_resumed_from"} <= indexes
    statements: list[str] = []
    connect = SqliteDialect.connect

    def traced(self: SqliteDialect):
        conn = connect(self)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(SqliteDialect, "connect", traced)
    BuildStore(store.db_path, dialect=store.dialect)

    assert [s for s in statements if s.startswith("BEGIN IMMEDIATE")] == ["BEGIN IMMEDIATE"]


class _MadeElsewhere(sqlite3.Connection):
    """A connection to a file another process is still making: its switch to WAL is
    refused at once, as SQLite refuses it then, without the busy handler."""

    refusals = 0
    message = "database is locked"

    def execute(self, sql, *args):  # ty: ignore[invalid-method-override]
        if sql == "PRAGMA journal_mode=WAL" and _MadeElsewhere.refusals:
            _MadeElsewhere.refusals -= 1
            raise sqlite3.OperationalError(_MadeElsewhere.message)
        return super().execute(sql, *args)


@pytest.fixture
def made_elsewhere(monkeypatch: pytest.MonkeyPatch) -> type[_MadeElsewhere]:
    connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **k: connect(*a, factory=_MadeElsewhere, **k)
    )
    monkeypatch.setattr(_MadeElsewhere, "refusals", 0)
    monkeypatch.setattr(_MadeElsewhere, "message", "database is locked")
    return _MadeElsewhere


@pytest.mark.parametrize("opener", ["artifact_store", "build_store", "create_tables"])
def test_every_sqlite_open_waits_out_another_process_making_the_file(
    tmp_path: Path, made_elsewhere: type[_MadeElsewhere], opener: str
):
    """Ranks opening a new ledger together: a BuildStore or create_tables that opened the
    file first failed at the WAL switch, where only ArtifactStore retried."""
    made_elsewhere.refusals = 3
    if opener == "artifact_store":
        ArtifactStore(tmp_path / "ledger").close()
    elif opener == "build_store":
        BuildStore(tmp_path / "ledger.sqlite")
    else:
        create_tables(SqliteDialect(tmp_path / "ledger.sqlite"), "t", "CREATE TABLE t (x);")
    assert made_elsewhere.refusals == 0


def test_a_sqlite_open_gives_up_on_a_lock_that_outlasts_the_wait_and_on_any_other_error(
    tmp_path: Path, made_elsewhere: type[_MadeElsewhere], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sql_backend, "_WAL_RETRY_SECONDS", 0.05)
    made_elsewhere.refusals = 10_000
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        SqliteDialect(tmp_path / "ledger.sqlite").connect()
    assert made_elsewhere.refusals > 0

    made_elsewhere.refusals, made_elsewhere.message = 2, "disk I/O error"
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        SqliteDialect(tmp_path / "ledger.sqlite").connect()
    assert made_elsewhere.refusals == 1


def _schema_names(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped", "stamped"])
def test_a_ledger_a_killed_0_4_first_boot_left_part_made_gets_the_rest(
    tmp_path: Path, stamped: bool
):
    """0.4.0 ran its first script statement by statement. One killed inside it left the
    versions without their later indexes and without the name pointers, and no open made
    them, whether it found the store unstamped or another 0.4.0 boot had stamped it."""
    store = ArtifactStore(tmp_path / "ledger")
    store.close()
    conn = sqlite3.connect(store.db_path)
    try:
        conn.executescript(
            "DROP INDEX idx_provenance; DROP INDEX idx_state; DROP INDEX idx_versions_tenant;"
            "DROP TABLE artifact_names;" + ("" if stamped else "DROP TABLE schema_version;")
        )
    finally:
        conn.close()

    store = ArtifactStore(tmp_path / "ledger")
    try:
        assert _FIRST_BOOT_OBJECTS <= _schema_names(store.db_path)
        version = store.create_artifact("x", "p")
        store.write_blob("x", version, b"x")
        store.finalize_artifact("x", version, "{}", None, 1)
        store.set_name("latest", "x", version)
        resolved = store.resolve_name("latest")
        assert resolved is not None and resolved.version == version
    finally:
        store.close()


def test_a_read_only_open_refuses_a_ledger_stamped_current_but_part_made(tmp_path: Path):
    """It would pass the stamp and then fail at a name lookup, "no such table"."""
    ArtifactStore(tmp_path / "ledger").close()
    conn = sqlite3.connect(tmp_path / "ledger" / "artifacts.sqlite")
    try:
        conn.execute("DROP TABLE artifact_names")
    finally:
        conn.close()

    with pytest.raises(StoreSchemaMismatch, match="artifact_names"):
        ArtifactStore(tmp_path / "ledger", read_only=True)
    ArtifactStore(tmp_path / "ledger").close()
    ArtifactStore(tmp_path / "ledger", read_only=True).close()


def test_a_ledger_a_killed_0_4_first_boot_stamped_without_its_publication_index_gets_it(
    tmp_path: Path,
):
    """0.4.0 stamped a new store current before it made the publication tables, so one
    killed between never got migration 10's index; duplicates live meanwhile are revoked."""
    store = ArtifactStore(tmp_path / "ledger")
    store.close()
    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute("DROP INDEX idx_publications_active")
        for token, at in (("a" * 64, 1.0), ("b" * 64, 2.0)):
            conn.execute(
                "INSERT INTO artifact_publications (token, artifact_id, version, published_at) "
                "VALUES (?, 'x', 1, ?)",
                (token, at),
            )
        conn.commit()
    finally:
        conn.close()

    ArtifactStore(tmp_path / "ledger").close()

    assert "idx_publications_active" in _schema_names(store.db_path)
    conn = sqlite3.connect(store.db_path)
    try:
        live = conn.execute(
            "SELECT token FROM artifact_publications WHERE revoked_at IS NULL"
        ).fetchall()
        withdrawn = conn.execute(
            "SELECT value FROM registry_audit WHERE action = 'withdraw'"
        ).fetchall()
    finally:
        conn.close()
    assert live == [("a" * 64,)]
    assert withdrawn == [("b" * 64,)]


def test_one_returns_the_row_and_refuses_none(tmp_path: Path):
    """A query whose row the caller's lock or lease guarantees: its absence is named,
    not a TypeError from subscripting None."""
    conn = SqliteDialect(tmp_path / "db.sqlite").connect()
    try:
        assert one(conn.execute("SELECT 7 AS n"))["n"] == 7
        conn.execute("CREATE TABLE t (x)")
        with pytest.raises(LookupError, match="returned none"):
            one(conn.execute("SELECT x FROM t"))
    finally:
        conn.close()
