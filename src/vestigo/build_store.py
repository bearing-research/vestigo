"""Build state store for server-orchestrated transforms, in the artifact store's database.

States: ``pending`` -> ``building`` (under a runner's lease) -> ``ready`` or ``failed``.

Every writer of a build's state, owner or attempt runs under the build's write key
(``begin_write``), so none lands between a release's fence check and its commit;
``renew_lease``, which only moves the deadline, does not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vestigo.sql_backend import SqlDialect, SqliteDialect, StoreConnection

_BUILD_SCHEMA_SQL = """
-- Build state: tracks async build lifecycle
CREATE TABLE IF NOT EXISTS artifact_builds (
    build_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    executor_ref TEXT NOT NULL,
    executor_url TEXT,  -- Resolved executor URL from registry
    tenant_id TEXT,  -- For tenant-based access control
    principal_id TEXT,  -- Who initiated the build
    created_at REAL NOT NULL,
    started_at REAL,  -- When execution started
    completed_at REAL,  -- When execution finished
    error_message TEXT,  -- Error details if failed
    error_code TEXT,  -- Error code for programmatic handling
    input_byte_count INTEGER,  -- Total input size
    output_byte_count INTEGER,  -- Total output size
    lease_owner TEXT,  -- Unique identifier of the runner node holding the lease
    lease_expires_at REAL,  -- Unix timestamp when the lease expires
    input_uris TEXT,  -- JSON array of input artifact URIs (for pull model)
    params TEXT,  -- JSON object of transform params (for pull model)
    name TEXT,  -- Optional name pointer to set after completion
    logs TEXT,  -- Executor stderr/stdout logs (text blob for debugging)
    occurrence TEXT,  -- Which instance of a repeated site this is (a sweep's variant)
    attempt TEXT,  -- The attempt holding the lease; a release from any other is rejected
    computation_key TEXT,  -- The bound computation this build runs (a resume finds it by this)
    -- 'runner' is offered to the consumer's runner; 'external' the consumer runs itself
    dispatch TEXT NOT NULL DEFAULT 'runner',
    FOREIGN KEY (artifact_id, version) REFERENCES artifact_versions(id, version)
);

-- Index for finding the open build of a computation (a resume)
CREATE INDEX IF NOT EXISTS idx_build_computation ON artifact_builds(computation_key);

-- Index for state queries (e.g., find pending builds)
CREATE INDEX IF NOT EXISTS idx_build_state ON artifact_builds(state);

-- Index for tenant queries
CREATE INDEX IF NOT EXISTS idx_build_tenant ON artifact_builds(tenant_id);

-- Index for artifact lookup (find builds for an artifact)
CREATE INDEX IF NOT EXISTS idx_build_artifact ON artifact_builds(artifact_id, version);

-- Index for expired lease queries (orphan recovery)
CREATE INDEX IF NOT EXISTS idx_build_lease_expires ON artifact_builds(state, lease_expires_at);
"""

# Every blob key a build attempt may write. Blob stores cannot list keys, so
# this is the only way to find an attempt that is never promoted (lost its lease,
# or uploaded and never finalized). Kept apart from artifact_builds so existing
# databases gain it too.
_ATTEMPT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS build_attempts (
    build_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    attempt TEXT NOT NULL,
    writable_until REAL NOT NULL,  -- Nothing can write the attempt's key after this
    resumed_from TEXT,  -- The accepted release this attempt continues from, if any
    principal TEXT,  -- Who runs this attempt (a resume may be another's than the build's)
    opened_at REAL,  -- When the attempt was first recorded
    PRIMARY KEY (build_id, attempt)
);
"""

# What the lineage lookups read: an executor's builds, the attempts resumed from a release.
# Created once the columns they index exist, so apart from the tables.
_LINEAGE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_build_executor ON artifact_builds(executor_ref, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_attempt_resumed_from ON build_attempts(resumed_from)",
)

# Console chunks a worker posted to a node other than the one that dispatched the
# build; that node polls them out. Rows live only while the build runs.
_CONSOLE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS build_console_chunks (
    build_id TEXT NOT NULL,
    stream TEXT NOT NULL,
    seq INTEGER NOT NULL,
    chunk TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (build_id, stream, seq)
);
"""


@dataclass
class BuildState:
    """One build row. Timestamps are Unix seconds; ``input_uris`` and ``params`` serve pull mode."""

    build_id: str
    artifact_id: str
    version: int
    state: str
    executor_ref: str
    executor_url: str | None
    tenant_id: str | None
    principal_id: str | None
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    error_code: str | None = None
    input_byte_count: int | None = None
    output_byte_count: int | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    input_uris: list[str] | None = None
    params: dict | None = None
    name: str | None = None
    logs: str | None = None
    occurrence: str | None = None
    attempt: str | None = None
    computation_key: str | None = None
    dispatch: str = "runner"

    # Used by auth checks.
    @property
    def tenant(self) -> str | None:
        """Alias for tenant_id."""
        return self.tenant_id

    def to_dict(self) -> dict:
        """Return the API response shape."""
        return {
            "build_id": self.build_id,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "state": self.state,
            "executor_ref": self.executor_ref,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "input_byte_count": self.input_byte_count,
            "output_byte_count": self.output_byte_count,
        }


@dataclass
class AttemptState:
    """One ``build_attempts`` row: a try at writing the build's bytes."""

    build_id: str
    attempt: str
    artifact_id: str
    version: int
    writable_until: float
    resumed_from: str | None = None
    principal: str | None = None
    opened_at: float | None = None


_ATTEMPT_COLUMNS = (
    "build_id, artifact_id, version, attempt, writable_until, resumed_from, principal, opened_at"
)


def _row_to_attempt(row: Any) -> AttemptState:
    return AttemptState(
        build_id=row["build_id"],
        attempt=row["attempt"],
        artifact_id=row["artifact_id"],
        version=row["version"],
        writable_until=row["writable_until"],
        resumed_from=row["resumed_from"],
        principal=row["principal"],
        opened_at=row["opened_at"],
    )


# The SELECT list of a BuildState; a new column is added here and in the dataclass.
_BUILD_FIELDS = (
    "build_id",
    "artifact_id",
    "version",
    "state",
    "executor_ref",
    "executor_url",
    "tenant_id",
    "principal_id",
    "created_at",
    "started_at",
    "completed_at",
    "error_message",
    "error_code",
    "input_byte_count",
    "output_byte_count",
    "lease_owner",
    "lease_expires_at",
    "input_uris",
    "params",
    "name",
    "logs",
    "occurrence",
    "attempt",
    "computation_key",
    "dispatch",
)
_BUILD_COLUMNS = ", ".join(_BUILD_FIELDS)


def _row_to_build_state(row: Any) -> BuildState:
    """A ``BuildState`` from a row holding ``_BUILD_FIELDS``, by name."""
    values = {field: row[field] for field in _BUILD_FIELDS}
    values["input_uris"] = json.loads(row["input_uris"]) if row["input_uris"] else None
    values["params"] = json.loads(row["params"]) if row["params"] else None
    return BuildState(**values)


# What a build's ``dispatch`` may be.
DISPATCHES = ("runner", "external")


def _dispatch_filter(dispatch: str | None) -> tuple[str, tuple[str, ...]]:
    """The ``AND dispatch = ?`` clause and its parameter, or nothing for any dispatch."""
    if dispatch is None:
        return "", ()
    if dispatch not in DISPATCHES:
        raise ValueError(f"dispatch must be None or one of {DISPATCHES}, not {dispatch!r}")
    return " AND dispatch = ?", (dispatch,)


# Before the ``dispatch`` column, a consumer marked a build its own worker runs with
# this key in ``params``; the column is backfilled from it once, as a ``LIKE``
# literal (a bare ``_`` matches any character).
_EXTERNAL_DISPATCH_LIKE = json.dumps({"_dispatch_mode": "external"})[1:-1].replace("_", "\\_")


class BuildStore:
    """Build state store sharing the artifact store's dialect and database.

    Thread-safe: a connection per operation. Sharing the database lets a build
    claimed on one node be seen from another when metadata is on Postgres.
    """

    def __init__(
        self,
        db_path: Path,
        dialect: SqlDialect | None = None,
        clock: Callable[[], float] = time.time,
    ):
        """Open the store and create its schema.

        Args:
            db_path: SQLite database; ignored when ``dialect`` is given.
            dialect: Metadata backend; pass the artifact store's to share its database.
            clock: Wall-clock source in seconds, injectable so tests can expire a
                lease without sleeping.
        """
        self.db_path = db_path
        self._dialect: SqlDialect = dialect if dialect is not None else SqliteDialect(db_path)
        self._clock = clock
        self._init_schema()

    def _get_connection(self) -> StoreConnection:
        """Open a connection configured by the active dialect."""
        return self._dialect.connect()

    def _init_schema(self) -> None:
        """Create the build tables; on SQLite, add columns missing from older databases."""
        conn = self._get_connection()
        try:
            # Only SQLite has older databases to migrate. Other backends create the schema
            # under the schema lock: CREATE TABLE IF NOT EXISTS races in Postgres, and every
            # node runs this at startup.
            if not self._dialect.supports_legacy_migration:
                if not self._dialect.schema_exists(conn, "artifact_builds"):
                    self._dialect.begin_write(conn, "__build_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_BUILD_SCHEMA_SQL))
                    conn.commit()
                if not self._dialect.schema_exists(conn, "build_attempts"):
                    self._dialect.begin_write(conn, "__build_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_ATTEMPT_SCHEMA_SQL))
                    for statement in _LINEAGE_INDEXES:
                        conn.execute(statement)
                    conn.commit()
                if not self._dialect.schema_exists(conn, "build_console_chunks"):
                    self._dialect.begin_write(conn, "__build_schema__")
                    conn.executescript(self._dialect.adapt_ddl(_CONSOLE_SCHEMA_SQL))
                    conn.commit()
                self._add_release_columns(conn)
                return

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_builds'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                cursor = conn.execute("PRAGMA table_info(artifact_builds)")
                columns = {row["name"] for row in cursor.fetchall()}

                if "lease_owner" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_owner TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN lease_expires_at REAL")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_build_lease_expires "
                        "ON artifact_builds(state, lease_expires_at)"
                    )
                    conn.commit()

                # Pull-model columns.
                if "input_uris" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN input_uris TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN params TEXT")
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN name TEXT")
                    conn.commit()

                if "logs" not in columns:
                    conn.execute("ALTER TABLE artifact_builds ADD COLUMN logs TEXT")
                    conn.commit()
            else:
                conn.executescript(_BUILD_SCHEMA_SQL)
                conn.commit()

            conn.executescript(_ATTEMPT_SCHEMA_SQL)
            conn.executescript(_CONSOLE_SCHEMA_SQL)
            conn.commit()
            self._add_release_columns(conn)
            for statement in _LINEAGE_INDEXES:
                conn.execute(statement)
            conn.commit()
        finally:
            conn.close()

    def _add_release_columns(self, conn: StoreConnection) -> None:
        """The columns added since the first schema, on a database from before them."""
        columns = {
            ("artifact_builds", "occurrence"): "TEXT",
            ("artifact_builds", "attempt"): "TEXT",
            ("artifact_builds", "computation_key"): "TEXT",
            ("build_attempts", "resumed_from"): "TEXT",
            ("artifact_builds", "dispatch"): "TEXT NOT NULL DEFAULT 'runner'",
            ("build_attempts", "principal"): "TEXT",
            # REAL is single precision on Postgres: seconds would round to minutes.
            ("build_attempts", "opened_at"): self._dialect.float_type,
        }
        if all(self._dialect.column_exists(conn, table, column) for table, column in columns):
            return
        self._dialect.begin_write(conn, "__build_schema__")
        # Re-checked under the lock: another node starting at once may have added them.
        for (table, column), declaration in columns.items():
            if not self._dialect.column_exists(conn, table, column):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
                if column == "dispatch":
                    conn.execute(
                        "UPDATE artifact_builds SET dispatch = 'external' "
                        "WHERE params LIKE ? ESCAPE '\\'",
                        (f"%{_EXTERNAL_DISPATCH_LIKE}%",),
                    )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_build_computation ON artifact_builds(computation_key)"
        )
        for statement in _LINEAGE_INDEXES:
            conn.execute(statement)
        conn.commit()

    def create_build(
        self,
        build_id: str,
        artifact_id: str,
        version: int,
        executor_ref: str,
        executor_url: str | None = None,
        tenant_id: str | None = None,
        principal_id: str | None = None,
        input_uris: list[str] | None = None,
        params: dict | None = None,
        name: str | None = None,
        occurrence: str | None = None,
        computation_key: str | None = None,
        dispatch: str = "runner",
    ) -> BuildState:
        """Create a build record in ``pending`` state.

        ``name`` is a name pointer to set once the build completes. ``occurrence``
        names which instance of a repeated site this is (a sweep's variant), so two
        builds of one structure at one site stay distinct executions.
        ``computation_key`` is the bound computation the build runs; with the
        occurrence, it is what :meth:`find_open_build` looks a resume up by.
        ``dispatch`` says who runs it: ``"runner"`` offers it through
        :meth:`list_runnable_builds`; ``"external"`` is a build the consumer runs
        itself, which nothing here picks up.

        Raises:
            ValueError: If ``dispatch`` is not one of :data:`DISPATCHES`.
        """
        if dispatch not in DISPATCHES:
            raise ValueError(f"dispatch must be one of {DISPATCHES}, not {dispatch!r}")
        conn = self._get_connection()
        try:
            created_at = self._clock()
            input_uris_json = json.dumps(input_uris) if input_uris else None
            params_json = json.dumps(params) if params else None

            conn.execute(
                """
                INSERT INTO artifact_builds
                    (build_id, artifact_id, version, state, executor_ref,
                     executor_url, tenant_id, principal_id, created_at,
                     input_uris, params, name, occurrence, computation_key, dispatch)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    build_id,
                    artifact_id,
                    version,
                    executor_ref,
                    executor_url,
                    tenant_id,
                    principal_id,
                    created_at,
                    input_uris_json,
                    params_json,
                    name,
                    occurrence,
                    computation_key,
                    dispatch,
                ),
            )
            conn.commit()

            return BuildState(
                build_id=build_id,
                artifact_id=artifact_id,
                version=version,
                state="pending",
                executor_ref=executor_ref,
                executor_url=executor_url,
                tenant_id=tenant_id,
                principal_id=principal_id,
                created_at=created_at,
                input_uris=input_uris,
                params=params,
                name=name,
                occurrence=occurrence,
                computation_key=computation_key,
                dispatch=dispatch,
            )
        finally:
            conn.close()

    def open_attempt(
        self,
        build_id: str,
        attempt: str,
        *,
        writable_until: float,
        lease_owner: str | None = None,
        lease_duration_seconds: float = 60.0,
        resumed_from: str | None = None,
        principal: str | None = None,
    ) -> bool:
        """Make *attempt* the build's current attempt, fencing every earlier one.

        A pending build starts; a building one continues in *attempt*, and every
        other attempt's unsettled release is abandoned. :meth:`ReleaseStore.resume
        <vestigo.releases.ReleaseStore.resume>` is the resume: it also checks that
        ``resumed_from`` is an accepted release of the build. The attempt's write
        window is recorded as ``record_attempt`` does, with ``principal`` as who runs
        it, which the build's ``principal_id`` need not be.

        A new attempt takes the lease as ``lease_owner`` (None when not given) for
        ``lease_duration_seconds``. The same attempt re-opened, or a claim's owner
        opening its first, keeps the owner when none is given and the later of the
        two deadlines; a re-opened attempt takes ``resumed_from`` when one is given.
        Returns False when the build is not active, its lease is live under another
        attempt, or *attempt* is the build's and its lease has lapsed (it continues
        only in a new attempt); a claim's live lease (no attempt yet) lets only its
        own ``lease_owner`` through.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            opened = self.open_attempt_within(
                conn,
                build_id,
                attempt,
                writable_until=writable_until,
                lease_owner=lease_owner,
                lease_expires_at=now + lease_duration_seconds,
                now=now,
                resumed_from=resumed_from,
                principal=principal,
            )
            conn.commit()
            return opened
        finally:
            conn.close()

    def open_attempt_within(
        self,
        conn: StoreConnection,
        build_id: str,
        attempt: str,
        *,
        writable_until: float,
        lease_owner: str | None,
        lease_expires_at: float,
        now: float,
        resumed_from: str | None = None,
        principal: str | None = None,
    ) -> bool:
        """As :meth:`open_attempt`, on the caller's transaction, which holds the build's key."""
        row = conn.execute(
            "SELECT artifact_id, version, state, attempt, lease_owner, lease_expires_at "
            "FROM artifact_builds WHERE build_id = ?",
            (build_id,),
        ).fetchone()
        if row is None or row["state"] not in ("pending", "building"):
            return False
        live = row["lease_expires_at"] is not None and row["lease_expires_at"] >= now
        own_claim = (
            row["attempt"] is None and lease_owner is not None and lease_owner == row["lease_owner"]
        )
        continuing = row["attempt"] == attempt or own_claim
        if live and not continuing:
            return False
        # A lapsed attempt was fenced, so a release it sent meanwhile was rejected.
        if row["attempt"] == attempt and not live:
            return False
        if continuing:
            if lease_owner is None:
                lease_owner = row["lease_owner"]
            if row["lease_expires_at"] is not None:
                lease_expires_at = max(lease_expires_at, row["lease_expires_at"])
        conn.execute(
            """
            UPDATE artifact_builds
            SET state = 'building', attempt = ?,
                started_at = COALESCE(started_at, ?),
                lease_owner = ?,
                lease_expires_at = ?
            WHERE build_id = ?
            """,
            (attempt, now, lease_owner, lease_expires_at, build_id),
        )
        # Only a release's own attempt may settle it, so a fenced attempt's would
        # block every later release of its kind for good.
        conn.execute(
            "UPDATE build_releases SET state = 'abandoned', settled_at = ? "
            "WHERE build_id = ? AND attempt <> ? AND state IN ('preparing', 'prepared')",
            (now, build_id, attempt),
        )
        conn.execute(
            """
            INSERT INTO build_attempts
                (build_id, artifact_id, version, attempt, writable_until, resumed_from,
                 principal, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(build_id, attempt) DO UPDATE SET
                writable_until = CASE
                    WHEN excluded.writable_until > build_attempts.writable_until
                    THEN excluded.writable_until
                    ELSE build_attempts.writable_until
                END,
                resumed_from = COALESCE(excluded.resumed_from, build_attempts.resumed_from),
                principal = COALESCE(excluded.principal, build_attempts.principal)
            """,
            (
                build_id,
                row["artifact_id"],
                row["version"],
                attempt,
                writable_until,
                resumed_from,
                principal,
                now,
            ),
        )
        return True

    def get_attempt(self, build_id: str, attempt: str) -> AttemptState | None:
        """One attempt's record, or None."""
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"SELECT {_ATTEMPT_COLUMNS} FROM build_attempts WHERE build_id = ? AND attempt = ?",
                (build_id, attempt),
            ).fetchone()
            return None if row is None else _row_to_attempt(row)
        finally:
            conn.close()

    def attempts_of(self, build_id: str) -> list[AttemptState]:
        """The build's attempts in the order they opened."""
        return self._attempts_where("build_id = ?", (build_id,))

    def attempts_resumed_from(self, release_id: str) -> list[AttemptState]:
        """The attempts that continued from *release_id*, in the order they opened."""
        return self._attempts_where("resumed_from = ?", (release_id,))

    def _attempts_where(self, clause: str, params: tuple[Any, ...]) -> list[AttemptState]:
        conn = self._get_connection()
        try:
            rows = conn.execute(
                f"SELECT {_ATTEMPT_COLUMNS} FROM build_attempts WHERE {clause} "
                # Rows from before opened_at came first; among them, by window.
                "ORDER BY COALESCE(opened_at, 0), writable_until, attempt",
                params,
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_attempt(row) for row in rows]

    def find_open_build(
        self,
        computation_key: str,
        *,
        occurrence: str | None = None,
        tenant_id: str | None = None,
    ) -> BuildState | None:
        """The newest ``building`` build of a computation at an occurrence, or None.

        What a run of the same computation resumes, whether or not its lease has
        lapsed: the caller reads ``lease_expires_at`` to tell a crashed run from one
        still going.
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                f"""
                SELECT {_BUILD_COLUMNS}
                FROM artifact_builds
                WHERE computation_key = ? AND state = 'building'
                  AND COALESCE(occurrence, '') = COALESCE(?, '')
                  AND COALESCE(tenant_id, '') = COALESCE(?, '')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (computation_key, occurrence, tenant_id),
            ).fetchone()
            return None if row is None else _row_to_build_state(row)
        finally:
            conn.close()

    def complete_attempt_within(
        self,
        conn: StoreConnection,
        build_id: str,
        attempt: str,
        *,
        artifact_id: str | None,
        version: int | None,
        now: float,
    ) -> bool:
        """Complete the build on the caller's transaction, fenced on *attempt*.

        The build is repointed at ``artifact_id``/``version`` when given (a release's
        first output). Returns True if *attempt* was current and the build is now ready.
        """
        cursor = conn.execute(
            """
            UPDATE artifact_builds
            SET state = 'ready', completed_at = ?,
                artifact_id = COALESCE(?, artifact_id), version = COALESCE(?, version)
            WHERE build_id = ? AND state = 'building' AND attempt = ?
            """,
            (now, artifact_id, version, build_id, attempt),
        )
        return cursor.rowcount > 0

    def get_build(self, build_id: str) -> BuildState | None:
        """Get a build by ID, or None if not found."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"""
                SELECT {_BUILD_COLUMNS}
                FROM artifact_builds
                WHERE build_id = ?
                """,
                (build_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return _row_to_build_state(row)
        finally:
            conn.close()

    def update_build_output(self, build_id: str, artifact_id: str, version: int) -> bool:
        """Repoint a build at the existing artifact finalization deduplicated it to."""
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET artifact_id = ?, version = ?
                WHERE build_id = ?
                """,
                (artifact_id, version, build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def start_build(self, build_id: str) -> bool:
        """Move a build from pending to building with neither a lease nor an attempt.

        For a build its creator runs itself: no fence applies to it, so only
        ``complete_build`` and ``fail_build`` without an owner end it, and it has no
        releases. A runner claims with ``claim_build``; a release's attempt opens
        with ``open_attempt``. Returns False if not found or not pending.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'building', started_at = ?
                WHERE build_id = ? AND state = 'pending'
                """,
                (self._clock(), build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def claim_build(
        self,
        build_id: str,
        lease_owner: str,
        lease_duration_seconds: float = 60.0,
    ) -> bool:
        """Claim a pending build under a lease, atomically; False if another runner won.

        Renew with ``renew_lease()`` or another runner may reclaim the build. A claimed
        build has no attempt, so no release of it is accepted, until ``open_attempt``
        with the same ``lease_owner``, which the claim's live lease lets through.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET state = 'building',
                    started_at = ?,
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempt = NULL
                WHERE build_id = ? AND state = 'pending'
                """,
                (now, lease_owner, lease_expires_at, build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def renew_lease(
        self,
        build_id: str,
        lease_owner: str,
        lease_duration_seconds: float = 60.0,
    ) -> bool:
        """Extend a live lease from now.

        Returns False unless ``lease_owner`` holds it, it has not lapsed and the build
        is building. A lapsed lease stays lapsed: its attempt is fenced for good, and
        the build goes on only in a new attempt or a reclaim.
        """
        conn = self._get_connection()
        try:
            now = self._clock()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET lease_expires_at = ?
                WHERE build_id = ?
                  AND state = 'building'
                  AND lease_owner = ?
                  AND (lease_expires_at IS NULL OR lease_expires_at >= ?)
                """,
                (lease_expires_at, build_id, lease_owner, now),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def lapse_lease(self, build_id: str, lease_owner: str) -> bool:
        """End ``lease_owner``'s lease now, for good, from any thread or process.

        Unlike a renewal to zero seconds, a renewal that read the clock first cannot
        revive it: the lease has no owner left to renew. Returns False unless
        ``lease_owner`` holds it and the build is building.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                UPDATE artifact_builds
                SET lease_owner = NULL, lease_expires_at = ?
                WHERE build_id = ? AND state = 'building' AND lease_owner = ?
                """,
                (self._clock(), build_id, lease_owner),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def reclaim_expired_build(
        self,
        build_id: str,
        new_lease_owner: str,
        lease_duration_seconds: float = 60.0,
        *,
        dispatch: str | None = "runner",
    ) -> bool:
        """Take over a building build whose lease has expired (a dead runner's orphan).

        The lapsed attempt is fenced: the build has none until ``open_attempt`` with
        ``new_lease_owner``, so a release the old runner still sends is rejected.
        Only a build of ``dispatch`` is taken (None: any), so a runner's orphan
        recovery leaves alone the builds a consumer runs itself. Returns False if
        the lease has not expired, the build is not building, or is another dispatch's.

        Raises:
            ValueError: If ``dispatch`` is neither None nor one of :data:`DISPATCHES`.
        """
        dispatch_clause, dispatch_params = _dispatch_filter(dispatch)
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            now = self._clock()
            lease_expires_at = now + lease_duration_seconds
            cursor = conn.execute(
                f"""
                UPDATE artifact_builds
                SET lease_owner = ?,
                    lease_expires_at = ?,
                    started_at = ?,
                    attempt = NULL
                WHERE build_id = ?
                  AND state = 'building'
                  AND lease_expires_at < ?{dispatch_clause}
                """,
                (new_lease_owner, lease_expires_at, now, build_id, now, *dispatch_params),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_expired_leases(
        self, limit: int = 10, *, dispatch: str | None = "runner"
    ) -> list[BuildState]:
        """List building builds of ``dispatch`` (None: any) whose lease has expired.

        Likely orphans. By default only the runner's: a consumer's own build whose
        attempt paused past its lease is not a runner's to reclaim.

        Raises:
            ValueError: If ``dispatch`` is neither None nor one of :data:`DISPATCHES`.
        """
        dispatch_clause, dispatch_params = _dispatch_filter(dispatch)
        conn = self._get_connection()
        try:
            now = self._clock()
            cursor = conn.execute(
                f"""
                SELECT {_BUILD_COLUMNS}
                FROM artifact_builds
                WHERE state = 'building'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < ?{dispatch_clause}
                ORDER BY lease_expires_at ASC
                LIMIT ?
                """,
                (now, *dispatch_params, limit),
            )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def complete_build(
        self,
        build_id: str,
        output_byte_count: int | None = None,
        logs: str | None = None,
        lease_owner: str | None = None,
    ) -> bool:
        """Mark a build ready (building -> ready).

        Args:
            lease_owner: When given, apply only if this owner still holds the lease,
                so a runner whose lease was taken over cannot publish over the new owner.

        Returns:
            False if not found, not building, or the lease is held by someone else.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            sql = """
                UPDATE artifact_builds
                SET state = 'ready', completed_at = ?, output_byte_count = ?, logs = ?
                WHERE build_id = ? AND state = 'building'
            """
            params: list[Any] = [self._clock(), output_byte_count, logs, build_id]
            if lease_owner is not None:
                sql += " AND lease_owner = ?"
                params.append(lease_owner)
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def complete_within(
        self,
        conn: Any,
        build_id: str,
        *,
        artifact_id: str,
        version: int,
        lease_owner: str,
        lease_expires_at: float | None = None,
        output_byte_count: int | None = None,
        logs: str | None = None,
    ) -> bool:
        """Complete a build on the caller's connection, without committing.

        Lets ``ArtifactStore.finalize_artifact`` mark the artifact ready and complete
        the build in one transaction. ``lease_expires_at`` pins the exact claim, since
        every executor holds the lease as ``external:manifest``. The build is
        repointed at ``artifact_id``/``version`` (they differ after dedup).

        Returns:
            True if this caller held the lease and the build is now complete.
        """
        sql = """
            UPDATE artifact_builds
            SET state = 'ready', completed_at = ?, output_byte_count = ?, logs = ?,
                artifact_id = ?, version = ?
            WHERE build_id = ? AND state = 'building' AND lease_owner = ?
        """
        params: list[Any] = [
            self._clock(),
            output_byte_count,
            logs,
            artifact_id,
            version,
            build_id,
            lease_owner,
        ]
        if lease_expires_at is not None:
            sql += " AND lease_expires_at = ?"
            params.append(lease_expires_at)
        return conn.execute(sql, params).rowcount > 0

    def record_attempt(
        self,
        build_id: str,
        artifact_id: str,
        version: int,
        attempt: str,
        *,
        writable_until: float,
    ) -> None:
        """Record a blob key a build attempt may write, before anything can write it.

        ``writable_until`` is when the last write capability expires (a signed URL's
        expiry, or now for a runner). Re-recording keeps the later time.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO build_attempts
                    (build_id, artifact_id, version, attempt, writable_until, opened_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(build_id, attempt) DO UPDATE SET writable_until = CASE
                    WHEN excluded.writable_until > build_attempts.writable_until
                    THEN excluded.writable_until
                    ELSE build_attempts.writable_until
                END
                """,
                (build_id, artifact_id, version, attempt, writable_until, self._clock()),
            )
            conn.commit()
        finally:
            conn.close()

    def settled_attempts(self, limit: int = 100) -> list[tuple[str, str, int, str, bool]]:
        """List attempts whose build is over and whose key nothing can still write.

        Each is ``(build_id, artifact_id, version, attempt, promoted)``; only a
        promoted attempt's bytes must be kept.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT a.build_id, a.artifact_id, a.version, a.attempt,
                       CASE WHEN v.blob_attempt = a.attempt THEN 1 ELSE 0 END AS promoted
                FROM build_attempts a
                LEFT JOIN artifact_builds b ON b.build_id = a.build_id
                LEFT JOIN artifact_versions v ON v.id = a.artifact_id AND v.version = a.version
                WHERE a.writable_until < ?
                  AND (b.build_id IS NULL OR b.state NOT IN ('pending', 'building'))
                LIMIT ?
                """,
                (self._clock(), limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            (r["build_id"], r["artifact_id"], r["version"], r["attempt"], bool(r["promoted"]))
            for r in rows
        ]

    def forget_attempt(self, build_id: str, attempt: str) -> None:
        """Drop an attempt from the ledger once its bytes are dealt with."""
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM build_attempts WHERE build_id = ? AND attempt = ?",
                (build_id, attempt),
            )
            conn.commit()
        finally:
            conn.close()

    def append_console_chunk(self, build_id: str, stream: str, seq: int, chunk: str) -> bool:
        """Leave a running build's console chunk for the node that dispatched it.

        Returns False, storing nothing, when the build is not running (a late or
        stale worker) or the chunk is already stored.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO build_console_chunks (build_id, stream, seq, chunk, created_at)
                SELECT ?, ?, ?, ?, ?
                WHERE EXISTS (
                    SELECT 1 FROM artifact_builds
                    WHERE build_id = ? AND state IN ('pending', 'building')
                )
                ON CONFLICT (build_id, stream, seq) DO NOTHING
                """,
                (build_id, stream, seq, chunk, self._clock(), build_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def take_console_chunks(self, build_id: str) -> list[tuple[str, int, str]]:
        """Remove and return a build's stored console chunks as ``(stream, seq, chunk)``."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT stream, seq, chunk FROM build_console_chunks
                WHERE build_id = ? ORDER BY stream, seq
                """,
                (build_id,),
            ).fetchall()
            taken = [(r["stream"], r["seq"], r["chunk"]) for r in rows]
            # Only the rows read: a chunk another node stores between the read and the
            # delete must stay for the next take, or it is lost unseen.
            for stream, seq, _ in taken:
                conn.execute(
                    "DELETE FROM build_console_chunks "
                    "WHERE build_id = ? AND stream = ? AND seq = ?",
                    (build_id, stream, seq),
                )
            conn.commit()
            return taken
        finally:
            conn.close()

    def delete_console_chunks(self, build_id: str) -> None:
        """Drop whatever console a build left in the store."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM build_console_chunks WHERE build_id = ?", (build_id,))
            conn.commit()
        finally:
            conn.close()

    def fail_build(
        self,
        build_id: str,
        error_message: str,
        error_code: str | None = None,
        logs: str | None = None,
        lease_owner: str | None = None,
        lease_expires_at: float | None = None,
    ) -> bool:
        """Mark a build failed (building -> failed).

        Args:
            lease_owner: When given, apply only if this owner still holds the lease,
                as in ``complete_build``.
            lease_expires_at: With ``lease_owner``, also pin the exact claim, as in
                ``complete_within``.

        Returns:
            False if not found, not building, or the lease is held by someone else.
        """
        conn = self._get_connection()
        try:
            self._dialect.begin_write(conn, build_id)
            sql = """
                UPDATE artifact_builds
                SET state = 'failed', completed_at = ?, error_message = ?, error_code = ?, logs = ?
                WHERE build_id = ? AND state IN ('pending', 'building')
            """
            params: list[Any] = [self._clock(), error_message, error_code, logs, build_id]
            if lease_owner is not None:
                sql += " AND lease_owner = ?"
                params.append(lease_owner)
                if lease_expires_at is not None:
                    sql += " AND lease_expires_at = ?"
                    params.append(lease_expires_at)
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_runnable_builds(self, limit: int, per_tenant: int) -> list[BuildState]:
        """Pending builds the runner executes: the oldest ``per_tenant`` of each tenant,
        oldest first.

        Per tenant, so one tenant's backlog cannot fill the window and hide every other
        tenant's builds. Builds a notebook dispatches to its own worker are left out:
        nothing here runs them, and they would hold their tenant's places.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"""
                SELECT {_BUILD_COLUMNS}
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY tenant_id ORDER BY created_at ASC
                    ) AS tenant_rank
                    FROM artifact_builds
                    WHERE state = 'pending' AND dispatch = 'runner'
                ) ranked
                WHERE tenant_rank <= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (per_tenant, limit),
            )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def builds_of(self, executor_ref: str) -> list[BuildState]:
        """Every build of an executor, of every tenant, oldest first."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                f"SELECT {_BUILD_COLUMNS} FROM artifact_builds WHERE executor_ref = ? "
                "ORDER BY created_at, build_id",
                (executor_ref,),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_build_state(row) for row in rows]

    def list_builds_by_tenant(
        self,
        tenant_id: str,
        limit: int = 100,
        state: str | None = None,
    ) -> list[BuildState]:
        """List a tenant's builds, newest first, optionally filtered by state."""
        conn = self._get_connection()
        try:
            if state:
                cursor = conn.execute(
                    f"""
                    SELECT {_BUILD_COLUMNS}
                    FROM artifact_builds
                    WHERE tenant_id = ? AND state = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tenant_id, state, limit),
                )
            else:
                cursor = conn.execute(
                    f"""
                    SELECT {_BUILD_COLUMNS}
                    FROM artifact_builds
                    WHERE tenant_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (tenant_id, limit),
                )

            return [_row_to_build_state(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def cleanup_old_builds(self, max_age_days: float = 7.0) -> int:
        """Delete ready/failed builds completed more than ``max_age_days`` ago; return the count.

        Their console chunks go with them. Their attempts stay for
        :meth:`settled_attempts`, which lists an attempt whose build is gone: an
        attempt row is the only record of a key the attempt may have written. Their
        releases stay too, the record of what was published, which
        :meth:`ReleaseStore.collectable_checkpoints
        <vestigo.releases.ReleaseStore.collectable_checkpoints>` still reads.
        """
        conn = self._get_connection()
        try:
            cutoff = self._clock() - (max_age_days * 86400)
            conn.execute(
                """
                DELETE FROM build_console_chunks
                WHERE build_id IN (
                    SELECT build_id FROM artifact_builds
                    WHERE state IN ('ready', 'failed') AND completed_at < ?
                )
                """,
                (cutoff,),
            )
            cursor = conn.execute(
                """
                DELETE FROM artifact_builds
                WHERE state IN ('ready', 'failed')
                  AND completed_at < ?
                """,
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return build counts, total and by state."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN state = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN state = 'building' THEN 1 END) as building,
                    COUNT(CASE WHEN state = 'ready' THEN 1 END) as ready,
                    COUNT(CASE WHEN state = 'failed' THEN 1 END) as failed
                FROM artifact_builds
                """
            )
            row = cursor.fetchone()
            return {
                "total": row["total"],
                "pending": row["pending"],
                "building": row["building"],
                "ready": row["ready"],
                "failed": row["failed"],
            }
        finally:
            conn.close()


_build_store: BuildStore | None = None


def get_build_store(
    db_path: Path | None = None,
    dialect: SqlDialect | None = None,
) -> BuildStore | None:
    """Get the build store singleton, or None if not yet created.

    ``db_path`` and ``dialect`` apply only on the call that creates it.
    """
    global _build_store
    if _build_store is None and db_path is not None:
        _build_store = BuildStore(db_path, dialect=dialect)
    return _build_store


def reset_build_store() -> None:
    """Reset the build store singleton (for testing)."""
    global _build_store
    _build_store = None
