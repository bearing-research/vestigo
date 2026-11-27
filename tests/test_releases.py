"""The release lifecycle (ADR 002 §7), on both backends.

Gate 2 of the extraction (ADR 002 §8) is this module: a build with attempts publishes
checkpoint and final releases with typed operations, resolves a bundle in one read,
recovers from a crash between prepare and accept, is fenced after a resume, orders
overlapping releases, resolves out-of-order head updates by revision, and accepts the
object-store path only after a manifest check. The ``ledger`` fixture runs every test on
SQLite and on Postgres. Nothing here imports Strata.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from vestigo.build_store import BuildState, BuildStore
from vestigo.operations import (
    AdvanceReference,
    AssertReferenceRevision,
    AssignAuthority,
    AuditEvent,
    CheckBuildLease,
    InsertAuditEvent,
)
from vestigo.provenance import TransformSpec, checkpoint_output_key, derive_subkey, run_output_key
from vestigo.releases import (
    HEAD_ROOT_KIND,
    OperationFailed,
    Release,
    ReleaseBundle,
    ReleaseError,
    ReleaseIncomplete,
    ReleaseOutOfOrder,
    ReleaseOvertaken,
    ReleaseRejected,
    ReleaseStore,
)
from vestigo.sql_backend import SqlDialect, SqliteDialect
from vestigo.store import ArtifactStore, StagedVersion

PARENT = hashlib.sha256(b"train").hexdigest()
SPEC = TransformSpec(executor="train@v1", params={"epochs": "3"}, inputs=[])


class Clock:
    """A clock the tests advance, so a write window can close without sleeping."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Ledger:
    store: ArtifactStore
    builds: BuildStore
    releases: ReleaseStore
    clock: Clock
    build_ids: list[str] = field(default_factory=list)


@pytest.fixture
def ledger(backend_store) -> Ledger:
    clock = Clock()
    builds = BuildStore(backend_store.db_path, dialect=backend_store.dialect, clock=clock)
    return Ledger(backend_store, builds, ReleaseStore(backend_store, builds, clock=clock), clock)


def _build(
    ledger: Ledger,
    attempt: str | None = "a1",
    *,
    occurrence: str | None = None,
    computation_key: str = PARENT,
    tenant: str | None = None,
) -> str:
    """A building build under *attempt* (runner-1's 60 s lease), targeting its final output.

    Pending, with no attempt, when *attempt* is None.
    """
    build_id = uuid.uuid4().hex
    target = f"final-{build_id}"
    version = ledger.store.create_artifact(
        target,
        run_output_key(computation_key, build_id, "model"),
        SPEC,
        tenant=tenant,
        computation_key=computation_key,
        minted=True,
    )
    ledger.builds.create_build(
        build_id,
        target,
        version,
        executor_ref="train@v1",
        tenant_id=tenant,
        occurrence=occurrence,
        computation_key=computation_key,
    )
    if attempt is not None:
        assert ledger.builds.open_attempt(
            build_id, attempt, writable_until=ledger.clock() + 3600, lease_owner="runner-1"
        )
    return build_id


def _open(
    ledger: Ledger,
    key: str,
    attempt: str = "a1",
    *,
    occurrence: str | None = None,
    tenant: str | None = None,
    build_id: str | None = None,
    beside: bool = False,
) -> tuple[str, BuildState]:
    """``open_build`` of a new build of *key* under *attempt* (runner-1's 60 s lease): the new
    build's id, and the computation's open build the call returned."""
    build_id = build_id or uuid.uuid4().hex
    target = f"final-{build_id}"
    version = ledger.store.create_artifact(
        target,
        run_output_key(key, build_id, "model"),
        SPEC,
        tenant=tenant,
        computation_key=key,
        minted=True,
    )
    build = ledger.builds.open_build(
        build_id,
        target,
        version,
        attempt,
        computation_key=key,
        executor_ref="train@v1",
        writable_until=ledger.clock() + 3600,
        lease_owner="runner-1",
        occurrence=occurrence,
        tenant_id=tenant,
        beside=beside,
    )
    return build_id, build


def _resume(ledger: Ledger, build_id: str, attempt: str, **kwargs) -> int:
    """Resume the build in *attempt* once its lease has lapsed, as a crashed run's successor."""
    ledger.clock.now += 120
    return ledger.releases.resume(build_id, attempt, writable_until=ledger.clock() + 3600, **kwargs)


@contextmanager
def _another_process(ledger: Ledger) -> Iterator[Ledger]:
    """A second ledger on the same database, opened as another process would, that never waits.

    One of its connections finding a lock held fails at once with the dialect's
    operational error instead of queueing behind the holder.
    """
    dialect = ledger.store.dialect
    if dialect.name == "sqlite":
        other: SqlDialect = SqliteDialect(ledger.store.db_path)
        no_wait = "PRAGMA busy_timeout = 0"
    else:
        other = type(dialect)(dialect.dsn)  # type: ignore[attr-defined]
        no_wait = "SET LOCAL lock_timeout = 200"
    connect = other.connect

    def connect_without_waiting():
        conn = connect()
        conn.execute(no_wait)
        return conn

    other.connect = connect_without_waiting  # type: ignore[method-assign]
    store = ArtifactStore(ledger.store.artifact_dir, dialect=other)
    builds = BuildStore(store.db_path, dialect=other, clock=ledger.clock)
    try:
        yield Ledger(store, builds, ReleaseStore(store, builds, clock=ledger.clock), ledger.clock)
    finally:
        other.close()


def _interleave(
    ledger: Ledger,
    first: Callable[[Ledger], object],
    second: Callable[[Ledger], object],
    *,
    after: str,
) -> tuple[object, object, bool]:
    """Run ``first(ledger)``; right after its statement containing *after*, ``second(other)``.

    ``other`` is another process's ledger (:func:`_another_process`). A ``second``
    that finds a lock held runs again once ``first`` is done, as a process queued on
    the lock would. Returns both outcomes (a :class:`ReleaseError` in place of a
    result) and whether ``second`` was locked out.
    """
    outcomes: dict[str, object] = {}
    state = {"fired": False, "locked": False}

    def call(key: str, fn: Callable[[Ledger], object], target: Ledger) -> None:
        try:
            outcomes[key] = fn(target)
        except ReleaseError as exc:
            outcomes[key] = exc

    with _another_process(ledger) as other:

        class Interleaved:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, attr):
                return getattr(self._inner, attr)

            def execute(self, sql, params=()):
                cursor = self._inner.execute(sql, params)
                if after in sql and not state["fired"]:
                    state["fired"] = True
                    try:
                        call("second", second, other)
                    except other.store.dialect.operational_error:
                        state["locked"] = True
                return cursor

        connect = ledger.releases._connect
        ledger.releases._connect = lambda: Interleaved(connect())
        try:
            call("first", first, ledger)
        finally:
            ledger.releases._connect = connect
        if state["locked"]:
            call("second", second, other)
    assert state["fired"], f"no statement of first matched {after!r}"
    return outcomes["first"], outcomes["second"], state["locked"]


AFTER_FENCE = "FROM artifact_builds b"
"""The statement ``lease_held`` runs: the first read of an acceptance under the build's key."""


def _staged_row(artifact_id: str, version: int, data: bytes) -> StagedVersion:
    return StagedVersion(
        artifact_id=artifact_id,
        version=version,
        schema_json="{}",
        row_count=None,
        byte_size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )


def _state(ledger: Ledger, row) -> str:
    """The stored state of a staged or stored version."""
    artifact_id = getattr(row, "artifact_id", None) or row.id
    return ledger.store.get_artifact(artifact_id, row.version).state


def _stage(
    ledger: Ledger, release: Release, name: str, data: bytes, *, write: bool = True
) -> StagedVersion:
    """One output of a checkpoint release: its own key, its bytes written unless told not to."""
    artifact_id = f"{release.build_id}-{release.kind}-{release.sequence}-{name}"
    key = (
        run_output_key(PARENT, release.build_id, name)
        if release.kind == "final"
        else checkpoint_output_key(PARENT, release.build_id, release.sequence, name)
    )
    version = ledger.store.create_artifact(
        artifact_id, key, SPEC, computation_key=PARENT, minted=True, release_id=release.release_id
    )
    if write:
        ledger.store.write_blob(artifact_id, version, data)
    return _staged_row(artifact_id, version, data)


def _checkpoint(ledger: Ledger, build_id: str, attempt: str = "a1", *, operations=()):
    """Prepare, stage and accept one checkpoint with a model and its metrics."""
    release = ledger.releases.prepare(build_id, attempt, "checkpoint")
    staged = [
        _stage(ledger, release, "model", b"weights" + bytes([release.sequence])),
        _stage(ledger, release, "metrics", b"{}"),
    ]
    return ledger.releases.accept(release.release_id, staged, list(operations))


class TestCheckpointsAndFinals:
    """A build's checkpoint and final releases: what accept checks and makes current."""

    def test_two_checkpoints_and_a_final_with_typed_operations(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [
            _stage(ledger, release, "model", b"w1"),
            _stage(ledger, release, "metrics", b"m1"),
        ]
        first = ledger.releases.accept(
            release.release_id,
            staged,
            [
                CheckBuildLease(build_id, "a1"),
                AssertReferenceRevision("train/latest", 0),
                AdvanceReference("train/latest", release.release_id),
                InsertAuditEvent(AuditEvent("release.accept", "alice", {"sequence": 1})),
            ],
        )
        assert first.release.state == "accepted"
        # Two outputs of one invocation: the same release id, both ready.
        assert {v.state for v in first.outputs} == {"ready"}
        assert {_state(ledger, v) for v in first.outputs} == {"ready"}
        assert ledger.releases.reference("train/latest") == (release.release_id, 1)
        assert ledger.builds.get_build(build_id).state == "building"

        second = ledger.releases.prepare(build_id, "a1", "checkpoint")
        assert second.sequence == 2
        staged = [_stage(ledger, second, "model", b"w2"), _stage(ledger, second, "metrics", b"m2")]
        bundle = ledger.releases.accept(
            second.release_id,
            staged,
            [
                AssertReferenceRevision("train/latest", 1),
                AdvanceReference("train/latest", second.release_id),
            ],
        )
        assert ledger.releases.reference("train/latest") == (second.release_id, 2)
        # Outputs read oldest first: the model was staged before its metrics.
        assert [v.provenance_hash for v in bundle.outputs] == [
            checkpoint_output_key(PARENT, build_id, 2, "model"),
            checkpoint_output_key(PARENT, build_id, 2, "metrics"),
        ]

        final = ledger.releases.prepare(build_id, "a1", "final")
        build = ledger.builds.get_build(build_id)
        assert (
            ledger.releases.attach(final.release_id, "a1", [(build.artifact_id, build.version)])
            == 1
        )
        ledger.store.write_blob(build.artifact_id, build.version, b"final-weights")
        staged = [_staged_row(build.artifact_id, build.version, b"final-weights")]
        done = ledger.releases.accept(
            final.release_id, staged, [AssignAuthority(f"slot:{build_id}", final.release_id)]
        )
        assert done.release.kind == "final"
        assert ledger.builds.get_build(build_id).state == "ready"
        assert ledger.releases.authority(f"slot:{build_id}") == final.release_id
        # A checkpoint's outputs never answer the computation's own cache lookup.
        assert ledger.store.find_by_provenance(derive_subkey(PARENT, "model")) is None

    def test_a_checkpoint_bundle_resolves_in_one_read(self, ledger: Ledger):
        build_id = _build(ledger)
        _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)

        bundle = ledger.releases.bundle(second.release.release_id)
        assert bundle is not None
        assert bundle.release.sequence == 2
        assert sorted(v.id.rsplit("-", 1)[1] for v in bundle.outputs) == ["metrics", "model"]
        assert all(v.state == "ready" for v in bundle.outputs)
        assert all(ledger.store.read_blob(v.id, v.version) is not None for v in bundle.outputs)

    def test_a_builds_checkpoints_and_their_outputs_resolve_in_one_read(self, ledger: Ledger):
        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)
        other = _build(ledger)
        _checkpoint(ledger, other)

        bundles = ledger.releases.bundles_of(build_id, "checkpoint")

        assert [b.release.release_id for b in bundles] == [
            first.release.release_id,
            second.release.release_id,
        ]
        for bundle, accepted in zip(bundles, (first, second), strict=True):
            assert bundle.release == ledger.releases.get(accepted.release.release_id)
            assert sorted(v.id for v in bundle.outputs) == sorted(v.id for v in accepted.outputs)
            assert {v.release_id for v in bundle.outputs} == {accepted.release.release_id}
        assert ledger.releases.bundles_of("no-such-build") == []

    def test_the_object_store_path_accepts_only_after_a_manifest_check(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        unwritten = _stage(ledger, release, "model", b"not yet", write=False)
        with pytest.raises(ReleaseIncomplete, match="none stored"):
            ledger.releases.accept(release.release_id, [unwritten])
        assert ledger.releases.get(release.release_id).state == "preparing"

        ledger.store.write_blob(unwritten.artifact_id, unwritten.version, b"short")
        with pytest.raises(ReleaseIncomplete, match="7 bytes declared, 5 stored"):
            ledger.releases.accept(release.release_id, [unwritten])

        ledger.store.write_blob(unwritten.artifact_id, unwritten.version, b"not yet")
        assert ledger.releases.accept(release.release_id, [unwritten]).release.state == "accepted"

    def test_an_output_staged_under_the_release_but_not_declared_blocks_it(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        model = _stage(ledger, release, "model", b"w")
        metrics = _stage(ledger, release, "metrics", b"{}")
        with pytest.raises(ReleaseIncomplete, match=f"{metrics.artifact_id}@v={metrics.version}"):
            ledger.releases.accept(
                release.release_id, [model], [AdvanceReference("train/latest", release.release_id)]
            )
        assert ledger.releases.get(release.release_id).state == "preparing"
        assert ledger.releases.reference("train/latest") is None
        bundle = ledger.releases.accept(release.release_id, [model, metrics])
        assert {v.state for v in bundle.outputs} == {"ready"}

    def test_overlapping_releases_are_accepted_in_sequence_order(self, ledger: Ledger):
        build_id = _build(ledger)
        first = ledger.releases.prepare(build_id, "a1", "checkpoint")
        second = ledger.releases.prepare(build_id, "a1", "checkpoint")
        first_staged = [_stage(ledger, first, "model", b"w1")]
        second_staged = [_stage(ledger, second, "model", b"w2")]  # finishes uploading first

        with pytest.raises(ReleaseOutOfOrder):
            ledger.releases.accept(second.release_id, second_staged)
        assert ledger.releases.get(second.release_id).state == "preparing"
        ledger.releases.accept(first.release_id, first_staged)
        ledger.releases.accept(second.release_id, second_staged)
        assert [r.state for r in ledger.releases.releases_of(build_id, "checkpoint")] == [
            "accepted",
            "accepted",
        ]

    def test_a_taken_sequence_or_release_id_is_a_release_error(self, ledger: Ledger):
        build_id = _build(ledger)
        taken = ledger.releases.prepare(build_id, "a1", "checkpoint", sequence=1)
        with pytest.raises(ReleaseError, match="checkpoint sequence 1"):
            ledger.releases.prepare(build_id, "a1", "checkpoint", sequence=1)
        with pytest.raises(ReleaseError, match=taken.release_id):
            ledger.releases.prepare(build_id, "a1", "final", release_id=taken.release_id)
        assert [r.release_id for r in ledger.releases.releases_of(build_id)] == [taken.release_id]
        assert ledger.releases.prepare(build_id, "a1", "checkpoint").sequence == 2

    def test_a_retry_after_a_lost_response_returns_the_original_release(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w1")]
        first = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/latest", release.release_id)]
        )
        # The reference moves on before the lost response is retried.
        later = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            later.release_id,
            [_stage(ledger, later, "model", b"w2")],
            [AdvanceReference("train/latest", later.release_id)],
        )

        retried = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/latest", release.release_id)]
        )

        assert retried.release == first.release
        assert [v.id for v in retried.outputs] == [v.id for v in first.outputs]
        assert ledger.releases.reference("train/latest") == (later.release_id, 2)

    def test_an_alias_move_leaves_a_bound_run_on_its_exact_inputs(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        bound = {"vestigo://artifact/features@v=3": "features@v=3"}
        version = ledger.store.create_artifact(
            "bound-output",
            checkpoint_output_key(PARENT, build_id, 1, "model"),
            SPEC,
            input_versions=bound,
            release_id=release.release_id,
            minted=True,
        )
        ledger.store.write_blob("bound-output", version, b"w")
        staged = [_staged_row("bound-output", version, b"w")]
        ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("features/latest", release.release_id)]
        )
        other = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            other.release_id,
            [_stage(ledger, other, "model", b"w2")],
            [AdvanceReference("features/latest", other.release_id)],
        )
        (output,) = ledger.releases.bundle(release.release_id).outputs
        assert output.input_versions is not None and "features@v=3" in output.input_versions

    def test_the_final_release_retargets_the_build_and_fails_its_placeholder(self, ledger: Ledger):
        build_id = _build(ledger)
        placeholder = ledger.builds.get_build(build_id)
        assert _state(ledger, placeholder) == "building"

        final = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, final, "model", b"done"), _stage(ledger, final, "metrics", b"{}")]
        ledger.releases.accept(final.release_id, staged)

        build = ledger.builds.get_build(build_id)
        assert (build.artifact_id, build.version) == (staged[0].artifact_id, staged[0].version)
        assert build.state == "ready"
        assert (
            ledger.store.get_artifact(placeholder.artifact_id, placeholder.version).state
            == "failed"
        )

    def test_an_empty_final_release_ends_the_build_and_retires_its_placeholder(
        self, ledger: Ledger
    ):
        """A run that stores nothing still completes: nothing is left building."""
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = _build(ledger, computation_key=key)
        placeholder = ledger.builds.get_build(build_id)

        final = ledger.releases.prepare(build_id, "a1", "final")
        ledger.releases.accept(final.release_id, [])

        build = ledger.builds.get_build(build_id)
        assert build.state == "ready"
        assert (build.artifact_id, build.version) == (placeholder.artifact_id, placeholder.version)
        assert _state(ledger, placeholder) == "failed"
        assert ledger.builds.find_open_build(key) is None

    def test_a_release_belongs_to_its_builds_tenant(self, ledger: Ledger):
        build_id = _build(ledger, tenant="acme")
        with pytest.raises(ValueError, match="belongs to tenant 'acme'"):
            ledger.releases.prepare(build_id, "a1", "checkpoint", tenant="other")
        with pytest.raises(ValueError, match="belongs to tenant 'acme'"):
            ledger.releases.apply(build_id, "a1", [], tenant="other")
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        assert release.tenant == "acme"
        ledger.releases.accept(
            release.release_id,
            [_stage(ledger, release, "model", b"w")],
            [AdvanceReference("train/head", release.release_id)],
        )
        assert ledger.releases.reference("train/head", tenant="acme") == (release.release_id, 1)
        assert ledger.releases.reference("train/head") is None
        # The default tenant's build likewise refuses another's release.
        with pytest.raises(ValueError, match="belongs to tenant ''"):
            ledger.releases.prepare(_build(ledger), "a1", "checkpoint", tenant="acme")


class TestAttemptsAndFences:
    """Crashes, resumes and leases: which attempt may touch a build's releases."""

    def test_recovery_after_a_crash_between_prepare_and_accept(self, ledger: Ledger):
        build_id = _build(ledger)
        _checkpoint(ledger, build_id)
        crashed = ledger.releases.prepare(build_id, "a1", "checkpoint")
        _stage(ledger, crashed, "model", b"w3")
        # The process dies here. A new attempt resumes the same execution.
        assert _resume(ledger, build_id, "a2", lease_owner="runner-2") == 1
        assert ledger.releases.get(crashed.release_id).state == "abandoned"
        assert ledger.builds.get_build(build_id).attempt == "a2"

        # The old attempt is fenced: it can neither prepare nor accept.
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a1", "checkpoint")
        with pytest.raises(ValueError, match="abandoned"):
            ledger.releases.accept(crashed.release_id, [])
        # The new attempt continues the sequence; the earlier checkpoint stays valid.
        resumed = _checkpoint(ledger, build_id, "a2")
        assert resumed.release.sequence == 3
        (first,) = [r for r in ledger.releases.releases_of(build_id) if r.sequence == 1]
        assert first.state == "accepted"
        assert all(v.state == "ready" for v in ledger.releases.bundle(first.release_id).outputs)

    def test_a_stale_attempt_publishing_after_a_resume_is_rejected(self, ledger: Ledger):
        build_id = _build(ledger)
        stale = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, stale, "model", b"w1")]
        _resume(ledger, build_id, "a2")
        # resume abandoned it; an attempt whose window merely closed is rejected at accept.
        assert ledger.releases.get(stale.release_id).state == "abandoned"

        late = ledger.releases.prepare(build_id, "a2", "checkpoint")
        staged = [_stage(ledger, late, "model", b"w2")]
        ledger.clock.now += 7200  # a2's write window closes
        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(late.release_id, staged)
        assert ledger.releases.get(late.release_id).state == "rejected"
        assert _state(ledger, staged[0]) == "building"

    def test_a_crash_before_acceptance_leaves_an_orphan_and_no_authority(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, release, "model", b"all-written")]
        # Every byte is written; the process dies before accept.
        row = ledger.store.get_artifact(staged[0].artifact_id, staged[0].version)
        assert row.state == "building"
        assert ledger.store.read_blob(row.id, row.version) == b"all-written"
        assert ledger.releases.authority(f"slot:{build_id}") is None
        assert ledger.releases.get(release.release_id).state == "preparing"
        assert ledger.builds.get_build(build_id).state == "building"
        assert ledger.releases.abandon(release.release_id, "a1")

    def test_a_resume_binds_the_attempt_to_an_explicit_accepted_release(self, ledger: Ledger):
        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        unsettled = ledger.releases.prepare(build_id, "a1", "checkpoint")

        with pytest.raises(ValueError, match="not an accepted release"):
            _resume(ledger, build_id, "a2", from_release=unsettled.release_id)
        _resume(ledger, build_id, "a2", from_release=first.release.release_id)

        attempt = ledger.builds.get_attempt(build_id, "a2")
        assert attempt is not None and attempt.resumed_from == first.release.release_id
        assert ledger.builds.get_attempt(build_id, "a1").resumed_from is None
        assert ledger.releases.latest_accepted(build_id).release_id == first.release.release_id

    def test_a_computation_finds_its_open_build_until_the_final_release(self, ledger: Ledger):
        # Other tests leave builds of PARENT open in a shared store; this computation is its own.
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = _build(ledger, occurrence="variant=a", computation_key=key)
        assert ledger.builds.find_open_build(key) is None
        assert ledger.builds.find_open_build(key, occurrence="variant=a").build_id == build_id

        # A lapsed lease is still the computation's open build: what a resume continues.
        ledger.clock.now += 120
        assert ledger.builds.find_open_build(key, occurrence="variant=a").build_id == build_id

        ledger.releases.resume(build_id, "a2", writable_until=ledger.clock() + 3600)
        final = ledger.releases.prepare(build_id, "a2", "final")
        staged = [_stage(ledger, final, "model", b"done")]
        ledger.releases.accept(final.release_id, staged)
        assert ledger.builds.find_open_build(key, occurrence="variant=a") is None

    def test_a_live_lease_is_taken_over_by_no_other_attempt(self, ledger: Ledger):
        build_id = _build(ledger)
        with pytest.raises(ValueError, match="lease"):
            ledger.releases.resume(build_id, "a2", writable_until=ledger.clock() + 3600)
        assert not ledger.builds.open_attempt(build_id, "a2", writable_until=ledger.clock() + 3600)
        assert ledger.builds.get_build(build_id).attempt == "a1"
        # Its own attempt re-opens (a longer write window); once lapsed, any attempt may.
        assert ledger.builds.open_attempt(build_id, "a1", writable_until=ledger.clock() + 7200)
        assert ledger.builds.get_attempt(build_id, "a1").writable_until == ledger.clock() + 7200
        assert _resume(ledger, build_id, "a2") == 0
        assert ledger.builds.get_build(build_id).attempt == "a2"

    def test_a_lapsed_lease_fences_its_attempt_for_good(self, ledger: Ledger):
        build_id = _build(ledger)
        pending = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, pending, "model", b"w")]
        ledger.clock.now += 120
        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(pending.release_id, staged)
        # Nothing took the build over, yet a runner that merely stalled is fenced too.
        assert not ledger.builds.renew_lease(build_id, "runner-1", 60.0)
        writable_until = ledger.clock() + 3600
        assert not ledger.builds.open_attempt(
            build_id, "a1", writable_until=writable_until, lease_owner="runner-1"
        )
        # No other attempt's lease is live: the refusal says the attempt itself is spent.
        with pytest.raises(ValueError, match="attempt a1 lapsed or was taken over"):
            ledger.releases.resume(build_id, "a1", writable_until=writable_until)
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a1", "checkpoint")
        assert ledger.releases.get(pending.release_id).state == "rejected"

        # The build goes on in a new attempt, which renews as any live lease does.
        assert ledger.builds.open_attempt(
            build_id, "a2", writable_until=writable_until, lease_owner="runner-1"
        )
        ledger.clock.now += 30
        assert ledger.builds.renew_lease(build_id, "runner-1", 60.0)
        ledger.clock.now += 50
        late = ledger.releases.prepare(build_id, "a2", "checkpoint")
        assert ledger.releases.accept(late.release_id, []).release.state == "accepted"

        # One whose build was reclaimed is fenced as well: the build has no attempt.
        late = ledger.releases.prepare(build_id, "a2", "checkpoint")
        staged = [_stage(ledger, late, "model", b"w2")]
        ledger.clock.now += 120
        assert ledger.builds.reclaim_expired_build(build_id, "runner-2", 60.0)
        assert ledger.builds.get_build(build_id).attempt is None
        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(late.release_id, staged)
        assert ledger.releases.get(late.release_id).state == "rejected"
        assert not ledger.builds.renew_lease(build_id, "runner-1", 60.0)
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a2", "checkpoint")

    def test_a_renewal_reads_the_clock_under_the_build_s_key(self, ledger: Ledger, monkeypatch):
        """A reading taken before the key could predate a lapse an acceptance fenced on,
        and the renewal would revive the lease of an attempt whose release was rejected."""
        build_id = _build(ledger)
        order: list[str] = []
        begin_write = ledger.builds._dialect.begin_write
        clock = ledger.builds._clock

        def locking(conn, key):
            begin_write(conn, key)
            order.append(f"lock {key}")

        def reading():
            order.append("clock")
            return clock()

        monkeypatch.setattr(ledger.builds._dialect, "begin_write", locking)
        monkeypatch.setattr(ledger.builds, "_clock", reading)

        assert ledger.builds.renew_lease(build_id, "runner-1", 60.0)
        assert order == [f"lock {build_id}", "clock"]

    def test_a_fenced_attempt_does_not_renew_a_successor_that_shares_its_owner(
        self, ledger: Ledger
    ):
        build_id = _build(ledger)  # a1, under runner-1
        # a1 stalls past its lease; the same runner resumes the build in a2.
        _resume(ledger, build_id, "a2", lease_owner="runner-1")

        assert not ledger.builds.renew_lease(build_id, "runner-1", 600.0, attempt="a1")
        assert ledger.builds.renew_lease(build_id, "runner-1", 600.0, attempt="a2")

    def test_a_fenced_attempt_does_not_lapse_a_successor_that_shares_its_owner(
        self, ledger: Ledger
    ):
        build_id = _build(ledger)
        _resume(ledger, build_id, "a2", lease_owner="runner-1")

        assert not ledger.builds.lapse_lease(build_id, "runner-1", attempt="a1")
        assert ledger.builds.renew_lease(build_id, "runner-1", 600.0, attempt="a2")
        assert ledger.builds.lapse_lease(build_id, "runner-1", attempt="a2")

    def test_a_fenced_attempt_does_not_complete_a_successor_that_shares_its_owner(
        self, ledger: Ledger
    ):
        build_id = _build(ledger)
        _resume(ledger, build_id, "a2", lease_owner="runner-1")

        assert not ledger.builds.complete_build(build_id, lease_owner="runner-1", attempt="a1")
        assert ledger.builds.get_build(build_id).state == "building"
        assert ledger.builds.complete_build(build_id, lease_owner="runner-1", attempt="a2")

    def test_an_attempt_whose_lease_lapsed_does_not_complete_its_build(self, ledger: Ledger):
        build_id = _build(ledger)
        ledger.clock.now += 120  # a1's lease lapses; nobody has resumed the build yet

        assert not ledger.builds.complete_build(build_id, lease_owner="runner-1", attempt="a1")
        assert ledger.builds.get_build(build_id).state == "building"

    def test_a_renewal_queued_for_the_key_past_the_deadline_is_refused(self, ledger: Ledger):
        """Any fence check that held the key meanwhile saw the lease lapsed, so it stays so."""
        build_id = _build(ledger)
        ledger.clock.now += 59
        clock = ledger.builds._clock
        read = threading.Event()

        def reading():
            value = clock()
            read.set()
            return value

        outcome: dict[str, object] = {}

        def renew() -> None:
            try:
                outcome["renewed"] = ledger.builds.renew_lease(build_id, "runner-1", 600.0)
            except Exception as exc:
                outcome["raised"] = exc

        with _another_process(ledger) as other:
            holder = other.store.dialect.connect()
            try:
                other.store.dialect.begin_write(holder, build_id)
                if other.store.dialect.name != "sqlite":
                    # The row too: a renewal outside the key would queue on it instead.
                    holder.execute(
                        "SELECT 1 FROM artifact_builds WHERE build_id = ? FOR UPDATE", (build_id,)
                    )
                ledger.builds._clock = reading
                renewer = threading.Thread(target=renew)
                renewer.start()
                # A renewal that reads the clock before queuing has read it by now.
                read.wait(0.5)
                ledger.clock.now += 2  # past the deadline while it waits
            finally:
                holder.commit()
                holder.close()
            renewer.join(10)
            ledger.builds._clock = clock

        assert not renewer.is_alive()
        assert outcome == {"renewed": False}
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a1", "checkpoint")

    def test_only_its_own_attempt_abandons_attaches_or_marks_a_release(self, ledger: Ledger):
        build_id = _build(ledger)
        _resume(ledger, build_id, "a2")
        live = ledger.releases.prepare(build_id, "a2", "checkpoint")
        target = ledger.builds.get_build(build_id)
        rows = [(target.artifact_id, target.version)]
        for touch in (
            lambda: ledger.releases.abandon(live.release_id, "a1"),
            lambda: ledger.releases.attach(live.release_id, "a1", rows),
            lambda: ledger.releases.mark_prepared(live.release_id, "a1"),
        ):
            with pytest.raises(ValueError, match="attempt a2's, not a1's"):
                touch()
        assert ledger.releases.get(live.release_id).state == "preparing"
        assert ledger.store.get_artifact(*rows[0]).release_id is None

        assert ledger.releases.attach(live.release_id, "a2", rows) == 1
        assert ledger.releases.mark_prepared(live.release_id, "a2")
        assert ledger.releases.abandon(live.release_id, "a2")
        assert ledger.releases.get(live.release_id).state == "abandoned"
        with pytest.raises(ValueError, match="does not exist"):
            ledger.releases.abandon("no-such-release", "a2")

    def test_a_head_set_between_releases_is_fenced_on_the_attempt(self, ledger: Ledger):
        build_id = _build(ledger)
        best = _checkpoint(ledger, build_id)
        _checkpoint(ledger, build_id)

        ledger.releases.apply(
            build_id, "a1", [AdvanceReference("train/best", best.release.release_id)]
        )
        assert ledger.releases.reference("train/best") == (best.release.release_id, 1)

        # The new attempt may point a head at the old attempt's checkpoint; the old one may not.
        _resume(ledger, build_id, "a2")
        with pytest.raises(ReleaseRejected):
            ledger.releases.apply(
                build_id, "a1", [AdvanceReference("train/best", best.release.release_id)]
            )
        ledger.releases.apply(
            build_id, "a2", [AdvanceReference("train/best", best.release.release_id)]
        )
        assert ledger.releases.reference("train/best") == (best.release.release_id, 2)
        with pytest.raises(OperationFailed):
            ledger.releases.apply(build_id, "a2", [AssertReferenceRevision("train/best", 1)])

    def test_a_final_on_a_build_that_stopped_building_is_rejected(self, ledger: Ledger):
        """What acceptance does when a writer that took no key ended the build anyway."""
        build_id = _build(ledger)
        final = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, final, "model", b"done")]
        real_check = ReleaseStore._check_outputs

        def failing_underneath(self, conn, release, staged_, sizes):
            real_check(self, conn, release, staged_, sizes)
            conn.execute(
                "UPDATE artifact_builds SET state = 'failed' WHERE build_id = ?", (build_id,)
            )

        ReleaseStore._check_outputs = failing_underneath
        try:
            with pytest.raises(ReleaseRejected, match="no longer building"):
                ledger.releases.accept(final.release_id, staged)
        finally:
            ReleaseStore._check_outputs = real_check
        assert ledger.releases.get(final.release_id).state == "rejected"
        assert _state(ledger, staged[0]) == "building"
        assert ledger.builds.get_build(build_id).state == "building"

    def test_a_build_writer_waits_for_an_acceptance_in_flight(self, ledger: Ledger):
        """A fail landing between a final's fence check and its commit is held off and loses."""
        build_id = _build(ledger)
        final = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, final, "model", b"done")]

        bundle, failed, locked = _interleave(
            ledger,
            lambda ledger_: ledger_.releases.accept(final.release_id, staged),
            lambda other: other.builds.fail_build(build_id, "boom"),
            after=AFTER_FENCE,
        )

        assert locked, "fail_build ran inside the acceptance's transaction"
        assert isinstance(bundle, ReleaseBundle) and bundle.release.state == "accepted"
        assert failed is False
        assert ledger.builds.get_build(build_id).state == "ready"

    def test_a_lapse_waits_for_an_acceptance_in_flight_and_fences_what_follows(
        self, ledger: Ledger
    ):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w")]

        bundle, lapsed, locked = _interleave(
            ledger,
            lambda ledger_: ledger_.releases.accept(release.release_id, staged),
            lambda other: other.builds.lapse_lease(build_id, "runner-1"),
            after=AFTER_FENCE,
        )

        assert locked, "lapse_lease ran inside the acceptance's transaction"
        assert isinstance(bundle, ReleaseBundle) and bundle.release.state == "accepted"
        assert lapsed is True
        # At the lapse's own clock reading, not a tick later.
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a1", "checkpoint")

    def test_a_runner_checkpoints_under_the_attempt_its_claim_opens(self, ledger: Ledger):
        build_id = _build(ledger, None)
        assert ledger.builds.claim_build(build_id, "runner-1", 60.0)
        assert ledger.builds.open_attempt(
            build_id, "a1", writable_until=ledger.clock() + 3600, lease_owner="runner-1"
        )
        first = _checkpoint(ledger, build_id)
        assert first.release.state == "accepted"

        # The orphan's reclaimer continues it the same way, from the accepted checkpoint.
        ledger.clock.now += 120
        assert ledger.builds.reclaim_expired_build(build_id, "runner-2", 60.0)
        resumed = ledger.releases.resume(
            build_id,
            "a2",
            writable_until=ledger.clock() + 3600,
            lease_owner="runner-2",
            from_release=first.release.release_id,
        )
        assert resumed == 0
        assert _checkpoint(ledger, build_id, "a2").release.sequence == 2

    def test_a_new_attempt_abandons_what_the_fenced_ones_left_unsettled(self, ledger: Ledger):
        """However the attempt opens; else every later checkpoint waits on a dead release."""
        build_id = _build(ledger)
        _checkpoint(ledger, build_id)
        crashed = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.clock.now += 120
        assert ledger.builds.open_attempt(
            build_id, "a2", writable_until=ledger.clock() + 3600, lease_owner="runner-2"
        )
        assert ledger.releases.get(crashed.release_id).state == "abandoned"
        assert _checkpoint(ledger, build_id, "a2").release.state == "accepted"

        stuck = ledger.releases.prepare(build_id, "a2", "checkpoint")
        ledger.clock.now += 120
        assert ledger.builds.reclaim_expired_build(build_id, "runner-3", 60.0)
        assert ledger.builds.open_attempt(
            build_id, "a3", writable_until=ledger.clock() + 3600, lease_owner="runner-3"
        )
        assert ledger.releases.get(stuck.release_id).state == "abandoned"
        assert _checkpoint(ledger, build_id, "a3").release.state == "accepted"

    def test_attach_waits_for_an_acceptance_and_refuses_a_settled_release(self, ledger: Ledger):
        """A row attached mid-acceptance would sit ``building`` under an accepted release."""
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w")]
        late = ledger.store.create_artifact("late-row", "late-key", SPEC, minted=True)

        def attaching(other: Ledger) -> object:
            try:
                return other.releases.attach(release.release_id, "a1", [("late-row", late)])
            except ValueError as exc:
                return exc

        bundle, attached, locked = _interleave(
            ledger,
            lambda ledger_: ledger_.releases.accept(release.release_id, staged),
            attaching,
            after="SELECT state, release_id FROM artifact_versions",
        )

        assert locked, "attach ran inside the acceptance's transaction"
        assert isinstance(bundle, ReleaseBundle) and bundle.release.state == "accepted"
        assert isinstance(attached, ValueError) and "is accepted" in str(attached)
        assert ledger.store.get_artifact("late-row", late).release_id is None
        assert {v.state for v in ledger.releases.bundle(release.release_id).outputs} == {"ready"}

    def test_a_rejection_never_lands_on_a_retry_accepted_meanwhile(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
    ):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w")]
        ledger.clock.now += 120  # the lease lapses
        real_settle = ReleaseStore._settle
        retried: list[ReleaseBundle] = []

        def settle_after_a_retry(self, release_id: str, state: str) -> None:
            # Between the fence's rollback and the rejection, a caller whose clock lags
            # sees the lease live and its retry is accepted.
            if not retried:
                ledger.clock.now -= 120
                retried.append(ledger.releases.accept(release_id, staged))
                ledger.clock.now += 120
            real_settle(self, release_id, state)

        monkeypatch.setattr(ReleaseStore, "_settle", settle_after_a_retry)
        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(release.release_id, staged)

        assert retried[0].release.state == "accepted"
        assert ledger.releases.get(release.release_id).state == "accepted"
        assert {_state(ledger, v) for v in staged} == {"ready"}

    def test_outputs_are_sized_before_the_builds_key_is_taken(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
    ):
        """On SQLite the key is the database's write lock, and a remote size is a request."""
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w"), _stage(ledger, release, "metrics", b"{}")]
        events: list[str] = []
        blob_size = ledger.store.blob_store.blob_size
        begin_write = ledger.store.dialect.begin_write

        def sizing(artifact_id: str, version: int) -> int | None:
            events.append("size")
            return blob_size(artifact_id, version)

        def locking(conn, key: str) -> None:
            events.append(f"key:{key}")
            begin_write(conn, key)

        monkeypatch.setattr(ledger.store.blob_store, "blob_size", sizing)
        monkeypatch.setattr(ledger.store.dialect, "begin_write", locking)
        ledger.releases.accept(release.release_id, staged)

        assert events[: events.index(f"key:{build_id}")] == ["size", "size"]
        assert {_state(ledger, v) for v in staged} == {"ready"}


class TestOneOpenExecution:
    """``open_build``: of two opens of one computation that both found nothing open, the
    second returns the first's build and writes nothing."""

    def test_a_second_open_returns_the_first_and_writes_nothing(self, ledger: Ledger):
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        first_id, first = _open(ledger, key)
        assert (first.build_id, first.state, first.attempt) == (first_id, "building", "a1")
        assert (first.lease_owner, first.dispatch) == ("runner-1", "external")
        assert ledger.builds.get_attempt(first_id, "a1") is not None

        second_id, second = _open(ledger, key, "a2")

        assert second == first
        assert ledger.builds.get_build(second_id) is None
        assert ledger.builds.get_attempt(first_id, "a2") is None
        assert ledger.builds.find_open_build(key) == first

    def test_each_occurrence_and_tenant_is_an_execution_of_its_own(self, ledger: Ledger):
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        for occurrence, tenant in ((None, None), ("variant=a", None), (None, "acme")):
            build_id, build = _open(ledger, key, occurrence=occurrence, tenant=tenant)
            assert build.build_id == build_id
            assert (
                ledger.builds.find_open_build(key, occurrence=occurrence, tenant_id=tenant) == build
            )

    def test_a_build_that_is_not_building_leaves_the_computation_free(self, ledger: Ledger):
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        _build(ledger, None, computation_key=key)  # pending
        failed = _build(ledger, computation_key=key)
        assert ledger.builds.fail_build(failed, "crashed")
        finished = _build(ledger, computation_key=key)
        final = ledger.releases.prepare(finished, "a1", "final")
        ledger.releases.accept(final.release_id, [])

        build_id, build = _open(ledger, key)

        assert build.build_id == build_id

    def test_a_lapsed_open_build_comes_back_with_its_lease(self, ledger: Ledger):
        """What the caller resumes rather than open beside."""
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        first_id, first = _open(ledger, key)
        ledger.clock.now += 120

        _, found = _open(ledger, key, "a2")

        assert found.build_id == first_id
        assert found.lease_expires_at == first.lease_expires_at
        assert found.lease_expires_at is not None and found.lease_expires_at < ledger.clock()
        assert _resume(ledger, first_id, "a2") == 0

    def test_beside_opens_next_to_an_open_build(self, ledger: Ledger):
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        first_id, _ = _open(ledger, key)
        ledger.clock.now += 1  # the newer, so find_open_build returns it

        second_id, second = _open(ledger, key, "a2", beside=True)

        assert (second.build_id, second.attempt) == (second_id, "a2")
        assert ledger.builds.get_build(first_id).state == "building"
        assert ledger.builds.find_open_build(key) == second

    @pytest.mark.parametrize("beside", [False, True])
    def test_an_open_that_fails_leaves_no_build(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch, beside: bool
    ):
        lost = ledger.store.dialect.operational_error

        def dropped(*args: object, **kwargs: object) -> bool:
            raise lost("the connection dropped")

        monkeypatch.setattr(BuildStore, "open_attempt_within", dropped)
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = uuid.uuid4().hex
        with pytest.raises(lost):
            _open(ledger, key, build_id=build_id, beside=beside)

        assert ledger.builds.get_build(build_id) is None

    @pytest.mark.parametrize("beside", [False, True])
    def test_a_refused_first_attempt_is_an_error_and_leaves_no_build(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch, beside: bool
    ):
        """A build inserted a moment before cannot be refused: a broken invariant, and no
        caller may go on to a release its attempt does not hold."""
        monkeypatch.setattr(BuildStore, "open_attempt_within", lambda *args, **kwargs: False)
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = uuid.uuid4().hex
        with pytest.raises(RuntimeError, match="did not open its first attempt"):
            _open(ledger, key, build_id=build_id, beside=beside)

        assert ledger.builds.get_build(build_id) is None


class TestReferencesAndAuthorities:
    """The typed operations a release runs: heads by revision, slots all or none, audit order."""

    def test_head_updates_out_of_order_resolve_by_revision(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            release.release_id,
            [_stage(ledger, release, "model", b"w1")],
            [AdvanceReference("train/head", release.release_id)],
        )
        # Two updaters read revision 1; the slower one's assertion is stale.
        fast = ledger.releases.prepare(build_id, "a1", "checkpoint")
        slow = ledger.releases.prepare(build_id, "a1", "checkpoint")
        fast_staged = [_stage(ledger, fast, "model", b"w2")]
        slow_staged = [_stage(ledger, slow, "model", b"w3")]
        ledger.releases.accept(
            fast.release_id,
            fast_staged,
            [
                AssertReferenceRevision("train/head", 1),
                AdvanceReference("train/head", fast.release_id),
            ],
        )
        with pytest.raises(OperationFailed, match="revision 2, not 1"):
            ledger.releases.accept(
                slow.release_id,
                slow_staged,
                [
                    AssertReferenceRevision("train/head", 1),
                    AdvanceReference("train/head", slow.release_id),
                ],
            )
        assert ledger.releases.reference("train/head") == (fast.release_id, 2)
        assert ledger.releases.get(slow.release_id).state == "preparing"
        assert _state(ledger, slow_staged[0]) == "building"

    def test_two_builds_asserting_one_revision_have_one_winner(self, ledger: Ledger):
        """Two processes assert a head's revision and advance it; the loser's advance fails.

        On Postgres the two acceptances run under different build keys, so the
        assertion's read alone proves nothing: the advance is guarded on it.
        """
        b1, b2 = _build(ledger), _build(ledger)
        seed = _checkpoint(ledger, b1).release.release_id
        ledger.releases.apply(b1, "a1", [AdvanceReference("shared/head", seed)])
        assert ledger.releases.reference("shared/head") == (seed, 1)
        r1 = ledger.releases.prepare(b1, "a1", "checkpoint")
        r2 = ledger.releases.prepare(b2, "a1", "checkpoint")
        s1, s2 = [_stage(ledger, r1, "model", b"1")], [_stage(ledger, r2, "model", b"2")]

        def accepting(release, staged):
            return lambda ledger_: ledger_.releases.accept(
                release.release_id,
                staged,
                [
                    AssertReferenceRevision("shared/head", 1),
                    AdvanceReference("shared/head", release.release_id),
                ],
            )

        first, second, _ = _interleave(
            ledger, accepting(r1, s1), accepting(r2, s2), after="SELECT revision FROM"
        )

        (loser,) = [r for r in (first, second) if isinstance(r, OperationFailed)]
        (winner,) = [r for r in (first, second) if isinstance(r, ReleaseBundle)]
        assert "shared/head" in str(loser)
        assert ledger.releases.reference("shared/head") == (winner.release.release_id, 2)
        lost = r2 if winner.release.release_id == r1.release_id else r1
        assert ledger.releases.get(lost.release_id).state == "preparing"
        assert _state(ledger, (s1 if lost is r1 else s2)[0]) == "building"

    def test_a_reference_moves_only_to_an_accepted_release_of_its_tenant(self, ledger: Ledger):
        """Else it names nothing and its old head's outputs lose their root."""
        build_id = _build(ledger)
        head = _checkpoint(ledger, build_id).release.release_id
        ledger.releases.apply(build_id, "a1", [AdvanceReference("train/best", head)])
        roots = ledger.store.retention_roots(kind=HEAD_ROOT_KIND)
        assert {r["holder"] for r in roots} == {"train/best"} and len(roots) == 2

        unsettled = ledger.releases.prepare(build_id, "a1", "checkpoint").release_id
        foreign = _checkpoint(ledger, _build(ledger, tenant="acme")).release.release_id
        for release_id in ("no-such-release", unsettled, foreign):
            with pytest.raises(OperationFailed, match="not an accepted release"):
                ledger.releases.apply(build_id, "a1", [AdvanceReference("train/best", release_id)])

        assert ledger.releases.reference("train/best") == (head, 1)
        assert ledger.store.retention_roots(kind=HEAD_ROOT_KIND) == roots

    def test_multi_slot_authority_is_all_or_none(self, ledger: Ledger):
        taken = _build(ledger)
        winner = ledger.releases.prepare(taken, "a1", "final")
        ledger.releases.accept(
            winner.release_id,
            [_stage(ledger, winner, "model", b"w")],
            [AssignAuthority("slot:b", winner.release_id)],
        )
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, release, "model", b"w")]
        with pytest.raises(OperationFailed, match="slot:b already belongs"):
            ledger.releases.accept(
                release.release_id,
                staged,
                [
                    AssignAuthority("slot:a", release.release_id),
                    AssignAuthority("slot:b", release.release_id),
                ],
            )
        assert ledger.releases.authority("slot:a") is None
        assert ledger.releases.authority("slot:b") == winner.release_id
        assert ledger.releases.get(release.release_id).state == "preparing"
        assert ledger.builds.get_build(build_id).state == "building"

    def test_two_attempts_racing_for_a_slot_have_one_winner(self, ledger: Ledger):
        """Two processes accept finals claiming one slot at once; the slot has one winner."""
        releases, staged = [], []
        for _ in range(2):
            release = ledger.releases.prepare(_build(ledger), "a1", "final")
            releases.append(release)
            staged.append([_stage(ledger, release, "model", b"w")])

        def accepting(index):
            return lambda ledger_: ledger_.releases.accept(
                releases[index].release_id,
                staged[index],
                [AssignAuthority("slot:race", releases[index].release_id)],
            )

        first, second, _ = _interleave(ledger, accepting(0), accepting(1), after=AFTER_FENCE)

        (loser,) = [r for r in (first, second) if isinstance(r, OperationFailed)]
        (winner,) = [r for r in (first, second) if isinstance(r, ReleaseBundle)]
        assert "already belongs" in str(loser)
        assert ledger.releases.authority("slot:race") == winner.release.release_id
        states = {ledger.releases.get(r.release_id).state for r in releases}
        assert states == {"accepted", "preparing"}

    def test_an_audit_event_holds_the_audit_lock_until_the_release_commits(self, ledger: Ledger):
        """Audit rows commit in ``seq`` order: another writer's event waits for the release's."""
        build_id = _build(ledger)
        named = _checkpoint(ledger, build_id).outputs[0]
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w")]

        bundle, _, locked = _interleave(
            ledger,
            lambda ledger_: ledger_.releases.accept(
                release.release_id,
                staged,
                [InsertAuditEvent(AuditEvent("release.accept", "alice"))],
            ),
            lambda other: other.store.set_name("audited", named.id, named.version, actor="bob"),
            after="INSERT INTO registry_audit",
        )

        assert locked, "set_name's audit row was written inside the release's transaction"
        assert isinstance(bundle, ReleaseBundle) and bundle.release.state == "accepted"
        actions = [(e["action"], e["actor"]) for e in ledger.store.read_events()]
        assert actions[-2:] == [("release.accept", "alice"), ("name_set", "bob")]


def _value(ledger: Ledger, artifact_id: str, provenance_hash: str, data: bytes) -> int:
    """A ready version of *artifact_id* holding *data*, outside any build: an id's value."""
    version = ledger.store.create_artifact(
        artifact_id, provenance_hash, SPEC, computation_key=PARENT
    )
    ledger.store.write_blob(artifact_id, version, data)
    ledger.store.finalize_artifact(artifact_id, version, "{}", row_count=None, byte_size=len(data))
    return version


class TestPromotionsInARelease:
    """A revert stages copies of earlier versions, and a release makes them current together."""

    def _reverted(self, ledger: Ledger) -> tuple[Release, list[StagedVersion]]:
        """Two ids at their second values, with copies of their first staged in a final."""
        for artifact_id in ("x", "y"):
            _value(ledger, artifact_id, f"{artifact_id}-k1", f"{artifact_id}1".encode())
            _value(ledger, artifact_id, f"{artifact_id}-k2", f"{artifact_id}2".encode())
        release = ledger.releases.prepare(_build(ledger), "a1", "final")
        staged = [
            ledger.store.stage_promotion(artifact_id, 1, release_id=release.release_id)
            for artifact_id in ("x", "y")
        ]
        assert all(item is not None for item in staged)
        return release, [item for item in staged if item is not None]

    def test_staged_copies_become_current_together_on_accept(self, ledger: Ledger):
        release, staged = self._reverted(ledger)
        for item in staged:
            assert _state(ledger, item) == "building"
            assert ledger.store.get_latest_version(item.artifact_id).version == 2

        bundle = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("cell", release.release_id)]
        )

        for artifact_id in ("x", "y"):
            latest = ledger.store.get_latest_version(artifact_id)
            assert (latest.version, latest.provenance_hash) == (3, f"{artifact_id}-k1")
            assert latest.computation_key == PARENT
            assert ledger.store.read_blob(artifact_id, 3) == f"{artifact_id}1".encode()
            assert ledger.store.get_artifact(artifact_id, 1).state == "superseded"
        assert {(o.id, o.version) for o in bundle.outputs} == {("x", 3), ("y", 3)}
        roots = ledger.store.retention_roots(kind=HEAD_ROOT_KIND)
        assert {(r["artifact_id"], r["version"]) for r in roots} == {("x", 3), ("y", 3)}

    def test_a_rejected_release_makes_no_copy_current(self, ledger: Ledger):
        """The lease lapses before acceptance: both ids keep their second values."""
        release, staged = self._reverted(ledger)
        ledger.clock.now += 120

        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(release.release_id, staged)

        for item in staged:
            assert _state(ledger, item) == "building"
            latest = ledger.store.get_latest_version(item.artifact_id)
            assert (latest.version, latest.provenance_hash) == (2, f"{item.artifact_id}-k2")

    def test_a_copy_that_fails_to_write_keeps_the_release_from_being_accepted(
        self, ledger: Ledger, monkeypatch
    ):
        """The second copy's bytes fail: the first alone cannot become current."""
        for artifact_id in ("x", "y"):
            _value(ledger, artifact_id, f"{artifact_id}-k1", f"{artifact_id}1".encode())
            _value(ledger, artifact_id, f"{artifact_id}-k2", f"{artifact_id}2".encode())
        release = ledger.releases.prepare(_build(ledger), "a1", "final")
        first = ledger.store.stage_promotion("x", 1, release_id=release.release_id)
        assert first is not None

        def unwritable(artifact_id: str, version: int):
            raise OSError(f"disk full writing {artifact_id}@v={version}")

        monkeypatch.setattr(ledger.store.blob_store, "open_blob_writer", unwritable)
        with pytest.raises(OSError, match="disk full"):
            ledger.store.stage_promotion("y", 1, release_id=release.release_id)
        monkeypatch.undo()

        with pytest.raises(ReleaseIncomplete, match="not declared: y@v=3"):
            ledger.releases.accept(release.release_id, [first])
        assert ledger.releases.abandon(release.release_id, "a1")
        for artifact_id in ("x", "y"):
            assert ledger.store.get_latest_version(artifact_id).version == 2

    def test_an_output_with_a_newer_value_rejects_the_release(self, ledger: Ledger):
        """Another writer makes a newer y current first: x must not revert alone."""
        release, staged = self._reverted(ledger)
        _value(ledger, "y", "y-k3", b"y3")

        with pytest.raises(ReleaseOvertaken, match="y@v=3"):
            ledger.releases.accept(
                release.release_id, staged, [AdvanceReference("cell", release.release_id)]
            )

        assert ledger.store.get_latest_version("x").provenance_hash == "x-k2"
        assert ledger.store.get_latest_version("y").provenance_hash == "y-k3"
        assert ledger.releases.get(release.release_id).state == "rejected"
        assert ledger.releases.reference("cell") is None
        assert all(_state(ledger, item) == "building" for item in staged)

    def test_two_releases_crossing_on_two_ids_have_one_winner(self, ledger: Ledger):
        """A revert and a run stage x and y in opposite version order; one set is current."""
        for artifact_id in ("x", "y"):
            _value(ledger, artifact_id, f"{artifact_id}-k1", f"{artifact_id}1".encode())
            _value(ledger, artifact_id, f"{artifact_id}-k2", f"{artifact_id}2".encode())
        revert = ledger.releases.prepare(_build(ledger), "a1", "final")
        run = ledger.releases.prepare(_build(ledger), "a1", "final")

        def computed(artifact_id: str) -> StagedVersion:
            data = f"{artifact_id}3".encode()
            version = ledger.store.create_artifact(
                artifact_id, f"{artifact_id}-k3", SPEC, release_id=run.release_id
            )
            ledger.store.write_blob(artifact_id, version, data)
            return _staged_row(artifact_id, version, data)

        def copied(artifact_id: str) -> StagedVersion:
            item = ledger.store.stage_promotion(artifact_id, 1, release_id=revert.release_id)
            assert item is not None
            return item

        # x: the revert's copy is v3, the run's v4; y: the run's v3, the revert's v4.
        reverted, ran = [copied("x")], [computed("x")]
        ran.append(computed("y"))
        reverted.append(copied("y"))

        first, second, _ = _interleave(
            ledger,
            lambda ledger_: ledger_.releases.accept(revert.release_id, reverted),
            lambda other: other.releases.accept(run.release_id, ran),
            after=AFTER_FENCE,
        )

        (winner,) = [r for r in (first, second) if isinstance(r, ReleaseBundle)]
        (loser,) = [r for r in (first, second) if isinstance(r, ReleaseOvertaken)]
        key = "k1" if winner.release.release_id == revert.release_id else "k3"
        for artifact_id in ("x", "y"):
            latest = ledger.store.get_latest_version(artifact_id)
            assert latest.provenance_hash == f"{artifact_id}-{key}", str(loser)

    def test_a_copy_is_staged_only_under_an_unsettled_final(self, ledger: Ledger):
        """A copy under a checkpoint would be a current value its collector hands out."""
        _value(ledger, "x", "x-k1", b"x1")
        _value(ledger, "x", "x-k2", b"x2")
        checkpoint = ledger.releases.prepare(_build(ledger), "a1", "checkpoint")
        settled = ledger.releases.prepare(_build(ledger), "a1", "final")
        ledger.releases.accept(settled.release_id, [])

        for release_id in (checkpoint.release_id, settled.release_id, "no-such-release"):
            with pytest.raises(ValueError, match="unsettled final"):
                ledger.store.stage_promotion("x", 1, release_id=release_id)
        assert ledger.store.get_artifact("x", 3) is None

    def test_nothing_is_staged_from_a_version_that_is_not_a_result(self, ledger: Ledger):
        """A missing or still building version, or one whose bytes are gone, leaves no row."""
        _value(ledger, "x", "x-k1", b"x1")
        building = ledger.store.create_artifact("x", "x-k2", SPEC)
        _value(ledger, "z", "z-k1", b"z1")
        ledger.store.blob_store.delete_blob("z", 1)
        release = ledger.releases.prepare(_build(ledger), "a1", "final")

        for artifact_id, version in (("x", 9), ("x", building), ("z", 1)):
            staged = ledger.store.stage_promotion(
                artifact_id, version, release_id=release.release_id
            )
            assert staged is None
        assert ledger.store.get_artifact("x", building + 1) is None
        assert ledger.store.get_artifact("z", 2) is None
        assert ledger.releases.bundle(release.release_id).outputs == []


class TestCheckpointRetention:
    """What a sweep may take: collectable checkpoints and the roots that hold them."""

    def test_a_checkpoint_inside_the_floor_keeps_what_it_read(self, ledger: Ledger):
        """A notebook checkpoint part is its id's current value and records the run's
        inputs; collectable but kept by the idle floor, it still holds them."""
        for provenance in ("p-up-1", "p-up-2"):
            version = ledger.store.create_artifact("up", provenance, SPEC)
            ledger.store.write_blob("up", version, b"u")
            ledger.store.finalize_artifact("up", version, "{}", None, 1)
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        part = f"{build_id}-checkpoint-{release.sequence}-model"
        version = ledger.store.create_artifact(
            part,
            checkpoint_output_key(PARENT, build_id, release.sequence, "model"),
            SPEC,
            computation_key=PARENT,
            release_id=release.release_id,
            input_versions={"vestigo://artifact/up@v=1": "up@v=1"},
        )
        ledger.store.write_blob(part, version, b"w")
        ledger.releases.accept(release.release_id, [_staged_row(part, version, b"w")])
        _checkpoint(ledger, build_id)  # newer, so the first is collectable
        # Accept records a use on the ledger's clock; the sweep's floor reads the wall's.
        conn = ledger.store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_versions SET created_at = 0, last_used_at = 0 WHERE id = 'up'"
            )
            conn.execute(
                "UPDATE artifact_versions SET last_used_at = ? WHERE id <> 'up'", (time.time(),)
            )
            conn.commit()
        finally:
            conn.close()
        collectable = ledger.releases.collectable_checkpoints()
        assert collectable == [(part, version)]

        ledger.store.garbage_collect(
            collectable=collectable, keep_superseded=0, min_idle_seconds=3600
        )

        assert ledger.store.get_artifact(part, version) is not None
        assert ledger.store.get_artifact("up", 1) is not None

    def test_gc_under_a_root_keeps_the_head_and_collects_the_rest(self, ledger: Ledger):
        build_id = _build(ledger)
        old = _checkpoint(ledger, build_id)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"best")]
        head = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/best", release.release_id)]
        )
        candidates = [(v.id, v.version) for v in old.outputs + head.outputs]

        result = ledger.store.garbage_collect(collectable=candidates, collect_latest=True)

        assert result["deleted_count"] == len(old.outputs)
        assert all(ledger.store.get_artifact(v.id, v.version) is None for v in old.outputs)
        assert all(_state(ledger, v) == "ready" for v in head.outputs)

    def test_collectable_checkpoints_spare_the_newest_of_an_open_build(self, ledger: Ledger):
        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)

        def rows(bundle):
            return sorted((v.id, v.version) for v in bundle.outputs)

        assert ledger.releases.collectable_checkpoints() == rows(first)

        # Collectable is not collected: a root still holds what it names.
        ledger.releases.apply(
            build_id, "a1", [AdvanceReference("train/best", first.release.release_id)]
        )
        final = ledger.releases.prepare(build_id, "a1", "final")
        ledger.releases.accept(final.release_id, [_stage(ledger, final, "model", b"done")])
        collectable = ledger.releases.collectable_checkpoints()
        assert collectable == sorted(rows(first) + rows(second))

        result = ledger.store.garbage_collect(collectable=collectable, collect_latest=True)
        assert result["deleted_count"] == len(second.outputs)
        assert all(_state(ledger, v) == "ready" for v in first.outputs)
        assert all(ledger.store.get_artifact(v.id, v.version) is None for v in second.outputs)

    def test_collectable_checkpoints_keep_the_newest_k_of_an_open_build(self, ledger: Ledger):
        """A latest-k policy is the same read with a deeper bound; a finished build keeps none."""
        build_id = _build(ledger)
        checkpoints = [_checkpoint(ledger, build_id) for _ in range(4)]

        def rows(*bundles):
            return sorted((v.id, v.version) for bundle in bundles for v in bundle.outputs)

        assert ledger.releases.collectable_checkpoints(keep=2) == rows(*checkpoints[:2])
        assert ledger.releases.collectable_checkpoints(keep=10) == []
        with pytest.raises(ValueError):
            ledger.releases.collectable_checkpoints(keep=0)

        final = ledger.releases.prepare(build_id, "a1", "final")
        ledger.releases.accept(final.release_id, [_stage(ledger, final, "model", b"done")])
        assert ledger.releases.collectable_checkpoints(keep=2) == rows(*checkpoints)

    def test_collectable_checkpoints_do_not_count_every_later_checkpoint(self, tmp_path):
        """Each checkpoint asks whether *keep* newer ones exist, not how many there are:
        twice the checkpoints is twice the work, where a count of the later ones is four times.
        """
        store = ArtifactStore(tmp_path / "artifacts")
        clock = Clock()
        builds = BuildStore(store.db_path, dialect=store.dialect, clock=clock)
        ledger = Ledger(store, builds, ReleaseStore(store, builds, clock=clock), clock)
        build_id = _build(ledger)

        def steps() -> int:
            """The SQLite VM instructions one read runs."""
            ran = 0

            def count() -> int:
                nonlocal ran
                ran += 1
                return 0

            connect = ledger.releases._connect

            def counting():
                conn = connect()
                conn.set_progress_handler(count, 1)
                return conn

            ledger.releases._connect = counting
            try:
                ledger.releases.collectable_checkpoints(keep=2)
            finally:
                ledger.releases._connect = connect
            return ran

        for _ in range(60):
            _checkpoint(ledger, build_id)
        sixty = steps()
        for _ in range(60):
            _checkpoint(ledger, build_id)

        assert steps() < 2.5 * sixty
        store.close()

    def test_a_builds_checkpoints_stay_collectable_once_its_row_is_gone(self, ledger: Ledger):
        """Old builds are cleaned up, and an empty final's placeholder takes its build row."""

        def rows(bundle):
            return sorted((v.id, v.version) for v in bundle.outputs)

        finished = _build(ledger)
        kept = _checkpoint(ledger, finished)
        final = ledger.releases.prepare(finished, "a1", "final")
        ledger.releases.accept(final.release_id, [_stage(ledger, final, "model", b"done")])
        empty = _build(ledger)
        stored_nothing = _checkpoint(ledger, empty)
        ledger.releases.accept(ledger.releases.prepare(empty, "a1", "final").release_id, [])
        everything = sorted(rows(kept) + rows(stored_nothing))
        assert ledger.releases.collectable_checkpoints() == everything

        ledger.store.cleanup_failed(max_age_seconds=-10)
        assert ledger.builds.get_build(empty) is None
        ledger.clock.now += 8 * 86400
        assert ledger.builds.cleanup_old_builds(max_age_days=7) == 1
        assert ledger.builds.get_build(finished) is None

        assert ledger.releases.collectable_checkpoints() == everything

    def test_collectable_checkpoints_spare_what_the_current_attempt_resumed_from(
        self, ledger: Ledger
    ):
        def rows(*bundles):
            return sorted((v.id, v.version) for bundle in bundles for v in bundle.outputs)

        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)
        _resume(ledger, build_id, "a2", from_release=first.release.release_id)
        assert ledger.releases.collectable_checkpoints() == []

        third = _checkpoint(ledger, build_id, "a2")
        assert ledger.releases.collectable_checkpoints() == rows(second)

        # Only the current attempt's origin is spared: a fenced attempt releases nothing.
        _resume(ledger, build_id, "a3", from_release=third.release.release_id)
        assert ledger.releases.collectable_checkpoints() == rows(first, second)
