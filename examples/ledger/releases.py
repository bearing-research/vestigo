"""A long computation that checkpoints, stalls, is taken over, and finishes.

    uv run python examples/ledger/releases.py ledger

A build is one run of a computation, and an attempt is one worker's hold on it under a
lease. Each checkpoint is a release: its outputs become current together, and only
while the attempt holds the lease. Here the first worker checkpoints twice and stalls
past its lease. A second worker resumes the build from the latest checkpoint, which
fences the first, so the first's next checkpoint is rejected. The second worker moves
a head with compare-and-set and publishes the final result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

from vestigo.build_store import BuildStore
from vestigo.operations import AdvanceReference, AssertReferenceRevision
from vestigo.provenance import (
    TransformSpec,
    checkpoint_output_key,
    compute_provenance_hash,
    run_output_key,
)
from vestigo.releases import OperationFailed, Release, ReleaseRejected, ReleaseStore
from vestigo.store import ArtifactStore, StagedVersion

SPEC = TransformSpec("train@v1", {"steps": 4}, [])
KEY = compute_provenance_hash([], SPEC)
HEAD = "example/best"


def staged(store: ArtifactStore, artifact_id: str, version: int, state: dict) -> StagedVersion:
    """Write *state* as the version's bytes; what acceptance checks them against."""
    data = json.dumps(state).encode()
    store.write_blob(artifact_id, version, data)
    return StagedVersion(
        artifact_id, version, "{}", None, len(data), hashlib.sha256(data).hexdigest()
    )


def checkpoint(
    store: ArtifactStore, releases: ReleaseStore, build_id: str, attempt: str, state: dict
) -> Release:
    """One checkpoint release of the build, holding the training state."""
    release = releases.prepare(build_id, attempt, "checkpoint")
    artifact_id = f"run_{build_id}@checkpoint={release.sequence}"
    version = store.create_artifact(
        artifact_id,
        # Its own key, so a checkpoint is never the cache hit for the computation.
        checkpoint_output_key(KEY, build_id, release.sequence, "state"),
        SPEC,
        minted=True,
        computation_key=KEY,
        release_id=release.release_id,
    )
    bundle = releases.accept(release.release_id, [staged(store, artifact_id, version, state)])
    print(f"{attempt}: checkpoint {release.sequence} {state}")
    return bundle.release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()

    store = ArtifactStore(args.ledger)
    builds = BuildStore(store.db_path, dialect=store.dialect)
    releases = ReleaseStore(store, builds)
    try:
        # The build's target is the row its final release makes current.
        build_id = uuid.uuid4().hex
        target = f"run_{build_id}"
        version = store.create_artifact(
            target, run_output_key(KEY, build_id, "model"), SPEC, minted=True, computation_key=KEY
        )
        builds.create_build(
            build_id,
            target,
            version,
            executor_ref="train@v1",
            computation_key=KEY,
            dispatch="external",
        )

        # The first worker: a short lease, which it stops renewing. A live worker calls
        # builds.renew_lease(build_id, "first") well inside the lease.
        builds.open_attempt(
            build_id,
            "first",
            writable_until=time.time() + 3600,
            lease_owner="first",
            lease_duration_seconds=1.0,
        )
        for step in (1, 2):
            checkpoint(store, releases, build_id, "first", {"step": step, "loss": 1 / step})
        time.sleep(1.5)

        # The second worker finds the unfinished build by its computation and resumes it.
        found = builds.find_open_build(KEY)
        assert found is not None and found.build_id == build_id
        latest = releases.latest_accepted(build_id)
        assert latest is not None
        releases.resume(
            build_id,
            "second",
            writable_until=time.time() + 3600,
            lease_owner="second",
            from_release=latest.release_id,
        )
        bundle = releases.bundle(latest.release_id)
        assert bundle is not None
        [output] = bundle.outputs
        state = json.loads(store.read_blob(output.id, output.version) or b"{}")
        print(f"second: resumed from checkpoint {latest.sequence} {state}")

        try:
            checkpoint(store, releases, build_id, "first", {"step": 3, "loss": 0.3})
        except ReleaseRejected as exc:
            print(f"first: rejected ({exc})")

        best = None
        for step in (3, 4):
            best = checkpoint(store, releases, build_id, "second", {"step": step, "loss": 1 / step})

        # A head moves only from the revision its writer read; 0 is "does not exist".
        assert best is not None
        releases.apply(
            build_id,
            "second",
            [AssertReferenceRevision(HEAD, 0), AdvanceReference(HEAD, best.release_id)],
        )
        try:
            releases.apply(
                build_id,
                "second",
                [AssertReferenceRevision(HEAD, 0), AdvanceReference(HEAD, latest.release_id)],
            )
        except OperationFailed as exc:
            print(f"second: a stale head update failed ({exc})")
        print(f"{HEAD} -> {releases.reference(HEAD)}")

        final = releases.prepare(build_id, "second", "final")
        releases.attach(final.release_id, "second", [(target, version)])
        releases.accept(final.release_id, [staged(store, target, version, {"weights": "..."})])
        result = store.get_latest_version(target)
        assert result is not None
        print(f"second: final {result.id}@v={result.version} {result.state}")
        print(f"build {build_id}: {builds.get_build(build_id).state}")  # type: ignore[union-attr]
    finally:
        store.close()


if __name__ == "__main__":
    main()
