"""A revert of two outputs, made current together or not at all.

    uv run python examples/ledger/revert.py ledger

A model and its report are two ids, each at its second value. A revert stages a copy
of each one's first value under a build's final release, and the release's
acceptance makes both copies current in one transaction. A second revert is
overtaken: another writer makes a newer report current before it is accepted, so the
release is rejected and neither id reverts.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from vestigo.build_store import BuildStore
from vestigo.provenance import TransformSpec, compute_provenance_hash, run_output_key
from vestigo.releases import ReleaseOvertaken, ReleaseStore
from vestigo.store import ArtifactStore, StagedVersion

SPEC = TransformSpec("revert@v1", {}, [])
KEY = compute_provenance_hash([], SPEC)
IDS = ("model", "report")


def value(store: ArtifactStore, artifact_id: str, data: str) -> int:
    """A ready version of *artifact_id* holding *data*: what a run writes."""
    version = store.create_artifact(artifact_id, f"{artifact_id}:{data}", SPEC)
    store.write_blob(artifact_id, version, data.encode())
    store.finalize_artifact(artifact_id, version, "{}", row_count=None, byte_size=len(data))
    return version


def revert_build(store: ArtifactStore, builds: BuildStore) -> str:
    """A build for the revert, its attempt holding the lease."""
    build_id = uuid.uuid4().hex
    target = f"revert_{build_id}"
    version = store.create_artifact(
        target, run_output_key(KEY, build_id, "revert"), SPEC, minted=True, computation_key=KEY
    )
    builds.create_build(
        build_id,
        target,
        version,
        executor_ref="revert@v1",
        computation_key=KEY,
        dispatch="external",
    )
    builds.open_attempt(build_id, "worker", writable_until=time.time() + 3600, lease_owner="worker")
    return build_id


def stage(store: ArtifactStore, release_id: str, version: int) -> list[StagedVersion]:
    """A copy of each id's *version*, ``building`` under the release until accepted."""
    staged = [store.stage_promotion(i, version, release_id=release_id) for i in IDS]
    assert all(item is not None for item in staged), "a version to revert to is gone"
    return [item for item in staged if item is not None]


def current(store: ArtifactStore) -> str:
    values = []
    for artifact_id in IDS:
        latest = store.get_latest_version(artifact_id)
        assert latest is not None
        data = (store.read_blob(artifact_id, latest.version) or b"").decode()
        values.append(f"{artifact_id}@v={latest.version} {data}")
    return ", ".join(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()

    store = ArtifactStore(args.ledger)
    builds = BuildStore(store.db_path, dialect=store.dialect)
    releases = ReleaseStore(store, builds)
    try:
        for run in ("first", "second"):
            for artifact_id in IDS:
                value(store, artifact_id, run)
        print(f"current: {current(store)}")

        # Staged copies are building rows: until the release is accepted, nothing moves.
        final = releases.prepare(revert_build(store, builds), "worker", "final")
        staged = stage(store, final.release_id, 1)
        print(f"staged: {current(store)}")
        releases.accept(final.release_id, staged)
        print(f"reverted: {current(store)}")

        # Another writer's newer report lands between the staging and the acceptance.
        final = releases.prepare(revert_build(store, builds), "worker", "final")
        staged = stage(store, final.release_id, 2)
        value(store, "report", "third")
        try:
            releases.accept(final.release_id, staged)
        except ReleaseOvertaken as exc:
            print(f"overtaken: rejected ({exc})")
        print(f"current: {current(store)}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
