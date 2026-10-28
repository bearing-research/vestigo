"""PyTorch checkpoints as ledger releases (the ``torch`` extra).

A model and the training program around it are structural nodes, a run of the program
on concrete inputs is a build, and each checkpoint ``torch_checkpointing`` writes is
one release of that build: its files are ingested as versions under a manifest that
requires them, accepted from the writer's ``finalize_callback`` only while the run's
attempt holds the lease. A resume continues the build in a new attempt, fencing the old
one; a branch is a new build whose computation names the release it starts from.

Local filesystem and synchronous saves; the module imports neither torch nor
``torch_checkpointing``, reading a model and an optimizer through ``state_dict()`` and
``defaults``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from vestigo.build_store import BuildStore
from vestigo.graph import Edge, NodeId, NodeSpec
from vestigo.operations import AdvanceReference
from vestigo.provenance import (
    TransformSpec,
    checkpoint_output_key,
    compute_provenance_hash,
    run_output_key,
)
from vestigo.releases import Release, ReleaseRejected, ReleaseStore
from vestigo.store import ArtifactStore, StagedVersion

logger = logging.getLogger(__name__)

EXECUTOR = "vestigo.torch/train@v1"
MODEL_DOMAIN = "torch/model"
PROGRAM_DOMAIN = "torch/training-program"

_STEP_DIR = re.compile(r"step-(\d+)")
_CHUNK = 8 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fqn(obj: object) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            hasher.update(chunk)
            size += len(chunk)
    return size, hasher.hexdigest()


class CallbackArgs(Protocol):
    """What the adapter reads of ``torch_checkpointing``'s ``CallbackArgs``."""

    path: str


@dataclass(frozen=True)
class Definition:
    """A stored model or training-program node: its id and the spec it was stored from."""

    node_id: NodeId
    spec: NodeSpec


class Ledger:
    """A ledger under *artifact_dir* (SQLite and local blobs) seen as training runs.

    ``lease_seconds`` is how long a run's lease outlives its last renewal (a renewal
    thread renews it at a third of that); ``writable_seconds`` bounds an attempt's
    whole write window, past which nothing it releases is accepted.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        lease_seconds: float = 60.0,
        writable_seconds: float = 7 * 86400.0,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.lease_seconds = lease_seconds
        self.writable_seconds = writable_seconds
        self.store = ArtifactStore(self.artifact_dir)
        self.builds = BuildStore(self.store.db_path, dialect=self.store.dialect)
        self.releases = ReleaseStore(self.store, self.builds)
        self.staging_dir = self.artifact_dir / "torch"

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- Definitions ---

    def define_model(self, model: Any, *, version: str) -> Definition:
        """The model's node: its class and ``state_dict`` schema, under *version*.

        The schema is every key's shape and dtype, the interface a checkpoint keys on.
        Source is not hashed: dependencies and configuration change behaviour without
        changing it, and a comment would change it without changing behaviour, so the
        caller names the version.
        """
        schema = [
            [key, list(value.shape), str(value.dtype)]
            for key, value in sorted(model.state_dict().items())
        ]
        payload = {"class": _fqn(model), "state_dict": schema, "version": version}
        return self._put(NodeSpec(MODEL_DOMAIN, _canonical(payload)))

    def define_program(
        self,
        model: Definition,
        *,
        loss: Any,
        optimizer: Any,
        config: Mapping[str, Any] | None = None,
    ) -> Definition:
        """The training program's node: the model it implements, its loss and update.

        *loss* is a loss module, class or name; *optimizer* an optimizer, read as its
        class and ``defaults``; *config* whatever else the caller declares part of the
        program (schedule, accumulation, precision), as JSON. Two programs that differ
        in any of them are different programs over one model.

        Raises:
            TypeError: a value is not JSON (an optimizer default held as a tensor).
        """
        payload = {
            "loss": loss if isinstance(loss, str) else _fqn(loss),
            "optimizer": {"class": _fqn(optimizer), "defaults": dict(optimizer.defaults)},
            "config": dict(config or {}),
        }
        return self._put(
            NodeSpec(PROGRAM_DOMAIN, _canonical(payload), edges=(Edge("impl", 0, model.node_id),))
        )

    def _put(self, spec: NodeSpec) -> Definition:
        [node_id] = self.store.graph.put_batch([spec])
        return Definition(node_id, spec)

    def _definition(self, node_id: NodeId) -> Definition:
        record = self.store.graph.get(node_id)
        if record is None:
            raise ValueError(f"node {node_id} is not in the ledger")
        edges = tuple(
            Edge(e.role, e.position, e.target_node_id) for e in self.store.graph.inputs(node_id)
        )
        spec = NodeSpec(record.domain, record.payload, edges, record.payload_version)
        return Definition(node_id, spec)

    # --- Executions ---

    def computation_key(
        self, program: Definition, inputs: Mapping[str, str], init: str | None = None
    ) -> str:
        """The bound computation: the program, its inputs by role, the release it starts from.

        *inputs* name concrete versions (an artifact URI with its version, a table at a
        snapshot); two runs of one program on two snapshots share the program's node
        and differ here.
        """
        return compute_provenance_hash([], self._spec(program, inputs, init))

    @staticmethod
    def _spec(program: Definition, inputs: Mapping[str, str], init: str | None) -> TransformSpec:
        return TransformSpec(
            EXECUTOR,
            {"program": program.node_id, "inputs": dict(inputs), "init": init},
            [],
        )

    def start(self, program: Definition, inputs: Mapping[str, str] | None = None) -> Run:
        """A new execution of *program* on *inputs*, in its first attempt."""
        return self._open(program, dict(inputs or {}), None)

    def branch(
        self, program: Definition, from_release: str, inputs: Mapping[str, str] | None = None
    ) -> Run:
        """A new execution starting from *from_release*, an accepted release of any run.

        The release is part of the computation, so a branch never dedups against its
        source or another branch of it; :meth:`Run.restore` hands back its files.

        Raises:
            ValueError: *from_release* is not an accepted release.
        """
        source = self.releases.get(from_release)
        if source is None or source.state != "accepted":
            raise ValueError(f"release {from_release} is not an accepted release")
        return self._open(program, dict(inputs or {}), from_release)

    def _open(self, program: Definition, inputs: dict[str, str], init: str | None) -> Run:
        spec = self._spec(program, inputs, init)
        key = compute_provenance_hash([], spec)
        build_id = uuid.uuid4().hex
        # A build references a version, so a run whose outputs do not exist yet targets
        # a placeholder its final release fails.
        placeholder = f"torch_{build_id}"
        version = self.store.create_artifact(
            placeholder,
            run_output_key(key, build_id, "run"),
            spec,
            minted=True,
            definition=[program.spec],
            computation_key=key,
        )
        self.builds.create_build(
            build_id,
            placeholder,
            version,
            executor_ref=EXECUTOR,
            params=spec.params,
            computation_key=key,
            dispatch="external",
        )
        attempt = uuid.uuid4().hex
        self.builds.open_attempt(
            build_id,
            attempt,
            writable_until=time.time() + self.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.lease_seconds,
        )
        return Run(self, build_id, attempt, key, program, restores_from=init)

    def find(
        self, program: Definition, inputs: Mapping[str, str] | None = None, init: str | None = None
    ) -> str | None:
        """The newest unfinished execution of this computation, to resume; or None."""
        build = self.builds.find_open_build(self.computation_key(program, inputs or {}, init))
        return build.build_id if build is not None else None

    def resume(self, execution_id: str, from_release: str | None = None) -> Run:
        """Continue *execution_id* in a new attempt, fencing every earlier one.

        *from_release* defaults to the execution's latest accepted checkpoint; with
        none, the attempt starts from the execution's own initial state.

        Raises:
            ValueError: the execution does not exist, is finished, another attempt's
                lease on it is live, or *from_release* is not an accepted release of it.
        """
        build = self.builds.get_build(execution_id)
        if build is None or build.computation_key is None:
            raise ValueError(f"execution {execution_id} does not exist")
        if from_release is None:
            latest = self.releases.latest_accepted(execution_id)
            from_release = latest.release_id if latest is not None else None
        attempt = uuid.uuid4().hex
        self.releases.resume(
            execution_id,
            attempt,
            writable_until=time.time() + self.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.lease_seconds,
            from_release=from_release,
        )
        params = build.params or {}
        return Run(
            self,
            execution_id,
            attempt,
            build.computation_key,
            self._definition(params["program"]),
            restores_from=from_release or params.get("init"),
        )

    # --- Checkpoints ---

    def release_of(self, execution_id: str, step: int) -> Release | None:
        """The execution's newest accepted checkpoint of *step*, or None."""
        for bundle in reversed(self.releases.bundles_of(execution_id, "checkpoint")):
            if bundle.release.state == "accepted" and any(
                v.params.get("manifest") and v.params.get("step") == step for v in bundle.outputs
            ):
                return bundle.release
        return None

    def _accept(self, writer: Checkpoints, path: Path) -> Release:
        match = _STEP_DIR.fullmatch(path.name)
        if match is None or path.parent != writer.attempt_dir:
            raise ValueError(f"{path} is not a checkpoint id of attempt {writer.attempt}")
        step = int(match.group(1))
        done = self.release_of(writer.execution_id, step)
        if done is not None and done.attempt == writer.attempt:
            # The writer retried a finalize whose acceptance had landed.
            return done
        files = sorted(p for p in path.rglob("*") if p.is_file())
        release = self.releases.prepare(writer.execution_id, writer.attempt, "checkpoint")
        prefix = f"torch_{writer.execution_id}@step={step}"
        staged: list[StagedVersion] = []
        try:
            for file in files:
                rel = file.relative_to(path).as_posix()
                label = f"file={quote(rel, safe='')}"
                artifact_id = f"{prefix}@{label}"
                size, digest = _digest(file)
                version = self._create(writer, release, artifact_id, label, {"file": rel}, step)
                self.store.publish_blob_from_path(artifact_id, version, file)
                staged.append(StagedVersion(artifact_id, version, "", None, size, digest))
            manifest = _canonical(
                {
                    "step": step,
                    "files": [
                        {
                            "file": s.artifact_id.split("@file=", 1)[1],
                            "artifact_id": s.artifact_id,
                            "version": s.version,
                            "byte_size": s.byte_size,
                            "sha256": s.content_sha256,
                        }
                        for s in staged
                    ],
                }
            )
            version = self._create(
                writer,
                release,
                prefix,
                "manifest",
                {"manifest": True},
                step,
                requires=[(s.artifact_id, s.version) for s in staged],
            )
            self.store.write_blob(prefix, version, manifest)
            staged.append(
                StagedVersion(
                    prefix, version, "", None, len(manifest), hashlib.sha256(manifest).hexdigest()
                )
            )
            self.releases.accept(release.release_id, staged)
        except ReleaseRejected:
            raise
        except Exception:
            self.releases.abandon(release.release_id, writer.attempt)
            raise
        # The ledger holds the bytes now; the written directory was only staging.
        shutil.rmtree(path)
        logger.info("Accepted step %d of execution %s", step, writer.execution_id)
        accepted = self.releases.get(release.release_id)
        assert accepted is not None
        return accepted

    def _create(
        self,
        writer: Checkpoints,
        release: Release,
        artifact_id: str,
        label: str,
        params: dict[str, Any],
        step: int,
        requires: list[tuple[str, int]] | None = None,
    ) -> int:
        return self.store.create_artifact(
            artifact_id,
            checkpoint_output_key(
                writer.computation_key, writer.execution_id, release.sequence, label
            ),
            TransformSpec(EXECUTOR, {**params, "step": step}, []),
            requires=requires or (),
            definition=[writer.program],
            release_id=release.release_id,
            computation_key=writer.computation_key,
        )


@dataclass(frozen=True)
class Checkpoints:
    """The ``finalize_callback`` that makes a written checkpoint a release.

    Holds paths and ids only, and opens the ledger per call, so it can cross into the
    process an asynchronous saver writes from.
    """

    artifact_dir: Path
    execution_id: str
    attempt: str
    computation_key: str
    program: NodeSpec

    @property
    def attempt_dir(self) -> Path:
        return self.artifact_dir / "torch" / self.execution_id / self.attempt

    def __call__(self, args: CallbackArgs) -> None:
        with Ledger(self.artifact_dir) as ledger:
            ledger._accept(self, Path(args.path))


@dataclass
class Run:
    """One attempt at an execution: where its checkpoints go and what it starts from.

    Renews its lease from a thread until :meth:`finish` or :meth:`close`; use it as a
    context manager. ``restores_from`` is the accepted release :meth:`restore` returns.
    """

    ledger: Ledger
    execution_id: str
    attempt: str
    computation_key: str
    program: Definition
    restores_from: str | None = None
    finished: bool = False
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renewal: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._renewal = threading.Thread(
            target=self._renew, name=f"lease:{self.execution_id}", daemon=True
        )
        self._renewal.start()

    def _renew(self) -> None:
        every = self.ledger.lease_seconds / 3
        while not self._stop.wait(every):
            if not self.ledger.builds.renew_lease(
                self.execution_id, self.attempt, self.ledger.lease_seconds
            ):
                logger.warning("Attempt %s lost execution %s", self.attempt, self.execution_id)
                return

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def finalize_callback(self) -> Checkpoints:
        """Pass as ``CheckpointManager.Config.finalize_callback``."""
        return Checkpoints(
            self.ledger.artifact_dir,
            self.execution_id,
            self.attempt,
            self.computation_key,
            self.program.spec,
        )

    def checkpoint_id(self, step: int) -> str:
        """Where to save *step*: under this attempt's own directory, never reused."""
        return str(self.finalize_callback.attempt_dir / f"step-{step}")

    def release_of(self, step: int) -> Release | None:
        """This execution's accepted checkpoint of *step*, once its save has returned."""
        return self.ledger.release_of(self.execution_id, step)

    def restore(self) -> Path | None:
        """The files of ``restores_from``, written back into a directory, or None.

        Pass the directory to ``CheckpointManager.load``. Every file is checked
        against the digest the ledger recorded for it.

        Raises:
            ValueError: a file's bytes are missing or differ from what was accepted.
        """
        if self.restores_from is None:
            return None
        bundle = self.ledger.releases.bundle(self.restores_from)
        assert bundle is not None
        target = self.finalize_callback.attempt_dir / "restore"
        if target.exists():
            shutil.rmtree(target)
        for version in bundle.outputs:
            rel = version.params.get("file")
            if rel is None:
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            reader = self.ledger.store.open_blob_reader(version.id, version.version)
            if reader is None:
                raise ValueError(f"{version.id}@v={version.version} has no bytes")
            with reader as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, _CHUNK)
            if _digest(dest)[1] != version.content_sha256:
                raise ValueError(f"{version.id}@v={version.version} differs from its release")
        return target

    def set_head(self, name: str, release_id: str) -> None:
        """Point the reference *name* at an accepted release (a best-validation head).

        Raises:
            ReleaseRejected: this attempt no longer holds the execution.
            OperationFailed: the release is not accepted.
        """
        self.ledger.releases.apply(
            self.execution_id, self.attempt, [AdvanceReference(name, release_id)]
        )

    def finish(self) -> None:
        """End the execution: an empty final release completes it; nothing resumes it.

        Raises:
            ReleaseRejected: this attempt no longer holds the execution.
        """
        release = self.ledger.releases.prepare(self.execution_id, self.attempt, "final")
        self.ledger.releases.accept(release.release_id, [])
        self.finished = True
        self._stop.set()

    def close(self) -> None:
        """Stop renewing. An unfinished execution with a checkpoint stays resumable,
        its lease dropped so a resume need not wait; one without fails."""
        self._stop.set()
        if self._renewal is not None:
            self._renewal.join()
        if self.finished:
            return
        builds = self.ledger.builds
        if self.ledger.releases.latest_accepted(self.execution_id) is not None:
            builds.renew_lease(self.execution_id, self.attempt, lease_duration_seconds=0.0)
        else:
            builds.fail_build(
                self.execution_id, "ended before a checkpoint", lease_owner=self.attempt
            )
