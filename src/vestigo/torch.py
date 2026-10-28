"""PyTorch checkpoints as ledger releases (the ``torch`` extra).

A model and the training program around it are structural nodes, a run of the program
on concrete inputs is a build, and each checkpoint ``torch_checkpointing`` writes is
one release of that build. The writer's storage is the ledger itself: every file a rank
writes is a ``building`` version created before its bytes, on whatever blob backend the
ledger has (local disk, S3, GCS, Azure), so a crash leaves rows garbage collection
reaps, never stray objects. Before the barrier each rank reports what it wrote; after
it, rank 0 gathers every report into one release under a manifest that requires the
files, accepted only while the run's attempt holds the lease. Nothing is renamed: the
writer runs with ``write_to_temp_dir=False``, the configuration ``torch_checkpointing``
names for a backend whose commit is an external registry.

A resume continues the build in a new attempt, fencing the old one; a branch is a new
build whose computation names the release it starts from. Loads read accepted releases
through the same storage, checking each file against its recorded digest.

Saves may be synchronous or asynchronous (the writer then runs in a subprocess, which is
why everything the writer holds is a picklable description of the ledger, opened per
call), on one rank or many (rank and world size from the default process group, else
``RANK`` and ``WORLD_SIZE``).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, cast
from urllib.parse import quote

try:
    import torch.distributed as dist
    from torch_checkpointing.storage.base_storage import ReadArgs, Storage, StorageConfig
except ImportError as exc:
    raise ImportError("vestigo.torch needs the torch extra: install vestigo[torch]") from exc

from vestigo.blob_store import (
    AzureBlobStore,
    BlobSettings,
    BlobStore,
    GCSBlobStore,
    S3BlobStore,
)
from vestigo.build_store import BuildStore
from vestigo.graph import Edge, NodeId, NodeSpec
from vestigo.operations import AdvanceReference
from vestigo.provenance import (
    TransformSpec,
    compute_provenance_hash,
    derive_subkey,
    run_output_key,
)
from vestigo.releases import Release, ReleaseBundle, ReleaseRejected, ReleaseStore
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore, ArtifactVersion, StagedVersion

logger = logging.getLogger(__name__)

EXECUTOR = "vestigo.torch/train@v1"
MODEL_DOMAIN = "torch/model"
PROGRAM_DOMAIN = "torch/training-program"
# The root of every checkpoint id; a path under it names no file on any disk.
ROOT = "vestigo-torch"

_STEP_DIR = re.compile(r"step-(\d+)")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fqn(obj: object) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _rank() -> tuple[int, int]:
    """This process's rank and the world size: the process group's, else the launcher's.

    An asynchronous writer's subprocess has no process group, but inherits the
    environment ``torchrun`` set.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


# --- Where the ledger is ---


@dataclass(frozen=True)
class RemoteBlobs:
    """An object store for the ledger's bytes; ``backend`` is ``s3``, ``gcs`` or ``azure``.

    ``bucket`` is the container on Azure.
    """

    backend: str
    bucket: str
    prefix: str = "artifacts"
    settings: BlobSettings = field(default_factory=BlobSettings)

    def __post_init__(self) -> None:
        if self.backend not in ("s3", "gcs", "azure"):
            raise ValueError(f"backend must be s3, gcs or azure, not {self.backend!r}")

    def open(self) -> BlobStore:
        if self.backend == "s3":
            return S3BlobStore.from_settings(self.settings, self.bucket, self.prefix)
        if self.backend == "gcs":
            return GCSBlobStore.from_settings(self.settings, self.bucket, self.prefix)
        return AzureBlobStore.from_settings(self.settings, self.bucket, self.prefix)


@dataclass(frozen=True)
class LedgerConfig:
    """A picklable description of a ledger, which every rank and writer opens itself.

    Metadata is SQLite under *artifact_dir* unless *dsn* names a Postgres database;
    bytes are local under it unless *blobs* names an object store. Ranks on more than
    one machine need both.
    """

    artifact_dir: Path
    dsn: str | None = None
    blobs: RemoteBlobs | None = None
    lease_seconds: float = 60.0
    writable_seconds: float = 7 * 86400.0

    @contextmanager
    def open(self) -> Generator[Ledger]:
        ledger = Ledger(
            self.artifact_dir,
            dsn=self.dsn,
            blobs=self.blobs,
            lease_seconds=self.lease_seconds,
            writable_seconds=self.writable_seconds,
        )
        try:
            yield ledger
        finally:
            ledger.close()


@dataclass(frozen=True)
class Definition:
    """A stored model or training-program node: its id and the spec it was stored from."""

    node_id: NodeId
    spec: NodeSpec


@dataclass(frozen=True)
class _Location:
    """What a path under :data:`ROOT` names: a step of an attempt, and a file in it."""

    execution_id: str
    attempt: str
    step: int
    file: str | None

    @classmethod
    def of(cls, path: Path | str) -> _Location:
        parts = PurePosixPath(path).parts
        match = _STEP_DIR.fullmatch(parts[3]) if len(parts) >= 4 else None
        if parts[:1] != (ROOT,) or match is None:
            raise FileNotFoundError(f"{path} is not a checkpoint id")
        return cls(parts[1], parts[2], int(match.group(1)), "/".join(parts[4:]) or None)

    @property
    def prefix(self) -> str:
        """The artifact id of the step's manifest, and the stem of every id under it."""
        return f"torch_{self.execution_id}@attempt={self.attempt}@step={self.step}"


class Ledger:
    """A ledger seen as training runs; see :class:`LedgerConfig` for the arguments.

    ``lease_seconds`` is how long a run's lease outlives its last renewal (a thread
    renews it at a third of that); ``writable_seconds`` bounds an attempt's whole write
    window, past which nothing it releases is accepted.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        dsn: str | None = None,
        blobs: RemoteBlobs | None = None,
        lease_seconds: float = 60.0,
        writable_seconds: float = 7 * 86400.0,
    ) -> None:
        self.config = LedgerConfig(Path(artifact_dir), dsn, blobs, lease_seconds, writable_seconds)
        self.store = ArtifactStore(
            self.config.artifact_dir,
            blob_store=blobs.open() if blobs is not None else None,
            dialect=PostgresDialect(dsn) if dsn is not None else None,
        )
        self.builds = BuildStore(self.store.db_path, dialect=self.store.dialect)
        self.releases = ReleaseStore(self.store, self.builds)

    @property
    def lease_seconds(self) -> float:
        return self.config.lease_seconds

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
        source or another branch of it; :meth:`Run.restore` names it.

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
            writable_until=time.time() + self.config.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.lease_seconds,
        )
        return Run(self, Checkpoints(self.config, build_id, attempt, key, program.spec), init)

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
            writable_until=time.time() + self.config.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.lease_seconds,
            from_release=from_release,
        )
        params = build.params or {}
        program = self._definition(params["program"])
        checkpoints = Checkpoints(
            self.config, execution_id, attempt, build.computation_key, program.spec
        )
        return Run(self, checkpoints, from_release or params.get("init"))

    # --- Checkpoints ---

    def _accepted(self, execution_id: str) -> Iterator[tuple[int, ReleaseBundle]]:
        """The execution's accepted checkpoints with their steps, newest first."""
        for bundle in reversed(self.releases.bundles_of(execution_id, "checkpoint")):
            if bundle.release.state != "accepted":
                continue
            for version in bundle.outputs:
                if version.params.get("manifest"):
                    yield version.params["step"], bundle

    def release_of(self, execution_id: str, step: int) -> Release | None:
        """The execution's newest accepted checkpoint of *step*, or None."""
        for at, bundle in self._accepted(execution_id):
            if at == step:
                return bundle.release
        return None

    def checkpoint_id(self, release_id: str) -> str:
        """The checkpoint id an accepted release loads from.

        Raises:
            ValueError: the release is not an accepted checkpoint.
        """
        release = self.releases.get(release_id)
        if release is not None and release.state == "accepted":
            for step, bundle in self._accepted(release.build_id):
                if bundle.release.release_id == release_id:
                    return f"{ROOT}/{release.build_id}/{release.attempt}/step-{step}"
        raise ValueError(f"release {release_id} is not an accepted checkpoint")

    def _files(self, at: _Location) -> dict[str, ArtifactVersion] | None:
        """The files of the accepted checkpoint *at* names, by path in it; None if none."""
        for step, bundle in self._accepted(at.execution_id):
            if step == at.step and bundle.release.attempt == at.attempt:
                return {v.params["file"]: v for v in bundle.outputs if "file" in v.params}
        return None

    def _stage(
        self, writer: Checkpoints, at: _Location, label: str, params: dict[str, Any], **kw: Any
    ) -> tuple[str, int]:
        """A ``building`` version for one output of the step, before its bytes exist."""
        artifact_id = f"{at.prefix}@{label}" if label != "manifest" else at.prefix
        version = self.store.create_artifact(
            artifact_id,
            derive_subkey(
                writer.computation_key,
                f"build={at.execution_id}",
                f"attempt={at.attempt}",
                f"step={at.step}",
                label,
            ),
            TransformSpec(EXECUTOR, {**params, "step": at.step}, []),
            definition=[writer.program],
            computation_key=writer.computation_key,
            **kw,
        )
        return artifact_id, version

    def _write_small(self, artifact_id: str, version: int, data: bytes) -> StagedVersion:
        self.store.write_blob(artifact_id, version, data)
        return StagedVersion(
            artifact_id, version, "", None, len(data), hashlib.sha256(data).hexdigest()
        )

    def _report(self, writer: Checkpoints, at: _Location, files: list[dict[str, Any]]) -> None:
        rank, _ = _rank()
        artifact_id, version = self._stage(writer, at, f"rank={rank}", {"rank": rank})
        self._write_small(artifact_id, version, _canonical({"rank": rank, "files": files}))

    def _accept(self, writer: Checkpoints, at: _Location) -> Release:
        for step, bundle in self._accepted(at.execution_id):
            if step == at.step and bundle.release.attempt == at.attempt:
                # The writer retried a finalize whose acceptance had landed.
                return bundle.release
        _, world = _rank()
        reports: list[tuple[str, int, bytes]] = []
        for rank in range(world):
            report_id = f"{at.prefix}@rank={rank}"
            pending = [
                v
                for v in self.store.list_versions_by_id_prefix(report_id)
                if v.id == report_id and v.state == "building" and v.release_id is None
            ]
            data = self.store.read_blob(report_id, pending[-1].version) if pending else None
            if data is None:
                raise ValueError(f"rank {rank} reported nothing for step {at.step}")
            reports.append((report_id, pending[-1].version, data))
        files = [entry for *_, data in reports for entry in json.loads(data)["files"]]
        if len({entry["file"] for entry in files}) != len(files):
            raise ValueError(f"two ranks wrote one file of step {at.step}")

        release = self.releases.prepare(at.execution_id, at.attempt, "checkpoint")
        try:
            outputs = [(e["artifact_id"], e["version"]) for e in files] + [
                (report_id, version) for report_id, version, _ in reports
            ]
            if self.releases.attach(release.release_id, at.attempt, outputs) != len(outputs):
                raise ValueError(f"a file of step {at.step} is no longer staged")
            staged = [
                StagedVersion(e["artifact_id"], e["version"], "", None, e["byte_size"], e["sha256"])
                for e in files
            ] + [
                StagedVersion(
                    report_id, version, "", None, len(data), hashlib.sha256(data).hexdigest()
                )
                for report_id, version, data in reports
            ]
            manifest = _canonical({"step": at.step, "files": files})
            manifest_id, version = self._stage(
                writer,
                at,
                "manifest",
                {"manifest": True},
                requires=[(e["artifact_id"], e["version"]) for e in files],
                release_id=release.release_id,
            )
            staged.append(self._write_small(manifest_id, version, manifest))
            self.releases.accept(release.release_id, staged)
        except ReleaseRejected:
            raise
        except Exception:
            self.releases.abandon(release.release_id, at.attempt)
            raise
        logger.info("Accepted step %d of execution %s", at.step, at.execution_id)
        accepted = self.releases.get(release.release_id)
        assert accepted is not None
        return accepted


# --- The writer's side: picklable, opened per call ---


@dataclass(frozen=True)
class Checkpoints:
    """One attempt's checkpoints, as every rank and writer process sees them.

    Rank 0 has it from its :class:`Run`; the others receive it (it pickles, as
    ``torch.distributed.broadcast_object_list`` needs) and pass it to
    :meth:`configure` like rank 0.
    """

    ledger: LedgerConfig
    execution_id: str
    attempt: str
    computation_key: str
    program: NodeSpec

    def checkpoint_id(self, step: int) -> str:
        """Where to save *step*: a name under this attempt, never reused."""
        return f"{ROOT}/{self.execution_id}/{self.attempt}/step-{step}"

    def configure(self, config: Any) -> Any:
        """Point a ``CheckpointManager.Config`` at the ledger; returns it.

        Sets its storage, both write callbacks and ``write_to_temp_dir=False`` (the
        barrier stays: rank 0 accepts only after every rank has written).
        """
        config.storage_config = LedgerStorageConfig(self)
        config.pre_finalize_callback = report
        config.finalize_callback = self
        config.save.writer_config.write_to_temp_dir = False
        return config

    def __call__(self, args: Any) -> None:
        """``finalize_callback``: on rank 0, make the step a release."""
        if _rank()[0] != 0:
            return
        with self.ledger.open() as ledger:
            ledger._accept(self, self._own(args.path))

    def _own(self, path: Path | str) -> _Location:
        at = _Location.of(path)
        if (at.execution_id, at.attempt) != (self.execution_id, self.attempt):
            raise PermissionError(f"{path} is not a checkpoint of attempt {self.attempt}")
        return at


def report(args: Any) -> None:
    """``pre_finalize_callback``: this rank records what it wrote of the step."""
    storage = args.storage
    if not isinstance(storage, LedgerStorage):
        raise TypeError("the writer's storage is not the ledger; use Checkpoints.configure")
    storage.report(Path(args.path))


@dataclass
class LedgerStorageConfig(StorageConfig):
    checkpoints: Checkpoints

    def create_storage(self) -> LedgerStorage:
        return LedgerStorage(self.checkpoints)


class _Upload(io.RawIOBase):
    """One file's bytes into its version's blob; committed only on a clean close."""

    def __init__(self, storage: LedgerStorage, at: _Location, artifact_id: str, version: int):
        self._storage, self._at = storage, at
        self._artifact_id, self._version = artifact_id, version
        self._hasher = hashlib.sha256()
        self._size = 0
        self._writer = None  # set once open; the finalizer of a failed open needs it
        self._opened = storage.checkpoints.ledger.open()
        ledger = self._opened.__enter__()
        try:
            writer = ledger.store.open_blob_writer(artifact_id, version)
            self._sink = writer.__enter__()
        except BaseException:
            self._opened.__exit__(None, None, None)
            raise
        self._writer = writer

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        view = memoryview(data).cast("B")
        self._sink.write(view)
        self._hasher.update(view)
        self._size += len(view)
        return len(view)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # IOBase's exit closes as if all went well; a failed write must not commit.
        self._finish(exc_type, exc_val, exc_tb)
        super().close()

    def close(self) -> None:
        if not self.closed:
            self._finish(None, None, None)
        super().close()

    def _finish(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._writer is None:
            return
        writer, self._writer = self._writer, None
        try:
            writer.__exit__(exc_type, exc, tb)
        finally:
            self._opened.__exit__(None, None, None)
        if exc_type is None:
            self._storage._wrote(
                self._at,
                {
                    "file": self._at.file,
                    "artifact_id": self._artifact_id,
                    "version": self._version,
                    "byte_size": self._size,
                    "sha256": self._hasher.hexdigest(),
                },
            )


class LedgerStorage(Storage):
    """``torch_checkpointing``'s storage over the ledger: writes stage, reads verify.

    A write goes only under this attempt's own checkpoint ids, each file a new version
    of its own id. A read serves only accepted checkpoints, of any execution in the
    ledger, whole and checked against the digest the release recorded (so a file is held
    in memory once while it loads). Nothing is renamed, deleted or overwritten.
    """

    def __init__(self, checkpoints: Checkpoints) -> None:
        self.checkpoints = checkpoints
        self._lock = threading.Lock()
        self._written: dict[_Location, list[dict[str, Any]]] = {}

    @property
    def mmap_fill_workers(self) -> int | None:
        # Each stream reads its file whole to check it; ranged readers would each do so.
        return 1

    # --- Writing ---

    def stream_write(self, path: Path) -> io.IOBase:
        at = self.checkpoints._own(path)
        if at.file is None:
            raise IsADirectoryError(f"{path} is a checkpoint, not a file in one")
        label = f"file={quote(at.file, safe='')}"
        with self.checkpoints.ledger.open() as ledger:
            artifact_id, version = ledger._stage(self.checkpoints, at, label, {"file": at.file})
        return _Upload(self, at, artifact_id, version)

    def write(self, path: Path, data: Any) -> None:
        with self.stream_write(path) as stream:
            stream.write(data)

    def _wrote(self, at: _Location, entry: dict[str, Any]) -> None:
        step = _Location(at.execution_id, at.attempt, at.step, None)
        with self._lock:
            self._written.setdefault(step, []).append(entry)

    def report(self, path: Path) -> None:
        """Record, as this rank's report for the step, every file it wrote of it."""
        at = self.checkpoints._own(path)
        with self._lock:
            files = self._written.pop(at, [])
        with self.checkpoints.ledger.open() as ledger:
            ledger._report(self.checkpoints, at, files)

    def mkdir(self, path: Path, recursive: bool = True) -> None:
        # Ids, not directories: nothing to make.
        return None

    def rmdir(self, path: Path) -> None:
        raise PermissionError("the ledger's checkpoints are not removed by the writer")

    def delete(self, path: Path) -> None:
        raise PermissionError("the ledger's checkpoints are not removed by the writer")

    def rename(
        self,
        src_path: Path,
        dst_path: Path,
        is_directory: bool = False,
        background_cleanup: bool = False,
    ) -> None:
        raise OSError("the ledger commits by release, not rename: write_to_temp_dir=False")

    # --- Reading ---

    def _accepted_files(self, path: Path) -> tuple[_Location, dict[str, ArtifactVersion]]:
        try:
            at = _Location.of(path)
        except FileNotFoundError:
            return _Location("", "", 0, None), {}
        with self.checkpoints.ledger.open() as ledger:
            files = ledger._files(at)
        return at, files or {}

    def _version(self, path: Path) -> ArtifactVersion:
        at, files = self._accepted_files(path)
        if at.file not in files:
            raise FileNotFoundError(f"{path} is not a file of an accepted checkpoint")
        return files[at.file]

    def exists(self, path: Path) -> bool:
        at, files = self._accepted_files(path)
        if at.file is None:
            return bool(files)
        return at.file in files or any(f.startswith(at.file + "/") for f in files)

    def isdir(self, path: Path) -> bool:
        at, files = self._accepted_files(path)
        return bool(files) and (at.file is None or any(f.startswith(at.file + "/") for f in files))

    def ls(self, path: Path) -> list[str]:
        at, files = self._accepted_files(path)
        below = f"{at.file}/" if at.file else ""
        return sorted({f[len(below) :].split("/")[0] for f in files if f.startswith(below)})

    def getsize(self, path: Path) -> int:
        size = self._version(path).byte_size
        assert size is not None
        return size

    def stream_read(self, path: Path, read_args: ReadArgs | None = None) -> io.RawIOBase:
        version = self._version(path)
        with self.checkpoints.ledger.open() as ledger:
            data = ledger.store.read_blob(version.id, version.version)
        if data is None or hashlib.sha256(data).hexdigest() != version.content_sha256:
            raise OSError(f"{path} differs from the checkpoint the ledger accepted")
        # What the reader asks of a stream, readinto and seek, a BytesIO has.
        return cast(io.RawIOBase, io.BytesIO(data))

    def remap_path(self, path: Path) -> str:
        return str(path)


# --- The trainer's side ---


@dataclass
class Run:
    """One attempt at an execution, held by rank 0: its lease, heads and end.

    Renews its lease from a thread until :meth:`finish` or :meth:`close`; use it as a
    context manager. ``restores_from`` is the accepted release the attempt starts from.
    """

    ledger: Ledger
    checkpoints: Checkpoints
    restores_from: str | None = None
    finished: bool = False
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renewal: threading.Thread | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._renewal = threading.Thread(
            target=self._renew, name=f"lease:{self.execution_id}", daemon=True
        )
        self._renewal.start()

    @property
    def execution_id(self) -> str:
        return self.checkpoints.execution_id

    @property
    def attempt(self) -> str:
        return self.checkpoints.attempt

    @property
    def computation_key(self) -> str:
        return self.checkpoints.computation_key

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

    def configure(self, config: Any) -> Any:
        """As :meth:`Checkpoints.configure`."""
        return self.checkpoints.configure(config)

    def checkpoint_id(self, step: int) -> str:
        return self.checkpoints.checkpoint_id(step)

    def release_of(self, step: int) -> Release | None:
        """This execution's accepted checkpoint of *step*, once its save has completed."""
        return self.ledger.release_of(self.execution_id, step)

    def restore(self) -> str | None:
        """The checkpoint id of ``restores_from``, for ``CheckpointManager.load``; or None."""
        if self.restores_from is None:
            return None
        return self.ledger.checkpoint_id(self.restores_from)

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
