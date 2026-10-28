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

With a signing key (the ``sign`` extra), each rank signs its report with Ed25519 and
rank 0 signs the checkpoint's manifest from the reports that verify; a ledger with the
matching verify key finds a checkpoint's files in its signed manifest and loads one only
once its bytes match the size and digest signed there, so neither the blob store nor
the metadata database, during the save or after it, can change what a load
deserializes.

Saves may be synchronous or asynchronous (the writer then runs in a subprocess, which is
why everything the writer holds is a picklable description of the ledger, opened per
call), on one rank or many (rank and world size from the default process group, else
``RANK`` and ``WORLD_SIZE``, read when each rank configures its manager).

An ``ExportedProgram`` the run captured is recorded beside its checkpoints as an
observed implementation: the archive with its versions and constraints, and a declared
scope and coverage. It never enters a node, so no definition changes because of it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, cast
from urllib.parse import quote

try:
    import torch.distributed as dist
    import torch.export
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
from vestigo.releases import (
    UNSETTLED,
    Release,
    ReleaseBundle,
    ReleaseRejected,
    ReleaseStore,
    lease_held,
)
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore, ArtifactVersion, BuildLeaseLost, StagedVersion

logger = logging.getLogger(__name__)

EXECUTOR = "vestigo.torch/train@v1"
DERIVE_EXECUTOR = "vestigo.torch/derive@v1"
MODEL_DOMAIN = "torch/model"
PROGRAM_DOMAIN = "torch/training-program"
# The root of every checkpoint id; a path under it names no file on any disk.
ROOT = "vestigo-torch"

_STEP_DIR = re.compile(r"step-(\d+)")
# An execution's placeholder, and any output a step of one of its attempts staged.
_PLACEHOLDER = re.compile(r"torch_([0-9a-f]{32})")
_STAGED = re.compile(r"torch_([0-9a-f]{32})@attempt=([0-9a-f]{32})@step=\d+")
_CHUNK = 8 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fqn(obj: object) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _archive_metadata(path: Path) -> dict[str, Any]:
    """What a PT2 archive says of itself: its own versions, and each program's.

    The layout is torch's and unversioned; what an archive lacks is None.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = names[0].split("/", 1)[0] if names else ""

        def text(name: str) -> str | None:
            try:
                return archive.read(f"{root}/{name}").decode().strip()
            except KeyError:
                return None

        programs = {}
        for name in names:
            entry = PurePosixPath(name)
            if entry.parent == PurePosixPath(root, "models") and entry.suffix == ".json":
                program = json.loads(archive.read(name))
                programs[entry.stem] = {
                    "schema_version": program.get("schema_version"),
                    "opset_version": program.get("opset_version"),
                    "torch_version": program.get("torch_version"),
                    "range_constraints": program.get("range_constraints"),
                    "verifiers": program.get("verifiers"),
                    "guards": len(program.get("guards_code") or []),
                }
        # The program's identity is its graph, weights and constants: the zip's
        # timestamps differ per save, the sample inputs and serialization id per export.
        contents = hashlib.sha256()
        for name in sorted(names):
            inner = name.split("/", 1)[-1]
            if inner.startswith("data/sample_inputs/") or inner == ".data/serialization_id":
                continue
            entry_hash = hashlib.sha256()
            with archive.open(name) as stream:
                while chunk := stream.read(_CHUNK):
                    entry_hash.update(chunk)
            contents.update(f"{name}\0{entry_hash.hexdigest()}\0".encode())
        return {
            "contents_sha256": contents.hexdigest(),
            "archive_version": text("archive_version"),
            "data_version": text(".data/version"),
            "serialization_id": text(".data/serialization_id"),
            "programs": programs,
        }


def _ed25519() -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:
        raise ImportError("signed checkpoints need the sign extra: install vestigo[sign]") from exc
    return ed25519


def _signed(manifest: Mapping[str, Any]) -> bytes:
    """What a manifest's signature covers: every field but the signature."""
    return _canonical({k: v for k, v in manifest.items() if k != "signature"})


def _file_digest(path: Path) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            hasher.update(chunk)
            size += len(chunk)
    return size, hasher.hexdigest()


def _rank() -> tuple[int, int]:
    """This process's rank and the world size: the process group's, else the launcher's.

    Read when a rank configures its manager: an asynchronous writer's subprocess has
    no process group, nor the environment when the group was given its rank directly.
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
    one machine need both. *signing_key* and *verify_key* are raw Ed25519 keys (32
    bytes each); see :class:`Ledger`.
    """

    artifact_dir: Path
    dsn: str | None = None
    blobs: RemoteBlobs | None = None
    lease_seconds: float = 60.0
    writable_seconds: float = 7 * 86400.0
    signing_key: bytes | None = field(default=None, repr=False)
    verify_key: bytes | None = None

    @contextmanager
    def open(self) -> Generator[Ledger]:
        ledger = Ledger(
            self.artifact_dir,
            dsn=self.dsn,
            blobs=self.blobs,
            lease_seconds=self.lease_seconds,
            writable_seconds=self.writable_seconds,
            signing_key=self.signing_key,
            verify_key=self.verify_key,
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


@dataclass(frozen=True)
class _File:
    """A file of an accepted checkpoint: the version holding it, and its size and digest."""

    artifact_id: str
    version: int
    byte_size: int | None
    sha256: str | None


class Ledger:
    """A ledger seen as training runs; see :class:`LedgerConfig` for the arguments.

    ``lease_seconds`` is how long a run's lease outlives its last renewal (a thread
    renews it at a third of that); ``writable_seconds`` bounds an attempt's whole write
    window, past which nothing it releases is accepted.

    With ``signing_key`` (a raw Ed25519 private key) each rank signs its report and
    rank 0 every checkpoint it accepts, and the ledger verifies with the key's public
    half. With ``verify_key``
    alone it verifies and cannot sign: a reader (an evaluation job, another team).
    Either way every load is verified, and an unsigned checkpoint is refused. The
    signing key travels with :class:`Checkpoints` to every rank and the async writer;
    give it to the training job, never to readers.

    Raises:
        ValueError: a key is not a raw Ed25519 key, or *verify_key* is not
            *signing_key*'s public half.
    """

    def __init__(
        self,
        artifact_dir: Path,
        *,
        dsn: str | None = None,
        blobs: RemoteBlobs | None = None,
        lease_seconds: float = 60.0,
        writable_seconds: float = 7 * 86400.0,
        signing_key: bytes | None = None,
        verify_key: bytes | None = None,
    ) -> None:
        if signing_key is not None:
            own = (
                _ed25519()
                .Ed25519PrivateKey.from_private_bytes(signing_key)
                .public_key()
                .public_bytes_raw()
            )
            if verify_key is not None and verify_key != own:
                raise ValueError("verify_key is not signing_key's public half")
            verify_key = own
        elif verify_key is not None:
            _ed25519().Ed25519PublicKey.from_public_bytes(verify_key)
        self.config = LedgerConfig(
            Path(artifact_dir),
            dsn,
            blobs,
            lease_seconds,
            writable_seconds,
            signing_key,
            verify_key,
        )
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
        """A new execution starting from *from_release*, an accepted checkpoint of any run.

        The release is part of the computation, so a branch never dedups against its
        source or another branch of it; :meth:`Run.restore` names it.

        Raises:
            ValueError: *from_release* is not an accepted checkpoint, or was collected.
        """
        self.checkpoint_id(from_release)
        source = self.releases.bundle(from_release)
        assert source is not None
        # A use now, so a collection with min_idle_seconds spares it until the load.
        for output in source.outputs:
            self.store.record_use(output.id, output.version)
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
                lease on it is live, or *from_release* is not an accepted checkpoint of
                it, or was collected.
        """
        build = self.builds.get_build(execution_id)
        if build is None or build.computation_key is None:
            raise ValueError(f"execution {execution_id} does not exist")
        if from_release is None:
            latest = self.releases.latest_accepted(execution_id)
            from_release = latest.release_id if latest is not None else None
        if from_release is not None:
            self.checkpoint_id(from_release)
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
                if version.params.get("manifest") and "step" in version.params:
                    yield version.params["step"], bundle

    def release_of(self, execution_id: str, step: int) -> Release | None:
        """The execution's newest accepted checkpoint of *step*, or None."""
        for at, bundle in self._accepted(execution_id):
            if at == step:
                return bundle.release
        return None

    def final_checkpoint(self, execution_id: str) -> Release | None:
        """The checkpoint a finished execution ended on, which :meth:`collect` keeps."""
        head = self.releases.reference(f"{ROOT}/{execution_id}")
        return self.releases.get(head[0]) if head is not None else None

    def checkpoint_id(self, release_id: str) -> str:
        """The checkpoint id an accepted release loads from.

        Raises:
            ValueError: the release is not an accepted checkpoint, or was collected.
        """
        release = self.releases.get(release_id)
        if release is not None and release.state == "accepted":
            for step, bundle in self._accepted(release.build_id):
                if bundle.release.release_id == release_id:
                    return f"{ROOT}/{release.build_id}/{release.attempt}/step-{step}"
        raise ValueError(f"release {release_id} is not an accepted checkpoint, or was collected")

    def configure(self, config: Any) -> Any:
        """Point a ``CheckpointManager.Config`` at the ledger for loads only; returns it.

        For a job that loads checkpoints and saves none, such as an evaluation or an
        export: no run, no lease, and a save through it raises. Load by
        :meth:`checkpoint_id`; with a verify key, every load checks the signature.
        """
        config.storage_config = LedgerStorageConfig(self.config)
        return config

    def _files(self, at: _Location) -> dict[str, _File] | None:
        """The files of the accepted checkpoint *at* names, by path in it; None if none.

        With a verify key, as its signed manifest lists them: the database is not trusted.

        Raises:
            OSError: with a verify key, the manifest is missing, malformed, unsigned,
                does not verify, or is signed for another checkpoint.
        """
        bundle = self._checkpoint(at)
        if bundle is None:
            return None
        if self.config.verify_key is None:
            return {
                v.params["file"]: _File(v.id, v.version, v.byte_size, v.content_sha256)
                for v in bundle.outputs
                if "file" in v.params
            }
        where = f"step {at.step} of execution {at.execution_id}"
        rows = [v for v in bundle.outputs if v.params.get("manifest")]
        if len(rows) != 1:
            raise OSError(f"{where} has {len(rows)} manifests")
        raw = self.store.read_blob(rows[0].id, rows[0].version)
        manifest = self._verified(raw, f"the manifest of {where}")
        named = (manifest["execution"], manifest["attempt"], manifest["step"], manifest["release"])
        if named != (at.execution_id, at.attempt, at.step, bundle.release.release_id):
            raise OSError(f"the manifest of {where} is signed for another checkpoint")
        return {
            e["file"]: _File(e["artifact_id"], e["version"], e["byte_size"], e["sha256"])
            for e in manifest["files"]
        }

    def _checkpoint(self, at: _Location) -> ReleaseBundle | None:
        for step, bundle in self._accepted(at.execution_id):
            if step == at.step and bundle.release.attempt == at.attempt:
                return bundle
        return None

    def _sign(self, document: dict[str, Any]) -> dict[str, Any]:
        """*document* with its signature, given a signing key; else as it is."""
        if self.config.signing_key is None:
            return document
        key = _ed25519().Ed25519PrivateKey.from_private_bytes(self.config.signing_key)
        return {**document, "signature": key.sign(_signed(document)).hex()}

    def _verified(self, raw: bytes | None, what: str) -> dict[str, Any]:
        """The signed document *raw* holds, once its signature verifies under the verify key.

        Its fields are trusted only after that, so only its envelope is checked first.

        Raises:
            OSError: *raw* is missing, is not a JSON object, is unsigned, or its
                signature does not verify.
        """
        from cryptography.exceptions import InvalidSignature

        assert self.config.verify_key is not None
        if raw is None:
            raise OSError(f"{what} is missing")
        try:
            document = json.loads(raw)
        except ValueError as exc:
            raise OSError(f"{what} is malformed") from exc
        if not isinstance(document, dict):
            raise OSError(f"{what} is malformed")
        if "signature" not in document:
            raise OSError(f"{what} is not signed")
        public = _ed25519().Ed25519PublicKey.from_public_bytes(self.config.verify_key)
        try:
            public.verify(bytes.fromhex(document["signature"]), _signed(document))
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise OSError(f"the signature on {what} does not verify") from exc
        return document

    # --- Derived releases ---

    def derive(
        self, from_release: str, name: str, outputs: Mapping[str, bytes | Path]
    ) -> ReleaseBundle:
        """Publish what was computed from an accepted checkpoint as a release of its own.

        An export, evaluation metrics, an inference package: made once the checkpoint
        was accepted, so never part of it. *outputs* maps a label to the bytes or a
        file. They become current together, as the final release of a build keyed by
        the checkpoint's release and *name*, which each output's spec names (``from``).
        That is the only edge: it is not an input, so keeping a derived release keeps
        only its own outputs, never the checkpoint. Deriving *name* from the same
        checkpoint again returns the release already there and stores nothing.

        Raises:
            ValueError: *from_release* is not an accepted checkpoint, *name* or
                *outputs* is empty, or *name* was derived from a later checkpoint of
                the execution, whose outputs this would replace as current.
        """
        if not name or not outputs or not all(outputs):
            raise ValueError("a derived release needs a name and labelled outputs")
        # Before the checkpoint is looked up: it may since have been collected.
        existing = self.derived(from_release, name)
        if existing is not None:
            return existing
        execution_id = self.checkpoint_id(from_release).split("/")[1]
        source = self.releases.get(from_release)
        assert source is not None
        # A trimmed derivation is not found again, so this also refuses to redo one.
        prefix = f"torch_{execution_id}@derived={quote(name, safe='')}@output="
        for origin in {
            v.params["from"]
            for v in self.store.list_versions_by_id_prefix(prefix)
            if v.state == "ready"
        }:
            later = self.releases.get(origin)
            if later is not None and later.sequence > source.sequence:
                raise ValueError(
                    f"{name!r} is derived from a later checkpoint of execution "
                    f"{execution_id}; derive the earlier one under another name"
                )
        key, spec = self._derivation(from_release, name)

        build_id = uuid.uuid4().hex
        placeholder = f"torch_{build_id}"
        target = self.store.create_artifact(
            placeholder,
            run_output_key(key, build_id, "derived"),
            spec,
            minted=True,
            computation_key=key,
        )
        self.builds.create_build(
            build_id,
            placeholder,
            target,
            executor_ref=DERIVE_EXECUTOR,
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
        release = self.releases.prepare(build_id, attempt, "final")
        created: list[tuple[str, int]] = []
        try:
            staged = []
            for label, value in outputs.items():
                artifact_id = (
                    f"torch_{execution_id}@derived={quote(name, safe='')}"
                    f"@output={quote(label, safe='')}"
                )
                version = self.store.create_artifact(
                    artifact_id,
                    derive_subkey(key, f"output={quote(label, safe='')}"),
                    TransformSpec(DERIVE_EXECUTOR, {**spec.params, "output": label}, []),
                    computation_key=key,
                    release_id=release.release_id,
                )
                created.append((artifact_id, version))
                if isinstance(value, Path):
                    size, digest = _file_digest(value)
                    self.store.publish_blob_from_path(artifact_id, version, value)
                    staged.append(StagedVersion(artifact_id, version, "", None, size, digest))
                else:
                    staged.append(self._write_small(artifact_id, version, value))
            # Nothing resumes a derivation, so its lease need only be live at acceptance,
            # however long the uploads took.
            self.builds.renew_lease(build_id, attempt, self.lease_seconds)
            return self.releases.accept(release.release_id, staged)
        except BaseException:
            self.releases.abandon(release.release_id, attempt)
            for artifact_id, version in created:
                self.store.fail_artifact(artifact_id, version)
            self.builds.fail_build(build_id, "derivation failed", lease_owner=attempt)
            raise

    def derived(self, from_release: str, name: str) -> ReleaseBundle | None:
        """The release derived from *from_release* under *name*, or None.

        Found by the derivation's key alone, so still once the checkpoint is collected.
        """
        key, _ = self._derivation(from_release, name)
        stored = [v for v in self.store.versions_of_computation(key) if v.release_id]
        if not stored:
            return None
        # Two racing derivations: the later holds the outputs ready, the earlier superseded.
        stored.sort(key=lambda v: v.state != "ready")
        release_id = stored[0].release_id
        assert release_id is not None
        return self.releases.bundle(release_id)

    @staticmethod
    def _derivation(from_release: str, name: str) -> tuple[str, TransformSpec]:
        """The key and spec of *name* derived from a release; its id is unique, so enough."""
        spec = TransformSpec(DERIVE_EXECUTOR, {"from": from_release, "name": name}, [])
        return compute_provenance_hash([], spec), spec

    # --- Retention ---

    def collect(self, *, keep: int, min_idle_seconds: float = 0.0, dry_run: bool = False) -> dict:
        """Delete the checkpoints no execution needs any more; ``garbage_collect``'s counts.

        Kept: of each unfinished execution, its newest *keep* checkpoints and the one
        its attempt resumed from; the release each unfinished branch starts from; the
        checkpoint a finished execution ended on; any release a head points at. The
        rows staged for a save that can no longer be released are failed and collected
        (``orphaned`` counts them; a dry run only counts): every row a fenced or ended
        attempt staged and never released, and of an attempt still holding its
        execution, those under an abandoned release and those of a step at or before
        the newest it accepted. Failed rows go too, but for the placeholder a build
        still names (a finished execution's record). Observations stay.

        A branch opened while this runs may lose the release it starts from; run it
        with *min_idle_seconds* or while none opens.

        Raises:
            ValueError: *keep* is below 1.
        """
        # Versions before builds: a row listed here was staged by an attempt the build
        # read below already names, so a live attempt's rows are never taken for orphans.
        versions = self.store.list_versions_by_id_prefix("torch_")
        builds = {
            match.group(1): self.builds.get_build(match.group(1))
            for v in versions
            if (match := _PLACEHOLDER.fullmatch(v.id))
        }
        # From the listing itself: a release accepted before it holds no row it shows
        # unreleased, and a step accepted takes no later save.
        accepted: dict[tuple[str, str], int] = {}
        for v in versions:
            match = _STAGED.fullmatch(v.id)
            if match is not None and v.state == "ready" and v.params.get("manifest"):
                attempt = (match.group(1), match.group(2))
                accepted[attempt] = max(v.params["step"], accepted.get(attempt, -1))
        settled: dict[str, bool] = {}

        def given_up(release_id: str) -> bool:
            if release_id not in settled:
                release = self.releases.get(release_id)
                settled[release_id] = release is not None and release.state in (
                    "abandoned",
                    "rejected",
                )
            return settled[release_id]

        orphans, failed = [], []
        for v in versions:
            if v.state == "failed":
                # A placeholder its build still names is that build's record.
                match = _PLACEHOLDER.fullmatch(v.id)
                build = builds.get(match.group(1)) if match is not None else None
                if build is None or (build.artifact_id, build.version) != (v.id, v.version):
                    failed.append((v.id, v.version))
                continue
            match = _STAGED.match(v.id)
            if match is None or v.state != "building":
                continue
            build = builds.get(match.group(1))
            if build is None or build.state != "building" or build.attempt != match.group(2):
                orphans.append((v.id, v.version))
            elif v.release_id is not None:
                if given_up(v.release_id):
                    orphans.append((v.id, v.version))
            elif v.params["step"] <= accepted.get((match.group(1), match.group(2)), -1):
                orphans.append((v.id, v.version))
        spare = []
        for build in builds.values():
            if build is None or build.state != "building":
                continue
            init = (build.params or {}).get("init")
            source = self.releases.bundle(init) if init else None
            if source is not None:
                spare += [(v.id, v.version) for v in source.outputs]
        if not dry_run:
            for artifact_id, version in orphans:
                self.store.fail_artifact(artifact_id, version)
        result = self.store.garbage_collect(
            collectable=[*self.releases.collectable_checkpoints(keep=keep), *orphans, *failed],
            spare=spare,
            min_idle_seconds=min_idle_seconds,
            dry_run=dry_run,
        )
        return {**result, "orphaned": len(orphans)}

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

    def observations(self, execution_id: str) -> list[ArtifactVersion]:
        """The implementations the execution's attempts recorded, oldest first."""
        build = self.builds.get_build(execution_id)
        if build is None or build.computation_key is None:
            raise ValueError(f"execution {execution_id} does not exist")
        # Every execution of one computation shares its key; the id is the execution's.
        prefix = f"torch_{execution_id}@observed="
        return [
            v
            for v in self.store.versions_of_computation(build.computation_key)
            if v.state == "ready" and v.id.startswith(prefix)
        ]

    def _require_lease(self, writer: Checkpoints) -> None:
        conn = self.store._get_connection()
        try:
            held = lease_held(conn, writer.execution_id, writer.attempt, time.time())
        finally:
            conn.close()
        if not held:
            raise BuildLeaseLost(
                f"attempt {writer.attempt} no longer holds execution {writer.execution_id}: "
                "it finished, or another attempt took it over"
            )

    def _observe(
        self, writer: Checkpoints, archive: Path, scope: str, coverage: str
    ) -> ArtifactVersion:
        # Before anything is looked up or staged: a stale attempt records nothing.
        self._require_lease(writer)
        metadata = _archive_metadata(archive)
        artifact_id = f"torch_{writer.execution_id}@observed={quote(scope, safe='')}"
        key = derive_subkey(
            writer.computation_key,
            f"build={writer.execution_id}",
            f"attempt={writer.attempt}",
            f"scope={quote(scope, safe='')}",
            f"coverage={quote(coverage, safe='')}",
            f"observed={metadata['contents_sha256']}",
        )
        recorded = self.store.find_version_by_provenance(artifact_id, key)
        if recorded is not None:
            return recorded
        version = self.store.create_artifact(
            artifact_id,
            key,
            TransformSpec(
                EXECUTOR,
                {
                    "observed": scope,
                    "coverage": coverage,
                    "attempt": writer.attempt,
                    **metadata,
                },
                [],
            ),
            definition=[writer.program],
            computation_key=writer.computation_key,
        )
        size, digest = _file_digest(archive)
        try:
            self.store.publish_blob_from_path(artifact_id, version, archive)
            stored = self.store.finalize_artifact(
                artifact_id,
                version,
                "",
                None,
                size,
                digest,
                fence=lambda conn, *_: lease_held(
                    conn, writer.execution_id, writer.attempt, time.time()
                ),
            )
        except Exception:
            # Failed, not left building: cleanup takes the row and its archive.
            self.store.fail_artifact(artifact_id, version)
            raise
        assert stored is not None
        return stored

    def _report(
        self, writer: Checkpoints, at: _Location, rank: int, files: list[dict[str, Any]]
    ) -> None:
        artifact_id, version = self._stage(writer, at, f"rank={rank}", {"rank": rank})
        # Signed by the rank that hashed the bytes, so rank 0 signs no digest a bucket
        # writer swapped in between this report and the acceptance.
        report = {
            "execution": at.execution_id,
            "attempt": at.attempt,
            "step": at.step,
            "rank": rank,
            "files": files,
        }
        self._write_small(artifact_id, version, _canonical(self._sign(report)))

    def _files_reported(self, at: _Location, rank: int, raw: bytes) -> list[dict[str, Any]]:
        """The files *rank* reported writing of the step; verified, given a signing key.

        Raises:
            OSError: the report does not verify, or is signed for another rank or step.
        """
        if self.config.signing_key is None:
            return json.loads(raw)["files"]
        what = f"rank {rank}'s report of step {at.step} of execution {at.execution_id}"
        report = self._verified(raw, what)
        named = (report["execution"], report["attempt"], report["step"], report["rank"])
        if named != (at.execution_id, at.attempt, at.step, rank):
            raise OSError(f"{what} is signed for another checkpoint")
        return report["files"]

    def _accept(self, writer: Checkpoints, at: _Location, world: int) -> Release:
        for step, bundle in self._accepted(at.execution_id):
            if step == at.step and bundle.release.attempt == at.attempt:
                # The writer retried a finalize whose acceptance had landed.
                return bundle.release
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
        files = [
            entry
            for rank, (*_, data) in enumerate(reports)
            for entry in self._files_reported(at, rank, data)
        ]
        if len({entry["file"] for entry in files}) != len(files):
            raise ValueError(f"two ranks wrote one file of step {at.step}")

        # A release that neither settled nor was abandoned (the database failed under
        # both) would hold up every later one; a writer has one save in flight, so it
        # is dead.
        for stale in self.releases.releases_of(at.execution_id, "checkpoint"):
            if stale.attempt == at.attempt and stale.state in UNSETTLED:
                self.releases.abandon(stale.release_id, at.attempt)
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
            manifest: dict[str, Any] = {
                "execution": at.execution_id,
                "attempt": at.attempt,
                "step": at.step,
                "release": release.release_id,
                "computation_key": writer.computation_key,
                "files": files,
            }
            manifest = self._sign(manifest)
            manifest_id, version = self._stage(
                writer,
                at,
                "manifest",
                {"manifest": True},
                requires=[(e["artifact_id"], e["version"]) for e in files],
                release_id=release.release_id,
            )
            staged.append(self._write_small(manifest_id, version, _canonical(manifest)))
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
        barrier stays: rank 0 accepts only after every rank has written). Call it once
        the process group is up: the rank and world size are read here.
        """
        rank, world = _rank()
        config.storage_config = LedgerStorageConfig(self.ledger, self, rank, world)
        config.pre_finalize_callback = report
        config.finalize_callback = self
        config.save.writer_config.write_to_temp_dir = False
        return config

    def __call__(self, args: Any) -> None:
        """``finalize_callback``: on rank 0, make the step a release."""
        at = self._own(args.path)
        storage = _ledger_storage(args)
        if storage.rank != 0:
            return
        with self.ledger.open() as ledger:
            ledger._accept(self, at, storage.world)

    def _own(self, path: Path | str) -> _Location:
        at = _Location.of(path)
        if (at.execution_id, at.attempt) != (self.execution_id, self.attempt):
            raise PermissionError(f"{path} is not a checkpoint of attempt {self.attempt}")
        return at


def report(args: Any) -> None:
    """``pre_finalize_callback``: this rank records what it wrote of the step."""
    _ledger_storage(args).report(Path(args.path))


def _ledger_storage(args: Any) -> LedgerStorage:
    storage = args.storage
    if not isinstance(storage, LedgerStorage):
        raise TypeError("the writer's storage is not the ledger; use Checkpoints.configure")
    return storage


@dataclass
class LedgerStorageConfig(StorageConfig):
    ledger: LedgerConfig
    # None for a storage that only loads (Ledger.configure).
    checkpoints: Checkpoints | None = None
    rank: int = 0
    world: int = 1

    def create_storage(self) -> LedgerStorage:
        return LedgerStorage(self.ledger, self.checkpoints, self.rank, self.world)


class _Upload(io.RawIOBase):
    """One file's bytes into its version's blob; committed only on a clean close."""

    def __init__(self, storage: LedgerStorage, at: _Location, artifact_id: str, version: int):
        self._storage, self._at = storage, at
        self._artifact_id, self._version = artifact_id, version
        self._hasher = hashlib.sha256()
        self._size = 0
        self._writer = None  # set once open; the finalizer of a failed open needs it
        self._opened = storage.ledger.open()
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
    in memory once while it loads). Nothing is renamed, deleted or overwritten. A storage
    without *checkpoints* (:meth:`Ledger.configure`) only loads.
    """

    def __init__(
        self,
        ledger: LedgerConfig,
        checkpoints: Checkpoints | None = None,
        rank: int = 0,
        world: int = 1,
    ) -> None:
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.rank, self.world = rank, world
        self._lock = threading.Lock()
        self._written: dict[_Location, list[dict[str, Any]]] = {}

    @property
    def mmap_fill_workers(self) -> int | None:
        # Each stream reads its file whole to check it; ranged readers would each do so.
        return 1

    # --- Writing ---

    def _writer(self) -> Checkpoints:
        if self.checkpoints is None:
            raise PermissionError("this storage only loads; save through a run's configure")
        return self.checkpoints

    def stream_write(self, path: Path) -> io.IOBase:
        writer = self._writer()
        at = writer._own(path)
        if at.file is None:
            raise IsADirectoryError(f"{path} is a checkpoint, not a file in one")
        label = f"file={quote(at.file, safe='')}"
        with self.ledger.open() as ledger:
            artifact_id, version = ledger._stage(writer, at, label, {"file": at.file})
        return _Upload(self, at, artifact_id, version)

    def write(self, path: Path, data: Any) -> None:
        with self.stream_write(path) as stream:
            stream.write(data)

    def _wrote(self, at: _Location, entry: dict[str, Any]) -> None:
        step = _Location(at.execution_id, at.attempt, at.step, None)
        with self._lock:
            # A save of the step that failed before its report wrote some files too.
            written = self._written.setdefault(step, [])
            written[:] = [e for e in written if e["file"] != entry["file"]] + [entry]

    def report(self, path: Path) -> None:
        """Record, as this rank's report for the step, every file it wrote of it."""
        writer = self._writer()
        at = writer._own(path)
        with self._lock:
            files = self._written.pop(at, [])
        with self.ledger.open() as ledger:
            ledger._report(writer, at, self.rank, files)

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

    def _accepted_files(self, path: Path) -> tuple[_Location, dict[str, _File]]:
        try:
            at = _Location.of(path)
        except FileNotFoundError:
            return _Location("", "", 0, None), {}
        with self.ledger.open() as ledger:
            files = ledger._files(at)
        return at, files or {}

    def _file(self, path: Path) -> _File:
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
        # The loader reads this many bytes, so with a verify key it is the signed size.
        size = self._file(path).byte_size
        assert size is not None
        return size

    def stream_read(self, path: Path, read_args: ReadArgs | None = None) -> io.RawIOBase:
        file = self._file(path)
        with self.ledger.open() as ledger:
            data = ledger.store.read_blob(file.artifact_id, file.version)
        if data is None or hashlib.sha256(data).hexdigest() != file.sha256:
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
    # Held by a renewal and by finish's final release, so no renewal reads a finish
    # as a lost lease.
    _renewing: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
            with self._renewing:
                if self._stop.is_set():
                    return
                try:
                    renewed = self.ledger.builds.renew_lease(
                        self.execution_id, self.attempt, self.ledger.lease_seconds
                    )
                except Exception:
                    # A locked database, a dropped connection: the next tick retries.
                    logger.warning(
                        "Attempt %s could not renew its lease on execution %s",
                        self.attempt,
                        self.execution_id,
                        exc_info=True,
                    )
                    continue
            if not renewed:
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

    def observe(self, program: Any, *, scope: str, coverage: str) -> ArtifactVersion:
        """Record an ``ExportedProgram`` this attempt captured, as what it observed.

        The PT2 archive ``torch.export.save`` writes is stored as a version of the
        execution, its archive, schema and opset versions, torch version and range
        constraints as metadata. *scope* says what the graph is of (``model_forward``:
        not the training loop, the update or the checkpointing) and *coverage* how much
        of it (``complete_under_declared_constraints``); the ledger records both and
        reads neither. No node changes, so neither the program's nor the model's id
        moves. The same program exported again by the same attempt under the same
        scope and coverage, from example inputs of the same shapes, is one version:
        the example values and the zip's timestamps are not compared. Inputs of
        another shape change the recorded graph, and each attempt records its own.

        Raises:
            BuildLeaseLost: this attempt no longer holds the execution (it finished,
                or another attempt took it over).
        """
        # On disk, not in memory: an archive carries the weights.
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "program.pt2"
            torch.export.save(program, archive)
            return self.ledger._observe(self.checkpoints, archive, scope, coverage)

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

        The checkpoint it ended on (this attempt's last, else the one it restored)
        stays, under the reference ``vestigo-torch/<execution>``;
        :meth:`Ledger.final_checkpoint` names it.

        Raises:
            ReleaseRejected: this attempt no longer holds the execution.
        """
        ended_on = next(
            (
                bundle.release.release_id
                for _, bundle in self.ledger._accepted(self.execution_id)
                if bundle.release.attempt == self.attempt
            ),
            self.restores_from,
        )
        operations = [AdvanceReference(f"{ROOT}/{self.execution_id}", ended_on)] if ended_on else []
        release = self.ledger.releases.prepare(self.execution_id, self.attempt, "final")
        with self._renewing:
            self.ledger.releases.accept(release.release_id, [], operations)
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
