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
why everything the writer holds is a picklable description of the ledger, opened there
per call; in the trainer's process it works through the run's ledger), on one rank or
many. In a process group every rank calls :meth:`Ledger.run`
and gets its own :class:`Run`; rank 0 starts or resumes the build and holds its lease.

An ``ExportedProgram`` the run captured is recorded beside its checkpoints as an
observed implementation: the archive with its versions and constraints, and a declared
scope and coverage. It never enters a node, so no definition changes because of it.

Where another system holds the execution, as a Strata cell's server does, a
:class:`HostedRun` writes each checkpoint into a folder that system hands it, which
publishes the files as one release in the layout of :mod:`vestigo.checkpoint_files`;
:meth:`Ledger.load` reads such a checkpoint as it reads its own.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import io
import json
import logging
import os
import pickle
import re
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Collection, Generator, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path, PurePath, PurePosixPath
from types import BuiltinFunctionType, FunctionType, MethodType, TracebackType
from typing import Any, Protocol, cast
from urllib.parse import quote

try:
    import torch.distributed as dist
    import torch.export
    from torch.distributed import distributed_c10d
    from torch.distributed.distributed_c10d import _get_default_group, _get_default_store
    from torch_checkpointing.checkpoint_manager import CheckpointManager
    from torch_checkpointing.config import SyncCheckpointSaverConfig
    from torch_checkpointing.storage.base_storage import ReadArgs, Storage, StorageConfig
    from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
except ImportError as exc:
    raise ImportError("vestigo.torch needs the torch extra: install vestigo[torch]") from exc

import vestigo.digests
from vestigo import checkpoint_files
from vestigo.blob_store import (
    AzureBlobStore,
    BlobSettings,
    BlobStore,
    GCSBlobStore,
    S3BlobStore,
)
from vestigo.build_store import AttemptState, BuildState, BuildStore
from vestigo.digests import FileChanged, FileDigests, _stat_fields, _trusts
from vestigo.graph import Edge, NodeId, NodeSpec, Traversal
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
from vestigo.store import (
    _USE_RESOLUTION_SECONDS,
    ArtifactStore,
    ArtifactVersion,
    BuildLeaseLost,
    GcResult,
    StagedVersion,
)

logger = logging.getLogger(__name__)


class CollectResult(GcResult):
    """What :meth:`Ledger.collect` did: the store's sweep, and the orphans it found."""

    orphaned: int


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
_OBSERVED = re.compile(r"torch_([0-9a-f]{32})@observed=")
_CHUNK = 8 * 1024 * 1024
# The fields of each kind of signed document, and the one the other kind has and it
# lacks: one key signs both, so a report is never taken for a manifest or back.
_SIGNED = {
    "manifest": (("execution", "attempt", "step", "release", "files"), "rank"),
    "report": (("execution", "attempt", "step", "rank", "files", "inputs"), "release"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(value: Any) -> bytes:
    """*value* canonical, as a definition's payload or a term the ranks agree on.

    Raises:
        TypeError: it is not JSON (NaN and the infinities are not either).
    """
    try:
        return _canonical(value)
    except ValueError as exc:
        # What json raises for NaN, an infinity, or a value that contains itself.
        raise TypeError(f"{exc}: not JSON") from exc


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


# --- Verified inputs ---


class InputChanged(Exception):
    """A verified input's bytes are not the ones its run resolved, or are gone."""


# A file is (where its bytes are, its stat: what of it changes with any write is its key).
_Scanned = tuple[str, os.stat_result]
# What an operating system writes into a folder someone browses; never a dataset's.
_JUNK = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _reraise(exc: OSError) -> None:
    raise exc


def _real(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except RuntimeError as exc:
        # A link loop raises RuntimeError before 3.13, OSError after.
        raise OSError(f"{path} is a link loop") from exc


def _scan(path: Path) -> tuple[Path, list[tuple[str, _Scanned]]]:
    """Every file of the input at *path*, by its name under it ("" for a file), and
    where *path* leads.

    The path named may be a link. Inside a directory a link may name a file of the
    tree; any other link, or a file that is not a regular one, is refused. Files an
    operating system writes into a folder it shows (``.DS_Store``, ``Thumbs.db``,
    ``desktop.ini``, ``._`` AppleDouble files) are not the input's.

    Raises:
        OSError: the input, or a file of it, cannot be read.
        ValueError: a directory holds no file, a link it refuses, or a special file.
    """
    root = _real(path)
    if not root.is_dir():
        return root, [("", _regular(root))]
    files = []
    for top, dirs, names in os.walk(root, onerror=_reraise):
        for name in dirs:
            if os.path.islink(os.path.join(top, name)):
                raise ValueError(f"{os.path.join(top, name)} links to a directory")
        for name in names:
            if name in _JUNK or name.startswith("._"):
                continue
            entry = Path(top, name)
            # Under the real root, only a link's path can lead elsewhere.
            real = _real(entry) if entry.is_symlink() else entry
            if not real.is_relative_to(root):
                raise ValueError(f"{entry} links outside {root}")
            files.append((entry.relative_to(root).as_posix(), _regular(real)))
    if not files:
        raise _Empty(f"{path} holds no file")
    return root, sorted(files)


def _regular(real: Path) -> _Scanned:
    st = os.stat(real)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{real} is not a regular file")
    return str(real), st


def _fingerprint(files: Sequence[tuple[str, _Scanned]]) -> str:
    return hashlib.sha256(
        _canonical([[name, real, list(_stat_fields(st))] for name, (real, st) in files])
    ).hexdigest()


def _trusted(files: Sequence[tuple[str, _Scanned]], scanned_at: int) -> str:
    """*files*' fingerprint, which a later check takes as the same bytes; "" while a
    file was written too lately before the scan, as the digest cache judges it."""
    if all(_trusts(st, scanned_at) for _, (_, st) in files):
        return _fingerprint(files)
    return ""


class _Empty(ValueError):
    """A directory holds no file: refused as an input, and gone from one that held some."""


def _pin(files: list[tuple[str, _Scanned]], walk: tuple[Path, int] | None = None) -> dict[str, Any]:
    """What the key holds of a verified input: its bytes' digest, not where they are.

    A file is read only where the machine's digest cache has none for its stat, so a
    relaunch on the same data costs a stat per file. Given a directory's *walk* (its root,
    and the clock read before its scan began), the cache forgets its files that are gone.
    """
    reals = [real for _, (real, _) in files]
    with closing(FileDigests()) as cache:
        try:
            # Settled only: a file written as it is read stops the reads not yet started.
            stats = [st for _, (_, st) in files]
            digests = cache.digest_many(reals, stats=stats, settled_only=True)
        except FileChanged as exc:
            raise InputChanged(f"{exc} changed while it was hashed") from None
        if walk is not None:
            root, walked_at = walk
            cache.prune(root, reals, walked_at=walked_at)
    if files[0][0] == "":
        return {"sha256": digests[0].sha256, "size": files[0][1][1].st_size}
    listing = [
        [name, st.st_size, digest.sha256]
        for (name, (_, st)), digest in zip(files, digests, strict=True)
    ]
    return {"tree": hashlib.sha256(_canonical(listing)).hexdigest(), "files": len(listing)}


@dataclass(frozen=True)
class Remote:
    """A verified input in an object store: ``s3://``, ``gs://`` or ``az://``, the bucket
    (the container on Azure), then a key, or a prefix ending in ``/`` for every object
    under it. *settings* are the credentials and endpoint, as for :class:`RemoteBlobs`;
    every rank needs them, and they travel from rank 0 to every rank.

    Nothing is downloaded. Each object is pinned by what its store says names its bytes:
    S3's ETag; GCS's MD5, or a composite object's generation; Azure's ETag, which a
    metadata write changes too. The URI is part of the pin: unlike a local path, it
    names the same objects everywhere.

    Raises:
        ValueError: the URI has another scheme, or no bucket.
    """

    uri: str
    settings: BlobSettings = field(default_factory=BlobSettings, repr=False)

    def __post_init__(self) -> None:
        scheme, _, rest = self.uri.partition("://")
        if scheme not in _REMOTE_STORES or not rest.partition("/")[0]:
            raise ValueError(f"{self.uri!r} is not s3://, gs:// or az:// with a bucket")

    def objects(self) -> list[tuple[str, int, str]]:
        """Each object the URI names: its key, size and its bytes' name, from its store.

        A backend error propagates.
        """
        scheme, _, rest = self.uri.partition("://")
        bucket, _, path = rest.partition("/")
        store = _REMOTE_STORES[scheme].from_settings(self.settings, bucket, "")
        return store.objects(path)


_REMOTE_STORES: dict[str, Any] = {"s3": S3BlobStore, "gs": GCSBlobStore, "az": AzureBlobStore}
Input = str | Path | Remote


def _pin_remote(remote: Remote, objects: list[tuple[str, int, str]]) -> dict[str, Any]:
    if not objects:
        raise ValueError(f"{remote.uri} holds no object")
    if not remote.uri.endswith("/"):
        [(_, size, name)] = objects
        return {"uri": remote.uri, "size": size, "object": name}
    under = len(remote.uri.partition("://")[2].partition("/")[2])
    listing = sorted([key[under:], size, name] for key, size, name in objects)
    return {
        "uri": remote.uri,
        "tree": hashlib.sha256(_canonical(listing)).hexdigest(),
        "objects": len(listing),
    }


@dataclass(frozen=True)
class _Held:
    """A verified input as its run resolved it: its scan's fingerprint (a few bytes to
    every rank and writer, however many files; "" while a file was too fresh to trust
    its stat) and the bytes it was pinned to; for a remote one, the object store's,
    listed again at each check."""

    role: str
    path: str
    scanned: str
    pinned: dict[str, Any]
    remote: Remote | None = None


@dataclass(frozen=True)
class _Inputs:
    """A run's inputs: by role as the key hashes them (a label, or a path's pinned
    bytes), and the verified ones as they were scanned."""

    spec: dict[str, Any]
    held: tuple[_Held, ...] = ()


def _resolve(inputs: Mapping[str, Input] | None) -> _Inputs:
    """*inputs* with each ``Path`` and :class:`Remote` pinned to its bytes; each ``str``
    is a label.

    Raises:
        OSError: a path cannot be read.
        ValueError: a directory holds no file, a link it refuses, or a special file; a
            remote URI names no object.
        TypeError: an input is none of them.
    """
    spec: dict[str, Any] = {}
    held = []
    for role, value in (inputs or {}).items():
        if isinstance(value, Path):
            scanned_at = time.time_ns()
            root, files = _scan(value)
            # Once, at launch: a check whose stat moved need not walk the cache too.
            spec[role] = _pin(files, (root, scanned_at) if files[0][0] != "" else None)
            held.append(_Held(role, str(value.absolute()), _trusted(files, scanned_at), spec[role]))
        elif isinstance(value, Remote):
            spec[role] = _pin_remote(value, value.objects())
            held.append(_Held(role, value.uri, "", spec[role], value))
        elif isinstance(value, str):
            spec[role] = value
        else:
            raise TypeError(f"input {role!r} is a str label, a Path or a Remote, not {type(value)}")
    return _Inputs(spec, tuple(held))


def _remote_errors(remote: Remote) -> tuple[type[Exception], ...]:
    """What *remote*'s SDK raises when its store does not answer: an outage, a refusal,
    no credentials. ``OSError`` is a dropped connection under any of them."""
    scheme = remote.uri.partition("://")[0]
    if scheme == "s3":
        from botocore.exceptions import BotoCoreError, ClientError

        return OSError, BotoCoreError, ClientError
    if scheme == "gs":
        from google.api_core.exceptions import GoogleAPIError
        from google.auth.exceptions import GoogleAuthError

        return OSError, GoogleAPIError, GoogleAuthError
    from azure.core.exceptions import AzureError

    return OSError, AzureError


def _seen(held: Sequence[_Held], known: dict[str, str]) -> dict[str, Any]:
    """Each verified input as this process finds it now: its path, and its bytes' pin
    (None once they are gone) or why they cannot be read.

    An unchanged stat is the same bytes. A changed one (a chmod, a hardlink, an
    attribute a file browser adds) is read again, from the digest cache where it can
    be; *known* keeps the new stat's fingerprint, so the next check is a stat again.
    """
    seen: dict[str, Any] = {}
    for each in held:
        if each.remote is not None:
            remote = each.remote
            try:
                objects = remote.objects()
            except _remote_errors(remote) as exc:
                # The store did not answer (an outage among them): this step only.
                seen[each.role] = {"path": each.path, "unreadable": str(exc)}
            else:
                pin = _pin_remote(remote, objects) if objects else None
                seen[each.role] = {"path": each.path, "pin": pin}
            continue
        try:
            scanned_at = time.time_ns()
            _, files = _scan(Path(each.path))
            scanned = _fingerprint(files)
            if scanned in (each.scanned, known.get(each.role)):
                pin = each.pinned
            else:
                pin = _pin(files)
                if pin == each.pinned and _trusted(files, scanned_at):
                    known[each.role] = scanned
            seen[each.role] = {"path": each.path, "pin": pin}
        except (FileNotFoundError, _Empty):
            # Removed, or emptied: other bytes, as a file removed from a tree is.
            seen[each.role] = {"path": each.path, "pin": None}
        except (OSError, ValueError, InputChanged) as exc:
            seen[each.role] = {"path": each.path, "unreadable": str(exc)}
    return seen


def _stateful(value: object) -> bool:
    """A module, optimizer or scheduler: saved by its ``state_dict``, loaded through it."""
    return callable(getattr(value, "state_dict", None)) and callable(
        getattr(value, "load_state_dict", None)
    )


def _saved(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.state_dict() if _stateful(value) else value for key, value in state.items()}


def _load(manager: Any, checkpoint_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """*state* loaded from the checkpoint: objects in place, plain values replaced."""
    # A module's own tensors are the template, so its weights load where they live; an
    # optimizer's state is empty until a step, so it is read as saved and loaded.
    into = {
        key: value.state_dict() if isinstance(value, torch.nn.Module) else None
        for key, value in state.items()
    }
    loaded = manager.load(checkpoint_id, into=into)
    restored = {}
    for key, value in state.items():
        if isinstance(value, torch.nn.Module):
            restored[key] = value
        elif _stateful(value):
            value.load_state_dict(loaded[key])
            restored[key] = value
        else:
            restored[key] = loaded[key]
    return restored


def _group() -> tuple[int, int]:
    """This process's rank and the world size in its process group; one rank without.

    Raises:
        RuntimeError: the launcher started several ranks and this one has no group.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        raise RuntimeError(f"WORLD_SIZE is {world}: initialize the process group before the run")
    return 0, 1


def _portable(exc: BaseException) -> BaseException:
    """*exc* as another rank can raise it: itself if it pickles back, else its repr."""
    try:
        return pickle.loads(pickle.dumps(exc))
    except (pickle.PicklingError, TypeError, AttributeError):
        # A driver's error with arguments its class cannot be built from again.
        return RuntimeError(repr(exc))


class _Apart(RuntimeError):
    """The ranks did not meet at one of vestigo's exchanges within its timeout."""


def _term_digests(terms: Mapping[str, Any]) -> dict[str, str]:
    """Raises TypeError: a term is not JSON (an optimizer default held as a tensor)."""
    return {key: hashlib.sha256(_json(value)).hexdigest() for key, value in terms.items()}


@dataclass(frozen=True)
class _Ranks:
    """This process's place in its process group, and how vestigo's ranks exchange
    what they must agree on.

    *group* is a gloo group of every rank, for objects on the CPU whatever the
    training backend; it opens and finishes a run, at the process group's timeout,
    as rank 0 may hash a dataset meanwhile. A save's outcome goes through *store*,
    the process group's own: rank 0 posts it whoever is listening, and a rank that
    passed the barrier waits for it at most *barrier_seconds*. A gloo broadcast
    would wait for a rank that died forever, and one that timed out can abort the
    process when the late rank's bytes land.
    """

    rank: int = 0
    world: int = 1
    # A dist.ProcessGroup; new_group is typed to return a marker for a rank outside it.
    group: Any = None
    # A dist.Store, whose stub says set takes a str; it takes the bytes posted here.
    store: Any = None
    barrier_seconds: float = 0.0

    @classmethod
    def joined(cls, rank: int, world: int, barrier_seconds: float) -> _Ranks:
        """Every rank's, together: each rank makes the group at the same point."""
        store = dist.PrefixStore(f"{ROOT}/", _get_default_store())
        default = _get_default_group()
        # new_group's own default is 30 minutes, whatever the launcher chose.
        timeout = default._get_backend(default._device_types[0]).options._timeout
        group = dist.new_group(backend="gloo", timeout=timeout)
        return cls(rank, world, group, store, barrier_seconds)

    def leave(self) -> None:
        """Destroy the group, this rank's alone: each run opens its own."""
        # Not twice, nor once the default group's destruction took it with it.
        if self.group is not None and self.group in distributed_c10d._world.pg_map:
            dist.destroy_process_group(self.group)

    def share(self, result: Any = None, error: BaseException | None = None) -> Any:
        """Rank 0's *result* or *error*, sent by rank 0; returned or raised on the rest."""
        box = [result, None if error is None else _portable(error)]
        dist.broadcast_object_list(box, src=0, group=self.group)
        if box[1] is not None and self.rank:
            raise box[1]
        return box[0]

    def post(self, key: str, result: Any = None, error: BaseException | None = None) -> None:
        """Rank 0's outcome under *key*, for the ranks still waiting for it."""
        outcome = (result, None if error is None else _portable(error))
        self.store.set(key, pickle.dumps(outcome))

    def posted(self, key: str, at: str) -> Any:
        """Rank 0's outcome under *key*: returned, or its error raised.

        Raises:
            _Apart: rank 0 posted none within the barrier's timeout.
        """
        try:
            self.store.wait([key], timedelta(seconds=self.barrier_seconds))
        except RuntimeError as exc:
            # The store's timeout, a DistStoreError.
            raise _Apart(f"rank {self.rank} heard nothing from rank 0 {at}") from exc
        result, error = pickle.loads(self.store.get(key))
        if error is not None:
            raise error
        return result

    def from_rank_0[T](self, work: Callable[[], T]) -> T:
        """``work()`` run on rank 0; on every rank its result, or its error raised."""
        if self.world == 1:
            return work()
        if self.rank:
            return self.share()
        try:
            result = work()
        except BaseException as exc:
            # An interrupt too: the others are waiting for word of it.
            self.share(error=exc)
            raise
        return self.share(result)

    def raise_any(self, error: BaseException | None) -> None:
        """Raises on every rank once any rank has an *error*: its own, else the first."""
        if self.world == 1:
            if error is not None:
                raise error
            return
        errors: list[BaseException | None] = [None] * self.world
        dist.all_gather_object(
            errors, None if error is None else _portable(error), group=self.group
        )
        failed = next((each for each in errors if each is not None), None)
        if failed is not None:
            raise error or failed

    def agree(self, terms: Callable[[], Mapping[str, Any]]) -> None:
        """Raises on every rank: a rank's *terms* are not rank 0's (ValueError), or are
        not JSON (TypeError)."""
        digests: dict[str, str] = {}
        error: TypeError | None = None
        try:
            digests = _term_digests(terms())
        except TypeError as exc:
            error = exc
        self.raise_any(error)
        everyone: list[Any] = [None] * self.world
        dist.all_gather_object(everyone, digests, group=self.group)
        differ = [
            f"rank {rank}'s {', '.join(k for k in digests if theirs[k] != everyone[0][k])}"
            for rank, theirs in enumerate(everyone)
            if theirs != everyone[0]
        ]
        if differ:
            raise ValueError(f"every rank must run what rank 0 runs: {'; '.join(differ)} differ")


def _kinds(inputs: Mapping[str, Input] | None) -> dict[str, Any]:
    """*inputs* as every rank must name them: a label or URI as given, a path by role."""
    return {
        role: value
        if isinstance(value, str)
        else value.uri
        if isinstance(value, Remote)
        else type(value).__name__
        for role, value in (inputs or {}).items()
    }


def _copies(inputs: Mapping[str, Input], held: Sequence[_Held], rank: int) -> tuple[_Held, ...]:
    """This rank's own copies of rank 0's verified inputs; a remote one is the same.

    A copy is read only when its scan is not rank 0's (another machine's disk), or
    rank 0's was too fresh to trust.

    Raises:
        InputChanged: a copy has other bytes than rank 0 pinned.
        OSError, ValueError: a copy cannot be read, or holds what a scan refuses.
    """
    copies = []
    for each in held:
        if each.remote is not None:
            copies.append(each)
            continue
        path = cast(Path, inputs[each.role])
        scanned_at = time.time_ns()
        _, files = _scan(path)
        if _fingerprint(files) != each.scanned and _pin(files) != each.pinned:
            raise InputChanged(f"input {each.role!r} at {path} has other bytes on rank {rank}")
        copies.append(
            _Held(each.role, str(path.absolute()), _trusted(files, scanned_at), each.pinned)
        )
    return tuple(copies)


def _params(build: BuildState) -> dict[str, Any]:
    assert build.params is not None, "an execution's build holds its params"
    return build.params


def _attempt(state: AttemptState) -> Attempt:
    return Attempt(state.build_id, state.attempt, state.resumed_from)


def _model_payload(model: torch.nn.Module, version: str) -> dict[str, Any]:
    schema = [
        [key, list(value.shape), str(value.dtype)]
        if isinstance(value, torch.Tensor)
        # A module's get_extra_state: any object, keyed on by its type.
        else [key, None, _fqn(value)]
        for key, value in sorted(model.state_dict().items())
    ]
    return {"class": _fqn(model), "state_dict": schema, "version": version}


def _loss_payload(loss: Any, *, tensors: bool = True) -> Any:
    """*loss* as its program records it: a name as given; a class or function by its
    qualified name; a bound method by its function's and its object's, a module's
    without its tensors (a model's ``compute_loss`` is not its weights); a ``partial``
    with its arguments; a loss module by every module in it (its name there, class and
    public attributes) and the digest of each tensor in its ``state_dict``; any other
    object by its class and public attributes. A compiled loss is what it compiles.

    Raises:
        TypeError: an attribute or argument is not JSON, or a tensor has no values.
        ValueError: the loss is a lambda, or a nested function, whose captured values
            have no name.
    """
    if isinstance(loss, str):
        return loss
    # torch.compile's wrapper computes what it wraps.
    if isinstance(loss, torch.nn.Module):
        loss = getattr(loss, "_orig_mod", loss)
    elif isinstance(loss, FunctionType):
        loss = getattr(loss, "_torchdynamo_orig_callable", loss)
    if isinstance(loss, functools.partial):
        return {
            "function": _loss_payload(loss.func),
            "args": list(loss.args),
            "keywords": dict(loss.keywords),
        }
    if isinstance(loss, MethodType):
        return {"method": _named(loss), "self": _loss_payload(loss.__self__, tensors=False)}
    if isinstance(loss, FunctionType) and loss.__closure__:
        # A decorator's wrapper, or a function made by another: what it captured.
        raise ValueError(f"{_named(loss)} captures values a loss's name does not record; {_PASS}")
    if isinstance(loss, (type, FunctionType, BuiltinFunctionType)):
        return _named(loss)
    if isinstance(loss, torch.nn.Module):
        modules = [
            [name, _fqn(module), _attributes(module)] for name, module in loss.named_modules()
        ]
        if not tensors:
            return {"modules": modules}
        return {
            "modules": modules,
            "tensors": {
                key: _tensor_digest(key, value) if isinstance(value, torch.Tensor) else value
                for key, value in loss.state_dict().items()
            },
        }
    return {"class": _fqn(loss), "settings": _attributes(loss), "tensors": {}}


_PASS = "pass a loss module, a functools.partial, or the loss's name as a str"


def _named(loss: Any) -> str:
    """Raises ValueError: *loss* is a lambda or is defined in a function."""
    if loss.__name__ == "<lambda>":
        raise ValueError("a lambda loss has no name; pass a named function or a str")
    if "<locals>" in loss.__qualname__:
        raise ValueError(f"{loss.__qualname__} is defined in a function; {_PASS}")
    return f"{loss.__module__}.{loss.__qualname__}"


def _attributes(obj: object) -> dict[str, Any]:
    """*obj*'s public attributes, in its ``__dict__`` or its ``__slots__``; a module's
    ``training`` flag left out, and its submodules, which are not among them."""
    found = dict(getattr(obj, "__dict__", {}))
    for cls in type(obj).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in [slots] if isinstance(slots, str) else slots:
            if hasattr(obj, name):
                found[name] = getattr(obj, name)
    if isinstance(obj, torch.nn.Module):
        found.pop("training", None)
    return {key: value for key, value in found.items() if not key.startswith("_")}


def _tensor_digest(key: str, tensor: torch.Tensor) -> list[Any]:
    """Raises TypeError: the tensor is on the meta device, with no values to digest."""
    if tensor.is_meta:
        raise TypeError(f"tensor {key!r} is on the meta device and has no values to record")
    data = tensor.detach().cpu()
    if data.layout != torch.strided:
        data = data.to_dense()  # a sparse tensor, by its values
    data = data.contiguous()
    # Viewed as bytes, as numpy has no bfloat16.
    raw = data.reshape(-1).view(torch.uint8).numpy()
    return [list(data.shape), str(data.dtype), hashlib.sha256(raw).hexdigest()]


def _program_payload(
    loss: str | Callable[..., Any],
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # A group's own settings (a per-layer learning rate) are outside the defaults.
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    return {
        "loss": _loss_payload(loss),
        "optimizer": {
            "class": _fqn(optimizer),
            "defaults": dict(optimizer.defaults),
            "param_groups": groups,
        },
        "config": dict(config or {}),
    }


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
class Attempt:
    """One attempt of an execution, and the checkpoint it continued from.

    With none, it started from the execution's ``init``: a branch's checkpoint, or
    nothing.
    """

    execution_id: str
    attempt: str
    resumed_from: str | None


@dataclass(frozen=True)
class Execution:
    """An execution as the ledger recorded it.

    *inputs* are each role's pin as the computation key holds it, *init* the
    checkpoint it branched from. *state* is ``building`` until it finishes
    (``ready``) or fails. *attempts* are in the order they opened. *checkpoints* are
    the accepted checkpoints still stored, as (step, release_id) in the order they
    were accepted: a step saved again after a resume from an earlier one is there
    twice, and :meth:`Ledger.release_of` returns the later.
    """

    execution_id: str
    program: NodeId
    inputs: Mapping[str, Any]
    init: str | None
    state: str
    attempts: tuple[Attempt, ...]
    checkpoints: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class _Location:
    """What a path under :data:`ROOT` names: a step of an attempt, and a file in it."""

    execution_id: str
    attempt: str
    step: int
    file: str | None

    @classmethod
    def of(cls, path: Path | str) -> _Location:
        # torch_checkpointing hands over a native path, or its str: on Windows, "\\" splits.
        parts = (path if isinstance(path, PurePath) else PurePath(path)).parts
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
    window, past which nothing it releases is accepted and the run lets its lease lapse.

    With ``signing_key`` (a raw Ed25519 private key) each rank signs its report and
    rank 0 every checkpoint it accepts, and the ledger verifies with the key's public
    half. With ``verify_key``
    alone it verifies and cannot sign: a reader (an evaluation job, another team).
    Either way every load is verified, and an unsigned checkpoint is refused. The
    signing key travels from rank 0 to every rank and the async writer; give it to the
    training job, never to readers.

    Raises:
        ValueError: a key is not a raw Ed25519 key, or *verify_key* is not
            *signing_key*'s public half.
    """

    def __init__(
        self,
        artifact_dir: Path | str,
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

    def define_model(self, model: torch.nn.Module, *, version: str) -> Definition:
        """The model's node: its class and ``state_dict`` schema, under *version*.

        The schema is every key's shape and dtype, the interface a checkpoint keys on.
        Source is not hashed: dependencies and configuration change behaviour without
        changing it, and a comment would change it without changing behaviour, so the
        caller names the version.
        """
        return self._model(_model_payload(model, version))

    def _model(self, payload: Mapping[str, Any]) -> Definition:
        return self._put(NodeSpec(MODEL_DOMAIN, _json(payload)))

    def define_program(
        self,
        model: Definition,
        *,
        loss: str | Callable[..., Any],
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any] | None = None,
    ) -> Definition:
        """The training program's node: the model it implements, its loss and update.

        *loss* is a loss module (read as every module in it, with its settings, and its
        tensors), a class, a module-level function, a bound method, a ``partial`` of
        one, or a name; *optimizer* an optimizer, read as its class, ``defaults`` and
        each param group's settings; *config* every other setting that changes the
        trained weights (batch size, schedule, seed, accumulation, precision), as
        JSON. Two programs that differ in any of them are different programs over one
        model.

        Raises:
            TypeError: a value is not JSON (an optimizer default held as a tensor, a
                NaN), or a loss's tensor is on the meta device.
            ValueError: the loss is a lambda, or a function defined in a function or
                capturing values (a decorator's wrapper).
        """
        return self._program(model, _program_payload(loss, optimizer, config))

    def _program(self, model: Definition, payload: Mapping[str, Any]) -> Definition:
        return self._put(
            NodeSpec(PROGRAM_DOMAIN, _json(payload), edges=(Edge("impl", 0, model.node_id),))
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
        self,
        program: Definition,
        inputs: Mapping[str, Input] | None = None,
        init: str | None = None,
    ) -> str:
        """The bound computation: the program, its inputs by role, the release it starts from.

        *inputs* name concrete versions. A ``Path`` is pinned to its bytes: a file's
        sha256, or a directory's over its files' names and digests; where they are is
        not part of it, so the same bytes on another machine are the same computation.
        A :class:`Remote` is pinned to what its object store says names each object's
        bytes, with its URI. A ``str`` is a label taken as given (an artifact URI with
        its version, a table at a snapshot): nothing checks it names what the run reads.

        Raises:
            OSError: a path cannot be read.
            ValueError: a directory holds no file, a link out of it or to a directory,
                or a special file; a remote URI names no object.
        """
        return compute_provenance_hash([], self._spec(program, _resolve(inputs).spec, init))

    @staticmethod
    def _spec(program: Definition, inputs: Mapping[str, Any], init: str | None) -> TransformSpec:
        return TransformSpec(
            EXECUTOR,
            {"program": program.node_id, "inputs": dict(inputs), "init": init},
            [],
        )

    def run(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        loss: str | Callable[..., Any],
        version: str,
        inputs: Mapping[str, Input] | None = None,
        config: Mapping[str, Any] | None = None,
        from_release: str | None = None,
        checkpointing: CheckpointManager.Config | None = None,
    ) -> Run:
        """Resume this computation's unfinished execution, or start one.

        The computation is the program over *model* (:meth:`define_model` under
        *version*; :meth:`define_program` with *loss*, *optimizer* and *config*), its
        *inputs* (see :meth:`computation_key`), and *from_release*, an accepted
        checkpoint of any run to branch from. *checkpointing* is the
        ``CheckpointManager.Config`` the run saves and loads with, synchronous saves
        when None.

        Each ``Path`` or :class:`Remote` input is pinned once here (a digest cache on
        the machine spares a relaunch the read), and its bytes must stay: a checkpoint
        or finish after they change raises :class:`InputChanged`. Data changed after a
        crash is another computation, so the run starts fresh rather than resume on it.

        In a process group every rank calls it, as it calls ``DistributedDataParallel``.
        Rank 0 starts or resumes the run and holds its lease; each rank has its own
        :class:`Run` and saves its own files. A rank passes its own copy of each
        ``Path`` input, which must be the bytes rank 0 pinned.

        Raises:
            ValueError: the unfinished execution's lease is live under another attempt
                (the old process is still running, or was killed under
                ``lease_seconds`` ago), *from_release* is not an accepted checkpoint, or
                a rank's model, program, inputs or *from_release* are not rank 0's.
            InputChanged: a rank's copy of a ``Path`` input has other bytes.
            RuntimeError: ``WORLD_SIZE`` names several ranks and the group is not up.
        """
        model_payload = _model_payload(model, version)
        program_payload = _program_payload(loss, optimizer, config)
        terms = {"model": model_payload, "program": program_payload, "from_release": from_release}

        def opened(resolved: _Inputs, ranks: _Ranks) -> Run:
            program = self._program(self._model(model_payload), program_payload)
            execution = self._find(program, resolved, from_release)
            if execution is not None:
                return self._resume(execution, None, resolved, checkpointing, ranks)
            if from_release is not None:
                return self._branch(
                    program, from_release, resolved, checkpointing, ranks, alone=True
                )
            return self._open(program, resolved, None, checkpointing, ranks, alone=True)

        return self._collective(terms, inputs, opened, checkpointing)

    def _collective(
        self,
        terms: dict[str, Any],
        inputs: Mapping[str, Input] | None,
        opened: Callable[[_Inputs, _Ranks], Run],
        checkpointing: CheckpointManager.Config | None,
    ) -> Run:
        """The run *opened* makes, on every rank once all agree on *terms* and inputs.

        Every rank checks it runs what rank 0 runs. Rank 0 pins the inputs, every
        other rank checks its own copies against the pins, and only then does rank 0
        open the run; each other rank checks its own ledger holds it. A failure on any
        rank raises on all, and once the run is open rank 0 closes it.
        """
        saver = (checkpointing or CheckpointManager.Config.with_sync_save()).save
        barrier = saver.writer_config.checkpoint_write_barrier_timeout_sec

        def agreed() -> dict[str, Any]:
            verify_key = self.config.verify_key
            return {
                **terms,
                "inputs": _kinds(inputs),
                # A save is collective only when synchronous, and meets at the barrier.
                "saves": [isinstance(saver, SyncCheckpointSaverConfig), barrier],
                "verify_key": verify_key.hex() if verify_key is not None else None,
            }

        rank, world = _group()
        if world == 1:
            # Checked before the inputs are hashed: a bad term fails at once.
            _term_digests(agreed())
            return opened(_resolve(inputs), _Ranks())
        ranks = _Ranks.joined(rank, world, barrier)
        own: list[Run] = []
        try:
            ranks.agree(agreed)
            resolved = ranks.from_rank_0(lambda: _resolve(inputs))
            copies = resolved.held
            error: BaseException | None = None
            if rank:
                try:
                    copies = _copies(inputs or {}, resolved.held, rank)
                except (OSError, ValueError, InputChanged) as exc:
                    error = exc
            ranks.raise_any(error)

            def start() -> tuple[_Checkpoints, str | None]:
                run = opened(resolved, ranks)
                # Open from here, even if sending it to the others fails.
                own.append(run)
                return run._checkpoints, run.restores_from

            checkpoints, restores_from = ranks.from_rank_0(start)
            if rank:
                try:
                    # This rank writes through its own ledger, which must be rank 0's.
                    if self.builds.get_build(checkpoints.execution_id) is None:
                        error = ValueError(
                            f"rank {rank}'s ledger does not hold execution"
                            f" {checkpoints.execution_id}: every rank must open rank 0's ledger"
                        )
                except BaseException as exc:
                    # Any error: rank 0 waits below for word of it.
                    error = exc
                writer = replace(checkpoints, ledger=self.config, inputs=copies)
                own.append(Run(self, writer, restores_from, checkpointing, ranks))
            ranks.raise_any(error)
        except BaseException:
            # The run closes with its group; without a run, the group alone.
            if own:
                own[0].close()
            else:
                ranks.leave()
            raise
        return own[0]

    def start(
        self,
        program: Definition,
        inputs: Mapping[str, Input] | None = None,
        *,
        checkpointing: CheckpointManager.Config | None = None,
    ) -> Run:
        """A new execution of *program* on *inputs*, in its first attempt; in a process
        group, on every rank as :meth:`run` is."""
        return self._collective(
            {"program": program.node_id},
            inputs,
            lambda resolved, ranks: self._open(program, resolved, None, checkpointing, ranks),
            checkpointing,
        )

    def branch(
        self,
        program: Definition,
        from_release: str,
        inputs: Mapping[str, Input] | None = None,
        *,
        checkpointing: CheckpointManager.Config | None = None,
    ) -> Run:
        """A new execution starting from *from_release*, an accepted checkpoint of any run.

        The release is part of the computation, so a branch never dedups against its
        source or another branch of it; :meth:`Run.restore` loads it. In a process
        group, on every rank as :meth:`run` is.

        Raises:
            ValueError: *from_release* is not an accepted checkpoint, or was collected.
        """
        return self._collective(
            {"program": program.node_id, "from_release": from_release},
            inputs,
            lambda resolved, ranks: self._branch(
                program, from_release, resolved, checkpointing, ranks
            ),
            checkpointing,
        )

    def _branch(
        self,
        program: Definition,
        from_release: str,
        inputs: _Inputs,
        checkpointing: CheckpointManager.Config | None,
        ranks: _Ranks,
        *,
        alone: bool = False,
    ) -> Run:
        self.checkpoint_id(from_release)
        self._use(from_release)
        return self._open(program, inputs, from_release, checkpointing, ranks, alone=alone)

    def _use(self, release_id: str) -> None:
        """A use now of the checkpoint's outputs, so a collection with min_idle_seconds
        spares it until the load."""
        source = self.releases.bundle(release_id)
        assert source is not None
        for output in source.outputs:
            self.store.record_use(output.id, output.version)

    def _open(
        self,
        program: Definition,
        inputs: _Inputs,
        init: str | None,
        checkpointing: CheckpointManager.Config | None,
        ranks: _Ranks,
        *,
        alone: bool = False,
    ) -> Run:
        """A new execution in its first attempt. With *alone*, only if no other execution
        of the computation is unfinished as it opens: that one is resumed instead
        (refused while its lease is live), and the new one is failed unopened."""
        spec = self._spec(program, inputs.spec, init)
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
        conn = self.store._get_connection()
        try:
            # Keyed on the computation, so of two runs that both found nothing to resume
            # one opens and the other sees it.
            self.store.dialect.begin_write(conn, f"{ROOT}/{key}")
            # As find_open_build, on this transaction: Postgres lends no second
            # connection while it holds the lock.
            other = (
                conn.execute(
                    "SELECT build_id FROM artifact_builds"
                    " WHERE computation_key = ? AND state = 'building'"
                    " ORDER BY created_at DESC LIMIT 1",
                    (key,),
                ).fetchone()
                if alone
                else None
            )
            now = time.time()
            opened = other is None and self.builds.open_attempt_within(
                conn,
                build_id,
                attempt,
                writable_until=now + self.config.writable_seconds,
                lease_owner=attempt,
                lease_expires_at=now + self.lease_seconds,
                now=now,
            )
            conn.commit()
        finally:
            conn.close()
        if other is not None:
            self.builds.fail_build(build_id, f"execution {other['build_id']} opened first")
            self.store.fail_artifact(placeholder, version)
            return self._resume(other["build_id"], None, inputs, checkpointing, ranks)
        if not opened:
            raise RuntimeError(f"execution {build_id} did not open its first attempt")
        checkpoints = _Checkpoints(
            self.config, build_id, attempt, key, program.spec, inputs=inputs.held
        )
        return Run(self, checkpoints, init, checkpointing, ranks)

    def find(
        self,
        program: Definition,
        inputs: Mapping[str, Input] | None = None,
        init: str | None = None,
    ) -> str | None:
        """The newest unfinished execution of this computation, to resume; or None."""
        return self._find(program, _resolve(inputs), init)

    def _find(self, program: Definition, inputs: _Inputs, init: str | None) -> str | None:
        key = compute_provenance_hash([], self._spec(program, inputs.spec, init))
        build = self.builds.find_open_build(key)
        return build.build_id if build is not None else None

    def resume(
        self,
        execution_id: str,
        from_release: str | None = None,
        *,
        inputs: Mapping[str, Input] | None = None,
        checkpointing: CheckpointManager.Config | None = None,
    ) -> Run:
        """Continue *execution_id* in a new attempt, fencing every earlier one.

        *from_release* defaults to the execution's latest accepted checkpoint; with
        none, the attempt starts from the execution's own initial state. *inputs*, when
        given, must be the execution's; an execution with a ``Path`` or
        :class:`Remote` input needs them, since the attempt holds them as the first
        resolved them. In a process
        group, on every rank as :meth:`run` is.

        Raises:
            ValueError: the execution does not exist, is finished, another attempt's
                lease on it is live, *inputs* are not its own or are missing, or
                *from_release* is not an accepted checkpoint of it, or was collected.
        """
        return self._collective(
            {"execution": execution_id, "from_release": from_release, "given": inputs is not None},
            inputs,
            lambda resolved, ranks: self._resume(
                execution_id,
                from_release,
                None if inputs is None else resolved,
                checkpointing,
                ranks,
            ),
            checkpointing,
        )

    def _resume(
        self,
        execution_id: str,
        from_release: str | None,
        inputs: _Inputs | None,
        checkpointing: CheckpointManager.Config | None,
        ranks: _Ranks,
    ) -> Run:
        # Before the lease is taken: a build of another executor is no execution.
        build = self._build(execution_id)
        if build.computation_key is None:
            raise ValueError(f"execution {execution_id} does not exist")
        params = build.params or {}
        recorded = params.get("inputs", {})
        if inputs is None and any(isinstance(value, dict) for value in recorded.values()):
            raise ValueError(
                f"execution {execution_id} holds verified inputs (a Path or a Remote): "
                "resume it with its inputs"
            )
        if inputs is not None and inputs.spec != recorded:
            raise ValueError(f"these are not the inputs of execution {execution_id}")
        if from_release is None:
            latest = self.releases.latest_accepted(execution_id)
            from_release = latest.release_id if latest is not None else None
        if from_release is not None:
            self.checkpoint_id(from_release)
            self._use(from_release)
        program = self._definition(params["program"])
        attempt = uuid.uuid4().hex
        self.releases.resume(
            execution_id,
            attempt,
            writable_until=time.time() + self.config.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.lease_seconds,
            from_release=from_release,
        )
        checkpoints = _Checkpoints(
            self.config,
            execution_id,
            attempt,
            build.computation_key,
            program.spec,
            inputs=inputs.held if inputs is not None else (),
        )
        return Run(self, checkpoints, from_release or params.get("init"), checkpointing, ranks)

    # --- Checkpoints ---

    def _accepted(self, execution_id: str) -> Iterator[tuple[int, ReleaseBundle]]:
        """The execution's accepted checkpoints with their steps, newest first."""
        for bundle in reversed(self.releases.bundles_of(execution_id, "checkpoint")):
            if bundle.release.state != "accepted":
                continue
            step = checkpoint_files.step_of(bundle.outputs)
            if step is not None:
                yield step, bundle

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
            ValueError: the release is not an accepted checkpoint, or was collected, or
                its attempt accepted its step again after it (a host can), so its id
                loads the later one.
        """
        release = self.releases.get(release_id)
        if release is not None and release.state == "accepted":
            later = set()
            for step, bundle in self._accepted(release.build_id):
                if bundle.release.release_id == release_id:
                    if (bundle.release.attempt, step) in later:
                        raise ValueError(
                            f"release {release_id} is step {step} of an attempt that "
                            "accepted that step again; its id loads the later one"
                        )
                    return f"{ROOT}/{release.build_id}/{release.attempt}/step-{step}"
                later.add((bundle.release.attempt, step))
        raise ValueError(f"release {release_id} is not an accepted checkpoint, or was collected")

    def _newest_step(self, execution_id: str, attempt: str) -> int | None:
        """The newest step *attempt* accepted, or None."""
        return max(
            (step for step, b in self._accepted(execution_id) if b.release.attempt == attempt),
            default=None,
        )

    def load(self, release_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """*state* loaded from the accepted checkpoint *release_id*, as :meth:`Run.restore` does.

        For a job that loads and saves nothing, such as an evaluation or an export: no
        run and no lease. With a verify key, the load checks the signature.

        Raises:
            ValueError: the release is not an accepted checkpoint, or was collected.
            FileNotFoundError: the checkpoint saved nothing under a key of *state*.
            OSError: with a verify key, the checkpoint is unsigned, signed with another
                key, or changed since it was signed.
        """
        checkpoint_id = self.checkpoint_id(release_id)
        config = CheckpointManager.Config.with_sync_save()
        config.storage_config = LedgerStorageConfig(self.config, opened=self)
        manager = config.build()
        try:
            return _load(manager, checkpoint_id, state)
        finally:
            manager.close()

    def configure(self, config: CheckpointManager.Config) -> CheckpointManager.Config:
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
                path: _File(v.id, v.version, v.byte_size, v.content_sha256)
                for path, v in checkpoint_files.files_of(bundle.outputs).items()
            }
        where = f"step {at.step} of execution {at.execution_id}"
        rows = [v for v in bundle.outputs if v.params.get("manifest")]
        if len(rows) != 1:
            raise OSError(f"{where} has {len(rows)} manifests")
        raw = self.store.read_blob(rows[0].id, rows[0].version)
        manifest = self._verified(raw, f"the manifest of {where}", "manifest")
        named = (manifest["execution"], manifest["attempt"], manifest["step"], manifest["release"])
        if named != (at.execution_id, at.attempt, at.step, bundle.release.release_id):
            raise OSError(f"the manifest of {where} is signed for another checkpoint")
        try:
            return {
                e["file"]: _File(e["artifact_id"], e["version"], e["byte_size"], e["sha256"])
                for e in manifest["files"]
            }
        except (KeyError, TypeError) as exc:
            raise OSError(f"the manifest of {where} is malformed") from exc

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

    def _verified(self, raw: bytes | None, what: str, kind: str) -> dict[str, Any]:
        """The signed document of *kind* (``manifest`` or ``report``) *raw* holds, once
        its signature verifies under the verify key.

        Its fields are trusted only after that, so only its envelope is checked first.

        Raises:
            OSError: *raw* is missing, is not a JSON object, is unsigned, its signature
                does not verify, or it is not a document of *kind*.
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
        fields, other = _SIGNED[kind]
        if other in document or not all(name in document for name in fields):
            raise OSError(f"{what} is malformed")
        return document

    # --- Lineage ---

    def execution(self, execution_id: str) -> Execution:
        """What the ledger recorded of *execution_id*.

        Raises:
            ValueError: the execution does not exist.
        """
        return self._execution(self._build(execution_id))

    def _build(self, execution_id: str) -> BuildState:
        build = self.builds.get_build(execution_id)
        if build is None or build.executor_ref != EXECUTOR:
            raise ValueError(f"execution {execution_id} does not exist")
        return build

    def executions(
        self, of: Definition | None = None, *, reading: Input | None = None
    ) -> list[Execution]:
        """Every execution, oldest first, or those of *of* and reading *reading*.

        *of* is a program, or a model: the executions of every program over it.
        *reading* is pinned as :meth:`computation_key` pins an input, and matches an
        execution with an input of that pin under any role.

        Raises:
            OSError, ValueError: *reading* cannot be pinned (see :meth:`computation_key`).
        """
        programs = None
        if of is not None and of.spec.domain == MODEL_DOMAIN:
            # A program reads its model directly, so one hop finds every one.
            hop = Traversal(max_depth=1, max_nodes=2**31)
            programs = set(self.store.graph.dependents(of.node_id, traversal=hop))
        elif of is not None:
            programs = {of.node_id}
        pin = None if reading is None else _resolve({"input": reading}).spec["input"]
        found = []
        for build in self.builds.builds_of(EXECUTOR):
            params = _params(build)
            if programs is not None and params["program"] not in programs:
                continue
            if pin is not None and pin not in params["inputs"].values():
                continue
            found.append(self._execution(build))
        return found

    def resumed_from(self, release_id: str) -> list[Attempt]:
        """The attempts that continued from *release_id*, in the order they opened.

        Executions branched from it are not among them: :meth:`descendants` finds those.
        """
        return [_attempt(a) for a in self.builds.attempts_resumed_from(release_id)]

    def descendants(self, execution_id: str) -> list[Execution]:
        """The executions branched from a checkpoint of *execution_id*, at any depth.

        A branch of a branch is one; each names the checkpoint it starts from in
        ``init``. Oldest first.

        Raises:
            ValueError: the execution does not exist.
        """
        self._build(execution_id)
        branches: dict[str, list[BuildState]] = {}
        sources: dict[str, Release | None] = {}
        for build in self.builds.builds_of(EXECUTOR):
            init = _params(build).get("init")
            if init is None:
                continue
            # A release row outlives the checkpoint's bytes, so a collected one still links.
            if init not in sources:
                sources[init] = self.releases.get(init)
            source = sources[init]
            if source is not None:
                branches.setdefault(source.build_id, []).append(build)
        found, frontier = [], [execution_id]
        while frontier:
            children = [b for parent in frontier for b in branches.get(parent, [])]
            found += children
            frontier = [b.build_id for b in children]
        found.sort(key=lambda b: (b.created_at, b.build_id))
        return [self._execution(build) for build in found]

    def _execution(self, build: BuildState) -> Execution:
        params = _params(build)
        return Execution(
            build.build_id,
            params["program"],
            params["inputs"],
            params.get("init"),
            build.state,
            tuple(_attempt(a) for a in self.builds.attempts_of(build.build_id)),
            tuple(
                (step, bundle.release.release_id)
                for step, bundle in reversed(list(self._accepted(build.build_id)))
            ),
        )

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
                *outputs* is empty, or *name* was derived, or is being derived, from a
                later checkpoint of the execution, whose outputs this would replace as
                current.
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
        self._refuse_later(prefix, source, name)
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
        # Nothing resumes or reclaims a derivation, so its lease lasts its write window,
        # however long the uploads take.
        self.builds.open_attempt(
            build_id,
            attempt,
            writable_until=time.time() + self.config.writable_seconds,
            lease_owner=attempt,
            lease_duration_seconds=self.config.writable_seconds,
        )
        release = self.releases.prepare(build_id, attempt, "final")
        created: list[tuple[str, int]] = []
        try:
            for label in outputs:
                artifact_id = f"{prefix}{quote(label, safe='')}"
                version = self.store.create_artifact(
                    artifact_id,
                    derive_subkey(key, f"output={quote(label, safe='')}"),
                    TransformSpec(DERIVE_EXECUTOR, {**spec.params, "output": label}, []),
                    computation_key=key,
                    release_id=release.release_id,
                )
                created.append((artifact_id, version))
            # Again now these rows exist: a later checkpoint's derivation begun before
            # them is seen, and one begun after numbers its versions above them, so it
            # stays current whichever is accepted last.
            self._refuse_later(prefix, source, name, mine=set(created))
            staged = [
                self._write_path(artifact_id, version, value)
                if isinstance(value, Path)
                else self._write_small(artifact_id, version, value)
                for (artifact_id, version), value in zip(created, outputs.values(), strict=True)
            ]
            return self.releases.accept(release.release_id, staged)
        except BaseException:
            self.releases.abandon(release.release_id, attempt)
            for artifact_id, version in created:
                self.store.fail_artifact(artifact_id, version)
            # Failed, not left building: nothing would ever settle it.
            self.store.fail_artifact(placeholder, target)
            self.builds.fail_build(build_id, "derivation failed", lease_owner=attempt)
            raise

    def _refuse_later(
        self, prefix: str, source: Release, name: str, mine: Collection[tuple[str, int]] = ()
    ) -> None:
        """Raises ValueError: an output under *prefix* is derived, or being derived, from a
        checkpoint of the execution later than *source*."""
        for origin in {
            v.params["from"]
            for v in self.store.list_versions_by_id_prefix(prefix)
            if v.state in ("ready", "building") and (v.id, v.version) not in mine
        }:
            later = self.releases.get(origin)
            if later is not None and later.sequence > source.sequence:
                raise ValueError(
                    f"{name!r} is derived from a later checkpoint of execution "
                    f"{source.build_id}; derive the earlier one under another name"
                )

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

    def collect(
        self, *, keep: int, min_idle_seconds: float = 0.0, dry_run: bool = False
    ) -> CollectResult:
        """Delete the checkpoints no execution needs any more; ``garbage_collect``'s counts.

        Kept: of each unfinished execution, its newest *keep* checkpoints and the one
        its attempt resumed from; the release each unfinished branch starts from; the
        checkpoint a finished execution ended on; any release a head points at. The
        rows staged for a save that can no longer be released are failed and collected
        (``orphaned`` counts them; a dry run only counts): every row a fenced or ended
        attempt staged and never released, and of an attempt still holding its
        execution, those under an abandoned release and those of a step at or before
        the newest it accepted; an observation a fenced or ended attempt never
        finished storing; and the outputs of a derivation still building past its
        lease (its write window), whose process died: its release is abandoned and
        its build and placeholder fail. Failed rows go too, but for the placeholder a build still
        names (a finished execution's record). Observations stay. ``deleted_count``
        and ``deleted_bytes`` count the staged rows too, by their blobs' sizes, and a
        dry run lists them under ``collected``.

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
        # A derivation leases its whole write window, so one still building past its
        # lease died: nothing it releases would be accepted now.
        now = time.time()
        dead = {
            release.release_id: (build, release)
            for build in builds.values()
            if build is not None
            and build.executor_ref == DERIVE_EXECUTOR
            and build.state == "building"
            and build.lease_expires_at is not None
            and build.lease_expires_at <= now
            for release in self.releases.releases_of(build.build_id, "final")
            if release.state in UNSETTLED
        }
        settled: dict[str, bool] = {}

        def given_up(release_id: str) -> bool:
            if release_id not in settled:
                release = self.releases.get(release_id)
                settled[release_id] = release is not None and release.state in (
                    "abandoned",
                    "rejected",
                )
            return settled[release_id]

        orphans: list[ArtifactVersion] = []
        failed = []
        for v in versions:
            if v.state == "failed":
                # A placeholder its build still names is that build's record.
                match = _PLACEHOLDER.fullmatch(v.id)
                build = builds.get(match.group(1)) if match is not None else None
                if build is None or (build.artifact_id, build.version) != (v.id, v.version):
                    failed.append((v.id, v.version))
                continue
            if v.state != "building":
                continue
            if v.release_id in dead:
                orphans.append(v)
                continue
            observed = _OBSERVED.match(v.id)
            if observed is not None:
                # Its attempt died mid-upload; while that attempt holds, it may be storing.
                build = builds.get(observed.group(1))
                if (
                    build is None
                    or build.state != "building"
                    or build.attempt != v.params.get("attempt")
                ):
                    orphans.append(v)
                continue
            match = _STAGED.match(v.id)
            if match is None:
                continue
            build = builds.get(match.group(1))
            if build is None or build.state != "building" or build.attempt != match.group(2):
                orphans.append(v)
            elif v.release_id is not None:
                if given_up(v.release_id):
                    orphans.append(v)
            elif v.params["step"] <= accepted.get((match.group(1), match.group(2)), -1):
                orphans.append(v)
        spare = []
        for build in builds.values():
            if build is None or build.state != "building":
                continue
            init = (build.params or {}).get("init")
            source = self.releases.bundle(init) if init else None
            if source is not None:
                spare += [(v.id, v.version) for v in source.outputs]
        # A row never finalized records no size, so its blob is measured before it goes.
        sizes = {(v.id, v.version): self.store.blob_size(v.id, v.version) or 0 for v in orphans}
        if not dry_run:
            for v in orphans:
                self.store.fail_artifact(v.id, v.version)
            for build, release in dead.values():
                self.releases.abandon(release.release_id, release.attempt)
                self.store.fail_artifact(build.artifact_id, build.version)
                self.builds.fail_build(build.build_id, "derivation died", lease_owner=build.attempt)
        result = self.store.garbage_collect(
            collectable=[
                *self.releases.collectable_checkpoints(keep=keep),
                *((v.id, v.version) for v in orphans),
                *failed,
            ],
            spare=spare,
            min_idle_seconds=min_idle_seconds,
            dry_run=dry_run,
        )
        if dry_run:
            # Not failed, so the store's pass did not see them: those its idle floor
            # would not spare are counted here.
            floor = min_idle_seconds + _USE_RESOLUTION_SECONDS if min_idle_seconds else 0.0
            taken = [v for v in orphans if (v.created_at or 0.0) <= time.time() - floor]
            result["deleted_count"] += len(taken)
            result["collected"] += [
                {
                    "artifact_id": v.id,
                    "version": v.version,
                    "byte_size": sizes[(v.id, v.version)],
                    "last_used_at": v.created_at,
                }
                for v in taken
            ]
        else:
            taken = [v for v in orphans if self.store.get_artifact(v.id, v.version) is None]
        result["deleted_bytes"] += sum(sizes[(v.id, v.version)] for v in taken)
        return {**result, "orphaned": len(orphans)}

    def _stage(
        self, writer: _Checkpoints, at: _Location, label: str, params: dict[str, Any], **kw: Any
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

    def _write_path(self, artifact_id: str, version: int, path: Path) -> StagedVersion:
        """*path*'s bytes into the version, hashed as they are written: a file changed
        meanwhile is recorded with the digest of the bytes stored."""
        hasher, size = hashlib.sha256(), 0
        with open(path, "rb") as source, self.store.open_blob_writer(artifact_id, version) as sink:
            while chunk := source.read(_CHUNK):
                sink.write(chunk)
                hasher.update(chunk)
                size += len(chunk)
        return StagedVersion(artifact_id, version, "", None, size, hasher.hexdigest())

    def observations(self, execution_id: str) -> list[ArtifactVersion]:
        """The implementations the execution's attempts recorded, oldest first.

        Raises:
            ValueError: the execution does not exist.
        """
        self._build(execution_id)
        # By id, not by computation key: the key's outputs are every checkpoint file.
        found = [
            v
            for v in self.store.list_versions_by_id_prefix(f"torch_{execution_id}@observed=")
            if v.state == "ready"
        ]
        return sorted(found, key=lambda v: (v.created_at or 0.0, v.id, v.version))

    def _hold(self, writer: _Checkpoints, seen: Mapping[str, Any], where: str) -> None:
        """Raises InputChanged: a verified input, as *seen* *where*, cannot be read, or
        has bytes other than its pin or none."""
        for each in writer.inputs:
            now = seen.get(each.role, {})
            if now.get("pin") == each.pinned:
                continue
            path = now.get("path", each.path)
            if "unreadable" in now:
                # Perhaps for a moment (EIO on a shared mount): this step fails, the
                # lease stays, the next save checks again.
                raise InputChanged(
                    f"input {each.role!r} at {path} cannot be read {where}: {now['unreadable']}"
                )
            # The data this attempt trained on is gone: a resume need not wait for it.
            # A lapse, not a renewal to zero, as the run's own renewal goes on meanwhile.
            self.builds.lapse_lease(writer.execution_id, writer.attempt)
            changed = "is gone" if "pin" in now and now["pin"] is None else "has other bytes"
            raise InputChanged(f"input {each.role!r} at {path} {changed} {where}")

    def _closed(self, writer: _Checkpoints) -> str | None:
        """Why the attempt's write window refuses it, or None while the window is open."""
        attempt = self.builds.get_attempt(writer.execution_id, writer.attempt)
        if attempt is None or attempt.writable_until > time.time():
            return None
        return (
            f"attempt {writer.attempt}'s write window on execution {writer.execution_id} "
            f"closed {time.time() - attempt.writable_until:.0f}s ago (writable_seconds): "
            "resume it in a new attempt"
        )

    def _require_lease(self, writer: _Checkpoints) -> None:
        conn = self.store._get_connection()
        try:
            held = lease_held(conn, writer.execution_id, writer.attempt, time.time())
        finally:
            conn.close()
        if not held:
            raise BuildLeaseLost(
                self._closed(writer)
                or f"attempt {writer.attempt} no longer holds execution {writer.execution_id}: "
                "it finished, or another attempt took it over"
            )

    def _observe(
        self, writer: _Checkpoints, archive: Path, scope: str, coverage: str
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
        digest, _ = vestigo.digests._read_digest(archive)
        size = archive.stat().st_size
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
            # Failed, not left building: cleanup takes the row and its archive. Any error,
            # as each blob SDK and the database raise their own types.
            self.store.fail_artifact(artifact_id, version)
            raise
        assert stored is not None
        return stored

    def _report(
        self,
        writer: _Checkpoints,
        at: _Location,
        rank: int,
        files: list[dict[str, Any]],
        inputs: dict[str, Any] | None,
    ) -> None:
        artifact_id, version = self._stage(writer, at, f"rank={rank}", {"rank": rank})
        # Signed by the rank that hashed the bytes, so rank 0 signs no digest a bucket
        # writer swapped in between this report and the acceptance. *inputs* are this
        # rank's copies as it found them, held by rank 0 against what the run pinned
        # (None on rank 0, which checks its own at acceptance).
        report = {
            "execution": at.execution_id,
            "attempt": at.attempt,
            "step": at.step,
            "rank": rank,
            "files": files,
            "inputs": inputs,
        }
        self._write_small(artifact_id, version, _canonical(self._sign(report)))

    def _reported(self, at: _Location, rank: int, raw: bytes) -> dict[str, Any]:
        """*rank*'s report of the step: the files it wrote and its inputs as it found
        them; verified, given a signing key.

        Raises:
            OSError: the report does not verify, or is signed for another rank or step.
        """
        if self.config.signing_key is None:
            return json.loads(raw)
        what = f"rank {rank}'s report of step {at.step} of execution {at.execution_id}"
        report = self._verified(raw, what, "report")
        named = (report["execution"], report["attempt"], report["step"], report["rank"])
        if named != (at.execution_id, at.attempt, at.step, rank):
            raise OSError(f"{what} is signed for another checkpoint")
        return report

    def _pending(self, report_id: str) -> list[ArtifactVersion]:
        """The versions of a rank's report no release holds yet, oldest first."""
        return [
            v
            for v in self.store.list_versions_by_id_prefix(report_id)
            if v.id == report_id and v.state == "building" and v.release_id is None
        ]

    def _accept(
        self, writer: _Checkpoints, at: _Location, world: int, known: dict[str, str]
    ) -> Release:
        newest, done = None, None
        for step, bundle in self._accepted(at.execution_id):
            if bundle.release.attempt == at.attempt:
                newest = step if newest is None else max(newest, step)
                done = bundle.release if step == at.step else done
        if done is not None and not self._pending(f"{at.prefix}@rank=0"):
            # The writer retried a finalize whose acceptance had landed.
            return done
        if newest is not None and at.step <= newest:
            # Saved through the manager itself, or while a later save was in flight.
            raise ValueError(
                f"step {at.step} is not after step {newest}, which this attempt accepted"
            )
        reports: list[tuple[str, int, bytes]] = []
        for rank in range(world):
            report_id = f"{at.prefix}@rank={rank}"
            pending = self._pending(report_id)
            data = self.store.read_blob(report_id, pending[-1].version) if pending else None
            if data is None:
                raise ValueError(f"rank {rank} reported nothing for step {at.step}")
            reports.append((report_id, pending[-1].version, data))
        documents = [self._reported(at, rank, data) for rank, (*_, data) in enumerate(reports)]
        # Rank 0's copy now, past the barrier; every other rank's as it found it after
        # its write. Any one that does not hold fails the step.
        inputs = _seen(writer.inputs, known)
        self._hold(writer, inputs, "on rank 0")
        for rank, document in enumerate(documents[1:], start=1):
            self._hold(writer, document["inputs"], f"on rank {rank}")
        files = [entry for document in documents for entry in document["files"]]
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
                "inputs": inputs,
            }
            manifest = self._sign(manifest)
            manifest_id, version = self._stage(
                writer,
                at,
                "manifest",
                checkpoint_files.manifest_params(at.step),
                requires=[(e["artifact_id"], e["version"]) for e in files],
                release_id=release.release_id,
            )
            staged.append(self._write_small(manifest_id, version, _canonical(manifest)))
            self.releases.accept(release.release_id, staged)
        except ReleaseRejected:
            raise
        except Exception:
            # Any error, of the database, a blob SDK or the signer: the release must not
            # stay prepared, and each raises its own types.
            self.releases.abandon(release.release_id, at.attempt)
            raise
        logger.info("Accepted step %d of execution %s", at.step, at.execution_id)
        accepted = self.releases.get(release.release_id)
        assert accepted is not None
        return accepted


# --- The writer's side: picklable, opened per call ---


@dataclass(frozen=True)
class _Checkpoints:
    """One attempt's checkpoints, as a rank and its writer process see them.

    Rank 0 makes it and broadcasts it (it pickles); each rank's :class:`Run` holds its
    own, with that rank's copies of the inputs, and builds its manager from it.
    """

    ledger: LedgerConfig
    execution_id: str
    attempt: str
    computation_key: str
    program: NodeSpec
    # The verified inputs, this rank's copies: it scans them at each save.
    inputs: tuple[_Held, ...] = ()

    def checkpoint_id(self, step: int) -> str:
        """Where to save *step*: a name under this attempt, never reused."""
        return f"{ROOT}/{self.execution_id}/{self.attempt}/step-{step}"

    def configure(
        self,
        config: CheckpointManager.Config,
        rank: int = 0,
        world: int = 1,
        opened: Ledger | None = None,
    ) -> CheckpointManager.Config:
        """Point a ``CheckpointManager.Config`` at the ledger, as *rank* of *world*;
        returns it.

        Sets its storage, both write callbacks and ``write_to_temp_dir=False`` (the
        barrier stays: rank 0 accepts only after every rank has written). The storage
        carries the rank to an asynchronous writer's subprocess, which has no group.
        In this process the storage works through *opened*, given one.
        """
        config.storage_config = LedgerStorageConfig(self.ledger, self, rank, world, opened)
        config.pre_finalize_callback = _report_written
        config.finalize_callback = self
        config.save.writer_config.write_to_temp_dir = False
        return config

    def __call__(self, args: Any) -> None:
        """``finalize_callback``: on rank 0, make the step a release."""
        at = self._own(args.path)
        storage = _ledger_storage(args)
        if storage.rank != 0:
            return
        with storage.opened() as ledger:
            try:
                ledger._accept(self, at, storage.world, storage.known)
            except ReleaseRejected as exc:
                closed = ledger._closed(self)
                if closed is None:
                    raise
                raise ReleaseRejected(closed) from exc

    def _own(self, path: Path | str) -> _Location:
        at = _Location.of(path)
        if (at.execution_id, at.attempt) != (self.execution_id, self.attempt):
            raise PermissionError(f"{path} is not a checkpoint of attempt {self.attempt}")
        return at


def _report_written(args: Any) -> None:
    """``pre_finalize_callback``: this rank records what it wrote of the step."""
    _ledger_storage(args).report(Path(args.path))


def _ledger_storage(args: Any) -> LedgerStorage:
    storage = args.storage
    if not isinstance(storage, LedgerStorage):
        raise TypeError("the writer's storage is not the ledger; save through Ledger.run")
    return storage


@dataclass
class LedgerStorageConfig(StorageConfig):
    ledger: LedgerConfig
    # None for a storage that only loads (Ledger.configure).
    checkpoints: _Checkpoints | None = None
    rank: int = 0
    world: int = 1
    # A ledger open in this process, which its storages share; never pickled, so an
    # asynchronous writer's subprocess opens its own per call.
    opened: Ledger | None = field(default=None, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "opened": None}

    def create_storage(self) -> LedgerStorage:
        return LedgerStorage(self.ledger, self.checkpoints, self.rank, self.world, self.opened)


class _Upload(io.RawIOBase):
    """One file's bytes into its version's blob; committed only on a clean close."""

    def __init__(self, storage: LedgerStorage, at: _Location, artifact_id: str, version: int):
        self._storage, self._at = storage, at
        self._artifact_id, self._version = artifact_id, version
        self._hasher = hashlib.sha256()
        self._size = 0
        self._writer = None  # set once open; the finalizer of a failed open needs it
        self._opened = storage.opened()
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
    without *checkpoints* (:meth:`Ledger.configure`) only loads. With *opened*, every
    call goes through that ledger; without, each opens its own.
    """

    def __init__(
        self,
        ledger: LedgerConfig,
        checkpoints: _Checkpoints | None = None,
        rank: int = 0,
        world: int = 1,
        opened: Ledger | None = None,
    ) -> None:
        self.ledger = ledger
        self.checkpoints = checkpoints
        self.rank, self.world = rank, world
        self._opened = opened
        self._lock = threading.Lock()
        self._written: dict[_Location, list[dict[str, Any]]] = {}
        # By role, a stat fingerprint of this rank's copy that hashed to the pin.
        self.known: dict[str, str] = {}
        # The files of each accepted checkpoint read, by step: a load asks many times.
        self._found: dict[_Location, dict[str, _File]] = {}

    @contextmanager
    def opened(self) -> Generator[Ledger]:
        """The ledger to work through: the shared one, else one open for the call."""
        if self._opened is not None:
            yield self._opened
            return
        with self.ledger.open() as ledger:
            yield ledger

    @property
    def mmap_fill_workers(self) -> int | None:
        # Each stream reads its file whole to check it; ranged readers would each do so.
        return 1

    # --- Writing ---

    def _writer(self) -> _Checkpoints:
        if self.checkpoints is None:
            raise PermissionError("this storage only loads; save through Ledger.run")
        return self.checkpoints

    def stream_write(self, path: Path) -> io.IOBase:
        writer = self._writer()
        at = writer._own(path)
        if at.file is None:
            raise IsADirectoryError(f"{path} is a checkpoint, not a file in one")
        label = f"file={quote(at.file, safe='')}"
        with self.opened() as ledger:
            artifact_id, version = ledger._stage(
                writer, at, label, checkpoint_files.file_params(at.file, at.step)
            )
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
        # Recorded, not raised: a rank that raised here would hold every other one at
        # the barrier. Rank 0 checks its own after the barrier, in the manifest.
        inputs = _seen(writer.inputs, self.known) if self.rank else None
        with self.opened() as ledger:
            ledger._report(writer, at, self.rank, files, inputs)

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
        step = replace(at, file=None)
        files = self._found.get(step)
        if files is None:
            with self.opened() as ledger:
                files = ledger._files(at)
            # An accepted checkpoint's files never change; a step not accepted yet may be.
            if files:
                self._found[step] = files
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
        with self.opened() as ledger:
            data = ledger.store.read_blob(file.artifact_id, file.version)
        if data is None or hashlib.sha256(data).hexdigest() != file.sha256:
            raise OSError(f"{path} differs from the checkpoint the ledger accepted")
        # What the reader asks of a stream, readinto and seek, a BytesIO has.
        return cast(io.RawIOBase, io.BytesIO(data))

    def remap_path(self, path: Path) -> str:
        return PurePath(path).as_posix()


# --- The trainer's side ---


@dataclass
class Run:
    """One attempt at an execution, as one rank runs it: its saves, heads and end.

    Rank 0's renews the lease from a thread until :meth:`finish` or :meth:`close`; use
    it as a context manager. ``restores_from`` is the accepted release the attempt
    starts from. ``checkpointing`` is the ``CheckpointManager.Config`` :attr:`manager`
    is built from, synchronous saves when None.

    In a process group, :meth:`finish` and a synchronous :meth:`save` are collective:
    every rank calls them, and rank 0's outcome (the release, or its error) is every
    rank's. A rank that raises before one leaves the others waiting at it: at a save
    until the barrier's timeout, at a finish until the process group's, unless the
    launcher (``torchrun``) ends them first.
    """

    ledger: Ledger
    _checkpoints: _Checkpoints
    restores_from: str | None = None
    checkpointing: CheckpointManager.Config | None = None
    _ranks: _Ranks = field(default_factory=_Ranks, repr=False)
    finished: bool = False
    _manager: CheckpointManager | None = field(default=None, repr=False)
    # The last asynchronous save's future, which reports how that save went.
    _pending: Future[Any] | None = field(default=None, repr=False)
    # Every rank's count of its synchronous saves, which names each one's outcome: a
    # save of a step again after a failed one must not read that one's.
    _saves: int = field(default=0, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _renewal: threading.Thread | None = field(default=None, repr=False)
    # Held by a renewal and by finish's final release, so no renewal reads a finish
    # as a lost lease.
    _renewing: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.rank == 0:
            self._renewal = threading.Thread(
                target=self._renew, name=f"lease:{self.execution_id}", daemon=True
            )
            self._renewal.start()

    @property
    def rank(self) -> int:
        return self._ranks.rank

    @property
    def world(self) -> int:
        return self._ranks.world

    @property
    def execution_id(self) -> str:
        return self._checkpoints.execution_id

    @property
    def attempt(self) -> str:
        return self._checkpoints.attempt

    @property
    def computation_key(self) -> str:
        return self._checkpoints.computation_key

    def _renew(self) -> None:
        every = self.ledger.lease_seconds / 3
        builds = self.ledger.builds
        while not self._stop.wait(every):
            with self._renewing:
                if self._stop.is_set():
                    return
                try:
                    closed = self.ledger._closed(self._checkpoints)
                    if closed is not None:
                        # Nothing the attempt sends is accepted now: free the execution
                        # for a resume rather than hold it while the process lives.
                        builds.lapse_lease(self.execution_id, self.attempt)
                        logger.warning("The %s; its lease has lapsed", closed)
                        return
                    renewed = builds.renew_lease(
                        self.execution_id, self.attempt, self.ledger.lease_seconds
                    )
                except self.ledger.store.dialect.operational_error:
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

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> None:
        if exc is None:
            self.close()
            return
        try:
            self.close()
        except Exception:
            # The block's own error is the one to raise; this one is only logged.
            logger.warning(
                "Attempt %s did not close cleanly on execution %s",
                self.attempt,
                self.execution_id,
                exc_info=True,
            )

    @property
    def manager(self) -> CheckpointManager:
        """The run's ``CheckpointManager``, built on first use; the run closes it."""
        if self._manager is None:
            # A copy down to the writer's config, all configure sets: the caller's
            # config stays as it was, and each run's storage and callbacks its own.
            config = copy.copy(self._config)
            config.save = copy.copy(config.save)
            config.save.writer_config = copy.copy(config.save.writer_config)
            self._manager = self._checkpoints.configure(
                config, self.rank, self.world, self.ledger
            ).build()
        return self._manager

    @property
    def _config(self) -> CheckpointManager.Config:
        return self.checkpointing or CheckpointManager.Config.with_sync_save()

    @property
    def _synchronous(self) -> bool:
        return isinstance(self._config.save, SyncCheckpointSaverConfig)

    def save(self, step: int, state: Mapping[str, Any]) -> Release | Future[Any]:
        """Save *state* as checkpoint *step*, each module or optimizer by its ``state_dict``.

        A synchronous save returns the accepted release, on every rank. An
        asynchronous one returns the write's future once the state is copied, so an
        in-place update after it (the next optimizer step) is not saved; once the
        future resolves, :meth:`release_of` names the release.

        Raises:
            ValueError: *step* is not after every step this attempt has accepted.
            ReleaseRejected: this attempt no longer holds the execution, or its write
                window closed.
            InputChanged: a ``Path`` input's bytes changed since the run resolved
                it, on any rank (from rank 0's future, for an asynchronous save);
                nothing is released, and the attempt gives up its lease.
        """
        # Before anything is staged: a step again would be dropped for the one accepted,
        # and an earlier one's files are a collection's to take while they are written.
        newest = self.ledger._newest_step(self.execution_id, self.attempt)
        if newest is not None and step <= newest:
            raise ValueError(f"step {step} is not after step {newest}, which this attempt accepted")
        checkpoint_id = self._checkpoints.checkpoint_id(step)
        if not self._synchronous:
            pending = self.manager.save(checkpoint_id, _saved(state))
            assert pending is not None
            self._pending = pending
            # The stager copies the state on a thread, holding the staging lock for write
            # until it has; taking it for read waits for the copy, not the write.
            with self.manager.lock():
                pass
            return pending

        def accepted() -> Release:
            self.manager.save(checkpoint_id, _saved(state))
            release = self.release_of(step)
            assert release is not None
            return release

        if self.world == 1:
            return accepted()
        self._saves += 1
        key = f"{self.attempt}/save-{self._saves}"
        if self.rank:
            # Back from the barrier: rank 0 accepts, and says how it went.
            self.manager.save(checkpoint_id, _saved(state))
            return self._ranks.posted(key, f"after step {step}")
        try:
            release = accepted()
        except BaseException as exc:
            self._ranks.post(key, error=exc)
            raise
        self._ranks.post(key, release)
        return release

    def restore(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """*state* as of ``restores_from``; as given when the attempt starts fresh.

        Each module, optimizer or scheduler is loaded in place and returned as itself;
        each other value is a default, replaced by the one saved under its key. Keys
        not in *state* are not read.

        Raises:
            FileNotFoundError: the checkpoint saved nothing under a key of *state*.
        """
        if self.restores_from is None:
            return dict(state)
        return _load(self.manager, self.ledger.checkpoint_id(self.restores_from), state)

    def observe(
        self, program: torch.export.ExportedProgram, *, scope: str, coverage: str
    ) -> ArtifactVersion:
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
                another attempt took it over, or its write window closed).
        """
        # On disk, not in memory: an archive carries the weights.
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / "program.pt2"
            torch.export.save(program, archive)
            return self.ledger._observe(self._checkpoints, archive, scope, coverage)

    def release_of(self, step: int) -> Release | None:
        """This execution's accepted checkpoint of *step*, once its save has completed."""
        return self.ledger.release_of(self.execution_id, step)

    def set_head(self, name: str, release_id: str) -> None:
        """Point the reference *name* at an accepted release (a best-validation head).

        Raises:
            ValueError: *name* is under ``vestigo-torch/``, the ledger's own.
            ReleaseRejected: this attempt no longer holds the execution.
            OperationFailed: the release is not accepted.
        """
        # Each execution's final checkpoint is a reference there, which collect keeps.
        if name.split("/")[0] == ROOT:
            raise ValueError(f"{name!r} is under {ROOT}/, which the ledger reserves")
        self.ledger.releases.apply(
            self.execution_id, self.attempt, [AdvanceReference(name, release_id)]
        )

    def finish(self) -> None:
        """End the execution: an empty final release completes it; nothing resumes it.

        The checkpoint it ended on (this attempt's last, else the one it restored)
        stays, under the reference ``vestigo-torch/<execution>``;
        :meth:`Ledger.final_checkpoint` names it.

        Waits first for the save in flight, which lands as a release before the final one;
        if it failed, its future reported why, and the run ends on its last accepted
        checkpoint. In a process group every rank calls it, and rank 0 releases.

        Raises:
            ReleaseRejected: this attempt no longer holds the execution.
            InputChanged: a ``Path`` input's bytes changed since the run resolved
                it, on any rank; the attempt gives up its lease.
        """
        error: BaseException | None = None
        try:
            self._drain()
        except BaseException as exc:
            # An interrupt too: every rank meets at the gather below, and one that cannot
            # wait says so first.
            error = exc
        self._ranks.raise_any(error)
        # A remote input names the same objects on every rank: rank 0 lists them.
        held = self._checkpoints.inputs if self.rank == 0 else self._local
        seen = _seen(held, {})
        everyone: list[Any] = [seen]
        if self.world > 1:
            everyone = [None] * self.world if self.rank == 0 else []
            dist.gather_object(
                seen, everyone if self.rank == 0 else None, dst=0, group=self._ranks.group
            )
        self._ranks.from_rank_0(lambda: self._finish(everyone))
        self.finished = True

    @property
    def _local(self) -> tuple[_Held, ...]:
        return tuple(each for each in self._checkpoints.inputs if each.remote is None)

    def _finish(self, everyone: Sequence[Mapping[str, Any]]) -> None:
        local = replace(self._checkpoints, inputs=self._local)
        for rank, seen in enumerate(everyone):
            where = "now" if len(everyone) == 1 else f"now on rank {rank}"
            self.ledger._hold(self._checkpoints if rank == 0 else local, seen, where)
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
        try:
            with self._renewing:
                self.ledger.releases.accept(release.release_id, [], operations)
                self.finished = True
                self._stop.set()
        except ReleaseRejected:
            raise
        except Exception:
            # Left preparing, it would refuse every later final release as out of order.
            self.ledger.releases.abandon(release.release_id, self.attempt)
            raise

    def _drain(self) -> None:
        """Wait for the save in flight and close the manager.

        A failed asynchronous save's error is its future's to report: the caller had it
        there, so it is not raised again here.
        """
        if self._manager is None:
            return
        try:
            self._manager.close()
        except Exception as exc:
            # Whatever the save raised, of any type; anything else goes on up.
            pending = self._pending
            if pending is None or not pending.done() or pending.exception() is not exc:
                raise
            logger.info("Attempt %s's last save failed: %s", self.attempt, exc)

    def close(self) -> None:
        """Wait for the save in flight and stop renewing. An unfinished execution with a
        checkpoint stays resumable, its lease dropped so a resume need not wait; one
        without fails. Each rank closes its own; only rank 0's touches the lease."""
        try:
            self._drain()
        finally:
            try:
                self._close()
            finally:
                self._ranks.leave()

    def _close(self) -> None:
        self._stop.set()
        if self._renewal is not None:
            self._renewal.join()
        if self.finished or self.rank:
            return
        builds = self.ledger.builds
        if self.ledger.releases.latest_accepted(self.execution_id) is not None:
            builds.renew_lease(self.execution_id, self.attempt, lease_duration_seconds=0.0)
        elif builds.fail_build(
            self.execution_id, "ended before a checkpoint", lease_owner=self.attempt
        ):
            # Its placeholder too, once the build is this attempt's to fail: nothing
            # would ever settle it.
            build = builds.get_build(self.execution_id)
            assert build is not None
            self.ledger.store.fail_artifact(build.artifact_id, build.version)


# --- Inside another system's execution ---


class Host(Protocol):
    """What publishes a run's checkpoints where the trainer holds no ledger: a Strata
    cell's ``strata``, whose server holds the run's build and lease."""

    def checkpoint_files(self, step: int, write: Callable[[Path], object]) -> int:
        """Publish, as checkpoint *step*, the files *write* puts in the empty folder it
        is given; returns the checkpoint's sequence in the host's run."""
        ...

    def restore_files(self) -> Path | None:
        """The folder holding the files of the checkpoint the run starts from, or None."""
        ...


class HostedRun:
    """A run whose execution *host* holds: saves and restores as :class:`Run` does.

    Each save is synchronous and written to local disk in the folder the host gives;
    the host publishes it once written. Its checkpoints are unsigned, so a ledger with
    a verify key refuses them. One process: the host is one rank's.

    Raises:
        RuntimeError: the process is one rank of several.
    """

    def __init__(self, host: Host) -> None:
        if _group()[1] > 1:
            raise RuntimeError("a hosted run is one process; run several ranks with Ledger.run")
        self.host = host
        # The newest step the host released; a step again would shadow its checkpoint.
        self._newest: int | None = None
        config = CheckpointManager.Config.with_sync_save()
        # torch-checkpointing 0.1.1 writes unaligned buffers, which O_DIRECT refuses on
        # flush (EINVAL on Linux); macOS has no O_DIRECT.
        config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
        # Into the host's folder itself: a temporary sibling, renamed into place, would
        # be left beside it by a crash.
        config.save.writer_config.write_to_temp_dir = False
        self.manager = config.build()

    def save(self, step: int, state: Mapping[str, Any]) -> int:
        """Save *state* as checkpoint *step*; returns its sequence in the host's run.

        Raises:
            ValueError: *step* is not after every step this run has saved.
        """
        if self._newest is not None and step <= self._newest:
            raise ValueError(f"step {step} is not after step {self._newest}, which this run saved")
        saved = _saved(state)
        sequence = self.host.checkpoint_files(
            step, lambda folder: self.manager.save(str(folder), saved)
        )
        self._newest = step
        return sequence

    def restore(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """*state* as of the checkpoint the host's run starts from; as given without one.

        Raises:
            FileNotFoundError: the checkpoint saved nothing under a key of *state*.
        """
        folder = self.host.restore_files()
        if folder is None:
            return dict(state)
        return _load(self.manager, str(folder), state)

    def close(self) -> None:
        self.manager.close()

    def __enter__(self) -> HostedRun:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
