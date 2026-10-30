"""A checkpoint made of files, as its release records them.

Each file of the checkpoint is an output of the release whose params name its path in
the checkpoint and the checkpoint's step; one more output, the manifest, carries the
step too. ``vestigo.torch`` writes its checkpoints so, and so does a host that publishes
for a trainer with no ledger of its own (a Strata cell's server), so each reads what the
other wrote. Torch-free, for such a host.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from vestigo.store import ArtifactVersion

FILE = "file"
MANIFEST = "manifest"
STEP = "step"


def checked_path(path: str) -> str:
    """*path* if it names a file inside a checkpoint: relative, POSIX, no ``.`` or ``..``.

    A reader writes the files out under these paths, so none may leave the folder.

    Raises:
        ValueError: it does not.
    """
    if not isinstance(path, str) or "\x00" in path or path.startswith("/"):
        raise ValueError(f"{path!r} is not a path inside a checkpoint")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"{path!r} is not a path inside a checkpoint")
    return path


def file_params(path: str, step: int) -> dict[str, Any]:
    """The params of the version holding the checkpoint's file *path*."""
    return {FILE: checked_path(path), STEP: step}


def manifest_params(step: int) -> dict[str, Any]:
    """The params of the checkpoint's manifest."""
    return {MANIFEST: True, STEP: step}


def manifest(step: int, files: Mapping[str, tuple[int, str]]) -> bytes:
    """A manifest listing *files*, each path's size and sha256."""
    listed = [
        {"file": checked_path(path), "byte_size": size, "sha256": sha256}
        for path, (size, sha256) in sorted(files.items())
    ]
    return json.dumps({"step": step, "files": listed}, sort_keys=True).encode()


def step_of(outputs: Iterable[ArtifactVersion]) -> int | None:
    """The step of the checkpoint *outputs* make, or None if they make none."""
    for version in outputs:
        if version.params.get(MANIFEST) and STEP in version.params:
            return version.params[STEP]
    return None


def files_of(outputs: Iterable[ArtifactVersion]) -> dict[str, ArtifactVersion]:
    """The checkpoint's files among *outputs*, by path in it.

    Raises:
        ValueError: a path leaves the checkpoint.
    """
    return {checked_path(v.params[FILE]): v for v in outputs if FILE in v.params}
