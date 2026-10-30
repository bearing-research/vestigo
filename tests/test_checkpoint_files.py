"""The layout of a checkpoint made of files, shared by vestigo.torch and a host."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vestigo import checkpoint_files


@pytest.mark.parametrize(
    "path", ["", "/etc/passwd", "../up", "a/../../up", "a//b", "./a", "a/.", "a\x00b"]
)
def test_a_path_that_leaves_the_checkpoint_is_refused(path: str):
    with pytest.raises(ValueError, match="not a path inside a checkpoint"):
        checkpoint_files.file_params(path, 1)


def test_files_and_step_are_read_from_the_params_their_writer_recorded():
    outputs = [
        SimpleNamespace(params=checkpoint_files.file_params("model/0.pt", 3)),
        SimpleNamespace(params={"rank": 0, "step": 3}),
        SimpleNamespace(params=checkpoint_files.manifest_params(3)),
    ]
    assert checkpoint_files.step_of(outputs) == 3
    assert checkpoint_files.files_of(outputs) == {"model/0.pt": outputs[0]}
    # A rank's report carries a step but is not the manifest.
    assert checkpoint_files.step_of(outputs[:2]) is None


def test_a_recorded_path_that_leaves_the_checkpoint_is_not_read():
    outputs = [SimpleNamespace(params={"file": "../../.ssh/id_ed25519", "step": 1})]
    with pytest.raises(ValueError, match="not a path inside a checkpoint"):
        checkpoint_files.files_of(outputs)


def test_a_manifest_lists_each_file_once_in_path_order():
    raw = checkpoint_files.manifest(2, {"b.pt": (3, "bb"), "a.pt": (1, "aa")})
    assert json.loads(raw) == {
        "step": 2,
        "files": [
            {"file": "a.pt", "byte_size": 1, "sha256": "aa"},
            {"file": "b.pt", "byte_size": 3, "sha256": "bb"},
        ],
    }
