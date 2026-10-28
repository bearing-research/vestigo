"""The PyTorch adapter (``vestigo.torch``) over real torch and torch_checkpointing.

Needs the ``torch`` extra; CI runs this module in its own job and the matrix ignores
it. Every checkpoint is written through the ledger's own storage: on local blobs and
SQLite, on Postgres, on S3, GCS and Azure (their emulators), synchronously and
asynchronously, on one rank and on two (gloo, one process each).
"""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import re
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from torch import nn
from torch_checkpointing.checkpoint_manager import CheckpointManager

from vestigo.blob_store import BlobSettings
from vestigo.releases import ReleaseRejected, ReleaseStore
from vestigo.store import ArtifactStore, BuildLeaseLost
from vestigo.torch import (
    Checkpoints,
    Definition,
    Ledger,
    RemoteBlobs,
    Run,
    _archive_metadata,
    _canonical,
)

STEPS = 6


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgres", marks=[pytest.mark.integration, pytest.mark.slow]),
    ]
)
def ledger(request, tmp_path: Path) -> Iterator[Ledger]:
    dsn = None
    if request.param == "postgres":
        # The fixture empties the schema; this ledger opens its own pools on it.
        request.getfixturevalue("postgres_store")
        dsn = request.getfixturevalue("postgres_dsn")
    with Ledger(tmp_path / "ledger", dsn=dsn) as ledger:
        yield ledger


def _model(hidden: int = 8) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def _program(ledger: Ledger, model: nn.Module, optimizer: torch.optim.Optimizer) -> Definition:
    return ledger.define_program(
        ledger.define_model(model, version="tiny-v1"),
        loss=nn.MSELoss(),
        optimizer=optimizer,
        config={"batch": 16},
    )


def _manager(checkpoints: Run | Checkpoints, asynchronous: bool = False) -> CheckpointManager:
    config = (
        CheckpointManager.Config.with_async_save()
        if asynchronous
        else CheckpointManager.Config.with_sync_save()
    )
    return checkpoints.configure(config).build()


class Trainer:
    """A tiny regression trained one step per checkpoint, deterministically."""

    def __init__(self, ledger: Ledger) -> None:
        self.model = _model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.program = _program(ledger, self.model, self.optimizer)
        generator = torch.Generator().manual_seed(1)
        self.x = torch.randn(64, 4, generator=generator)
        self.y = self.x.sum(dim=1, keepdim=True)
        self.shuffle = torch.Generator().manual_seed(2)
        self.step = 0

    def train(self, run: Run, until: int, asynchronous: bool = False) -> None:
        manager = _manager(run, asynchronous)
        try:
            while self.step < until:
                self.step += 1
                for batch in torch.randperm(64, generator=self.shuffle).split(16):
                    self.optimizer.zero_grad()
                    nn.functional.mse_loss(self.model(self.x[batch]), self.y[batch]).backward()
                    self.optimizer.step()
                pending = manager.save(
                    run.checkpoint_id(self.step),
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optimizer.state_dict(),
                        "shuffle": self.shuffle.get_state(),
                        "step": self.step,
                    },
                )
                if pending is not None and self.step == until:
                    pending.result()
        finally:
            manager.close()

    def restore(self, run: Run) -> None:
        checkpoint = run.restore()
        assert checkpoint is not None
        manager = _manager(run)
        try:
            loaded = manager.load(
                checkpoint,
                into={
                    "model": self.model.state_dict(),
                    "optim": None,
                    "shuffle": None,
                    "step": None,
                },
            )
        finally:
            manager.close()
        self.model.load_state_dict(loaded["model"])
        self.optimizer.load_state_dict(loaded["optim"])
        self.shuffle.set_state(loaded["shuffle"])
        self.step = loaded["step"]

    def weights(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.model.parameters()]


def _uninterrupted(ledger: Ledger) -> list[torch.Tensor]:
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": "data@v=1"}) as run:
        trainer.train(run, STEPS)
        run.finish()
    return trainer.weights()


def _same(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _outputs(ledger: Ledger, run: Run, step: int):
    release = run.release_of(step)
    assert release is not None
    return ledger.releases.bundle(release.release_id).outputs


# --- Identity ---


def test_a_model_is_its_class_and_state_dict_schema_under_a_version(ledger: Ledger):
    first = ledger.define_model(_model(), version="v1")
    retrained = _model()
    with torch.no_grad():
        retrained[0].weight.add_(1.0)

    assert ledger.define_model(retrained, version="v1").node_id == first.node_id
    assert ledger.define_model(_model(), version="v2").node_id != first.node_id
    assert ledger.define_model(_model(hidden=16), version="v1").node_id != first.node_id


def test_a_program_is_its_loss_and_update_over_the_model(ledger: Ledger):
    model = _model()
    definition = ledger.define_model(model, version="v1")

    def program(loss: object, lr: float) -> Definition:
        return ledger.define_program(
            definition, loss=loss, optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
        )

    base = program(nn.MSELoss(), 1e-3)
    assert program(nn.MSELoss(), 1e-3).node_id == base.node_id
    assert program(nn.L1Loss(), 1e-3).node_id != base.node_id
    assert program(nn.MSELoss(), 1e-2).node_id != base.node_id
    [impl] = ledger.store.graph.inputs(base.node_id)
    assert (impl.role, impl.target_node_id) == ("impl", definition.node_id)


def test_runs_on_two_inputs_share_the_program_and_differ_in_computation(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": "data@v=1"}) as one:
        trainer.train(one, 1)
        with ledger.start(trainer.program, {"train": "data@v=2"}) as two:
            assert one.computation_key != two.computation_key
            assert one.computation_key == ledger.computation_key(
                trainer.program, {"train": "data@v=1"}
            )
    [output, *_] = _outputs(ledger, one, 1)
    assert ledger.store.definition_of(output.id, output.version) == trainer.program.node_id


# --- Checkpoints ---


def test_a_checkpoint_is_one_release_of_its_files_under_a_manifest(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        outputs = _outputs(ledger, run, 1)
        files = {v.params["file"]: v for v in outputs if "file" in v.params}
        [manifest] = [v for v in outputs if v.params.get("manifest")]
        [report] = [v for v in outputs if "rank" in v.params]

        assert set(files) == {"model_0.pt", "optim_0.pt", "shuffle_0.pt", "step_0.pt"}
        assert {v.state for v in outputs} == {"ready"}
        assert report.params["rank"] == 0
        conn = ledger.store._get_connection()
        try:
            required = {
                (row["to_artifact_id"], row["to_version"])
                for row in conn.execute(
                    "SELECT to_artifact_id, to_version FROM retention_edges "
                    "WHERE from_artifact_id = ? AND from_version = ?",
                    (manifest.id, manifest.version),
                )
            }
        finally:
            conn.close()
        assert required == {(v.id, v.version) for v in files.values()}


def test_a_retried_finalize_returns_the_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.checkpoints(SimpleNamespace(path=run.checkpoint_id(1)))

        assert len(ledger.releases.releases_of(run.execution_id, "checkpoint")) == 1


def test_an_attempt_writes_only_its_own_checkpoints(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run, ledger.start(trainer.program) as other:
        manager = _manager(run)
        try:
            with pytest.raises(PermissionError):
                manager.save(other.checkpoint_id(1), {"step": 1})
        finally:
            manager.close()
        with pytest.raises(PermissionError):
            run.checkpoints(SimpleNamespace(path=other.checkpoint_id(1)))


def test_a_checkpoint_not_accepted_does_not_load_though_its_bytes_are_written(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            patch.setattr(ReleaseStore, "accept", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                trainer.train(run, 1)
        written = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@")
        assert any(ledger.store.blob_exists(v.id, v.version) for v in written)

        storage = run.configure(CheckpointManager.Config()).storage_config.create_storage()
        assert not storage.exists(Path(run.checkpoint_id(1)))
        with pytest.raises(FileNotFoundError):
            storage.getsize(Path(run.checkpoint_id(1)) / "model_0.pt")


def test_only_rank_zero_accepts(ledger: Ledger, monkeypatch: pytest.MonkeyPatch):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")

        run.checkpoints(SimpleNamespace(path=run.checkpoint_id(1)))

        assert ledger.releases.releases_of(run.execution_id, "checkpoint") == []


# --- Resume and branch ---


def test_a_resumed_run_ends_with_the_uninterrupted_weights(ledger: Ledger):
    expected = _uninterrupted(ledger)
    trainer = Trainer(ledger)
    first = ledger.start(trainer.program, {"train": "data@v=1"})
    trainer.train(first, 3)
    first.close()  # the process dies after step 3

    trainer = Trainer(ledger)
    execution = ledger.find(trainer.program, {"train": "data@v=1"})
    assert execution == first.execution_id
    with ledger.resume(execution) as resumed:
        assert resumed.attempt != first.attempt
        assert resumed.restores_from == first.release_of(3).release_id
        trainer.restore(resumed)
        trainer.train(resumed, STEPS)
        resumed.finish()

    assert trainer.step == STEPS
    assert _same(trainer.weights(), expected)
    attempt = ledger.builds.get_attempt(execution, resumed.attempt)
    assert attempt is not None and attempt.resumed_from == resumed.restores_from


def test_a_branch_is_its_own_execution_from_its_release(ledger: Ledger):
    expected = _uninterrupted(ledger)
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": "data@v=1"}) as source:
        trainer.train(source, 4)
        source.finish()
    second, fourth = source.release_of(2).release_id, source.release_of(4).release_id

    trainer = Trainer(ledger)
    with ledger.branch(trainer.program, second, {"train": "data@v=1"}) as branch:
        assert branch.execution_id != source.execution_id
        assert branch.computation_key not in {
            source.computation_key,
            ledger.computation_key(trainer.program, {"train": "data@v=1"}, fourth),
        }
        trainer.restore(branch)
        assert trainer.step == 2
        trainer.train(branch, STEPS)
        branch.finish()

    assert _same(trainer.weights(), expected)
    assert ledger.builds.get_build(branch.execution_id).params["init"] == second


def test_a_load_refuses_bytes_that_differ_from_the_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as source:
        trainer.train(source, 1)
        release = source.release_of(1)
    [model] = [v for v in _outputs(ledger, source, 1) if v.params.get("file") == "model_0.pt"]
    ledger.store.write_blob(model.id, model.version, b"not the checkpoint")

    with ledger.branch(trainer.program, release.release_id) as branch:
        with pytest.raises(OSError, match="differs from the checkpoint"):
            Trainer(ledger).restore(branch)


# --- Signed checkpoints ---


def _keypair() -> tuple[bytes, bytes]:
    private = Ed25519PrivateKey.generate()
    return private.private_bytes_raw(), private.public_key().public_bytes_raw()


def _tamper(ledger: Ledger, run: Run, step: int) -> bytes:
    """Rewrite step's model file and the digest its row records, as a database writer can."""
    [model] = [v for v in _outputs(ledger, run, step) if v.params.get("file") == "model_0.pt"]
    forged = io.BytesIO()
    torch.save({k: torch.zeros_like(v) for k, v in _model().state_dict().items()}, forged)
    data = forged.getvalue()
    ledger.store.write_blob(model.id, model.version, data)
    conn = ledger.store._get_connection()
    try:
        conn.execute(
            "UPDATE artifact_versions SET content_sha256 = ?, byte_size = ? "
            "WHERE id = ? AND version = ?",
            (hashlib.sha256(data).hexdigest(), len(data), model.id, model.version),
        )
        conn.commit()
    finally:
        conn.close()
    return data


def test_a_signed_checkpoint_resumes_and_a_reader_verifies_it(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        expected = _uninterrupted(ledger)
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as first:
            trainer.train(first, 3)
        with ledger.resume(first.execution_id) as resumed:
            trainer = Trainer(ledger)
            trainer.restore(resumed)
            trainer.train(resumed, STEPS)
            resumed.finish()
        assert _same(trainer.weights(), expected)
        source = first.release_of(3).release_id

    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        with reader.branch(trainer.program, source) as branch:
            Trainer(reader).restore(branch)


def test_a_database_writer_cannot_change_a_signed_checkpoint(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
        _tamper(ledger, run, 1)
        source = run.release_of(1).release_id

    # Without a key the forgery loads: bytes and row agree.
    with Ledger(tmp_path / "ledger") as unkeyed:
        with unkeyed.branch(trainer.program, source) as branch:
            fooled = Trainer(unkeyed)
            fooled.restore(branch)
            assert not any(w.any() for w in fooled.weights())
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        with reader.branch(trainer.program, source) as branch:
            with pytest.raises(OSError, match="differs from the checkpoint"):
                Trainer(reader).restore(branch)


def test_a_rewritten_manifest_does_not_verify(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
        forged = _tamper(ledger, run, 1)
        # The forger fixes the manifest too, keeping the old signature.
        [manifest] = [v for v in _outputs(ledger, run, 1) if v.params.get("manifest")]
        content = json.loads(ledger.store.read_blob(manifest.id, manifest.version))
        for entry in content["files"]:
            if entry["file"] == "model_0.pt":
                entry["sha256"] = hashlib.sha256(forged).hexdigest()
        ledger.store.write_blob(manifest.id, manifest.version, json.dumps(content).encode())
        source = run.release_of(1).release_id

    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        with reader.branch(trainer.program, source) as branch:
            with pytest.raises(OSError, match="does not verify"):
                Trainer(reader).restore(branch)


def test_a_manifest_signed_for_another_step_is_refused(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 2)
        # Step 2's genuine manifest and signature, put where step 1's was.
        [one] = [v for v in _outputs(ledger, run, 1) if v.params.get("manifest")]
        [two] = [v for v in _outputs(ledger, run, 2) if v.params.get("manifest")]
        ledger.store.write_blob(one.id, one.version, ledger.store.read_blob(two.id, two.version))
        source = run.release_of(1).release_id

    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        with reader.branch(trainer.program, source) as branch:
            with pytest.raises(OSError, match="signed for another checkpoint"):
                Trainer(reader).restore(branch)


def test_a_verifying_ledger_refuses_unsigned_and_foreign_checkpoints(tmp_path: Path):
    with Ledger(tmp_path / "unsigned") as plain:
        trainer = Trainer(plain)
        with plain.start(trainer.program) as run:
            trainer.train(run, 1)
        unsigned = run.release_of(1).release_id
    with Ledger(tmp_path / "unsigned", verify_key=_keypair()[1]) as reader:
        with reader.branch(trainer.program, unsigned) as branch:
            with pytest.raises(OSError, match="is not signed"):
                Trainer(reader).restore(branch)

    with Ledger(tmp_path / "signed", signing_key=_keypair()[0]) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
        signed = run.release_of(1).release_id
    with Ledger(tmp_path / "signed", verify_key=_keypair()[1]) as stranger:
        with stranger.branch(trainer.program, signed) as branch:
            with pytest.raises(OSError, match="does not verify"):
                Trainer(stranger).restore(branch)


def test_keys_are_checked_and_the_private_one_is_never_shown(tmp_path: Path):
    signing, verify = _keypair()
    with pytest.raises(ValueError, match="public half"):
        Ledger(tmp_path / "ledger", signing_key=signing, verify_key=_keypair()[1])
    with pytest.raises(ValueError):
        Ledger(tmp_path / "ledger", verify_key=b"short")
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        assert ledger.config.verify_key == verify
        assert signing.hex() not in repr(ledger.config) and repr(signing) not in repr(ledger.config)


# --- Authority ---


def test_a_stale_attempt_cannot_publish_after_a_resume(ledger: Ledger):
    stale = Trainer(ledger)
    first = ledger.start(stale.program)
    stale.train(first, 1)
    first.close()  # its lease lapses: a pause, a partition
    with ledger.resume(first.execution_id) as resumed:
        with pytest.raises(ReleaseRejected):
            stale.train(first, 2)
        assert ledger.release_of(first.execution_id, 2) is None

        fresh = Trainer(ledger)
        fresh.restore(resumed)
        fresh.train(resumed, 2)
        assert resumed.release_of(2).attempt == resumed.attempt


class _Crash(BaseException):
    """The process dying inside acceptance: nothing after it runs."""


def test_a_crash_before_acceptance_leaves_no_authority(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    first = ledger.start(trainer.program)
    trainer.train(first, 1)

    def crash(*args: object, **kwargs: object) -> None:
        raise _Crash

    with monkeypatch.context() as patch:
        patch.setattr(ReleaseStore, "accept", crash)
        with pytest.raises(_Crash):
            trainer.train(first, 2)
    first.close()

    [_, orphan] = ledger.releases.bundles_of(first.execution_id, "checkpoint")
    assert orphan.release.state == "preparing"
    assert orphan.outputs and {v.state for v in orphan.outputs} == {"building"}
    with ledger.resume(first.execution_id) as resumed:
        assert resumed.restores_from == first.release_of(1).release_id
        assert ledger.releases.get(orphan.release.release_id).state == "abandoned"
        assert ledger.release_of(first.execution_id, 2) is None


def test_a_crash_while_writing_leaves_only_rows_to_collect(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        real = ArtifactStore.open_blob_writer
        calls = []

        def second_fails(self, *args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise OSError("disk full")
            return real(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(ArtifactStore, "open_blob_writer", second_fails)
            with pytest.raises(OSError):
                trainer.train(run, 1)

        assert ledger.releases.releases_of(run.execution_id, "checkpoint") == []
        staged = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@")
        assert staged and {v.state for v in staged} == {"building"}
        assert {v.release_id for v in staged} == {None}


def test_a_write_that_fails_midway_commits_no_bytes(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        manager = _manager(run)
        try:
            with pytest.raises(Exception, match="pickle"):
                # torch.save writes its archive header before the lambda refuses to pickle.
                manager.save(run.checkpoint_id(1), {"bad": {"f": lambda: 0}})
        finally:
            manager.close()

        [staged] = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@")
        assert staged.state == "building"
        assert not ledger.store.blob_exists(staged.id, staged.version)


def test_a_checkpoint_that_fails_to_release_does_not_hold_up_the_next(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            patch.setattr(ReleaseStore, "attach", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                trainer.train(run, 1)
        trainer.train(run, 2)

        [failed, accepted] = ledger.releases.releases_of(run.execution_id, "checkpoint")
        assert (failed.state, accepted.state) == ("abandoned", "accepted")
        assert run.release_of(2) == accepted


def _raise(error: type[Exception]) -> None:
    raise error


def test_a_head_points_at_an_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
        run.set_head("tiny/best", run.release_of(1).release_id)

    assert ledger.releases.reference("tiny/best") == (run.release_of(1).release_id, 1)


def test_a_finished_execution_is_not_resumed(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.finish()

    assert ledger.find(trainer.program) is None
    with pytest.raises(ValueError, match="not active"):
        ledger.resume(run.execution_id)


def test_a_run_that_ends_before_a_checkpoint_fails(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        pass

    assert ledger.builds.get_build(run.execution_id).state == "failed"
    assert ledger.find(trainer.program) is None


# --- Derived releases ---


def _manifest(ledger: Ledger, release_id: str):
    [manifest] = [v for v in ledger.releases.bundle(release_id).outputs if v.params.get("manifest")]
    return manifest


def _export(trainer: Trainer, tmp_path: Path) -> Path:
    path = tmp_path / "export.pt"
    torch.save(trainer.model.state_dict(), path)
    return path


def test_a_derived_release_holds_its_outputs_linked_to_the_checkpoint(
    ledger: Ledger, tmp_path: Path
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
        source = run.release_of(2).release_id
        export = _export(trainer, tmp_path)

        bundle = ledger.derive(source, "eval", {"metrics": b'{"loss": 0.1}', "weights": export})

    assert (bundle.release.kind, bundle.release.state) == ("final", "accepted")
    outputs = {v.params["output"]: v for v in bundle.outputs}
    assert ledger.store.read_blob(outputs["metrics"].id, 1) == b'{"loss": 0.1}'
    assert ledger.store.read_blob(outputs["weights"].id, 1) == export.read_bytes()
    # Named by the spec, not an input: nothing walks from an output to the checkpoint.
    assert {v.params["from"] for v in bundle.outputs} == {source}
    assert all(v.input_versions in (None, "{}") for v in bundle.outputs)
    assert ledger.derived(source, "eval").release == bundle.release
    assert ledger.derived(source, "export") is None


def test_deriving_the_same_name_again_stores_nothing(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
    source = run.release_of(1).release_id
    first = ledger.derive(source, "eval", {"metrics": b"1"})

    again = ledger.derive(source, "eval", {"metrics": b"2"})

    assert again.release == first.release
    [version] = ledger.store.list_versions_by_id_prefix(first.outputs[0].id)
    assert ledger.store.read_blob(version.id, version.version) == b"1"


def test_a_later_checkpoint_s_derivation_is_the_output_s_current_value(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
    one = ledger.derive(run.release_of(1).release_id, "eval", {"metrics": b"1"})
    two = ledger.derive(run.release_of(2).release_id, "eval", {"metrics": b"2"})

    assert one.release != two.release and one.outputs[0].id == two.outputs[0].id
    current = ledger.store.get_latest_version(one.outputs[0].id)
    assert (current.version, current.release_id) == (2, two.release.release_id)


def test_keeping_a_derived_release_does_not_keep_its_checkpoint(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        steps = _steps(ledger, run)
        run.finish()
    source = run.release_of(1).release_id
    derived = ledger.derive(source, "eval", {"metrics": b"1"})
    [output] = derived.outputs
    ledger.store.retain(output.id, output.version, kind="pin", holder="report")
    manifest = _manifest(ledger, source)

    ledger.collect(keep=1)

    assert _stored(ledger, steps) == [3]
    assert ledger.store.get_artifact(manifest.id, manifest.version) is None
    assert ledger.store.read_blob(output.id, output.version) == b"1"
    assert ledger.derived(source, "eval").release == derived.release


def test_only_an_accepted_checkpoint_is_derived_from(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.finish()
    final = ledger.releases.latest_accepted(run.execution_id, "final")

    for release_id in (final.release_id, "no-such-release"):
        with pytest.raises(ValueError, match="not an accepted checkpoint"):
            ledger.derive(release_id, "eval", {"metrics": b"1"})
    with pytest.raises(ValueError, match="labelled outputs"):
        ledger.derive(run.release_of(1).release_id, "eval", {})


def test_a_derivation_that_fails_leaves_nothing_current(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
    source = run.release_of(1).release_id

    def full(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ArtifactStore, "publish_blob_from_path", full)
    with pytest.raises(OSError):
        ledger.derive(source, "eval", {"metrics": b"1", "weights": _export(trainer, tmp_path)})

    assert ledger.derived(source, "eval") is None
    staged = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@derived=")
    assert staged and {v.state for v in staged} == {"failed"}


def test_a_derivation_slower_than_its_lease_is_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with Ledger(tmp_path / "ledger", lease_seconds=0.2) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
        real = ArtifactStore.publish_blob_from_path

        def slow(self, *args, **kwargs):
            time.sleep(0.5)  # an upload outlasting the lease
            return real(self, *args, **kwargs)

        monkeypatch.setattr(ArtifactStore, "publish_blob_from_path", slow)
        bundle = ledger.derive(
            run.release_of(1).release_id, "export", {"weights": _export(trainer, tmp_path)}
        )

        assert bundle.release.state == "accepted"


# --- Retention ---


def _steps(ledger: Ledger, run: Run) -> dict[int, list[tuple[str, int]]]:
    """Every accepted step's outputs, read before a collection removes any."""
    steps = {}
    for step in range(1, STEPS + 1):
        release = ledger.release_of(run.execution_id, step)
        if release is not None:
            outputs = ledger.releases.bundle(release.release_id).outputs
            steps[step] = [(v.id, v.version) for v in outputs]
    return steps


def _stored(ledger: Ledger, steps: dict[int, list[tuple[str, int]]]) -> list[int]:
    return [
        step
        for step, outputs in steps.items()
        if all(ledger.store.get_artifact(*output) is not None for output in outputs)
    ]


def test_collect_keeps_an_unfinished_run_s_newest_checkpoints_and_its_heads(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, STEPS)
        run.set_head("tiny/best", run.release_of(2).release_id)
        steps = _steps(ledger, run)

        result = ledger.collect(keep=2)

        assert _stored(ledger, steps) == [2, 5, 6]
        assert result["deleted_count"] == sum(len(steps[s]) for s in (1, 3, 4))
    with ledger.resume(run.execution_id) as resumed:
        fresh = Trainer(ledger)
        fresh.restore(resumed)
        assert fresh.step == STEPS and _same(fresh.weights(), trainer.weights())


def test_a_dry_run_collects_nothing(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        steps = _steps(ledger, run)

        result = ledger.collect(keep=1, dry_run=True)

        collected = {(c["artifact_id"], c["version"]) for c in result["collected"]}
        assert collected == set(steps[1] + steps[2])
        assert _stored(ledger, steps) == [1, 2, 3]


def test_a_finished_run_keeps_the_checkpoint_it_ended_on(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        steps = _steps(ledger, run)
        run.finish()

    ledger.collect(keep=1)

    final = ledger.final_checkpoint(run.execution_id)
    assert final == run.release_of(3) and _stored(ledger, steps) == [3]
    with ledger.branch(trainer.program, final.release_id) as branch:
        fresh = Trainer(ledger)
        fresh.restore(branch)
        assert _same(fresh.weights(), trainer.weights())


def test_a_resume_that_finishes_without_saving_keeps_what_it_restored(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
    with ledger.resume(run.execution_id) as resumed:
        resumed.finish()

    assert ledger.final_checkpoint(run.execution_id) == run.release_of(2)


def test_an_unfinished_branch_keeps_the_release_it_starts_from(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        steps = _steps(ledger, run)
        run.finish()
    # Opened, and killed before its first checkpoint: its resume restores the source.
    branch = ledger.branch(trainer.program, run.release_of(1).release_id)
    try:
        ledger.collect(keep=1)

        assert _stored(ledger, steps) == [1, 3]
        Trainer(ledger).restore(branch)
    finally:
        branch.close()


def test_a_branch_marks_its_source_used_before_it_loads(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.finish()
    source = run.release_of(1).release_id
    conn = ledger.store._get_connection()
    try:
        # Long idle, as an old checkpoint is.
        conn.execute(
            "UPDATE artifact_versions SET last_used_at = 0 WHERE release_id = ?", (source,)
        )
        conn.commit()
    finally:
        conn.close()

    with ledger.branch(trainer.program, source):
        conn = ledger.store._get_connection()
        try:
            rows = conn.execute(
                "SELECT last_used_at FROM artifact_versions WHERE release_id = ?", (source,)
            ).fetchall()
        finally:
            conn.close()

    assert rows and all(row["last_used_at"] > 0 for row in rows)


def test_collect_takes_what_an_ended_attempt_staged_and_spares_a_live_one(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    first = ledger.start(trainer.program)
    trainer.train(first, 1)
    real = ArtifactStore.open_blob_writer
    calls = []

    def second_fails(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ArtifactStore, "open_blob_writer", second_fails)
        with pytest.raises(OSError):
            trainer.train(first, 2)
    prefix = f"torch_{first.execution_id}@attempt={first.attempt}@step=2"
    staged = [(v.id, v.version) for v in ledger.store.list_versions_by_id_prefix(prefix)]
    assert staged

    assert ledger.collect(keep=1)["orphaned"] == 0
    assert all(ledger.store.get_artifact(*v).state == "building" for v in staged)

    first.close()
    with ledger.resume(first.execution_id):
        result = ledger.collect(keep=1)

    assert result["orphaned"] == len(staged)
    assert all(ledger.store.get_artifact(*v) is None for v in staged)


# --- Asynchronous saves ---


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="torch_checkpointing 0.1.1 stages for CUDA whenever an accelerator exists, MPS too",
)
def test_asynchronous_saves_release_in_order_and_resume(tmp_path: Path):
    # Signed: the key reaches the writer subprocess, which signs.
    with Ledger(tmp_path / "ledger", signing_key=_keypair()[0]) as ledger:
        expected = _uninterrupted(ledger)
        trainer = Trainer(ledger)
        first = ledger.start(trainer.program)
        trainer.train(first, 3, asynchronous=True)
        first.close()

        releases = ledger.releases.releases_of(first.execution_id, "checkpoint")
        assert [(r.sequence, r.state) for r in releases] == [(n, "accepted") for n in (1, 2, 3)]
        trainer = Trainer(ledger)
        with ledger.resume(first.execution_id) as resumed:
            trainer.restore(resumed)
            trainer.train(resumed, STEPS, asynchronous=True)
            resumed.finish()
        assert _same(trainer.weights(), expected)


# --- Object store ---


@pytest.fixture(params=["s3", "gcs", "azure"])
def remote(request) -> RemoteBlobs:
    """A fresh bucket on each backend's emulator, as the adapter reaches it."""
    bucket = f"ckpt-{uuid.uuid4().hex[:8]}"
    if request.param == "s3":
        endpoint = request.getfixturevalue("minio_endpoint")
        settings = BlobSettings(
            s3_region="us-east-1",
            s3_endpoint_url=endpoint,
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
        )
    elif request.param == "gcs":
        endpoint = request.getfixturevalue("fake_gcs_endpoint")
        settings = BlobSettings(gcs_anonymous=True, gcs_endpoint_override=endpoint)
    else:
        container = request.getfixturevalue("azurite_container")
        settings = BlobSettings(
            azure_account_name=container.account_name,
            azure_connection_string=container.get_connection_string(),
        )
    blobs = RemoteBlobs(request.param, bucket, "ledger", settings)
    store = blobs.open()
    if request.param == "s3":
        store._client.create_bucket(Bucket=bucket)  # type: ignore[attr-defined]
    elif request.param == "gcs":
        store._client.create_bucket(bucket)  # type: ignore[attr-defined]
    else:
        store._client.create_container()  # type: ignore[attr-defined]
    return blobs


@pytest.mark.integration
@pytest.mark.slow
def test_checkpoints_on_an_object_store_load_back(tmp_path: Path, remote: RemoteBlobs):
    with Ledger(tmp_path / "ledger", blobs=remote) as ledger:
        expected = _uninterrupted(ledger)
        trainer = Trainer(ledger)
        first = ledger.start(trainer.program)
        trainer.train(first, 3)
        first.close()
        with ledger.resume(first.execution_id) as resumed:
            trainer = Trainer(ledger)
            trainer.restore(resumed)
            trainer.train(resumed, STEPS)
            resumed.finish()

        assert _same(trainer.weights(), expected)
        # The bytes are in the bucket, through a client of its own, and nowhere local.
        [model] = [v for v in _outputs(ledger, first, 3) if v.params.get("file") == "model_0.pt"]
        assert remote.open().blob_exists(model.id, model.version)
        assert not (tmp_path / "ledger" / "blobs").exists()


# --- Two ranks ---


def _rank_main(rank: int, world: int, root: str, die_at: int | None) -> None:
    """One rank: rank 0 starts the run and hands its checkpoints to the other."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=world
    )
    signing = Ed25519PrivateKey.generate()
    ledger = (
        Ledger(Path(root) / "ledger", signing_key=signing.private_bytes_raw())
        if rank == 0
        else None
    )
    model = _model()
    run = None
    shared: list[object] = [None]
    if ledger is not None:
        optimizer = torch.optim.AdamW(model.parameters())
        run = ledger.start(_program(ledger, model, optimizer))
        shared = [run.checkpoints]
    dist.broadcast_object_list(shared, src=0)
    checkpoints = shared[0]
    assert isinstance(checkpoints, Checkpoints)
    config = checkpoints.configure(CheckpointManager.Config.with_sync_save())
    config.save.writer_config.checkpoint_write_barrier_timeout_sec = 5
    manager = config.build()
    outcome: dict[str, object] = {
        "execution": checkpoints.execution_id,
        "saved": [],
        "verify_key": signing.public_key().public_bytes_raw().hex(),
    }
    for step in (1, 2):
        if rank == 1 and step == die_at:
            os._exit(3)  # dies before writing: rank 0 never passes the barrier
        try:
            with torch.no_grad():
                model[0].weight.add_(rank + 1)
            manager.save(checkpoints.checkpoint_id(step), {"model": model.state_dict()})
            outcome["saved"].append(step)  # type: ignore[union-attr]
        except Exception as exc:
            outcome["error"] = type(exc).__name__
            break
    if run is not None:
        run.close()
        Path(root, "outcome.json").write_text(json.dumps(outcome))
    # A dead peer can hang the group's teardown; nothing below needs it.
    os._exit(0)


def _two_ranks(root: Path, die_at: int | None) -> tuple[list[int], dict]:
    context = multiprocessing.get_context("spawn")
    ranks = [context.Process(target=_rank_main, args=(r, 2, str(root), die_at)) for r in (0, 1)]
    for process in ranks:
        process.start()
    for process in ranks:
        process.join(120)
    return [p.exitcode for p in ranks], json.loads((root / "outcome.json").read_text())


def test_two_ranks_make_one_release_from_rank_zero(tmp_path: Path):
    exits, outcome = _two_ranks(tmp_path, die_at=None)

    assert exits == [0, 0] and outcome["saved"] == [1, 2]
    with Ledger(tmp_path / "ledger") as ledger:
        for step in (1, 2):
            release = ledger.release_of(outcome["execution"], step)
            assert release is not None
            outputs = ledger.releases.bundle(release.release_id).outputs
            assert {v.params["file"] for v in outputs if "file" in v.params} == {
                "model_0.pt",
                "model_1.pt",
            }
            assert {v.params["rank"] for v in outputs if "rank" in v.params} == {0, 1}
            # Rank 0 signed over both ranks' files.
            [manifest] = [v for v in outputs if v.params.get("manifest")]
            content = json.loads(ledger.store.read_blob(manifest.id, manifest.version))
            assert {e["file"] for e in content["files"]} == {"model_0.pt", "model_1.pt"}
            public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(outcome["verify_key"]))
            public.verify(bytes.fromhex(content.pop("signature")), _canonical(content))


def test_a_rank_that_dies_before_writing_leaves_no_release(tmp_path: Path):
    exits, outcome = _two_ranks(tmp_path, die_at=2)

    assert exits == [0, 3] and outcome["saved"] == [1] and "error" in outcome
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.release_of(outcome["execution"], 1) is not None
        assert ledger.release_of(outcome["execution"], 2) is None
        [release] = ledger.releases.releases_of(outcome["execution"], "checkpoint")
        assert release.sequence == 1


# --- Observed implementations ---


def _exported(model: nn.Module) -> torch.export.ExportedProgram:
    batch = torch.export.Dim("batch")
    return torch.export.export(model, (torch.randn(2, 4),), dynamic_shapes=({0: batch},))


def _node_count(ledger: Ledger) -> int:
    conn = ledger.store._get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM nodes").fetchone()["n"]
    finally:
        conn.close()


def test_an_exported_program_is_recorded_with_its_scope_and_versions(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        nodes = _node_count(ledger)

        recorded = run.observe(
            _exported(trainer.model),
            scope="model_forward",
            coverage="complete_under_declared_constraints",
        )

        assert _node_count(ledger) == nodes
        assert ledger.store.definition_of(recorded.id, recorded.version) == trainer.program.node_id
        assert ledger.observations(run.execution_id) == [recorded]
        params = recorded.params
        assert (params["observed"], params["coverage"], params["attempt"]) == (
            "model_forward",
            "complete_under_declared_constraints",
            run.attempt,
        )
        [program] = params["programs"].values()
        assert program["torch_version"] == torch.__version__
        assert program["schema_version"]["major"] >= 8
        assert "aten" in program["opset_version"]
        assert params["serialization_id"]
        # The record is the archive itself: it loads and computes what the model does.
        data = ledger.store.read_blob(recorded.id, recorded.version)
        loaded = torch.export.load(io.BytesIO(data)).module()
        x = torch.randn(5, 4)
        assert torch.equal(loaded(x), trainer.model(x))


def test_the_same_program_exported_again_is_one_version(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        # Two exports: other example inputs, another serialization id, one program.
        first = run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")
        second = run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")

        assert (second.id, second.version) == (first.id, first.version)
        other = run.observe(_exported(trainer.model), scope="model_forward", coverage="partial")
        assert ledger.observations(run.execution_id) == [first, other]


def test_one_program_under_two_scopes_is_two_records(ledger: Ledger):
    trainer = Trainer(ledger)
    exported = _exported(trainer.model)
    with ledger.start(trainer.program) as run:
        forward = run.observe(exported, scope="model_forward", coverage="complete")
        compiled = run.observe(exported, scope="model_forward_compiled", coverage="complete")

        assert [v.params["observed"] for v in ledger.observations(run.execution_id)] == [
            "model_forward",
            "model_forward_compiled",
        ]
        assert forward.id != compiled.id


def test_observations_are_the_execution_s_own(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as one, ledger.start(trainer.program) as two:
        one.observe(_exported(trainer.model), scope="model_forward", coverage="complete")

        assert one.computation_key == two.computation_key
        assert ledger.observations(two.execution_id) == []


def test_a_stale_attempt_cannot_record_an_observation(ledger: Ledger):
    trainer = Trainer(ledger)
    exported = _exported(trainer.model)
    first = ledger.start(trainer.program)
    trainer.train(first, 1)
    recorded = first.observe(exported, scope="model_forward", coverage="complete")
    first.close()
    with ledger.resume(first.execution_id):
        # Not even the record it made before: the attempt no longer speaks for the run.
        with pytest.raises(BuildLeaseLost, match="no longer holds"):
            first.observe(exported, scope="model_forward", coverage="complete")
        with pytest.raises(BuildLeaseLost, match="no longer holds"):
            first.observe(exported, scope="model_forward", coverage="other")

    staged = ledger.store.list_versions_by_id_prefix(f"torch_{first.execution_id}@observed=")
    assert [(v.version, v.state) for v in staged] == [(recorded.version, "ready")]


def test_a_finished_run_says_so_when_it_observes(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        run.finish()
        with pytest.raises(BuildLeaseLost, match="it finished"):
            run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")


def test_an_observation_that_fails_to_store_is_failed_for_cleanup(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            patch.setattr(ArtifactStore, "publish_blob_from_path", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")

        [staged] = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@observed=")
        assert staged.state == "failed"


def test_an_archive_without_torch_s_metadata_entries_reads_as_none(tmp_path: Path):
    archive = tmp_path / "bare.pt2"
    with zipfile.ZipFile(archive, "w") as bare:
        bare.writestr("archive/models/model.json", json.dumps({"torch_version": "x"}))

    metadata = _archive_metadata(archive)

    assert metadata["archive_version"] is None and metadata["serialization_id"] is None
    assert metadata["programs"]["model"]["torch_version"] == "x"
    assert metadata["programs"]["model"]["schema_version"] is None


# --- The examples ---

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "torch"


def _example(script: str, *args: str, ranks: int = 1) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(EXAMPLES / script), *args]
    if ranks > 1:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        launch = ["-m", "torch.distributed.run", f"--nproc-per-node={ranks}"]
        rendezvous = ["--master-addr=127.0.0.1", f"--master-port={port}"]
        command = [sys.executable, *launch, *rendezvous, *command[1:]]
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def _said(result: subprocess.CompletedProcess[str], pattern: str) -> str:
    match = re.search(pattern, result.stdout)
    assert match is not None, result.stdout + result.stderr
    return match.group(1)


def test_the_training_example_resumes_where_it_stopped_then_branches(tmp_path: Path):
    root = str(tmp_path / "ledger")
    stopped = _example("train.py", root, "--epochs=4", "--crash-at=2")
    assert stopped.returncode == 1 and "stopped after epoch 2" in stopped.stderr
    execution = _said(stopped, r"started (\w+)")

    resumed = _example("train.py", root, "--epochs=4", "--keep=1")
    assert resumed.returncode == 0, resumed.stderr
    assert _said(resumed, r"resumed (\w+) after epoch 2") == execution
    final = _said(resumed, r"finished \w+ on release (\w+)")

    with Ledger(Path(root)) as ledger:
        assert ledger.builds.get_build(execution).state == "ready"
        assert [r.sequence for r in ledger.releases.releases_of(execution, "checkpoint")] == [
            1,
            2,
            3,
            4,
        ]
        head = ledger.releases.reference("regression/best")
        assert head is not None
        assert ledger.releases.get(head[0]).build_id == execution
        [observed] = ledger.observations(execution)
        assert observed.params["observed"] == "model_forward"
        # Collected as it went: what is left is the last epoch and the best.
        manifests = ledger.store.list_versions_by_id_prefix(f"torch_{execution}@")
        kept = {v.release_id for v in manifests if v.params.get("manifest")}
        assert kept == {final, head[0]}
        evaluation = ledger.derived(final, "eval")
        assert evaluation is not None
        assert evaluation.release.release_id == _said(resumed, r"as release (\w+)")

    branched = _example("branch.py", root, "--epochs=1")
    assert branched.returncode == 0, branched.stderr
    branch = _said(branched, r"branched (\w+) from release")
    with Ledger(Path(root)) as ledger:
        assert branch != execution
        assert ledger.builds.get_build(branch).params["init"] == head[0]


def test_the_distributed_example_releases_every_ranks_files_and_resumes(tmp_path: Path):
    root = str(tmp_path / "ledger")
    stopped = _example("distributed.py", root, "--epochs=2", "--crash-at=1", ranks=2)
    assert stopped.returncode == 0, stopped.stderr
    resumed = _example("distributed.py", root, "--epochs=2", ranks=2)
    assert resumed.returncode == 0, resumed.stderr
    execution = _said(resumed, r"resumed (\w+) after epoch 1")

    with Ledger(Path(root)) as ledger:
        assert ledger.builds.get_build(execution).state == "ready"
        for step in (1, 2):
            release = ledger.release_of(execution, step)
            assert release is not None
            outputs = ledger.releases.bundle(release.release_id).outputs
            files = {v.params["file"] for v in outputs if "file" in v.params}
            assert files == {f"{k}_{r}.pt" for k in ("model", "optim", "epoch") for r in (0, 1)}
