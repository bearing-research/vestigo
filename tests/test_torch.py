"""The PyTorch adapter (``vestigo.torch``) over real torch and torch_checkpointing.

Needs the ``torch`` extra; CI runs this module in its own job and the matrix ignores
it. Saves are synchronous on the local filesystem.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

from vestigo.releases import ReleaseRejected, ReleaseStore
from vestigo.torch import Definition, Ledger, Run

STEPS = 6


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[Ledger]:
    with Ledger(tmp_path / "ledger") as ledger:
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


def _manager(run: Run) -> CheckpointManager:
    config = CheckpointManager.Config.with_sync_save()
    # O_DIRECT is Linux-only and refused by some filesystems; the suite needs neither.
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    config.finalize_callback = run.finalize_callback
    return config.build()


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

    def train(self, run: Run, until: int) -> None:
        manager = _manager(run)
        while self.step < until:
            self.step += 1
            for batch in torch.randperm(64, generator=self.shuffle).split(16):
                self.optimizer.zero_grad()
                nn.functional.mse_loss(self.model(self.x[batch]), self.y[batch]).backward()
                self.optimizer.step()
            manager.save(
                run.checkpoint_id(self.step),
                {
                    "model": self.model.state_dict(),
                    "optim": self.optimizer.state_dict(),
                    "shuffle": self.shuffle.get_state(),
                    "step": self.step,
                },
            )

    def restore(self, run: Run) -> None:
        path = run.restore()
        assert path is not None
        loaded = _manager(run).load(
            str(path),
            into={"model": self.model.state_dict(), "optim": None, "shuffle": None, "step": None},
        )
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
    [output, *_] = ledger.releases.bundle(one.release_of(1).release_id).outputs
    assert ledger.store.definition_of(output.id, output.version) == trainer.program.node_id


# --- Checkpoints ---


def test_a_checkpoint_is_one_release_of_its_files_under_a_manifest(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        release = run.release_of(1)
        assert release is not None and release.state == "accepted"
        outputs = ledger.releases.bundle(release.release_id).outputs
        files = {v.params["file"]: v for v in outputs if "file" in v.params}
        [manifest] = [v for v in outputs if v.params.get("manifest")]

        assert set(files) == {"model_0.pt", "optim_0.pt", "shuffle_0.pt", "step_0.pt"}
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
        # The ledger holds the bytes; the directory the writer filled is gone.
        assert not Path(run.checkpoint_id(1)).exists()


def test_a_retried_finalize_returns_the_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.finalize_callback(SimpleNamespace(path=run.checkpoint_id(1)))

        assert len(ledger.releases.releases_of(run.execution_id, "checkpoint")) == 1


def test_a_checkpoint_outside_the_attempt_is_refused(ledger: Ledger, tmp_path: Path):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        elsewhere = tmp_path / "elsewhere" / "step-1"
        elsewhere.mkdir(parents=True)

        with pytest.raises(ValueError, match="not a checkpoint id"):
            run.finalize_callback(SimpleNamespace(path=str(elsewhere)))


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


def test_a_restore_refuses_bytes_that_differ_from_the_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as source:
        trainer.train(source, 1)
        release = source.release_of(1)
    [model] = [
        v
        for v in ledger.releases.bundle(release.release_id).outputs
        if v.params.get("file") == "model_0.pt"
    ]
    ledger.store.write_blob(model.id, model.version, b"not the checkpoint")

    with ledger.branch(trainer.program, release.release_id) as branch:
        with pytest.raises(ValueError, match="differs from its release"):
            branch.restore()


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
    assert {v.state for v in orphan.outputs} == {"building"}
    with ledger.resume(first.execution_id) as resumed:
        assert resumed.restores_from == first.release_of(1).release_id
        assert ledger.releases.get(orphan.release.release_id).state == "abandoned"
        assert ledger.release_of(first.execution_id, 2) is None


def test_a_checkpoint_that_fails_to_ingest_does_not_hold_up_the_next(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            patch.setattr(
                type(ledger.store), "publish_blob_from_path", lambda *a, **k: _raise(OSError)
            )
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
