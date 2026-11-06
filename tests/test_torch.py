"""The PyTorch adapter (``vestigo.torch``) over real torch and torch_checkpointing.

Needs the ``torch`` extra; CI runs this module in its own job and the matrix ignores
it. Every checkpoint is written through the ledger's own storage: on local blobs and
SQLite, on Postgres, on S3, GCS and Azure (their emulators), synchronously and
asynchronously, on one rank and on two (gloo, one process each).
"""

from __future__ import annotations

import errno
import functools
import gc
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import re
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import uuid
import warnings
import zipfile
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import closing
from datetime import timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from botocore.exceptions import ClientError, EndpointConnectionError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from torch import nn
from torch.distributed import distributed_c10d
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.storage.filesystem import LocalFileSystemStorage

import vestigo.torch
from vestigo import checkpoint_files
from vestigo.blob_store import BlobSettings
from vestigo.build_store import BuildStore
from vestigo.provenance import TransformSpec, compute_provenance_hash
from vestigo.releases import Release, ReleaseRejected, ReleaseStore
from vestigo.store import ArtifactStore, BuildLeaseLost, StagedVersion
from vestigo.torch import (
    EXECUTOR,
    Attempt,
    Definition,
    Execution,
    HostedRun,
    InputChanged,
    Ledger,
    LedgerStorage,
    Remote,
    RemoteBlobs,
    Run,
    _archive_metadata,
    _canonical,
    _Checkpoints,
    _copies,
    _file_digest,
    _Held,
    _kinds,
    _Location,
    _portable,
    _report_written,
    _seen,
)

STEPS = 6
ASYNC = CheckpointManager.Config.with_async_save()
# Windows keeps only a read-only flag: chmod(0) leaves a file readable.
POSIX_MODES = pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX file modes")


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


def _manager(checkpoints: _Checkpoints) -> CheckpointManager:
    return checkpoints.configure(CheckpointManager.Config.with_sync_save()).build()


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
        while self.step < until:
            self.step += 1
            for batch in torch.randperm(64, generator=self.shuffle).split(16):
                self.optimizer.zero_grad()
                nn.functional.mse_loss(self.model(self.x[batch]), self.y[batch]).backward()
                self.optimizer.step()
            pending = run.save(
                self.step,
                {
                    "model": self.model,
                    "optim": self.optimizer,
                    "shuffle": self.shuffle.get_state(),
                    "step": self.step,
                },
            )
            if isinstance(pending, Future) and self.step == until:
                pending.result()

    def restore(self, run: Run) -> None:
        assert run.restores_from is not None
        state = run.restore(
            {"model": self.model, "optim": self.optimizer, "shuffle": None, "step": None}
        )
        self.shuffle.set_state(state["shuffle"])
        self.step = state["step"]

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


def test_without_torch_the_adapter_names_its_extra():
    hidden = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "try:\n"
        "    import vestigo.torch\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", hidden], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "vestigo.torch needs the torch extra: install vestigo[torch]\n"


# --- The run API ---


def _run(ledger: Ledger, trainer: Trainer, inputs: dict | None = None, **kwargs) -> Run:
    return ledger.run(
        trainer.model,
        trainer.optimizer,
        loss=nn.MSELoss(),
        version="tiny-v1",
        inputs=inputs or {"train": "data@v=1"},
        config={"batch": 16},
        **kwargs,
    )


def _state(trainer: Trainer) -> dict:
    return {"model": trainer.model, "optim": trainer.optimizer, "step": 0}


def test_run_resumes_the_unfinished_execution_and_starts_after_a_finish(ledger: Ledger):
    expected = _uninterrupted(ledger)
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as first:
        assert first.restore(_state(trainer)) == _state(trainer)
        trainer.train(first, 3)

    trainer = Trainer(ledger)
    with _run(ledger, trainer) as resumed:
        assert resumed.execution_id == first.execution_id
        assert resumed.computation_key == ledger.computation_key(
            trainer.program, {"train": "data@v=1"}
        )
        state = resumed.restore(_state(trainer))
        assert state["model"] is trainer.model and state["optim"] is trainer.optimizer
        assert state["step"] == 3
        assert trainer.optimizer.state_dict()["state"], "the optimizer is loaded in place"
        trainer.step = state["step"]
        trainer.shuffle.set_state(resumed.restore({"shuffle": None})["shuffle"])
        trainer.train(resumed, STEPS)
        resumed.finish()
    assert _same(trainer.weights(), expected)

    with _run(ledger, Trainer(ledger)) as again:
        assert again.execution_id != first.execution_id
        assert again.restores_from is None


def test_run_refuses_an_execution_whose_lease_is_live(ledger: Ledger):
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as first:
        trainer.train(first, 1)
        with pytest.raises(ValueError, match="another attempt's lease on it is live"):
            _run(ledger, Trainer(ledger))
        assert [a.attempt for a in ledger.execution(first.execution_id).attempts] == [first.attempt]


def test_run_from_a_release_branches_and_resumes_the_branch(ledger: Ledger):
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as source:
        trainer.train(source, 2)
        source.finish()
    origin = ledger.release_of(source.execution_id, 2)
    assert origin is not None

    branched = Trainer(ledger)
    with _run(ledger, branched, from_release=origin.release_id) as branch:
        assert branch.execution_id != source.execution_id
        assert branch.restores_from == origin.release_id
        branch.restore({"model": branched.model})
        assert _same(branched.weights(), trainer.weights())
        branched.train(branch, 1)  # its own step 1, then the process stops
    with _run(ledger, Trainer(ledger), from_release=origin.release_id) as again:
        assert again.execution_id == branch.execution_id
        assert again.restores_from == branch.release_of(1).release_id


def test_two_runs_that_both_found_nothing_to_resume_open_one_execution(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    find = Ledger._find
    both = threading.Barrier(2)

    def found_together(self: Ledger, *args: object) -> str | None:
        execution = find(self, *args)
        both.wait(10)  # each looked before either opened
        return execution

    monkeypatch.setattr(Ledger, "_find", found_together)
    outcomes: list[Run | ValueError] = []

    def start(on: Ledger) -> None:
        try:
            outcomes.append(_run(on, Trainer(on)))
        except ValueError as exc:
            outcomes.append(exc)

    with ledger.config.open() as other:
        threads = [threading.Thread(target=start, args=(each,)) for each in (ledger, other)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        [run] = [each for each in outcomes if isinstance(each, Run)]
        with run:
            # The other found the first one's execution as it opened, and its lease live.
            [refused] = [each for each in outcomes if isinstance(each, ValueError)]
            assert "lease on it is live" in str(refused)
            builds = {b.build_id: b.state for b in ledger.builds.builds_of(EXECUTOR)}
            assert builds.pop(run.execution_id) == "building"
            assert list(builds.values()) == ["failed"]


def test_a_synchronous_save_returns_its_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as run:
        release = run.save(1, {"model": trainer.model, "note": "plain"})
        assert release == run.release_of(1)
        assert release.state == "accepted"
        files = {
            v.params.get("file", "") for v in ledger.releases.bundle(release.release_id).outputs
        }
        assert any(f.startswith("model") for f in files)
        assert ledger.load(release.release_id, {"note": None}) == {"note": "plain"}


def test_a_run_shares_no_manager_config_with_another(ledger: Ledger):
    shared = CheckpointManager.Config.with_sync_save()
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, checkpointing=shared) as one:
        one.manager  # built first, so the next run's configure would rewire it
        with ledger.start(trainer.program, checkpointing=shared) as two:
            two.manager
            assert one.save(1, {"step": 1}).build_id == one.execution_id
            assert two.save(1, {"step": 1}).build_id == two.execution_id
    assert shared.storage_config is None
    assert shared.save.writer_config.write_to_temp_dir is True


def test_finish_and_close_wait_for_the_save_in_flight_while_the_lease_holds(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []
    prepare = ReleaseStore.prepare

    def preparing(self, build_id, attempt, kind, *args, **kwargs):
        events.append(kind)
        return prepare(self, build_id, attempt, kind, *args, **kwargs)

    def closing(run: Run):
        close = run.manager.close

        def closed() -> None:
            build = ledger.builds.get_build(run.execution_id)
            events.append("close, held" if build.lease_expires_at > time.time() else "close")
            close()

        monkeypatch.setattr(run.manager, "close", closed)

    monkeypatch.setattr(ReleaseStore, "prepare", preparing)
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as run:
        trainer.train(run, 1)
        closing(run)
        events.clear()
        run.finish()
    assert events[:2] == ["close, held", "final"]

    trainer = Trainer(ledger)
    with _run(ledger, trainer) as run:
        trainer.train(run, 1)
        closing(run)
        events.clear()
    assert events[:1] == ["close, held"]


def test_a_failed_manager_close_still_drops_the_lease(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    run = _run(ledger, trainer)
    trainer.train(run, 1)
    monkeypatch.setattr(run.manager, "close", lambda: _raise(OSError("the write failed")))
    with pytest.raises(OSError, match="the write failed"):
        run.close()
    with ledger.resume(run.execution_id) as resumed:  # no live lease in the way
        assert resumed.restores_from == run.release_of(1).release_id


def test_a_failed_close_does_not_hide_why_the_block_was_left(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with pytest.raises(ValueError, match="the loss diverged"):
        with _run(ledger, trainer) as run:
            trainer.train(run, 1)
            monkeypatch.setattr(run.manager, "close", lambda: _raise(OSError("the write failed")))
            raise ValueError("the loss diverged")
    with ledger.resume(run.execution_id) as resumed:  # closed all the same
        assert resumed.restores_from == run.release_of(1).release_id


@POSIX_MODES
def test_a_failed_asynchronous_save_is_reported_by_its_future_alone(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # torch_checkpointing 0.1.1 stages for CUDA whenever an accelerator exists: hide MPS.
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    tree = _tree(tmp_path / "tree")
    trainer = Trainer(ledger)
    asynchronous = CheckpointManager.Config.with_async_save()
    with _run(ledger, trainer, {"train": tree}, checkpointing=asynchronous) as run:
        trainer.train(run, 1)
        # Unreadable, not gone (a gone input is other bytes and lapses the lease).
        tree.chmod(0)
        try:
            with pytest.raises(InputChanged, match="cannot be read"):
                trainer.train(run, 2)
        finally:
            tree.chmod(0o700)
        # Handled, the step's error is not raised again: the run finishes.
        run.finish()
    assert ledger.builds.get_build(run.execution_id).state == "ready"
    assert ledger.final_checkpoint(run.execution_id).release_id == run.release_of(1).release_id


def test_load_reads_an_accepted_checkpoint_into_live_objects(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    closed: list[CheckpointManager] = []
    close = CheckpointManager.close

    def closing(self: CheckpointManager) -> None:
        closed.append(self)
        close(self)

    trainer = Trainer(ledger)
    with _run(ledger, trainer) as run:
        trainer.train(run, 2)
        run.finish()
    release = ledger.final_checkpoint(run.execution_id)
    assert release is not None
    model = _model(hidden=8)
    nn.init.zeros_(model[0].weight)
    monkeypatch.setattr(CheckpointManager, "close", closing)
    state = ledger.load(release.release_id, {"model": model, "step": None})
    assert state["model"] is model and state["step"] == 2
    assert _same([p.detach() for p in model.parameters()], trainer.weights())
    with pytest.raises(FileNotFoundError, match="never-saved"):
        ledger.load(release.release_id, {"never-saved": None})
    assert len(closed) == 2, "each load closes its manager, a failed one too"
    with pytest.raises(ValueError, match="not an accepted checkpoint"):
        ledger.load("no-such-release", {"model": model})


# --- Identity ---


def test_a_model_is_its_class_and_state_dict_schema_under_a_version(ledger: Ledger):
    first = ledger.define_model(_model(), version="v1")
    retrained = _model()
    with torch.no_grad():
        retrained[0].weight.add_(1.0)

    assert ledger.define_model(retrained, version="v1").node_id == first.node_id
    assert ledger.define_model(_model(), version="v2").node_id != first.node_id
    assert ledger.define_model(_model(hidden=16), version="v1").node_id != first.node_id


class _Calibrated(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def get_extra_state(self) -> dict[str, bool]:
        return {"calibrated": True}

    def set_extra_state(self, state: object) -> None:
        pass


def test_a_model_s_extra_state_is_in_its_schema_by_its_type(ledger: Ledger):
    model = ledger.define_model(_Calibrated(), version="v1")
    schema = json.loads(model.spec.payload)["state_dict"]
    assert ["_extra_state", None, "builtins.dict"] in schema


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


def test_a_param_group_s_own_settings_are_part_of_the_program(ledger: Ledger):
    model = _model()
    definition = ledger.define_model(model, version="v1")
    first, last = model[0], model[2]

    def program(lr: float) -> str:
        groups = [{"params": first.parameters(), "lr": lr}, {"params": last.parameters()}]
        optimizer = torch.optim.SGD(groups, lr=1e-3)
        return ledger.define_program(definition, loss="mse", optimizer=optimizer).node_id

    assert program(1e-5) == program(1e-5)
    assert program(1e-5) != program(1e-1)


class _Scaled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.tensor(2.0)  # an attribute, not a buffer


class _Combined(nn.Module):
    """A loss around a stock one, as a composite loss is."""

    def __init__(self, inner: nn.Module, alpha: float = 0.5) -> None:
        super().__init__()
        self.inner = inner
        self.alpha = alpha


class _Margin:
    def __init__(self, margin: float) -> None:
        self.margin = margin

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target + self.margin).clamp(min=0).mean()


class _Scored(nn.Module):
    """A model whose own method is the loss."""

    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin
        self.layer = nn.Linear(4, 3)

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target + self.margin).clamp(min=0).mean()


class _Slotted:
    __slots__ = ("margin",)

    def __init__(self, margin: float) -> None:
        self.margin = margin

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target + self.margin).clamp(min=0).mean()


def _focal(gamma: float):
    def loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (output - target).abs().pow(gamma).mean()

    return loss


def _wrapped(function):
    @functools.wraps(function)
    def wrapper(*args):
        return function(*args)

    return wrapper


_decorated_mse = _wrapped(nn.functional.mse_loss)


def test_a_program_records_its_loss_s_settings_and_tensors(ledger: Ledger):
    model = _model()
    definition = ledger.define_model(model, version="v1")

    def program(loss: object) -> str:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return ledger.define_program(definition, loss=loss, optimizer=optimizer).node_id

    huber = program(nn.HuberLoss(delta=1.0))
    assert program(nn.HuberLoss(delta=1.0)) == huber
    assert program(nn.HuberLoss(delta=1.0).eval()) == huber
    assert program(nn.HuberLoss(delta=2.0)) != huber

    weighted = program(nn.CrossEntropyLoss(weight=torch.ones(3)))
    assert program(nn.CrossEntropyLoss(weight=torch.ones(3))) == weighted
    assert program(nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 1.0]))) != weighted
    assert program(nn.CrossEntropyLoss(weight=torch.ones(3, dtype=torch.bfloat16))) != weighted
    assert program(nn.CrossEntropyLoss()) != weighted

    f = nn.functional
    assert program(f.mse_loss) == program(f.mse_loss)
    assert program(f.mse_loss) != program(f.l1_loss)
    assert program(nn.MSELoss) != program(nn.L1Loss)
    assert program(functools.partial(f.huber_loss, delta=1.0)) != program(
        functools.partial(f.huber_loss, delta=2.0)
    )
    with pytest.raises(ValueError, match="lambda"):
        program(lambda output, target: f.mse_loss(output, target))
    with pytest.raises(TypeError, match="JSON"):
        program(_Scaled())


def test_a_loss_s_submodules_and_the_objects_its_methods_hold_are_part_of_the_program(
    ledger: Ledger,
):
    model = _model()
    definition = ledger.define_model(model, version="v1")

    def program(loss: object) -> str:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return ledger.define_program(definition, loss=loss, optimizer=optimizer).node_id

    smoothed = program(_Combined(nn.CrossEntropyLoss(label_smoothing=0.1)))
    assert program(_Combined(nn.CrossEntropyLoss(label_smoothing=0.1))) == smoothed
    assert program(_Combined(nn.CrossEntropyLoss(label_smoothing=0.3))) != smoothed
    assert program(_Combined(nn.MSELoss())) != program(_Combined(nn.L1Loss()))
    triplet = nn.TripletMarginWithDistanceLoss
    assert program(triplet(distance_function=nn.PairwiseDistance(p=1))) != program(
        triplet(distance_function=nn.CosineSimilarity())
    )
    assert program(_Margin(1.0).loss) == program(_Margin(1.0).loss)
    assert program(_Margin(1.0).loss) != program(_Margin(9.0).loss)
    # A module's method records its settings, not its weights: two inits are one program.
    torch.manual_seed(1)
    first = _Scored(1.0)
    torch.manual_seed(2)
    assert program(first.compute_loss) == program(_Scored(1.0).compute_loss)
    assert program(first.compute_loss) != program(_Scored(9.0).compute_loss)
    assert program(_Slotted(1.0)) != program(_Slotted(9.0))
    assert program(torch.compile(nn.HuberLoss(delta=2.0))) == program(nn.HuberLoss(delta=2.0))

    # What a nested function or a wrapper captured has no name: refused, not recorded.
    with pytest.raises(ValueError, match="defined in a function"):
        program(_focal(2.0))
    with pytest.raises(ValueError, match="captures values"):
        program(_decorated_mse)


@pytest.mark.parametrize(
    ("loss", "message"),
    [
        (nn.CrossEntropyLoss(weight=torch.empty(3, device="meta")), "meta device"),
        (nn.HuberLoss(delta=float("nan")), "not JSON"),
    ],
    ids=["meta tensor", "nan"],
)
def test_a_loss_that_cannot_be_recorded_raises_type_error(ledger: Ledger, loss, message: str):
    model = _model()
    definition = ledger.define_model(model, version="v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(TypeError, match=message):
        ledger.define_program(definition, loss=loss, optimizer=optimizer)


def test_a_sparse_loss_tensor_is_recorded_by_its_values(ledger: Ledger):
    model = _model()
    definition = ledger.define_model(model, version="v1")

    def program(loss: object) -> str:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return ledger.define_program(definition, loss=loss, optimizer=optimizer).node_id

    weight = torch.tensor([1.0, 0.0, 2.0])
    sparse = _Combined(nn.MSELoss())
    sparse.register_buffer("weight", weight.to_sparse())
    dense = _Combined(nn.MSELoss())
    dense.register_buffer("weight", weight)
    other = _Combined(nn.MSELoss())
    other.register_buffer("weight", torch.tensor([1.0, 0.0, 3.0]).to_sparse())
    assert program(sparse) == program(dense) != program(other)


def test_a_loss_named_by_a_str_is_recorded_as_given_with_its_settings_in_config(
    ledger: Ledger,
):
    model = _model()
    definition = ledger.define_model(model, version="v1")

    def program(loss: str, config: dict) -> Definition:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return ledger.define_program(definition, loss=loss, optimizer=optimizer, config=config)

    scaled = program("scaled-mse", {"scale": 2.0})
    payload = json.loads(scaled.spec.payload)
    assert (payload["loss"], payload["config"]) == ("scaled-mse", {"scale": 2.0})
    assert program("scaled-mse", {"scale": 2.0}).node_id == scaled.node_id
    assert program("scaled-mse", {"scale": 3.0}).node_id != scaled.node_id
    assert program("scaled-mae", {"scale": 2.0}).node_id != scaled.node_id


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


# --- Verified inputs ---


def _file(path: Path, content: bytes = b"rows") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _tree(root: Path) -> Path:
    _file(root / "a.bin", b"a")
    _file(root / "sub" / "b.bin", b"b")
    return root


def _key(ledger: Ledger, path: Path) -> str:
    return ledger.computation_key(Trainer(ledger).program, {"train": path})


def _counted(monkeypatch: pytest.MonkeyPatch, name: str) -> list:
    """Calls of ``vestigo.torch.<name>``, each one's first argument."""
    calls = []
    real = getattr(vestigo.torch, name)

    def counted(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(vestigo.torch, name, counted)
    return calls


def test_a_path_input_is_its_bytes_wherever_they_are(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "one" / "train.bin")
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": data, "seed": "seed=1"}) as run:
        params = ledger.builds.get_build(run.execution_id).params
        assert params["inputs"] == {
            "train": {"sha256": hashlib.sha256(b"rows").hexdigest(), "size": 4},
            "seed": "seed=1",
        }

    assert _key(ledger, _file(tmp_path / "two" / "copy.bin")) == _key(ledger, data)
    assert _key(ledger, _file(tmp_path / "three" / "train.bin", b"rowz")) != _key(ledger, data)
    with pytest.raises(TypeError):
        ledger.computation_key(trainer.program, {"train": 3})


def test_a_directory_input_is_its_files_names_and_bytes(ledger: Ledger, tmp_path: Path):
    tree = _tree(tmp_path / "tree")
    key = _key(ledger, tree)
    assert _key(ledger, _tree(tmp_path / "elsewhere")) == key

    # A link to a file of the tree is that file's bytes under the link's name.
    linked = _tree(tmp_path / "linked")
    (linked / "c.bin").symlink_to(linked / "a.bin")
    copied = _tree(tmp_path / "copied")
    _file(copied / "c.bin", b"a")
    assert _key(ledger, linked) == _key(ledger, copied) != key

    # What a file browser writes into a folder it shows is not the input's.
    browsed = _tree(tmp_path / "browsed")
    for junk in (".DS_Store", "sub/._b.bin", "Thumbs.db", "desktop.ini"):
        _file(browsed / junk, b"junk")
    assert _key(ledger, browsed) == key

    changes = {
        # Kept in sorted order, so only the name in the listing differs.
        "renamed": lambda t: (t / "a.bin").rename(t / "aa.bin"),
        "written": lambda t: _file(t / "a.bin", b"A"),
        "removed": lambda t: (t / "sub" / "b.bin").unlink(),
        "hidden": lambda t: _file(t / ".hidden", b""),
    }
    for name, change in changes.items():
        changed = _tree(tmp_path / name)
        change(changed)
        assert _key(ledger, changed) != key, name


def test_a_directory_refuses_what_it_cannot_pin(ledger: Ledger, tmp_path: Path):
    outside = _file(tmp_path / "outside.bin")
    refused = {
        "a link out of the tree": lambda t: (t / "out.bin").symlink_to(outside),
        "a link to a directory": lambda t: (t / "dir").symlink_to(t / "sub"),
    }
    if sys.platform != "win32":  # no FIFOs there
        refused["a special file"] = lambda t: os.mkfifo(t / "pipe")
    for name, make in refused.items():
        tree = _tree(tmp_path / name.replace(" ", "-"))
        make(tree)
        with pytest.raises(ValueError):
            _key(ledger, tree)
    looped = _tree(tmp_path / "looped")
    (looped / "x").symlink_to(looped / "y")
    (looped / "y").symlink_to(looped / "x")
    with pytest.raises(OSError):
        _key(ledger, looped)
    _file(tmp_path / "empty" / "sub" / ".DS_Store")
    with pytest.raises(ValueError, match="holds no file"):
        _key(ledger, tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        _key(ledger, tmp_path / "missing")


def test_data_changed_after_a_crash_is_a_fresh_run_and_unchanged_resumes(
    ledger: Ledger, tmp_path: Path
):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as first:
        trainer.train(first, 1)

    with _run(ledger, Trainer(ledger), {"train": data}) as resumed:
        assert resumed.execution_id == first.execution_id
        assert resumed.restores_from == first.release_of(1).release_id

    _file(data, b"rowz")
    with _run(ledger, Trainer(ledger), {"train": data}) as fresh:
        assert fresh.execution_id != first.execution_id
        assert fresh.restores_from is None


@pytest.mark.parametrize("change", ["written", "added", "removed"])
def test_a_save_after_other_bytes_releases_nothing_and_lapses_the_lease(
    ledger: Ledger, tmp_path: Path, change: str
):
    tree = _tree(tmp_path / "tree")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": tree}) as run:
        trainer.train(run, 1)
        if change == "written":
            (tree / "a.bin").write_bytes(b"A")
        elif change == "added":
            (tree / "sub" / "c.bin").write_bytes(b"c")
        else:
            (tree / "sub" / "b.bin").unlink()

        with pytest.raises(InputChanged, match=f"at {tree.absolute()} has other bytes"):
            trainer.train(run, 2)
        assert run.release_of(2) is None
        assert ledger.builds.get_build(run.execution_id).lease_owner is None


@pytest.mark.parametrize("change", ["file removed", "directory removed", "directory emptied"])
def test_a_save_after_an_input_is_gone_releases_nothing_and_lapses_the_lease(
    ledger: Ledger, tmp_path: Path, change: str
):
    tree = _tree(tmp_path / "tree")
    data = tree / "a.bin" if change == "file removed" else tree
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as run:
        trainer.train(run, 1)
        if change == "file removed":
            data.unlink()
        elif change == "directory removed":
            tree.rename(tmp_path / "aside")
        else:
            for each in (tree / "a.bin", tree / "sub" / "b.bin"):
                each.unlink()
        with pytest.raises(InputChanged, match=f"at {data.absolute()} is gone"):
            trainer.train(run, 2)
        assert run.release_of(2) is None
        assert ledger.builds.get_build(run.execution_id).lease_owner is None


def test_a_save_while_an_input_cannot_be_read_fails_that_step_only(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tree = _tree(tmp_path / "tree")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": tree}) as run:
        trainer.train(run, 1)
        scan = vestigo.torch._scan

        def failing(path: Path) -> list:
            raise OSError(errno.EIO, "Input/output error", str(path))

        monkeypatch.setattr(vestigo.torch, "_scan", failing)
        with pytest.raises(InputChanged, match="cannot be read"):
            trainer.train(run, 2)
        # Perhaps a moment's error on a shared mount: the lease stays, the next save holds.
        assert ledger.builds.get_build(run.execution_id).lease_owner == run.attempt
        monkeypatch.setattr(vestigo.torch, "_scan", scan)
        trainer.train(run, 3)
        assert run.release_of(2) is None and run.release_of(3) is not None


def test_a_change_after_rank_zero_s_report_fails_the_step(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as run:
        config = run._checkpoints.configure(CheckpointManager.Config.with_sync_save())

        def reported_then_changed(args) -> None:
            _report_written(args)
            # As while rank 0 waits at the barrier for a slower rank.
            data.write_bytes(b"rowz")

        config.pre_finalize_callback = reported_then_changed
        manager = config.build()
        try:
            with pytest.raises(InputChanged, match="on rank 0"):
                manager.save(run._checkpoints.checkpoint_id(1), {"step": 1})
        finally:
            manager.close()
        assert run.release_of(1) is None


def test_a_file_too_fresh_to_trust_its_stat_is_read_at_each_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A rewrite within the stat's timestamp tick leaves it as it was, as on a coarse
    # clock: the mtime set back, and the first write's ctime kept.
    data = _file(tmp_path / "train.bin", b"AAAA")
    first = os.stat(data)
    [held] = vestigo.torch._resolve({"train": data}).held
    data.write_bytes(b"BBBB")
    os.utime(data, ns=(first.st_atime_ns, first.st_mtime_ns))
    real = vestigo.torch._stat_key
    monkeypatch.setattr(vestigo.torch, "_stat_key", lambda st: (*real(st)[:4], first.st_ctime_ns))
    assert vestigo.torch._seen([held], {})["train"]["pin"] != held.pinned


def test_a_stat_that_hashed_to_the_pin_is_not_read_again(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Older than a scan's timestamp tick, so the stat that hashed to the pin is trusted.
    monkeypatch.setattr("vestigo.torch._RACY_NS", 0)
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as run:
        data.chmod(0o600)
        pins = _counted(monkeypatch, "_pin")
        trainer.train(run, 2)
        assert len(pins) == 1


def test_a_save_after_the_same_bytes_under_another_stat_is_released(ledger: Ledger, tmp_path: Path):
    tree = _tree(tmp_path / "tree")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": tree}) as run:
        trainer.train(run, 1)
        (tree / "a.bin").chmod(0o600)
        os.utime(tree / "sub" / "b.bin")
        os.link(tree / "a.bin", tmp_path / "hardlink.bin")
        _file(tree / ".DS_Store")
        _file(tree / "sub" / "._b.bin")
        trainer.train(run, 2)
        assert run.release_of(2) is not None


def test_a_changed_input_lapses_the_lease_so_a_resume_need_not_wait(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    run = _run(ledger, trainer, {"train": data})
    trainer.train(run, 1)
    data.write_bytes(b"rowz")
    with pytest.raises(InputChanged):
        trainer.train(run, 2)

    # The bytes put back are the same computation, taken over though `run` is open.
    data.write_bytes(b"rows")
    with _run(ledger, Trainer(ledger), {"train": data}) as resumed:
        assert resumed.execution_id == run.execution_id
        assert resumed.restores_from == run.release_of(1).release_id
    run.close()


def test_finish_after_a_change_to_an_input_does_not_finish(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as run:
        trainer.train(run, 1)
        data.write_bytes(b"rowz")
        with pytest.raises(InputChanged):
            run.finish()
        assert ledger.builds.get_build(run.execution_id).lease_owner is None
    assert ledger.builds.get_build(run.execution_id).state == "building"
    assert ledger.final_checkpoint(run.execution_id) is None


def test_a_save_through_the_manager_itself_holds_the_inputs(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data}) as run:
        trainer.train(run, 1)
        data.write_bytes(b"rowz")
        # As another rank's script saves, or a write still in flight when it lands.
        manager = _manager(run._checkpoints)
        try:
            with pytest.raises(InputChanged):
                manager.save(run._checkpoints.checkpoint_id(2), {"step": 2})
        finally:
            manager.close()
        assert run.release_of(2) is None
        data.write_bytes(b"rows")
        with _run(ledger, Trainer(ledger), {"train": data}) as resumed:
            assert resumed.execution_id == run.execution_id


def test_a_save_scans_its_inputs_once_reads_none_and_ships_no_listing(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    small = _tree(tmp_path / "small")
    large = _tree(tmp_path / "large")
    for i in range(300):
        _file(large / "shards" / f"{i:04}.bin", b"%d" % i)
    # Files older than a scan's timestamp tick, whose stat a check trusts.
    monkeypatch.setattr("vestigo.torch._RACY_NS", 0)
    sizes = []
    scans = _counted(monkeypatch, "_scan")
    pins = _counted(monkeypatch, "_pin")
    reads = _counted(monkeypatch, "_file_digest")
    for tree in (small, large):
        trainer = Trainer(ledger)
        with _run(ledger, trainer, {"train": tree}) as run:
            sizes.append(len(pickle.dumps(run._checkpoints)))
            scans.clear()
            pins.clear()
            reads.clear()
            trainer.train(run, 1)
            assert scans == [Path(str(tree.absolute()))]
            # An unchanged stat reads none, not even the digest cache.
            assert not pins
            assert not [r for r in reads if Path(r).is_relative_to(tree.resolve())]
    # Rank 0's file list stays on rank 0: every rank and writer gets a fingerprint.
    assert abs(sizes[1] - sizes[0]) < 64


def test_rank_zero_records_its_inputs_in_the_manifest(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data, "seed": "seed=1"}) as run:
        trainer.train(run, 1)
        outputs = _outputs(ledger, run, 1)
        pinned = ledger.builds.get_build(run.execution_id).params["inputs"]["train"]
        [manifest] = [v for v in outputs if v.params.get("manifest")]
        content = json.loads(ledger.store.read_blob(manifest.id, manifest.version))
        assert content["inputs"] == {"train": {"path": str(data.absolute()), "pin": pinned}}
        # Its report, written before the barrier, leaves its inputs to the acceptance.
        [own] = [v for v in outputs if "rank" in v.params]
        assert json.loads(ledger.store.read_blob(own.id, own.version))["inputs"] is None


def test_a_rank_s_copies_must_be_the_bytes_rank_0_pinned(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    objects = _bucket(monkeypatch)
    objects["s3://lake/eval.bin"] = (1, "etag:e")
    # Files older than a scan's timestamp tick, whose stat a check trusts.
    monkeypatch.setattr("vestigo.torch._RACY_NS", 0)
    data = _file(tmp_path / "rank0" / "train.bin")
    copy = _file(tmp_path / "rank1" / "train.bin")
    other = _file(tmp_path / "other" / "train.bin", b"rowz")
    remote = Remote("s3://lake/eval.bin")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": data, "eval": remote}) as run:
        held = run._checkpoints.inputs

    copies = _copies({"train": copy, "eval": remote}, held, 1)
    assert {each.role: each.path for each in copies} == {
        "train": str(copy.absolute()),
        "eval": "s3://lake/eval.bin",
    }
    assert [each.pinned for each in copies] == [each.pinned for each in held]
    with pytest.raises(InputChanged, match=f"at {re.escape(str(other))} has other bytes on rank 1"):
        _copies({"train": other, "eval": remote}, held, 1)

    # Rank 0's own copy, as its scan found it, is not read again.
    def unread(files: object) -> None:
        raise AssertionError("read again")

    monkeypatch.setattr(vestigo.torch, "_pin", unread)
    assert _copies({"train": data, "eval": remote}, held, 1)[0].pinned == held[0].pinned


def test_ranks_agree_on_labels_and_uris_but_not_where_their_copies_are(tmp_path: Path):
    remote = Remote("s3://lake/eval/")
    terms = _kinds({"label": "data@v=1", "eval": remote, "train": tmp_path / "rank0"})
    assert terms == _kinds({"label": "data@v=1", "eval": remote, "train": tmp_path / "rank1"})
    assert terms != _kinds(
        {"label": "data@v=1", "eval": Remote("s3://lake/other/"), "train": tmp_path}
    )
    assert terms != _kinds({"label": "data@v=2", "eval": remote, "train": tmp_path})
    assert terms != _kinds({"label": "data@v=1", "eval": remote, "train": str(tmp_path)})


@pytest.mark.parametrize("seed", [torch.tensor(1), float("nan"), float("inf")])
def test_a_term_that_is_not_json_fails_before_any_input_is_read(
    ledger: Ledger, tmp_path: Path, seed: object
):
    trainer = Trainer(ledger)
    with pytest.raises(TypeError, match="JSON"):
        ledger.run(
            trainer.model,
            trainer.optimizer,
            loss=nn.MSELoss(),
            version="tiny-v1",
            inputs={"train": tmp_path / "missing"},
            config={"seed": seed},
        )


def test_a_launcher_s_ranks_need_their_process_group(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="initialize the process group"):
        _run(ledger, Trainer(ledger))


class _Unbuildable(Exception):
    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"{code}: {detail}")


def test_an_error_goes_to_other_ranks_as_itself_else_as_its_repr():
    sent = _portable(InputChanged("input 'train' has other bytes"))
    assert type(sent) is InputChanged and str(sent) == "input 'train' has other bytes"
    sent = _portable(_Unbuildable(503, "busy"))
    assert type(sent) is RuntimeError and "503: busy" in str(sent)


def test_resume_takes_the_execution_s_own_inputs_and_holds_them(ledger: Ledger, tmp_path: Path):
    data = _file(tmp_path / "train.bin")
    other = _file(tmp_path / "other.bin", b"rowz")
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": data}) as first:
        trainer.train(first, 1)

    with pytest.raises(ValueError, match="resume it with its inputs"):
        ledger.resume(first.execution_id)
    with pytest.raises(ValueError, match="not the inputs"):
        ledger.resume(first.execution_id, inputs={"train": other})
    resumed_trainer = Trainer(ledger)
    with ledger.resume(first.execution_id, inputs={"train": data}) as resumed:
        resumed_trainer.restore(resumed)
        data.write_bytes(b"rowz")
        with pytest.raises(InputChanged):
            resumed_trainer.train(resumed, 2)


def test_a_relaunch_reads_no_input_again_but_one_whose_stat_changed(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Every file here is settled, so the cache keeps each digest it reads.
    monkeypatch.setattr("vestigo.torch._RACY_NS", 0)
    reads = _counted(monkeypatch, "_file_digest")
    data = _file(tmp_path / "train.bin")
    key = _key(ledger, data)
    assert _key(ledger, data) == key
    assert len(reads) == 1

    # A new mtime: Windows reports no change time, which a chmod alone would move.
    os.utime(data, ns=(data.stat().st_atime_ns, data.stat().st_mtime_ns - 10**9))
    assert _key(ledger, data) == key
    assert len(reads) == 2


def test_a_file_changed_in_the_last_two_seconds_is_read_each_time(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reads = _counted(monkeypatch, "_file_digest")
    # An old mtime does not settle it: rsync and cp -p set one back, and the ctime is now.
    data = _file(tmp_path / "train.bin")
    old = time.time() - 3600
    os.utime(data, (old, old))
    _key(ledger, data)
    _key(ledger, data)
    assert len(reads) == 2


def test_a_digest_is_cached_only_for_a_file_settled_before_its_read_began(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A long pass: the clock is ten seconds on once the file is read.
    now = [time.time_ns()]
    hashed = vestigo.torch._hash

    def slow(scanned: object) -> str:
        now[0] += 10 * 10**9
        return hashed(scanned)

    monkeypatch.setattr("vestigo.torch.time.time_ns", lambda: now[0])
    monkeypatch.setattr("vestigo.torch._hash", slow)
    data = _file(tmp_path / "train.bin")
    reads = _counted(monkeypatch, "_file_digest")
    _key(ledger, data)
    _key(ledger, data)
    assert len(reads) == 2


@POSIX_MODES
def test_the_digest_cache_is_its_owner_s_alone(ledger: Ledger, tmp_path: Path):
    _key(ledger, _file(tmp_path / "train.bin"))
    cache = Path(os.environ["VESTIGO_CACHE_DIR"])
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert stat.S_IMODE((cache / "digests.sqlite").stat().st_mode) == 0o600


def test_a_corrupt_digest_cache_closes_its_connection(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The warning's record would hold the failed frame, and the connection, alive.
    monkeypatch.setattr(logging.getLogger("vestigo.torch"), "disabled", True)
    cache = Path(os.environ["VESTIGO_CACHE_DIR"])
    _file(cache / "digests.sqlite", b"not a database, " * 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _key(ledger, _file(tmp_path / "train.bin"))
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


def test_a_file_written_while_it_is_hashed_is_refused(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def raced(path: Path) -> tuple[int, str]:
        read = _file_digest(path)
        path.write_bytes(b"rowz")
        return read

    monkeypatch.setattr("vestigo.torch._file_digest", raced)
    with pytest.raises(InputChanged, match="while it was hashed"):
        _key(ledger, _file(tmp_path / "train.bin"))


def test_the_first_failed_read_stops_the_hashing(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tree = tmp_path / "tree"
    for i in range(20):
        _file(tree / f"{i:02}.bin", b"%d" % i)
    reads = []

    def failing(path: Path) -> tuple[int, str]:
        reads.append(path)
        raise PermissionError(path)

    monkeypatch.setattr("vestigo.torch._file_digest", failing)
    monkeypatch.setattr("vestigo.torch.os.cpu_count", lambda: 1)
    with pytest.raises(PermissionError):
        _key(ledger, tree)
    # One worker: the one that failed, and at most the one it took up meanwhile.
    assert len(reads) <= 2


@pytest.mark.parametrize("unusable", ["a file in the way", "no home"])
def test_an_unusable_digest_cache_hashes_every_input(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog, unusable: str
):
    if unusable == "no home":
        monkeypatch.delenv("VESTIGO_CACHE_DIR")
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: _raise(RuntimeError))
    else:
        blocked = _file(tmp_path / "a-file", b"")
        monkeypatch.setenv("VESTIGO_CACHE_DIR", str(blocked / "cache"))
    data = _file(tmp_path / "train.bin")
    with caplog.at_level(logging.WARNING, logger="vestigo.torch"):
        assert _key(ledger, data) == ledger.computation_key(
            Trainer(ledger).program, {"train": _file(tmp_path / "again" / "train.bin")}
        )
    assert "No digest cache" in caplog.text


def _bucket(monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[int, str]]:
    """An object store in memory, by URI: each object's size and its bytes' name."""
    objects: dict[str, tuple[int, str]] = {}

    def listed(remote: Remote) -> list[tuple[str, int, str]]:
        if remote.uri == "s3://down/":
            raise ConnectionError("the endpoint is not answering")
        bucket = remote.uri.split("/")[2]
        found = []
        for uri, (size, name) in objects.items():
            key = uri.removeprefix(f"s3://{bucket}/")
            if key != uri and (
                uri == remote.uri or remote.uri.endswith("/") and uri.startswith(remote.uri)
            ):
                found.append((key, size, name))
        return found

    monkeypatch.setattr(Remote, "objects", listed)
    return objects


def test_a_remote_input_is_pinned_to_its_store_s_names_with_its_uri(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    objects = _bucket(monkeypatch)
    objects["s3://lake/data/b.bin"] = (2, "etag:b")
    objects["s3://lake/data/a.bin"] = (1, "etag:a")
    objects["s3://copy/data/a.bin"] = (1, "etag:a")
    objects["s3://copy/data/b.bin"] = (2, "etag:b")

    def key(uri: str) -> str:
        return ledger.computation_key(Trainer(ledger).program, {"train": Remote(uri)})

    trainer = Trainer(ledger)
    inputs = {"shards": Remote("s3://lake/data/"), "one": Remote("s3://lake/data/a.bin")}
    with ledger.start(trainer.program, inputs) as run:
        pinned = ledger.builds.get_build(run.execution_id).params["inputs"]
    assert pinned["one"] == {"uri": "s3://lake/data/a.bin", "size": 1, "object": "etag:a"}
    assert pinned["shards"]["uri"] == "s3://lake/data/" and pinned["shards"]["objects"] == 2

    prefix = key("s3://lake/data/")
    # The same objects under another bucket are another input: a URI names them anywhere.
    assert key("s3://copy/data/") != prefix
    objects["s3://lake/data/a.bin"] = (1, "etag:A")
    assert key("s3://lake/data/") != prefix
    objects["s3://lake/data/a.bin"] = (1, "etag:a")
    assert key("s3://lake/data/") == prefix
    objects["s3://lake/data/c.bin"] = (3, "etag:c")
    assert key("s3://lake/data/") != prefix

    with pytest.raises(ValueError, match="holds no object"):
        key("s3://lake/missing/")
    with pytest.raises(ValueError, match="holds no object"):
        key("s3://lake/data/missing.bin")
    for bad in ("file:///data", "s3://", "hdfs://lake/data"):
        with pytest.raises(ValueError):
            Remote(bad)
    assert "secret" not in repr(Remote("s3://lake/", BlobSettings(s3_secret_key="secret")))


@pytest.mark.parametrize(
    ("change", "message"), [("replaced", "has other bytes"), ("removed", "is gone")]
)
def test_a_remote_input_with_other_objects_fails_the_save_and_lapses_the_lease(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch, change: str, message: str
):
    objects = _bucket(monkeypatch)
    objects["s3://lake/data/a.bin"] = (1, "etag:a")
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": Remote("s3://lake/data/")}) as run:
        trainer.train(run, 1)
        if change == "replaced":
            objects["s3://lake/data/a.bin"] = (1, "etag:A")
        else:
            del objects["s3://lake/data/a.bin"]
        with pytest.raises(InputChanged, match=f"at s3://lake/data/ {message}"):
            trainer.train(run, 2)
        assert run.release_of(2) is None
        assert ledger.builds.get_build(run.execution_id).lease_owner is None


def test_a_store_that_does_not_answer_fails_that_step_only(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    objects = _bucket(monkeypatch)
    objects["s3://down/a.bin"] = (1, "etag:a")
    listed = Remote.objects
    monkeypatch.setattr(Remote, "objects", lambda remote: [("a.bin", 1, "etag:a")])
    trainer = Trainer(ledger)
    with _run(ledger, trainer, {"train": Remote("s3://down/")}) as run:
        trainer.train(run, 1)
        monkeypatch.setattr(Remote, "objects", listed)
        with pytest.raises(InputChanged, match="cannot be read.*not answering"):
            trainer.train(run, 2)
        assert ledger.builds.get_build(run.execution_id).lease_owner == run.attempt
        monkeypatch.setattr(Remote, "objects", lambda remote: [("a.bin", 1, "etag:a")])
        trainer.train(run, 3)
        assert run.release_of(3) is not None


@pytest.mark.parametrize(
    "error",
    [
        ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "HeadObject"),
        EndpointConnectionError(endpoint_url="http://lake"),
        ConnectionResetError("reset"),
    ],
    ids=["refused", "unreachable", "dropped"],
)
def test_only_the_store_s_own_errors_read_as_unreadable(
    monkeypatch: pytest.MonkeyPatch, error: Exception
):
    remote = Remote("s3://lake/a.bin")
    held = _Held(
        "train", remote.uri, "", {"uri": remote.uri, "size": 1, "object": "etag:a"}, remote
    )
    monkeypatch.setattr(Remote, "objects", lambda remote: _raise(error))
    assert "unreadable" in _seen([held], {})["train"]
    # Anything else is no outage but a fault, and propagates.
    monkeypatch.setattr(Remote, "objects", lambda remote: _raise(KeyError("bug")))
    with pytest.raises(KeyError, match="bug"):
        _seen([held], {})


def test_an_execution_with_a_remote_input_resumes_only_with_its_inputs(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    objects = _bucket(monkeypatch)
    objects["s3://lake/a.bin"] = (1, "etag:a")
    inputs = {"train": Remote("s3://lake/a.bin")}
    trainer = Trainer(ledger)
    with _run(ledger, trainer, inputs) as run:
        trainer.train(run, 1)
    with pytest.raises(ValueError, match=r"verified inputs \(a Path or a Remote\)"):
        ledger.resume(run.execution_id)
    with ledger.resume(run.execution_id, inputs=inputs) as resumed:
        assert resumed.restores_from == run.release_of(1).release_id


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


def test_a_save_and_a_load_work_through_the_ledger_already_open(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    opened, read = [], []
    init, files = Ledger.__init__, Ledger._files

    def counted_init(self: Ledger, *args, **kwargs) -> None:
        opened.append(args)
        init(self, *args, **kwargs)

    def counted_files(self: Ledger, at):
        read.append(at)
        return files(self, at)

    monkeypatch.setattr(Ledger, "__init__", counted_init)
    monkeypatch.setattr(Ledger, "_files", counted_files)
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        source = run.release_of(1).release_id
        reader = Trainer(ledger)
        ledger.load(source, {"model": reader.model, "optim": reader.optimizer, "step": None})

    assert opened == [] and len(read) == 1
    assert _same(reader.weights(), trainer.weights())
    # A writer's subprocess gets no ledger with its config: it opens its own.
    config = run._checkpoints.configure(CheckpointManager.Config(), opened=ledger)
    assert pickle.loads(pickle.dumps(config.storage_config)).opened is None


def _finalize(run: Run, step: int) -> None:
    """Call the finalize callback as the writer does, with a storage *run* configured."""
    config = run._checkpoints.configure(CheckpointManager.Config())
    storage = config.storage_config.create_storage()
    config.finalize_callback(
        SimpleNamespace(path=run._checkpoints.checkpoint_id(step), storage=storage)
    )


def test_a_retried_finalize_returns_the_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        _finalize(run, 1)

        assert len(ledger.releases.releases_of(run.execution_id, "checkpoint")) == 1


def test_an_attempt_writes_only_its_own_checkpoints(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run, ledger.start(trainer.program) as other:
        manager = _manager(run._checkpoints)
        try:
            with pytest.raises(PermissionError):
                manager.save(other._checkpoints.checkpoint_id(1), {"step": 1})
        finally:
            manager.close()
        with pytest.raises(PermissionError):
            run._checkpoints(SimpleNamespace(path=other._checkpoints.checkpoint_id(1)))


@pytest.mark.parametrize("native", [PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
def test_a_checkpoint_path_is_read_by_its_own_parts(native):
    """torch_checkpointing hands the storage a native path or its str (a finalize callback's):
    on Windows they hold backslashes."""
    file = "vestigo-torch/e1/a1/step-3/model/0.pt"
    for path in (native(file), str(Path(file))):
        at = _Location.of(path)
        assert (at.execution_id, at.attempt, at.step, at.file) == ("e1", "a1", 3, "model/0.pt")
    assert LedgerStorage.remap_path(None, native("vestigo-torch/e1/a1/step-3")) == (
        "vestigo-torch/e1/a1/step-3"
    )


def test_the_storage_removes_and_renames_nothing(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
    checkpoint = Path(ledger.checkpoint_id(run.release_of(1).release_id))
    storage = LedgerStorage(ledger.config)

    with pytest.raises(PermissionError, match="not removed by the writer"):
        storage.delete(checkpoint / "model_0.pt")
    with pytest.raises(PermissionError, match="not removed by the writer"):
        storage.rmdir(checkpoint)
    with pytest.raises(OSError, match="write_to_temp_dir=False") as renamed:
        storage.rename(checkpoint, checkpoint.with_name("step_9"), is_directory=True)
    assert renamed.type is OSError
    assert storage.exists(checkpoint / "model_0.pt")
    assert ledger.load(run.release_of(1).release_id, {"step": None}) == {"step": 1}


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

        storage = run._checkpoints.configure(
            CheckpointManager.Config()
        ).storage_config.create_storage()
        assert not storage.exists(Path(run._checkpoints.checkpoint_id(1)))
        with pytest.raises(FileNotFoundError):
            storage.getsize(Path(run._checkpoints.checkpoint_id(1)) / "model_0.pt")


def test_only_rank_zero_accepts_by_the_rank_it_configured_with(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        config = run._checkpoints.configure(CheckpointManager.Config(), 1, 2)
        # An async writer's subprocess has no process group: the storage carries the rank.
        storage = pickle.loads(pickle.dumps(config.storage_config)).create_storage()

        config.finalize_callback(
            SimpleNamespace(path=run._checkpoints.checkpoint_id(1), storage=storage)
        )

        assert (storage.rank, storage.world) == (1, 2)
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


def test_a_branch_starts_only_from_a_stored_checkpoint(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        one = run.release_of(1).release_id
        derived = ledger.derive(one, "eval", {"metrics": b"1"})
        run.finish()
    final = ledger.releases.latest_accepted(run.execution_id, "final")
    ledger.collect(keep=1)

    for release_id in (final.release_id, derived.release.release_id, one):
        with pytest.raises(ValueError, match="not an accepted checkpoint"):
            ledger.branch(trainer.program, release_id)
        assert ledger.find(trainer.program, init=release_id) is None
    # Derived before the collection, found again after it.
    again = ledger.derive(one, "eval", {"metrics": b"1"})
    assert again.release.release_id == derived.release.release_id


def test_a_resume_from_a_collected_checkpoint_fences_nothing(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
        one = run.release_of(1).release_id
    ledger.collect(keep=1)

    with pytest.raises(ValueError, match="was collected"):
        ledger.resume(run.execution_id, from_release=one)
    assert ledger.builds.get_build(run.execution_id).attempt == run.attempt


# --- Lineage ---


def _ids(executions: list[Execution]) -> list[str]:
    return [e.execution_id for e in executions]


def test_an_execution_records_its_computation_attempts_and_checkpoints(ledger: Ledger):
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as first:
        trainer.train(first, 2)
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as resumed:
        trainer.restore(resumed)
        trainer.train(resumed, 3)
        resumed.finish()
    one, two, three = (ledger.release_of(first.execution_id, s).release_id for s in (1, 2, 3))

    execution = ledger.execution(first.execution_id)
    assert execution.checkpoints == ((1, one), (2, two), (3, three))
    assert execution.attempts == (
        Attempt(first.execution_id, first.attempt, None),
        Attempt(first.execution_id, resumed.attempt, two),
    )
    assert ledger.resumed_from(two) == [execution.attempts[1]]
    assert ledger.resumed_from(one) == []
    # The record is the bound computation: it gives the execution's key again.
    assert execution.program == trainer.program.node_id and execution.state == "ready"
    assert (execution.inputs, execution.init) == ({"train": "data@v=1"}, None)
    assert first.computation_key == ledger.computation_key(
        trainer.program, execution.inputs, execution.init
    )


def test_a_step_saved_again_after_a_resume_from_an_earlier_one_is_listed_again(
    ledger: Ledger,
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
    one, two, three = (run.release_of(step).release_id for step in (1, 2, 3))
    trainer = Trainer(ledger)
    with ledger.resume(run.execution_id, from_release=one) as resumed:
        trainer.restore(resumed)
        trainer.train(resumed, 2)

    again = ledger.release_of(run.execution_id, 2).release_id
    assert again != two
    assert ledger.execution(run.execution_id).checkpoints == (
        (1, one),
        (2, two),
        (3, three),
        (2, again),
    )


def test_executions_are_found_by_program_by_model_and_by_input(ledger: Ledger, tmp_path: Path):
    trainer = Trainer(ledger)
    model = ledger.define_model(trainer.model, version="tiny-v1")
    wide = ledger.define_program(
        model, loss=nn.MSELoss(), optimizer=trainer.optimizer, config={"batch": 32}
    )
    other = _program(ledger, _model(16), trainer.optimizer)
    data = _file(tmp_path / "here" / "rows.bin")
    with ledger.start(trainer.program, {"train": data}) as one:
        trainer.train(one, 1)
        derived = ledger.derive(one.release_of(1).release_id, "eval", {"metrics": b"1"})
    with ledger.start(wide, {"train": "data@v=1", "eval": data}) as two:
        pass
    with ledger.start(trainer.program, {"train": "data@v=2"}) as three:
        pass
    with ledger.start(other, {"train": "data@v=2"}) as four:
        pass
    one, two, three, four = (r.execution_id for r in (one, two, three, four))

    assert _ids(ledger.executions()) == [one, two, three, four]
    assert _ids(ledger.executions(model)) == [one, two, three]
    assert _ids(ledger.executions(trainer.program)) == [one, three]
    assert _ids(ledger.executions(wide)) == [two]
    # An input is its bytes: a copy elsewhere finds the executions that read them.
    copy = _file(tmp_path / "there" / "copy.bin")
    assert _ids(ledger.executions(reading=copy)) == [one, two]
    assert _ids(ledger.executions(trainer.program, reading=copy)) == [one]
    assert _ids(ledger.executions(reading="data@v=2")) == [three, four]
    # Each record is its bound computation: it hashes to the execution's key.
    for execution in ledger.executions():
        program = ledger._definition(execution.program)
        spec = Ledger._spec(program, execution.inputs, execution.init)
        build = ledger.builds.get_build(execution.execution_id)
        assert compute_provenance_hash([], spec) == build.computation_key
    for missing in ("0" * 32, derived.release.build_id):
        with pytest.raises(ValueError, match="does not exist"):
            ledger.execution(missing)


def test_descendants_are_the_branches_of_a_run_at_any_depth(ledger: Ledger):
    def branch(release: Release | None) -> str:
        assert release is not None
        trainer = Trainer(ledger)
        with ledger.branch(trainer.program, release.release_id) as run:
            trainer.restore(run)
            trainer.train(run, trainer.step + 1)
            run.finish()
        return run.execution_id

    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as source:
        trainer.train(source, 3)
        source.finish()
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"train": "data@v=2"}) as unrelated:
        trainer.train(unrelated, 1)
        unrelated.finish()
    source, unrelated = source.execution_id, unrelated.execution_id
    a = branch(ledger.release_of(source, 1))
    b = branch(ledger.release_of(a, 2))
    c = branch(ledger.release_of(source, 3))
    d = branch(ledger.release_of(unrelated, 1))

    assert _ids(ledger.descendants(source)) == [a, b, c]
    assert _ids(ledger.descendants(a)) == [b]
    assert _ids(ledger.descendants(c)) == []
    assert _ids(ledger.descendants(unrelated)) == [d]
    # A branch's first attempt resumed from nothing: it started from its init.
    record = ledger.execution(b)
    assert record.init == ledger.release_of(a, 2).release_id
    assert [attempt.resumed_from for attempt in record.attempts] == [None]

    # The checkpoint a branch started from may be collected; the branch still descends.
    ledger.collect(keep=1)
    assert [step for step, _ in ledger.execution(source).checkpoints] == [3]
    assert _ids(ledger.descendants(source)) == [a, b, c]
    with pytest.raises(ValueError, match="does not exist"):
        ledger.descendants("0" * 32)


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


# --- Hosted runs ---


class _Host:
    """A host as a Strata cell's server is one: it holds an attempt of an execution and
    publishes each folder the run writes as one checkpoint release of it, in the layout
    of ``vestigo.checkpoint_files``; it restores a release by writing its files out."""

    def __init__(self, ledger: Ledger, held: Run, root: Path, restore: str | None = None):
        self.ledger, self.held, self.root, self.restore = ledger, held, root, restore
        self.released: list[str] = []
        self.folders: list[set[str]] = []

    def checkpoint_files(self, step: int, write) -> int:
        folder = self.root / "checkpoints" / str(len(self.released) + 1)
        folder.mkdir(parents=True)
        write(folder)
        files = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*")}
        releases, store = self.ledger.releases, self.ledger.store
        release = releases.prepare(self.held.execution_id, self.held.attempt, "checkpoint")
        staged = []

        def put(artifact_id: str, params: dict, data: bytes) -> None:
            version = store.create_artifact(
                artifact_id,
                uuid.uuid4().hex,
                TransformSpec("host", params, []),
                release_id=release.release_id,
            )
            store.write_blob(artifact_id, version, data)
            digest = hashlib.sha256(data).hexdigest()
            staged.append(StagedVersion(artifact_id, version, "", None, len(data), digest))

        for path, data in files.items():
            put(f"host_{len(staged)}", checkpoint_files.file_params(path, step), data)
        listed = {
            path: (len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()
        }
        put(
            "host_manifest",
            checkpoint_files.manifest_params(step),
            checkpoint_files.manifest(step, listed),
        )
        releases.accept(release.release_id, staged)
        self.released.append(release.release_id)
        self.folders.append(set(files))
        return len(self.released)

    def restore_files(self) -> Path | None:
        if self.restore is None:
            return None
        folder = self.root / "restore"
        bundle = self.ledger.releases.bundle(self.restore)
        for path, version in checkpoint_files.files_of(bundle.outputs).items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_bytes(self.ledger.store.read_blob(version.id, version.version))
        return folder


def _hosted_restore(trainer: Trainer, run: HostedRun) -> None:
    state = run.restore(
        {"model": trainer.model, "optim": trainer.optimizer, "shuffle": None, "step": None}
    )
    trainer.shuffle.set_state(state["shuffle"])
    trainer.step = state["step"]


def test_a_hosted_checkpoint_is_one_release_the_ledger_loads(ledger: Ledger, tmp_path: Path):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as held:
        host = _Host(ledger, held, tmp_path / "run")
        with HostedRun(host) as run:
            assert run.restore({"step": 0}) == {"step": 0}
            trainer.train(run, 2)
    assert host.folders[1] == {"model_0.pt", "optim_0.pt", "shuffle_0.pt", "step_0.pt"}
    # The ledger finds it by step, as it finds its own.
    assert ledger.release_of(held.execution_id, 2).release_id == host.released[1]

    reader = Trainer(ledger)
    loaded = ledger.load(
        host.released[1], {"model": reader.model, "optim": reader.optimizer, "step": None}
    )
    assert loaded["step"] == 2
    assert _same(reader.weights(), trainer.weights())
    assert reader.optimizer.state_dict()["state"][0]["step"] == 2 * 4


def test_a_step_is_released_once_per_attempt_through_a_host(ledger: Ledger, tmp_path: Path):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as held:
        host = _Host(ledger, held, tmp_path / "run")
        with HostedRun(host) as run:
            run.save(1, {"step": 1})
            with pytest.raises(ValueError, match="step 1 is not after step 1"):
                run.save(1, {"step": 2})
        assert len(host.released) == 1
        # A host that publishes the step again anyway shadows the first: its id is refused.
        with HostedRun(host) as again:
            again.save(1, {"step": 3})
    first, second = host.released

    with pytest.raises(ValueError, match="accepted that step again"):
        ledger.checkpoint_id(first)
    assert ledger.load(second, {"step": None}) == {"step": 3}


def test_a_hosted_run_resumes_and_branches_from_either_writer_s_checkpoints(
    ledger: Ledger, tmp_path: Path
):
    expected = _uninterrupted(ledger)

    # Interrupted after step 3 of a hosted run, resumed from its checkpoint.
    first = Trainer(ledger)
    with ledger.start(first.program, {"train": "cell"}) as held:
        host = _Host(ledger, held, tmp_path / "first")
        with HostedRun(host) as run:
            first.train(run, 3)
        resumed = Trainer(ledger)
        with HostedRun(_Host(ledger, held, tmp_path / "again", host.released[2])) as run:
            _hosted_restore(resumed, run)
            assert resumed.step == 3
            resumed.train(run, STEPS)
    assert _same(resumed.weights(), expected)

    # Branched from step 2 of a run the ledger's own storage wrote.
    source = Trainer(ledger)
    with ledger.start(source.program, {"train": "job"}) as job:
        source.train(job, 2)
        branch = Trainer(ledger)
        host = _Host(ledger, job, tmp_path / "branch", job.release_of(2).release_id)
        with HostedRun(host) as run:
            _hosted_restore(branch, run)
            branch.train(run, STEPS)
    assert _same(branch.weights(), expected)


def test_a_verifying_ledger_refuses_a_hosted_checkpoint(tmp_path: Path):
    with Ledger(tmp_path / "ledger") as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as held:
            host = _Host(ledger, held, tmp_path / "run")
            with HostedRun(host) as run:
                trainer.train(run, 1)
    with Ledger(tmp_path / "ledger", verify_key=_keypair()[1]) as reader:
        with pytest.raises(OSError, match="is not signed"):
            reader.load(host.released[0], {"model": _model()})


def test_a_hosted_run_writes_only_in_the_folder_it_is_given(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    written: list[Path] = []
    stream_write = LocalFileSystemStorage.stream_write

    def recorded(self: LocalFileSystemStorage, path: Path) -> io.IOBase:
        written.append(Path(path))
        return stream_write(self, path)

    monkeypatch.setattr(LocalFileSystemStorage, "stream_write", recorded)
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as held:
        with HostedRun(_Host(ledger, held, tmp_path / "run")) as run:
            trainer.train(run, 1)
    # Not in a sibling renamed into place, which a crash would leave beside it.
    folder = tmp_path / "run" / "checkpoints" / "1"
    assert written and all(folder in path.parents for path in written)


def test_a_hosted_run_is_one_process(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vestigo.torch, "_group", lambda: (0, 2))
    with pytest.raises(RuntimeError, match="one process"):
        HostedRun(SimpleNamespace())


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


def test_without_cryptography_a_key_names_the_sign_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    signing, verify = _keypair()
    monkeypatch.setitem(sys.modules, "cryptography.hazmat.primitives.asymmetric", None)
    for keys in ({"signing_key": signing}, {"verify_key": verify}):
        with pytest.raises(
            ImportError, match=r"signed checkpoints need the sign extra: install vestigo\[sign\]"
        ):
            Ledger(tmp_path / "ledger", **keys)


def _load_only(ledger: Ledger, release_id: str) -> list[torch.Tensor]:
    model = _model()
    manager = ledger.configure(CheckpointManager.Config.with_sync_save()).build()
    try:
        state = manager.load(ledger.checkpoint_id(release_id), into={"model": model.state_dict()})
    finally:
        manager.close()
    model.load_state_dict(state["model"])
    return [p.detach().clone() for p in model.parameters()]


def test_a_load_only_ledger_reads_a_checkpoint_and_writes_nothing(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 2)
            run.finish()
        source = run.release_of(2).release_id

    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        before = reader.store.list_versions_by_id_prefix("torch_")
        assert _same(_load_only(reader, source), trainer.weights())
        assert reader.store.list_versions_by_id_prefix("torch_") == before
        manager = reader.configure(CheckpointManager.Config.with_sync_save()).build()
        try:
            with pytest.raises(PermissionError, match="only loads"):
                manager.save(
                    run._checkpoints.checkpoint_id(3), {"model": trainer.model.state_dict()}
                )
        finally:
            manager.close()

        _tamper(reader, run, 2)
        with pytest.raises(OSError, match="differs from the checkpoint"):
            _load_only(reader, source)
    with Ledger(tmp_path / "ledger", verify_key=_keypair()[1]) as rotated:
        with pytest.raises(OSError, match="does not verify"):
            _load_only(rotated, source)


def _execute(ledger: Ledger, sql: str, args: tuple) -> None:
    conn = ledger.store._get_connection()
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _report_of(ledger: Ledger, run: Run, step: int) -> tuple[str, int]:
    report_id = f"torch_{run.execution_id}@attempt={run.attempt}@step={step}@rank=0"
    [version] = [v for v in ledger.store.list_versions_by_id_prefix(report_id) if v.id == report_id]
    return report_id, version.version


def _save_meddled(ledger: Ledger, run: Run, step: int, meddle) -> None:
    """Save *step*, letting *meddle* at the store between the rank's report and rank 0."""

    def reported(args: SimpleNamespace) -> None:
        _report_written(args)
        meddle()

    config = run._checkpoints.configure(CheckpointManager.Config.with_sync_save())
    config.pre_finalize_callback = reported
    manager = config.build()
    try:
        manager.save(run._checkpoints.checkpoint_id(step), {"model": _model().state_dict()})
    finally:
        manager.close()


def _forge_report(ledger: Ledger, run: Run, step: int, rewrite_report: bool) -> None:
    """A bucket writer's forgery: the model file's bytes and, if asked, the report's digest.

    The forged file keeps its size, which acceptance checks.
    """
    report_id, version = _report_of(ledger, run, step)
    content = json.loads(ledger.store.read_blob(report_id, version))
    [entry] = [e for e in content["files"] if e["file"] == "model_0.pt"]
    data = bytearray(ledger.store.read_blob(entry["artifact_id"], entry["version"]))
    data[len(data) // 2] ^= 0xFF
    data = bytes(data)
    ledger.store.write_blob(entry["artifact_id"], entry["version"], data)
    entry["sha256"], entry["byte_size"] = hashlib.sha256(data).hexdigest(), len(data)
    if rewrite_report:
        ledger.store.write_blob(report_id, version, _canonical(content))


def test_a_bucket_writer_during_a_save_cannot_have_a_forgery_signed(tmp_path: Path):
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            with pytest.raises(OSError, match="rank 0's report of step 1 .* does not verify"):
                _save_meddled(ledger, run, 1, lambda: _forge_report(ledger, run, 1, True))
            assert run.release_of(1) is None

            # The report left alone, rank 0 signs what the rank hashed, not the bytes now there.
            _save_meddled(ledger, run, 2, lambda: _forge_report(ledger, run, 2, False))
            source = run.release_of(2).release_id

    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        with pytest.raises(OSError, match="differs from the checkpoint"):
            _load_only(reader, source)


def test_a_report_moved_from_another_step_is_refused(tmp_path: Path):
    with Ledger(tmp_path / "ledger", signing_key=_keypair()[0]) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            _save_meddled(ledger, run, 1, lambda: None)

            def replay() -> None:
                one, two = _report_of(ledger, run, 1), _report_of(ledger, run, 2)
                ledger.store.write_blob(*two, ledger.store.read_blob(*one))

            with pytest.raises(OSError, match="signed for another checkpoint"):
                _save_meddled(ledger, run, 2, replay)
            assert run.release_of(2) is None


def _signed_run(tmp_path: Path) -> tuple[Ledger, Run, str, bytes]:
    signing, verify = _keypair()
    with Ledger(tmp_path / "ledger", signing_key=signing) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
            run.finish()
    return ledger, run, run.release_of(1).release_id, verify


def test_a_database_writer_cannot_shorten_or_hide_a_signed_file(tmp_path: Path):
    ledger, run, source, verify = _signed_run(tmp_path)
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        expected = _load_only(reader, source)
        [model] = [v for v in _outputs(reader, run, 1) if v.params.get("file") == "model_0.pt"]
        # A shorter size would have the loader read a prefix of the signed bytes.
        _execute(
            reader,
            "UPDATE artifact_versions SET byte_size = ? WHERE id = ? AND version = ?",
            (model.byte_size - 100, model.id, model.version),
        )
        assert _same(_load_only(reader, source), expected)
        # Renamed in the database, the file is still where the signed manifest says.
        _execute(
            reader,
            "UPDATE artifact_versions SET transform_spec = REPLACE(transform_spec, "
            "'model_0.pt', 'hidden.pt') WHERE id = ? AND version = ?",
            (model.id, model.version),
        )
        assert _same(_load_only(reader, source), expected)


@pytest.mark.parametrize(
    ("content", "refusal"),
    [
        (b"\xff not json", "is malformed"),
        (b"[1, 2]", "is malformed"),
        (b'{"signature": 7}', "does not verify"),
        (b'{"signature": "00", "x": NaN}', "does not verify"),
    ],
)
def test_a_garbled_manifest_is_a_refusal(tmp_path: Path, content: bytes, refusal: str):
    ledger, run, source, verify = _signed_run(tmp_path)
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        [manifest] = [v for v in _outputs(reader, run, 1) if v.params.get("manifest")]
        reader.store.write_blob(manifest.id, manifest.version, content)
        with pytest.raises(OSError, match=refusal):
            _load_only(reader, source)


def test_a_signed_report_in_the_manifest_s_place_is_a_refusal(tmp_path: Path):
    ledger, run, source, verify = _signed_run(tmp_path)
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        [manifest] = [v for v in _outputs(reader, run, 1) if v.params.get("manifest")]
        report = reader.store.read_blob(*_report_of(reader, run, 1))
        # Genuinely signed, by the same key, but a rank's report and not a manifest.
        reader.store.write_blob(manifest.id, manifest.version, report)
        with pytest.raises(OSError, match="is malformed"):
            _load_only(reader, source)


def test_a_signed_manifest_in_a_report_s_place_is_a_refusal(tmp_path: Path):
    with Ledger(tmp_path / "ledger", signing_key=_keypair()[0]) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            _save_meddled(ledger, run, 1, lambda: None)

            def swap() -> None:
                [manifest] = [v for v in _outputs(ledger, run, 1) if v.params.get("manifest")]
                data = ledger.store.read_blob(manifest.id, manifest.version)
                ledger.store.write_blob(*_report_of(ledger, run, 2), data)

            with pytest.raises(OSError, match="is malformed"):
                _save_meddled(ledger, run, 2, swap)
            assert run.release_of(2) is None


def test_a_manifest_whose_bytes_are_gone_is_a_refusal(tmp_path: Path):
    ledger, run, source, verify = _signed_run(tmp_path)
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        [manifest] = [v for v in _outputs(reader, run, 1) if v.params.get("manifest")]
        assert reader.store.blob_store.delete_blob(manifest.id, manifest.version)
        with pytest.raises(OSError, match="the manifest of step 1 of execution .* is missing"):
            _load_only(reader, source)


def test_a_second_manifest_row_is_a_refusal(tmp_path: Path):
    ledger, run, source, verify = _signed_run(tmp_path)
    with Ledger(tmp_path / "ledger", verify_key=verify) as reader:
        [model] = [v for v in _outputs(reader, run, 1) if v.params.get("file") == "model_0.pt"]
        _execute(
            reader,
            "UPDATE artifact_versions SET transform_spec = REPLACE(transform_spec, "
            '\'"file":\', \'"manifest":true,"file":\') WHERE id = ? AND version = ?',
            (model.id, model.version),
        )
        with pytest.raises(OSError, match="has 2 manifests"):
            _load_only(reader, source)


# --- Authority ---


def test_a_stale_attempt_cannot_publish_after_a_resume(ledger: Ledger):
    stale = Trainer(ledger)
    first = ledger.start(stale.program)
    stale.train(first, 1)
    # Its lease lapses (a pause, a partition) while the process goes on saving.
    ledger.builds.renew_lease(first.execution_id, first.attempt, lease_duration_seconds=0.0)
    with ledger.resume(first.execution_id) as resumed:
        with pytest.raises(ReleaseRejected):
            stale.train(first, 2)
        assert ledger.release_of(first.execution_id, 2) is None

        fresh = Trainer(ledger)
        fresh.restore(resumed)
        fresh.train(resumed, 2)
        assert resumed.release_of(2).attempt == resumed.attempt
    first.close()


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
        manager = _manager(run._checkpoints)
        try:
            # Python 3.14 made a local object's refusal a PicklingError.
            refused = pickle.PicklingError if sys.version_info >= (3, 14) else AttributeError
            with pytest.raises(refused, match="local object") as raised:
                # torch.save writes its archive header before the lambda refuses to pickle.
                manager.save(run._checkpoints.checkpoint_id(1), {"bad": {"f": lambda: 0}})
            assert raised.type is refused
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


def test_a_release_the_database_left_unsettled_does_not_hold_up_the_next(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            # The database fails under the release and under its abandonment.
            patch.setattr(ReleaseStore, "attach", lambda *a, **k: _raise(OSError))
            patch.setattr(ReleaseStore, "abandon", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                trainer.train(run, 1)
        trainer.train(run, 2)

        [stuck, accepted] = ledger.releases.releases_of(run.execution_id, "checkpoint")
        assert (stuck.state, accepted.state) == ("abandoned", "accepted")
        assert run.release_of(2) == accepted


def test_a_step_saved_again_after_a_failed_write_is_released(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    written = LedgerStorage.stream_write
    calls: list[Path] = []

    def second_fails(self: LedgerStorage, path: Path) -> io.IOBase:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("upload failed")
        return written(self, path)

    state = {"model": trainer.model.state_dict(), "step": 1}
    with ledger.start(trainer.program) as run:
        manager = _manager(run._checkpoints)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(LedgerStorage, "stream_write", second_fails)
                with pytest.raises(OSError, match="upload failed"):
                    manager.save(run._checkpoints.checkpoint_id(1), state)
            manager.save(run._checkpoints.checkpoint_id(1), state)
        finally:
            manager.close()

        files = [v.params["file"] for v in _outputs(ledger, run, 1) if "file" in v.params]
        assert sorted(files) == ["model_0.pt", "step_0.pt"]


def test_a_step_at_or_before_one_the_attempt_accepted_is_refused_before_it_is_staged(
    ledger: Ledger,
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
        first = run.release_of(1)
        staged = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@")
        for step in (2, 1):
            with pytest.raises(ValueError, match=f"step {step} is not after step 2"):
                run.save(step, {"model": trainer.model, "step": step})
        assert ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@") == staged

        # Saved through the manager itself, such a step is refused at acceptance.
        manager = _manager(run._checkpoints)
        try:
            for step in (2, 1):
                with pytest.raises(ValueError, match=f"step {step} is not after step 2"):
                    manager.save(run._checkpoints.checkpoint_id(step), {"step": step})
        finally:
            manager.close()
        assert run.release_of(1) == first
        assert run.save(3, {"step": 3}) == run.release_of(3)


def _raise(error: type[Exception]) -> None:
    raise error


def test_a_head_points_at_an_accepted_release(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
        run.set_head("tiny/best", run.release_of(1).release_id)

    assert ledger.releases.reference("tiny/best") == (run.release_of(1).release_id, 1)


def test_a_head_is_not_set_under_the_ledger_s_own_names(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program, {"data": "a"}) as ended:
        trainer.train(ended, 1)
        ended.finish()
    final = ledger.final_checkpoint(ended.execution_id)
    with ledger.start(trainer.program, {"data": "b"}) as run:
        Trainer(ledger).train(run, 1)
        for name in (f"vestigo-torch/{ended.execution_id}", "vestigo-torch"):
            with pytest.raises(ValueError, match="reserves"):
                run.set_head(name, run.release_of(1).release_id)

    assert ledger.final_checkpoint(ended.execution_id) == final


def test_a_renewal_that_fails_once_keeps_the_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    renew = BuildStore.renew_lease
    calls: list[float] = []
    renewed = threading.Event()

    def locked_once(self: BuildStore, *args: object, **kwargs: object) -> bool:
        calls.append(time.time())
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        if len(calls) == 4:
            renewed.set()
        return renew(self, *args, **kwargs)

    monkeypatch.setattr(BuildStore, "renew_lease", locked_once)
    with Ledger(tmp_path / "ledger", lease_seconds=0.3) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            assert renewed.wait(10), "renewal stopped after one error"
            trainer.train(run, 1)
            assert run.release_of(1) is not None


def test_a_closed_write_window_lapses_the_lease_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    with Ledger(tmp_path / "ledger", lease_seconds=0.6, writable_seconds=2.0) as ledger:
        trainer = Trainer(ledger)
        with caplog.at_level(logging.WARNING, logger="vestigo.torch"):
            with _run(ledger, trainer) as run:
                trainer.train(run, 1)
                deadline = time.time() + 10
                while ledger.builds.get_build(run.execution_id).lease_owner is not None:
                    assert time.time() < deadline, "the lease outlived the write window"
                    time.sleep(0.05)
                with pytest.raises(ReleaseRejected, match="write window .* closed"):
                    trainer.train(run, 2)
                with pytest.raises(BuildLeaseLost, match="write window .* closed"):
                    run.observe(_exported(trainer.model), scope="model_forward", coverage="all")
                # The process lives on, and another resumes the execution meanwhile.
                with _run(ledger, Trainer(ledger)) as resumed:
                    assert resumed.restores_from == run.release_of(1).release_id
    assert "write window" in caplog.text


def test_finishing_is_not_reported_as_a_lost_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    accept = ReleaseStore.accept

    def then_linger(self: ReleaseStore, release_id: str, *args: object, **kwargs: object):
        bundle = accept(self, release_id, *args, **kwargs)
        if bundle.release.kind == "final":
            time.sleep(0.3)  # renewals fall due once the final release has landed
        return bundle

    monkeypatch.setattr(ReleaseStore, "accept", then_linger)
    with Ledger(tmp_path / "ledger", lease_seconds=0.3) as ledger:
        trainer = Trainer(ledger)
        with caplog.at_level(logging.WARNING, logger="vestigo.torch"):
            with ledger.start(trainer.program) as run:
                trainer.train(run, 1)
                run.finish()
    assert "lost execution" not in caplog.text


def test_a_renewal_the_ledger_refuses_is_reported_as_a_lost_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    with Ledger(tmp_path / "ledger", lease_seconds=0.3) as ledger:
        trainer = Trainer(ledger)
        with caplog.at_level(logging.WARNING, logger="vestigo.torch"):
            with ledger.start(trainer.program) as run, monkeypatch.context() as patch:
                # Another attempt took the execution: the ledger refuses this one's renewal.
                patch.setattr(BuildStore, "renew_lease", lambda *a, **k: False)
                deadline = time.monotonic() + 10
                while "lost execution" not in caplog.text and time.monotonic() < deadline:
                    time.sleep(0.05)
    assert f"Attempt {run.attempt} lost execution {run.execution_id}" in caplog.text


def test_a_finished_execution_is_not_resumed(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        run.finish()

    assert ledger.find(trainer.program) is None
    with pytest.raises(ValueError, match="not active"):
        ledger.resume(run.execution_id)


def test_a_final_release_that_fails_to_land_does_not_hold_up_the_retry(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    accept = ReleaseStore.accept
    failed: list[str] = []

    def locked_once(self: ReleaseStore, release_id: str, *args: object, **kwargs: object):
        if not failed and self.get(release_id).kind == "final":
            failed.append(release_id)
            raise sqlite3.OperationalError("database is locked")
        return accept(self, release_id, *args, **kwargs)

    monkeypatch.setattr(ReleaseStore, "accept", locked_once)
    trainer = Trainer(ledger)
    with _run(ledger, trainer) as run:
        trainer.train(run, 1)
        with pytest.raises(sqlite3.OperationalError):
            run.finish()
        run.finish()
    assert ledger.builds.get_build(run.execution_id).state == "ready"
    finals = ledger.releases.releases_of(run.execution_id, "final")
    assert [r.state for r in finals] == ["abandoned", "accepted"]


def test_a_build_of_another_executor_is_not_resumed(ledger: Ledger):
    ledger.store.create_artifact("foreign", "f" * 64, TransformSpec("other@v1", {}, []))
    build_id = uuid.uuid4().hex
    ledger.builds.create_build(
        build_id,
        "foreign",
        1,
        executor_ref="other@v1",
        computation_key="c" * 64,
        params={"x": 1},
        dispatch="external",
    )
    with pytest.raises(ValueError, match="does not exist"):
        ledger.resume(build_id)
    # Refused before its lease was taken.
    build = ledger.builds.get_build(build_id)
    assert build.state == "pending" and build.attempt is None and build.lease_owner is None


def test_a_run_that_ends_before_a_checkpoint_fails(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        pass

    build = ledger.builds.get_build(run.execution_id)
    assert build.state == "failed"
    assert ledger.store.get_artifact(build.artifact_id, build.version).state == "failed"
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


def test_a_name_derived_from_a_later_checkpoint_is_derived_from_no_earlier_one(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
    one, two, three = (run.release_of(step).release_id for step in (1, 2, 3))
    with monkeypatch.context() as patch:
        patch.setattr(ArtifactStore, "open_blob_writer", lambda *a, **k: _raise(OSError))
        with pytest.raises(OSError):
            ledger.derive(three, "eval-2", {"weights": _export(trainer, tmp_path)})
    [output] = ledger.derive(one, "eval", {"metrics": b"1"}).outputs
    ledger.derive(three, "eval", {"metrics": b"3"})
    ledger.store.garbage_collect(keep_superseded=0)
    assert ledger.store.get_artifact(output.id, output.version) is None

    for earlier in (one, two):
        with pytest.raises(ValueError, match="later checkpoint"):
            ledger.derive(earlier, "eval", {"metrics": b"0"})

    assert ledger.store.get_latest_version(output.id).params["from"] == three
    assert ledger.derive(two, "eval-2", {"metrics": b"2"}).release.state == "accepted"


@pytest.mark.parametrize("when", ["in flight", "accepted meanwhile"])
def test_an_earlier_checkpoint_racing_a_later_one_s_derivation_is_refused(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch, when: str
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 3)
    one, three = run.release_of(1).release_id, run.release_of(3).release_id
    refused = []

    def derive_one() -> None:
        with pytest.raises(ValueError, match="later checkpoint"):
            ledger.derive(one, "eval", {"metrics": b"1"})
        refused.append(one)

    if when == "in flight":
        # Three's outputs exist, building, while one's derivation looks.
        write = Ledger._write_small

        def then_one(self: Ledger, *args):
            if not refused:
                derive_one()
            return write(self, *args)

        monkeypatch.setattr(Ledger, "_write_small", then_one)
        ledger.derive(three, "eval", {"metrics": b"3"})
    else:
        # Three's derivation lands between one's first look and its outputs' rows.
        look = Ledger._refuse_later

        def three_first(self: Ledger, prefix, source, name, mine=()):
            look(self, prefix, source, name, mine)
            if source.release_id == one and not mine:
                monkeypatch.setattr(Ledger, "_refuse_later", look)
                ledger.derive(three, "eval", {"metrics": b"3"})

        monkeypatch.setattr(Ledger, "_refuse_later", three_first)
        derive_one()

    assert refused == [one]
    [current] = ledger.derived(three, "eval").outputs
    assert ledger.store.get_latest_version(current.id) == current
    assert ledger.store.read_blob(current.id, current.version) == b"3"


def test_a_file_rewritten_as_it_is_derived_is_recorded_by_the_bytes_stored(
    ledger: Ledger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
    export = tmp_path / "export.bin"
    export.write_bytes(b"A" * 64)

    def rewritten(real):
        def store(self, *args, **kwargs):
            export.write_bytes(b"B" * 64)  # an exporter still writing, the same size
            return real(self, *args, **kwargs)

        return store

    for name in ("open_blob_writer", "publish_blob_from_path"):
        monkeypatch.setattr(ArtifactStore, name, rewritten(getattr(ArtifactStore, name)))
    [output] = ledger.derive(run.release_of(1).release_id, "export", {"w": export}).outputs

    stored = ledger.store.read_blob(output.id, output.version)
    assert output.content_sha256 == hashlib.sha256(stored).hexdigest()


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

    monkeypatch.setattr(ArtifactStore, "open_blob_writer", full)
    with pytest.raises(OSError):
        ledger.derive(source, "eval", {"metrics": b"1", "weights": _export(trainer, tmp_path)})

    assert ledger.derived(source, "eval") is None
    staged = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@derived=")
    assert staged and {v.state for v in staged} == {"failed"}
    # Its build and the placeholder it targets: none is left building.
    [build] = ledger.builds.builds_of(vestigo.torch.DERIVE_EXECUTOR)
    assert build.state == "failed"
    assert ledger.store.get_artifact(build.artifact_id, build.version).state == "failed"


def test_a_derivation_slower_than_its_lease_is_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with Ledger(tmp_path / "ledger", lease_seconds=0.2) as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 1)
        real = ArtifactStore.open_blob_writer

        def slow(self, *args, **kwargs):
            time.sleep(0.5)  # an upload outlasting the lease
            return real(self, *args, **kwargs)

        monkeypatch.setattr(ArtifactStore, "open_blob_writer", slow)
        bundle = ledger.derive(
            run.release_of(1).release_id, "export", {"weights": _export(trainer, tmp_path)}
        )

        assert bundle.release.state == "accepted"


def test_collect_reaps_a_derivation_whose_process_died_once_its_lease_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with Ledger(tmp_path / "ledger") as ledger:
        trainer = Trainer(ledger)
        with ledger.start(trainer.program) as run:
            trainer.train(run, 3)
        one, three = run.release_of(1).release_id, run.release_of(3).release_id
        # A deriving job whose lease, its write window, is short.
        with Ledger(tmp_path / "ledger", writable_seconds=1.0) as job:
            with monkeypatch.context() as patch:
                # Killed mid-upload: nothing it would clean up on an error runs.
                patch.setattr(Ledger, "_write_small", lambda *a: _raise(_Crash))
                patch.setattr(ReleaseStore, "abandon", lambda *a: _raise(_Crash))
                with pytest.raises(_Crash):
                    job.derive(three, "eval", {"metrics": b"3"})
        [build] = ledger.builds.builds_of(vestigo.torch.DERIVE_EXECUTOR)
        outputs = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@derived=")
        assert [v.state for v in outputs] == ["building"]

        # While its lease holds it may still be uploading: nothing is touched.
        assert ledger.collect(keep=3)["orphaned"] == 0
        assert ledger.builds.get_build(build.build_id).state == "building"
        with pytest.raises(ValueError, match="later checkpoint"):
            ledger.derive(one, "eval", {"metrics": b"1"})

        time.sleep(1.1)
        dry = ledger.collect(keep=3, dry_run=True)
        assert {(c["artifact_id"], c["version"]) for c in dry["collected"]} >= {
            (v.id, v.version) for v in outputs
        }
        assert ledger.builds.get_build(build.build_id).state == "building"

        result = ledger.collect(keep=3)

        assert result["orphaned"] == len(outputs)
        assert all(ledger.store.get_artifact(v.id, v.version) is None for v in outputs)
        assert ledger.builds.get_build(build.build_id).state == "failed"
        assert ledger.store.get_artifact(build.artifact_id, build.version).state == "failed"
        [release] = ledger.releases.releases_of(build.build_id, "final")
        assert release.state == "abandoned"
        # The name is free again, from an earlier checkpoint and then a later one.
        assert ledger.derive(one, "eval", {"metrics": b"1"}).release.state == "accepted"
        [current] = ledger.derive(three, "eval", {"metrics": b"3"}).outputs
        assert ledger.store.get_latest_version(current.id) == current


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


def test_collect_counts_the_staged_files_it_takes_and_a_dry_run_lists_them(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        with monkeypatch.context() as patch:
            patch.setattr(ReleaseStore, "accept", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                trainer.train(run, 1)
        trainer.step = 0
        trainer.train(run, 1)
        prefix = f"torch_{run.execution_id}@attempt={run.attempt}@step=1"
        staged = {
            (v.id, v.version): ledger.store.blob_size(v.id, v.version)
            for v in ledger.store.list_versions_by_id_prefix(prefix)
            if v.state == "building"
        }
        assert staged and all(staged.values())

        dry = ledger.collect(keep=1, dry_run=True)

        listed = {(c["artifact_id"], c["version"]): c["byte_size"] for c in dry["collected"]}
        assert listed == staged
        assert (dry["deleted_count"], dry["deleted_bytes"]) == (len(staged), sum(staged.values()))
        # An idle floor they are within spares them, in a dry run as in a real one.
        assert ledger.collect(keep=1, min_idle_seconds=3600, dry_run=True)["collected"] == []

        result = ledger.collect(keep=1)

        assert (result["deleted_count"], result["deleted_bytes"]) == (
            len(staged),
            sum(staged.values()),
        )
        assert all(ledger.store.get_artifact(*v) is None for v in staged)


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


def test_a_resume_marks_the_checkpoint_it_restores_used(ledger: Ledger):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
    older = run.release_of(1).release_id
    conn = ledger.store._get_connection()
    try:
        conn.execute("UPDATE artifact_versions SET last_used_at = 0 WHERE release_id = ?", (older,))
        conn.commit()
    finally:
        conn.close()

    # Before the attempt opens: a collection reading the executions first would not keep it.
    with ledger.resume(run.execution_id, from_release=older):
        conn = ledger.store._get_connection()
        try:
            rows = conn.execute(
                "SELECT last_used_at FROM artifact_versions WHERE release_id = ?", (older,)
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


def test_collect_takes_a_live_attempt_s_saves_that_can_no_longer_be_released(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    written = LedgerStorage.stream_write
    calls: list[Path] = []

    def second_fails(self: LedgerStorage, path: Path) -> io.IOBase:
        calls.append(path)
        if len(calls) % 2 == 0:
            raise OSError("upload failed")
        return written(self, path)

    def rejected(self: ReleaseStore, release_id: str, *args: object) -> None:
        # The lease lapsed at acceptance; the renewal took it back after.
        self._settle(release_id, "rejected")
        raise ReleaseRejected("lease lapsed")

    def staged(run: Run, step: int) -> set[tuple[str, int]]:
        prefix = f"torch_{run.execution_id}@attempt={run.attempt}@step={step}"
        return {
            (v.id, v.version)
            for v in ledger.store.list_versions_by_id_prefix(prefix)
            if v.state == "building"
        }

    with ledger.start(trainer.program) as run:
        manager = _manager(run._checkpoints)

        def save(
            step: int,
            patch: tuple[type, str, object] | None = None,
            error: type[Exception] = OSError,
        ) -> None:
            state = {"model": trainer.model.state_dict(), "step": step}
            if patch is None:
                manager.save(run._checkpoints.checkpoint_id(step), state)
                return
            with monkeypatch.context() as patched:
                patched.setattr(*patch)
                with pytest.raises(error) as raised:
                    manager.save(run._checkpoints.checkpoint_id(step), state)
                assert raised.type is error

        upload_fails = (LedgerStorage, "stream_write", second_fails)
        try:
            save(0, upload_fails)
            # Nothing accepted yet: step 0 may still be saved again.
            assert ledger.collect(keep=1)["orphaned"] == 0
            save(1, (ReleaseStore, "accept", lambda *a, **k: _raise(OSError)))
            save(2, (ReleaseStore, "accept", rejected), ReleaseRejected)
            save(3, upload_fails)
            save(3)
            save(4, upload_fails)
        finally:
            manager.close()
        dead = [staged(run, step) for step in range(4)]
        pending = staged(run, 4)
        assert all(dead) and pending
        dead = set().union(*dead)

        result = ledger.collect(keep=1)

        assert result["orphaned"] == len(dead)
        assert all(ledger.store.get_artifact(*v) is None for v in dead)
        assert all(ledger.store.get_artifact(*v).state == "building" for v in pending)
        assert all(ledger.store.get_artifact(v.id, v.version) for v in _outputs(ledger, run, 3))


def test_collect_takes_failed_rows_but_the_placeholder_a_finished_run_s_record_names(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 1)
        with monkeypatch.context() as patch:
            patch.setattr(ArtifactStore, "publish_blob_from_path", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")
        run.finish()
    derived = ledger.derive(run.release_of(1).release_id, "eval", {"metrics": b"1"})
    failed = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@observed=")
    with ledger.start(trainer.program) as ended:
        with monkeypatch.context() as patch:
            patch.setattr(ReleaseStore, "accept", lambda *a, **k: _raise(OSError))
            with pytest.raises(OSError):
                Trainer(ledger).train(ended, 1)
    # Failed by a pass whose idle floor kept them.
    assert ledger.collect(keep=1, min_idle_seconds=3600)["orphaned"] > 0
    failed += [
        v
        for v in ledger.store.list_versions_by_id_prefix(f"torch_{ended.execution_id}@")
        if v.state == "failed"
    ]
    placeholders = [f"torch_{run.execution_id}", f"torch_{derived.release.build_id}"]
    assert [ledger.store.get_artifact(p, 1).state for p in placeholders] == ["failed"] * 2

    ledger.collect(keep=1)

    assert failed and all(ledger.store.get_artifact(v.id, v.version) is None for v in failed)
    assert ledger.store.get_artifact(placeholders[1], 1) is None
    assert ledger.store.get_artifact(placeholders[0], 1).state == "failed"
    assert ledger.builds.get_build(run.execution_id).state == "ready"
    assert ledger.builds.get_build(derived.release.build_id) is not None
    assert ledger.derived(run.release_of(1).release_id, "eval") == derived


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
        first = ledger.start(trainer.program, checkpointing=ASYNC)
        trainer.train(first, 3)
        first.close()

        releases = ledger.releases.releases_of(first.execution_id, "checkpoint")
        assert [(r.sequence, r.state) for r in releases] == [(n, "accepted") for n in (1, 2, 3)]
        trainer = Trainer(ledger)
        with ledger.resume(first.execution_id, checkpointing=ASYNC) as resumed:
            trainer.restore(resumed)
            trainer.train(resumed, STEPS)
            resumed.finish()
        assert _same(trainer.weights(), expected)


def test_an_asynchronous_save_returns_once_its_state_is_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The CPU stager, which macOS runs too once no accelerator shows (see above).
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    saved = threading.Event()
    with Ledger(tmp_path / "ledger") as ledger:
        model = _model()
        program = _program(ledger, model, torch.optim.SGD(model.parameters(), lr=0.1))
        with ledger.start(program, checkpointing=ASYNC) as run:
            # The first save starts the writer's subprocess, which save waits for.
            run.save(1, {"model": model}).result()
            stager = run.manager._saver.stager
            stage = stager._stage_on_worker

            def late(*args):
                # Copies once save has returned, or after a second if save waits for it.
                saved.wait(1)
                return stage(*args)

            monkeypatch.setattr(stager, "_stage_on_worker", late)
            expected = [p.detach().clone() for p in model.parameters()]
            pending = run.save(2, {"model": model})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)  # the next optimizer step, in place
            saved.set()
            pending.result()
            run.finish()
        loaded = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        ledger.load(run.release_of(2).release_id, {"model": loaded})

    assert _same(list(loaded.parameters()), expected)


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
    # The stores have no close of their own; each SDK client does.
    with closing(blobs.open()._client) as client:  # type: ignore[attr-defined]
        if request.param == "s3":
            client.create_bucket(Bucket=bucket)
        elif request.param == "gcs":
            client.create_bucket(bucket)
        else:
            client.create_container()
    return blobs


def _put(remote: RemoteBlobs, key: str, data: bytes) -> None:
    with closing(remote.open()._client) as client:  # type: ignore[attr-defined]
        if remote.backend == "s3":
            client.put_object(Bucket=remote.bucket, Key=key, Body=data)
        elif remote.backend == "gcs":
            client.bucket(remote.bucket).blob(key).upload_from_string(data)
        else:
            client.upload_blob(key, data, overwrite=True)


def test_remote_blobs_take_only_a_backend_the_ledger_has():
    with pytest.raises(ValueError, match="backend must be s3, gcs or azure, not 'hdfs'"):
        RemoteBlobs("hdfs", "bucket")


@pytest.mark.integration
@pytest.mark.slow
def test_a_store_names_each_object_s_bytes(remote: RemoteBlobs):
    scheme = {"s3": "s3", "gcs": "gs", "azure": "az"}[remote.backend]
    _put(remote, "data/a.bin", b"a")
    _put(remote, "data/sub/b.bin", b"bb")
    _put(remote, "data/marker/", b"")
    _put(remote, "other/c.bin", b"c")

    def listed(path: str) -> dict[str, tuple[int, str]]:
        found = Remote(f"{scheme}://{remote.bucket}/{path}", remote.settings).objects()
        return {key: (size, name) for key, size, name in found}

    under = listed("data/")
    assert {key: size for key, (size, _) in under.items()} == {
        "data/a.bin": 1,
        "data/sub/b.bin": 2,
    }
    assert listed("data/a.bin") == {"data/a.bin": under["data/a.bin"]}
    assert listed("data/none.bin") == {}
    # The same bytes again keep their name, but on Azure, whose ETag each write changes;
    # other bytes change it.
    _put(remote, "data/a.bin", b"a")
    again = listed("data/a.bin")
    assert (again == {"data/a.bin": under["data/a.bin"]}) == (remote.backend != "azure")
    _put(remote, "data/a.bin", b"A")
    assert listed("data/a.bin")["data/a.bin"] != under["data/a.bin"]


@pytest.mark.integration
@pytest.mark.slow
def test_a_run_holds_its_inputs_in_an_object_store(tmp_path: Path, remote: RemoteBlobs):
    scheme = {"s3": "s3", "gcs": "gs", "azure": "az"}[remote.backend]
    _put(remote, "data/a.bin", b"a")
    with Ledger(tmp_path / "ledger") as ledger:
        trainer = Trainer(ledger)
        data = Remote(f"{scheme}://{remote.bucket}/data/", remote.settings)
        with _run(ledger, trainer, {"train": data}) as run:
            trainer.train(run, 1)
            _put(remote, "data/a.bin", b"A")
            with pytest.raises(InputChanged, match="has other bytes"):
                trainer.train(run, 2)


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


def test_a_write_to_an_object_store_stages_where_collect_sweeps(tmp_path: Path):
    remote = RemoteBlobs(
        "s3",
        "ckpt",
        settings=BlobSettings(s3_region="us-east-1", s3_access_key="a", s3_secret_key="b"),
    )
    with Ledger(tmp_path / "ledger", blobs=remote) as ledger:
        staging = ledger.store.blob_store.staging_dir
        assert staging == tmp_path / "ledger" / "staging"
        # What a rank killed mid-upload leaves, an hour and more ago.
        staging.mkdir()
        left = staging / "vestigo_blob_killed.tmp"
        left.write_bytes(b"partial")
        os.utime(left, (time.time() - 7200,) * 2)

        ledger.collect(keep=1)

        assert not left.exists()


# --- Two ranks ---


def _rank_main(rank: int, world: int, root: str, case: str, signing_key: bytes) -> None:
    """One rank of a two-rank run, every rank calling ``Ledger.run``; writes what it saw.

    Cases: ``sync`` and ``async`` save and finish; in ``die`` rank 1 dies before
    writing step 2, and in ``slow`` comes to its barrier after rank 0 gave up; in
    ``mute`` rank 0 never says how a save went;
    ``copies`` and ``drift`` give each rank its own copy of a ``Path`` input, which in
    ``drift`` rank 1 rewrites before step 2 and in ``late`` before it finishes; in
    ``retry`` rank 1's copy cannot be read for one save of step 2, saved again. Rank 1
    has another config in ``mismatch``, one that is not JSON in ``nonjson``, async
    saves in ``modes``, another signing key in ``otherkey``, another ledger in
    ``otherledger``, a ledger that fails to answer in ``unreachable``, and a copy with
    other bytes in ``badcopy``. In ``unread`` rank 0's copy cannot be read for step 2's
    asynchronous acceptance, and in ``closefails`` rank 0's manager fails to close at
    the finish. Under a 10 second group timeout, rank 1 never comes to the run in
    ``stall``, and in ``lost`` dies as rank 0 sends it the opened run.

    Asynchronous, the group gets its rank directly and the environment names none, so
    the writer's subprocess has only what each rank configured.
    """
    if sys.platform != "win32":
        os.setsid()  # killed with its writer subprocess if it hangs; Windows kills the tree
    asynchronous = case in ("async", "unread") or (case == "modes" and rank == 1)
    if case == "async":
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
    else:
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world))
    if case == "unread":
        # torch_checkpointing 0.1.1 stages for CUDA whenever an accelerator exists.
        torch.accelerator.is_available = lambda: False
    dist.init_process_group(
        "gloo",
        init_method=Path(root, "rendezvous").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=10) if case in ("stall", "lost") else None,
    )
    if rank == 0 and case == "mute":
        vestigo.torch._Ranks.post = lambda *args, **kwargs: None  # type: ignore[method-assign]
    if rank == 1 and case == "otherkey":
        signing_key = Ed25519PrivateKey.generate().private_bytes_raw()
    elsewhere = rank == 1 and case == "otherledger"
    ledger = Ledger(Path(root) / ("elsewhere" if elsewhere else "ledger"), signing_key=signing_key)
    if rank == 1 and case == "unreachable":
        ledger.builds.get_build = lambda build_id: _raise(RuntimeError("the ledger is down"))
    if rank == 1 and case == "lost":
        share = vestigo.torch._Ranks.from_rank_0
        calls: list[object] = []

        def dying(self: vestigo.torch._Ranks, work: object) -> object:
            calls.append(work)
            if len(calls) == 2:  # rank 0 sends the run it opened
                os._exit(3)
            return share(self, work)

        vestigo.torch._Ranks.from_rank_0 = dying  # type: ignore[method-assign]
    if rank == 1 and case == "stall":
        # Never comes to the run, and lives on until rank 0 gave up.
        while not Path(root, "outcome0.json").exists():
            time.sleep(0.1)
        os._exit(0)
    model = _model()
    copy = Path(root, f"rank{rank}", "train.bin")
    if case == "badcopy" and rank == 1:
        copy.write_bytes(b"rowz")
    checkpointing = (
        CheckpointManager.Config.with_async_save()
        if asynchronous
        else CheckpointManager.Config.with_sync_save()
    )
    checkpointing.save.writer_config.checkpoint_write_barrier_timeout_sec = (
        2 if case in ("slow", "mute") else 5
    )
    config: dict[str, object] = {"batch": 16}
    if rank == 1 and case == "mismatch":
        config["batch"] = 32
    if rank == 1 and case == "nonjson":
        config["seed"] = torch.tensor(1)
    saved: list[int] = []
    returned: list[str] = []
    outcome: dict[str, object] = {"saved": saved, "returned": returned}
    groups = len(distributed_c10d._world.pg_map)
    started = time.monotonic()
    try:
        run = ledger.run(
            model,
            torch.optim.AdamW(model.parameters()),
            loss=nn.MSELoss(),
            version="tiny-v1",
            inputs={"train": copy}
            if case in ("copies", "drift", "late", "retry", "badcopy", "unread")
            else None,
            config=config,
            checkpointing=checkpointing,
        )
    except Exception as exc:
        outcome.update(error=type(exc).__name__, message=str(exc))
        outcome["seconds"] = time.monotonic() - started
        run = None
    if run is not None and case in ("unread", "closefails"):
        outcome["execution"] = run.execution_id
        for step in (1, 2):
            if rank == 0 and step == 2 and case == "unread":
                copy.chmod(0)  # unreadable, not gone, so the lease stays
            try:
                pending = run.save(step, {"model": model})
                if isinstance(pending, Future):
                    pending.result()
                saved.append(step)
            except InputChanged as exc:
                outcome.update(error=type(exc).__name__, message=str(exc))
            finally:
                copy.chmod(0o600)
        if rank == 0 and case == "closefails":
            close = run.manager.close

            def fails_once() -> None:
                run.manager.close = close  # type: ignore[method-assign]
                raise OSError("the manager did not close")

            run.manager.close = fails_once  # type: ignore[method-assign]
        try:
            run.finish()
            outcome["finished"] = True
        except (OSError, InputChanged) as exc:
            outcome.update(error=type(exc).__name__, message=str(exc))
        run.close()
    elif run is not None:
        outcome["execution"] = run.execution_id
        retried = False
        for index, step in enumerate((1, 2, 2) if case == "retry" else (1, 2)):
            if rank == 1 and case == "retry" and index:
                # Unreadable for the first save of step 2, then readable again.
                copy.chmod(0 if index == 1 else 0o644)
            if rank == 1 and step == 2 and case == "die":
                os._exit(3)  # dies before writing: rank 0 never passes the barrier
            if rank == 1 and step == 2 and case == "slow":
                time.sleep(5)  # past rank 0's barrier timeout and its word of it
            if rank == 1 and step == 2 and case == "drift":
                copy.write_bytes(b"rowz")
            try:
                with torch.no_grad():
                    model[0].weight.add_(rank + 1)
                release = run.save(step, {"model": model})
                if isinstance(release, Future):
                    release.result()
                else:
                    returned.append(release.release_id)
                saved.append(step)
            except Exception as exc:
                outcome.update(error=type(exc).__name__, message=str(exc))
                if case == "retry" and not retried:
                    retried = True
                    continue
                break
        else:
            if rank == 1 and case == "late":
                copy.write_bytes(b"rowz")
            try:
                run.finish()
            except InputChanged as exc:
                outcome.update(error=type(exc).__name__, message=str(exc))
                run.close()
                Path(root, f"outcome{rank}.json").write_text(json.dumps(outcome))
                os._exit(0)
            # Past finish every save is settled: an asynchronous one may land on rank 0
            # after this rank's future resolves.
            outcome["released"] = [run.release_of(step).release_id for step in saved]  # type: ignore[union-attr]
        run.close()
    # Each run's own group is gone with it, opened or refused.
    outcome["groups"] = len(distributed_c10d._world.pg_map) - groups
    Path(root, f"outcome{rank}.json").write_text(json.dumps(outcome))
    # A dead peer can hang the group's teardown; nothing below needs it.
    os._exit(0)


def _builds(ledger: Ledger) -> list[str]:
    conn = ledger.builds._get_connection()
    try:
        return [row["build_id"] for row in conn.execute("SELECT build_id FROM artifact_builds")]
    finally:
        conn.close()


def _kill_tree(pid: int) -> None:
    """Kill a process and every process it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False, capture_output=True)
    else:
        os.killpg(pid, signal.SIGKILL)


def _two_ranks(root: Path, case: str) -> tuple[list[int], list[dict], bytes]:
    """Each rank's exit code and outcome (None for a rank that wrote none), and the
    verify key."""
    for rank in (0, 1):
        _file(root / f"rank{rank}" / "train.bin")
    signing = Ed25519PrivateKey.generate()
    context = multiprocessing.get_context("spawn")
    ranks = [
        context.Process(
            target=_rank_main, args=(r, 2, str(root), case, signing.private_bytes_raw())
        )
        for r in (0, 1)
    ]
    for process in ranks:
        process.start()
    # One deadline for both, inside the test's own timeout.
    deadline = time.monotonic() + 120
    for process in ranks:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in ranks:
        if process.is_alive():
            # Hung (a peer waiting for one that never comes): its session, writer too.
            _kill_tree(process.pid)
            process.join()
    outcomes = [
        json.loads(path.read_text()) if (path := root / f"outcome{r}.json").exists() else None
        for r in (0, 1)
    ]
    return [p.exitcode for p in ranks], outcomes, signing.public_key().public_bytes_raw()


@pytest.mark.parametrize(
    "case",
    [
        "sync",
        pytest.param(
            "async",
            marks=pytest.mark.skipif(
                sys.platform == "darwin",
                reason="torch_checkpointing 0.1.1 stages for CUDA whenever an accelerator "
                "exists, MPS too",
            ),
        ),
    ],
)
def test_every_rank_runs_one_run_whose_releases_rank_zero_makes(tmp_path: Path, case: str):
    exits, outcomes, verify_key = _two_ranks(tmp_path, case)

    assert exits == [0, 0]
    zero, one = outcomes
    assert zero is not None and one is not None
    # One execution, each save one release on both ranks, which both finished.
    assert zero["execution"] == one["execution"] and zero["released"] == one["released"]
    assert zero["groups"] == one["groups"] == 0
    assert len(zero["released"]) == 2
    # A synchronous save returned it on every rank.
    if case == "sync":
        assert zero["returned"] == one["returned"] == zero["released"]
    with Ledger(tmp_path / "ledger") as ledger:
        execution = zero["execution"]
        assert ledger.builds.get_build(execution).state == "ready"
        assert _builds(ledger) == [execution]
        for step, release_id in enumerate(zero["released"], start=1):
            release = ledger.release_of(execution, step)
            assert release is not None and release.release_id == release_id
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
            public = Ed25519PublicKey.from_public_bytes(verify_key)
            public.verify(bytes.fromhex(content.pop("signature")), _canonical(content))


def test_a_rank_that_comes_to_the_barrier_late_hears_why_rank_0_gave_up(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "slow")

    # Rank 0 gave up at the barrier; rank 1 passed it later, and raises rank 0's error.
    assert exits == [0, 0]
    zero, one = outcomes
    assert zero is not None and one is not None
    assert len(zero["saved"]) == 1 and len(one["saved"]) == 1
    assert zero["message"] in one["message"]
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.release_of(zero["execution"], 2) is None


def test_a_rank_that_hears_nothing_from_rank_0_stops_at_the_barrier_s_timeout(
    tmp_path: Path,
):
    exits, outcomes, _ = _two_ranks(tmp_path, "mute")

    assert exits == [0, 0]
    zero, one = outcomes
    assert zero is not None and one is not None
    # Rank 0 accepted step 1, which rank 1 never heard of, and so met no one at step 2.
    assert len(zero["saved"]) == 1 and "error" in zero
    assert one["saved"] == [] and "heard nothing from rank 0 after step 1" in one["message"]
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.release_of(zero["execution"], 1) is not None


def test_a_rank_that_dies_before_writing_leaves_no_release(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "die")

    # Rank 0 is not left waiting for a rank that will never hear it.
    assert exits == [0, 3]
    zero = outcomes[0]
    assert zero is not None and len(zero["saved"]) == 1 and "error" in zero
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.release_of(zero["execution"], 1) is not None
        assert ledger.release_of(zero["execution"], 2) is None
        [release] = ledger.releases.releases_of(zero["execution"], "checkpoint")
        assert release.sequence == 1


@POSIX_MODES
def test_a_failed_asynchronous_save_does_not_stop_the_finish_on_any_rank(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "unread")

    assert exits == [0, 0]
    zero, one = outcomes
    assert zero is not None and one is not None
    # Rank 0's future said why step 2 was not released; rank 1's write was whole.
    assert zero["saved"] == [1] and "cannot be read" in zero["message"]
    assert one["saved"] == [1, 2]
    assert zero.get("finished") and one.get("finished")
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.builds.get_build(zero["execution"]).state == "ready"
        assert ledger.release_of(zero["execution"], 2) is None


def test_a_rank_that_cannot_wait_for_its_save_stops_every_rank_s_finish(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "closefails")

    assert exits == [0, 0]
    for each in outcomes:
        assert each is not None and each["saved"] == [1, 2] and "finished" not in each
        assert each["error"] == "OSError" and "the manager did not close" in each["message"]
    with Ledger(tmp_path / "ledger") as ledger:
        assert ledger.builds.get_build(outcomes[0]["execution"]).state == "building"


def test_a_rank_that_never_comes_stops_rank_0_at_the_group_s_timeout(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "stall")

    # At the 10 seconds the launcher gave, not the 30 minutes of a group's own default.
    assert exits == [0, 0]
    zero = outcomes[0]
    assert zero is not None and "error" in zero and zero["seconds"] < 60
    assert zero["groups"] == 0


def test_a_rank_lost_as_the_run_opens_leaves_it_failed_not_held(tmp_path: Path):
    exits, outcomes, _ = _two_ranks(tmp_path, "lost")

    assert exits == [0, 3]
    zero = outcomes[0]
    assert zero is not None and "error" in zero and zero["groups"] == 0
    with Ledger(tmp_path / "ledger") as ledger:
        [build] = [ledger.builds.get_build(each) for each in _builds(ledger)]
        # Rank 0 closed the run it opened, which no rank holds.
        assert build is not None and build.state == "failed"


@pytest.mark.parametrize(
    "case", ["copies", "drift", "late", pytest.param("retry", marks=POSIX_MODES)]
)
def test_every_rank_s_copy_of_the_inputs_must_hold_for_a_release(tmp_path: Path, case: str):
    exits, outcomes, _ = _two_ranks(tmp_path, case)

    assert exits == [0, 0]
    zero, one = outcomes
    assert zero is not None and one is not None
    with Ledger(tmp_path / "ledger") as ledger:
        execution = zero["execution"]
        assert ledger.release_of(execution, 1) is not None
        if case == "copies":
            assert len(zero["saved"]) == 2 and one["saved"] == zero["saved"]
            outputs = ledger.releases.bundle(ledger.release_of(execution, 2).release_id).outputs
            pinned = ledger.builds.get_build(execution).params["inputs"]["train"]
            # Each rank's own copy, as it found it: rank 1's in its report, rank 0's in
            # the manifest it signed past the barrier.
            [report] = [v for v in outputs if v.params.get("rank") == 1]
            [manifest] = [v for v in outputs if v.params.get("manifest")]
            found = {
                1: json.loads(ledger.store.read_blob(report.id, report.version))["inputs"],
                0: json.loads(ledger.store.read_blob(manifest.id, manifest.version))["inputs"],
            }
            for rank, inputs_found in found.items():
                assert inputs_found == {
                    "train": {"path": str(tmp_path / f"rank{rank}" / "train.bin"), "pin": pinned}
                }
        elif case == "retry":
            # The failed save of step 2 raised on both ranks; saved again, it landed.
            for each in (zero, one):
                assert "cannot be read on rank 1" in each["message"]
                assert each["saved"] == [1, 2] and each["returned"] == zero["released"]
            assert ledger.builds.get_build(execution).state == "ready"
        elif case == "drift":
            # Rank 0's acceptance failed, and both ranks raised it.
            for each in (zero, one):
                assert len(each["saved"]) == 1 and each["error"] == "InputChanged"
                assert "on rank 1" in each["message"]
            assert ledger.release_of(execution, 2) is None
            assert ledger.builds.get_build(execution).lease_owner is None
        else:
            # Both saves landed; the finish did not, on either rank.
            for each in (zero, one):
                assert len(each["saved"]) == 2 and each["error"] == "InputChanged"
                assert "now on rank 1" in each["message"]
            assert ledger.builds.get_build(execution).state == "building"


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("mismatch", "ValueError", "rank 1's program differ"),
        ("nonjson", "TypeError", "JSON"),
        ("modes", "ValueError", "rank 1's saves differ"),
        ("otherkey", "ValueError", "rank 1's verify_key differ"),
        ("badcopy", "InputChanged", "has other bytes on rank 1"),
        ("otherledger", "ValueError", "rank 1's ledger does not hold execution"),
        ("unreachable", "RuntimeError", "the ledger is down"),
    ],
)
def test_a_rank_that_runs_something_else_stops_every_rank(
    tmp_path: Path, case: str, error: str, message: str
):
    exits, outcomes, _ = _two_ranks(tmp_path, case)

    assert exits == [0, 0]
    for each in outcomes:
        assert each is not None and each["error"] == error and message in each["message"]
        assert each["groups"] == 0
    with Ledger(tmp_path / "ledger") as ledger:
        builds = _builds(ledger)
        if case not in ("otherledger", "unreachable"):
            # Refused before rank 0 opened anything.
            assert builds == []
        else:
            [build] = [ledger.builds.get_build(each) for each in builds]
            # Opened, then failed with no checkpoint: nothing to resume.
            assert build is not None and build.state == "failed"


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


def test_observations_read_the_execution_s_own_ids_not_every_output_of_its_key(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    with ledger.start(trainer.program) as run:
        trainer.train(run, 2)
        recorded = run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")
        # The key's outputs are every checkpoint file the computation has stored.
        monkeypatch.setattr(
            ledger.store, "versions_of_computation", lambda *a: _raise(AssertionError)
        )
        assert ledger.observations(run.execution_id) == [recorded]


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


def test_an_observation_an_ended_attempt_never_stored_is_collected(
    ledger: Ledger, monkeypatch: pytest.MonkeyPatch
):
    trainer = Trainer(ledger)
    run = ledger.start(trainer.program)
    trainer.train(run, 1)
    with monkeypatch.context() as patch:
        # The process dies mid-upload: nothing fails the row.
        patch.setattr(ArtifactStore, "publish_blob_from_path", lambda *a, **k: _raise(_Crash))
        with pytest.raises(_Crash):
            run.observe(_exported(trainer.model), scope="model_forward", coverage="complete")
    [staged] = ledger.store.list_versions_by_id_prefix(f"torch_{run.execution_id}@observed=")
    assert staged.state == "building"
    # While its attempt holds the execution, it may still be storing.
    assert ledger.collect(keep=1)["orphaned"] == 0
    run.close()

    with ledger.resume(run.execution_id):
        assert ledger.collect(keep=1)["orphaned"] == 1

    assert ledger.store.get_artifact(staged.id, staged.version) is None


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


def _example(
    script: str, *args: str, ranks: int = 1, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(EXAMPLES / script), *args]
    if ranks > 1:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        launch = ["-m", "torch.distributed.run", f"--nproc-per-node={ranks}"]
        rendezvous = ["--master-addr=127.0.0.1", f"--master-port={port}"]
        command = [sys.executable, *launch, *rendezvous, *command[1:]]
    # Its own session, so a timeout kills torchrun's ranks with the launcher.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_tree(process.pid)
            process.communicate()
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _said(result: subprocess.CompletedProcess[str], pattern: str) -> str:
    match = re.search(pattern, result.stdout)
    assert match is not None, result.stdout + result.stderr
    return match.group(1)


def test_the_quick_start_is_the_guide_s_script_and_resumes_to_the_uninterrupted_loss(
    tmp_path: Path,
):
    guide = (EXAMPLES.parent.parent / "docs" / "torch.md").read_text()
    assert f"```python\n{(EXAMPLES / 'quickstart.py').read_text()}```" in guide
    stopped_at, whole_at = tmp_path / "stopped", tmp_path / "whole"
    stopped_at.mkdir()
    whole_at.mkdir()

    stopped = _example("quickstart.py", "--stop-after=3", cwd=stopped_at)
    assert stopped.returncode == 1 and "stopped after epoch 3" in stopped.stderr
    execution = _said(stopped, r"started execution (\w+)")
    resumed = _example("quickstart.py", cwd=stopped_at)
    assert resumed.returncode == 0, resumed.stderr
    assert _said(resumed, r"resumed execution (\w+) after epoch 3") == execution
    assert "epoch 3:" not in resumed.stdout and "finished in 2 attempt(s)" in resumed.stdout

    whole = _example("quickstart.py", cwd=whole_at)
    assert whole.returncode == 0, whole.stderr
    last = r"epoch 6: loss (\S+),"
    assert _said(resumed, last) == _said(whole, last)
    # Finished, so the same computation again is a new execution.
    again = _example("quickstart.py", cwd=stopped_at)
    assert _said(again, r"started execution (\w+)") != execution


def test_keys_replaces_neither_key_of_a_pair(tmp_path: Path):
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "verify.key").write_text("the key old checkpoints verify with\n")

    refused = _example("keys.py", str(keys))

    assert refused.returncode == 1 and "verify.key exists" in refused.stderr
    assert (keys / "verify.key").read_text() == "the key old checkpoints verify with\n"
    assert not (keys / "signing.key").exists()


def test_the_gate_ships_what_the_trainer_signed_and_refuses_a_forgery(tmp_path: Path):
    root, keys = str(tmp_path / "ledger"), tmp_path / "keys"
    made = _example("keys.py", str(keys))
    assert made.returncode == 0, made.stderr
    if sys.platform != "win32":  # no POSIX modes there
        assert (keys / "signing.key").stat().st_mode & 0o777 == 0o600
    again = _example("keys.py", str(keys))
    assert again.returncode == 1 and "exists" in again.stderr
    trained = _example("train.py", root, f"--signing-key={keys / 'signing.key'}")
    assert trained.returncode == 0, trained.stderr
    verify = f"--verify-key={keys / 'verify.key'}"

    shipped = _example("evaluate.py", root, verify, f"--out={tmp_path / 'model.safetensors'}")
    assert shipped.returncode == 0, shipped.stderr
    assert "signature verified" in shipped.stdout
    assert (tmp_path / "model.safetensors").exists()

    tampered = _example("tamper.py", root)
    assert tampered.returncode == 0, tampered.stderr
    # Bytes and row agree, so the digest alone is fooled.
    fooled = _example("evaluate.py", root, f"--out={tmp_path / 'forged.safetensors'}")
    assert fooled.returncode == 0, fooled.stderr
    assert (tmp_path / "forged.safetensors").exists()
    refused = _example("evaluate.py", root, verify, f"--out={tmp_path / 'refused.safetensors'}")
    assert refused.returncode == 1
    assert "differs from the checkpoint the ledger accepted" in refused.stderr
    assert not (tmp_path / "refused.safetensors").exists()


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
    assert f"{branch} branched from {execution} at {head[0]}" in branched.stdout
    assert "executions reading the data: 2" in branched.stdout


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
