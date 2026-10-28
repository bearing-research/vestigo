"""Data-parallel training on several ranks, each checkpoint one release over all of them.

    uv run torchrun --nproc-per-node 2 --master-addr 127.0.0.1 \
        examples/torch/distributed.py ledger --crash-at 2
    uv run torchrun --nproc-per-node 2 --master-addr 127.0.0.1 \
        examples/torch/distributed.py ledger

Rank 0 owns the run: it starts or resumes it and holds the lease. Every rank writes
its own files through the ledger's storage; rank 0 accepts the step once the barrier
has seen every rank write. The other ranks get the run's checkpoints, and the
checkpoint to restore, from rank 0 by broadcast.

One machine and SQLite here. Ranks on several machines need a ledger every rank can
reach: ``Ledger(path, dsn="postgresql://...", blobs=RemoteBlobs("s3", "bucket"))``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch_checkpointing.checkpoint_manager import CheckpointManager

from vestigo.torch import Checkpoints, Ledger

INPUTS = {"train": "synthetic-regression@seed=1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--crash-at", type=int, help="stop after saving this epoch")
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    ledger = run = None
    shared: list[object] = [None, None]
    if rank == 0:
        ledger = Ledger(args.ledger)
        program = ledger.define_program(
            ledger.define_model(model, version="regression-v1"),
            loss=nn.MSELoss(),
            optimizer=optimizer,
            config={"batch": 32, "world_size": world},
        )
        execution = ledger.find(program, INPUTS)
        run = ledger.resume(execution) if execution else ledger.start(program, INPUTS)
        shared = [run.checkpoints, run.restore()]
    dist.broadcast_object_list(shared, src=0)
    checkpoints, checkpoint = shared
    assert isinstance(checkpoints, Checkpoints)

    ddp = DistributedDataParallel(model)
    manager = checkpoints.configure(CheckpointManager.Config.with_sync_save()).build()
    try:
        epoch = 0
        if isinstance(checkpoint, str):
            # Each rank reads back the files it wrote.
            state = manager.load(
                checkpoint, into={"model": model.state_dict(), "optim": None, "epoch": None}
            )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])
            epoch = state["epoch"]
            if rank == 0:
                print(f"resumed {checkpoints.execution_id} after epoch {epoch}")

        generator = torch.Generator().manual_seed(1 + rank)
        x = torch.randn(128, 4, generator=generator)
        y = x.sum(dim=1, keepdim=True)
        while epoch < args.epochs:
            epoch += 1
            for batch in torch.randperm(len(x), generator=generator).split(32):
                optimizer.zero_grad()
                nn.functional.mse_loss(ddp(x[batch]), y[batch]).backward()
                optimizer.step()
            manager.save(
                checkpoints.checkpoint_id(epoch),
                {"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch},
            )
            if run is not None:
                release = run.release_of(epoch)
                assert release is not None
                print(f"epoch {epoch}: release {release.release_id}")
            if epoch == args.crash_at:
                break
    finally:
        manager.close()

    if run is not None and ledger is not None:
        if epoch == args.epochs:
            run.finish()
        run.close()
        ledger.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
