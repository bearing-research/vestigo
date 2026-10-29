"""Data-parallel training on several ranks, each checkpoint one release over all of them.

    uv run torchrun --nproc-per-node 2 --master-addr 127.0.0.1 \
        examples/torch/distributed.py ledger --crash-at 2
    uv run torchrun --nproc-per-node 2 --master-addr 127.0.0.1 \
        examples/torch/distributed.py ledger

Every rank runs the same code. ``ledger.run`` is collective, as
``DistributedDataParallel`` is: rank 0 starts or resumes the run and holds the lease,
and each rank gets its own ``Run``. Every rank writes its own files through the
ledger's storage; rank 0 accepts the step once the barrier has seen every rank write,
and each rank's ``save`` returns that release.

The training data is a directory of shards, one per rank, that the first run writes
beside the ledger. It is a ``Path`` input: every rank's report says what it found
there, and rank 0 accepts a step only while every rank's copy is the bytes the run
pinned. On another machine a rank passes the path of its own copy.

One machine and SQLite here. Ranks on several machines need a ledger every rank can
reach: ``Ledger(path, dsn="postgresql://...", blobs=RemoteBlobs("s3", "bucket"))``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from vestigo.torch import Ledger


def shards(ledger: Path, world: int) -> Path:
    """One training shard per rank, beside *ledger*, written by rank 0 on the first run."""
    root = ledger.parent / f"regression-shards-{world}"
    if dist.get_rank() == 0 and not root.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and renamed whole, so a crash midway leaves no partial set.
        partial = Path(tempfile.mkdtemp(dir=ledger.parent))
        for rank in range(world):
            generator = torch.Generator().manual_seed(1 + rank)
            x = torch.randn(128, 4, generator=generator)
            torch.save((x, x.sum(dim=1, keepdim=True)), partial / f"{rank}.pt")
        partial.rename(root)
    dist.barrier()
    return root


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
    data = shards(args.ledger, world)

    with (
        Ledger(args.ledger) as ledger,
        ledger.run(
            model,
            optimizer,
            loss=nn.MSELoss(),
            version="regression-v1",
            inputs={"train": data},
            config={"batch": 32, "world_size": world},
        ) as run,
    ):
        # Each rank reads back the files it wrote.
        epoch = run.restore({"model": model, "optim": optimizer, "epoch": 0})["epoch"]
        if run.restores_from and rank == 0:
            print(f"resumed {run.execution_id} after epoch {epoch}")

        ddp = DistributedDataParallel(model)
        x, y = torch.load(data / f"{rank}.pt")
        generator = torch.Generator().manual_seed(1 + rank)
        while epoch < args.epochs:
            epoch += 1
            for batch in torch.randperm(len(x), generator=generator).split(32):
                optimizer.zero_grad()
                nn.functional.mse_loss(ddp(x[batch]), y[batch]).backward()
                optimizer.step()
            release = run.save(epoch, {"model": model, "optim": optimizer, "epoch": epoch})
            if rank == 0:
                print(f"epoch {epoch}: release {release.release_id}")
            if epoch == args.crash_at:
                break
        if epoch == args.epochs:
            run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
