"""Train a small model with every epoch a checkpoint in a vestigo ledger.

    uv run python examples/torch/quickstart.py --stop-after 3
    uv run python examples/torch/quickstart.py

The first command trains three epochs and stops, as a crash would; the second resumes
after epoch 3 and finishes. The ledger and the training data are written to the
current directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from vestigo.torch import Ledger

EPOCHS = 6
BATCH = 32
DATA = Path("quickstart-data.pt")


def make_data(path: Path) -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(512, 4, generator=generator)
    torch.save((x, x.sum(dim=1, keepdim=True)), path)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    shuffle: torch.Generator,
) -> float:
    """One pass over the data in shuffled batches: plain PyTorch, no vestigo."""
    total = 0.0
    for batch in torch.randperm(len(x), generator=shuffle).split(BATCH):
        optimizer.zero_grad()
        loss = criterion(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
    return total / len(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after", type=int, help="stop after this epoch, as a crash")
    args = parser.parse_args()

    if not DATA.exists():
        make_data(DATA)
    x, y = torch.load(DATA)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    shuffle = torch.Generator().manual_seed(1)

    with Ledger("ledger") as ledger:
        with ledger.run(
            model,
            optimizer,
            loss=criterion,
            version="quickstart-v1",
            inputs={"train": DATA},
            config={"batch": BATCH},
        ) as run:
            state = run.restore(
                {"model": model, "optim": optimizer, "shuffle": shuffle.get_state(), "epoch": 0}
            )
            shuffle.set_state(state["shuffle"])
            if run.restores_from is None:
                print(f"started execution {run.execution_id}")
            else:
                print(f"resumed execution {run.execution_id} after epoch {state['epoch']}")

            for epoch in range(state["epoch"] + 1, EPOCHS + 1):
                loss = train_one_epoch(model, optimizer, criterion, x, y, shuffle)
                release = run.save(
                    epoch,
                    {
                        "model": model,
                        "optim": optimizer,
                        "shuffle": shuffle.get_state(),
                        "epoch": epoch,
                    },
                )
                print(f"epoch {epoch}: loss {loss:.4f}, checkpoint {release.release_id}")
                if epoch == args.stop_after:
                    raise SystemExit(f"stopped after epoch {epoch}; run again to resume")

            run.finish()

        record = ledger.execution(run.execution_id)
        final = ledger.final_checkpoint(run.execution_id)
        assert final is not None
        print(
            f"finished in {len(record.attempts)} attempt(s); "
            f"the trained model is checkpoint {final.release_id}"
        )


if __name__ == "__main__":
    main()
