"""Fine-tune from the best epoch of ``train.py``, as a new execution of a new program.

    uv run python examples/torch/train.py ledger
    uv run python examples/torch/branch.py ledger --lr 1e-3

A branch starts from any accepted checkpoint. Its computation names that release, so it
never deduplicates against the run it came from; its program (the same model under a
lower learning rate) is a different node over the same model node. At the end it prints
the branches of the run it came from, and how many executions read the training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from train import HEAD, data

from vestigo.torch import Ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    path = data(args.ledger)
    x, y = torch.load(path)

    with Ledger(args.ledger) as ledger:
        head = ledger.releases.reference(HEAD)
        if head is None:
            raise SystemExit(f"no {HEAD} yet: run train.py first")
        source, _revision = head
        # Run again after a crash, and this resumes the branch instead of starting one.
        with ledger.run(
            model,
            optimizer,
            loss=nn.MSELoss(),
            version="regression-v1",
            inputs={"train": path},
            config={"batch": 32},
            from_release=source,
        ) as run:
            # Only the weights: the optimizer starts fresh under the new rate.
            run.restore({"model": model})
            print(f"branched {run.execution_id} from release {run.restores_from}")
            for epoch in range(1, args.epochs + 1):
                for batch in torch.randperm(len(x)).split(32):
                    optimizer.zero_grad()
                    nn.functional.mse_loss(model(x[batch]), y[batch]).backward()
                    optimizer.step()
                run.save(epoch, {"model": model})
                with torch.no_grad():
                    loss = nn.functional.mse_loss(model(x), y).item()
                print(f"epoch {epoch}: loss {loss:.4f}")
            run.finish()

        # The lineage, from what the ledger recorded of each run.
        parent = ledger.releases.get(source)
        assert parent is not None
        for execution in ledger.descendants(parent.build_id):
            print(f"{execution.execution_id} branched from {parent.build_id} at {execution.init}")
        print(f"executions reading the data: {len(ledger.executions(reading=path))}")


if __name__ == "__main__":
    main()
