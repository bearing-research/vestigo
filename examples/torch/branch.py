"""Fine-tune from the best epoch of ``train.py``, as a new execution of a new program.

    uv run python examples/torch/train.py ledger
    uv run python examples/torch/branch.py ledger --lr 1e-3

A branch starts from any accepted release. Its computation names that release, so it
never deduplicates against the run it came from; its program (the same model under a
lower learning rate) is a different node over the same model node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch_checkpointing.checkpoint_manager import CheckpointManager
from train import HEAD, INPUTS, data

from vestigo.torch import Ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x, y = data()

    with Ledger(args.ledger) as ledger:
        head = ledger.releases.reference(HEAD)
        if head is None:
            raise SystemExit(f"no {HEAD} yet: run train.py first")
        source, _revision = head
        program = ledger.define_program(
            ledger.define_model(model, version="regression-v1"),
            loss=nn.MSELoss(),
            optimizer=optimizer,
            config={"batch": 32},
        )
        with ledger.branch(program, source, INPUTS) as run:
            manager = run.configure(CheckpointManager.Config.with_sync_save()).build()
            try:
                checkpoint = run.restore()
                assert checkpoint is not None
                # Only the weights: the optimizer starts fresh under the new rate.
                state = manager.load(checkpoint, into={"model": model.state_dict()})
                model.load_state_dict(state["model"])
                print(f"branched {run.execution_id} from release {source}")
                for epoch in range(1, args.epochs + 1):
                    for batch in torch.randperm(len(x)).split(32):
                        optimizer.zero_grad()
                        nn.functional.mse_loss(model(x[batch]), y[batch]).backward()
                        optimizer.step()
                    manager.save(run.checkpoint_id(epoch), {"model": model.state_dict()})
                    with torch.no_grad():
                        loss = nn.functional.mse_loss(model(x), y).item()
                    print(f"epoch {epoch}: loss {loss:.4f}")
            finally:
                manager.close()
            run.finish()


if __name__ == "__main__":
    main()
