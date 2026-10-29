"""Train a small regression, every epoch a checkpoint in a vestigo ledger.

    uv run python examples/torch/train.py ledger --epochs 6 --crash-at 3
    uv run python examples/torch/train.py ledger --epochs 6 --keep 1

The first command stops after epoch 3; the second finds that run again by its
computation (the program and its inputs), resumes it from epoch 3 and finishes it,
deleting after each epoch every checkpoint but the newest one, the best and, until it
finishes, the one it resumed from.
The training data is a file the first run writes beside the ledger. It is a ``Path``
input, so the key holds its digest: edit the file and the next run starts fresh, and
edit it mid-run and the next save raises ``InputChanged``.
The lowest-loss epoch is the reference ``regression/best``, which ``branch.py``
fine-tunes from. Before finishing, the run records the model's exported forward pass;
after, an evaluation of the last checkpoint is a release derived from it.

    uv run python examples/torch/train.py ledger --signing-key keys/signing.key

With a signing key (``keys.py`` makes one), every checkpoint's manifest is signed, and
``evaluate.py`` with the public key loads only what this job signed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from vestigo.torch import Ledger

HEAD = "regression/best"


def data(ledger: Path) -> Path:
    """The training set beside *ledger*, written on the first run."""
    path = ledger.parent / "regression.pt"
    if not path.exists():
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(256, 4, generator=generator)
        y = x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1, generator=generator)
        torch.save((x, y), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--crash-at", type=int, help="stop after saving this epoch")
    parser.add_argument("--keep", type=int, help="checkpoints to keep beyond the heads")
    parser.add_argument("--signing-key", type=Path, help="a hex Ed25519 private key file")
    args = parser.parse_args()
    signing_key = bytes.fromhex(args.signing_key.read_text()) if args.signing_key else None

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    shuffle = torch.Generator().manual_seed(2)
    path = data(args.ledger)
    x, y = torch.load(path)

    with Ledger(args.ledger, signing_key=signing_key) as ledger:
        # Resumes this computation's unfinished execution, if there is one.
        with ledger.run(
            model,
            optimizer,
            loss=nn.MSELoss(),
            version="regression-v1",
            inputs={"train": path},
            config={"batch": 32},
        ) as run:
            # The model and optimizer load in place; the rest are defaults for a fresh start.
            state = run.restore(
                {
                    "model": model,
                    "optim": optimizer,
                    "shuffle": shuffle.get_state(),
                    "epoch": 0,
                    "best": math.inf,
                }
            )
            shuffle.set_state(state["shuffle"])
            epoch, best = state["epoch"], state["best"]
            if run.restores_from is not None:
                print(f"resumed {run.execution_id} after epoch {epoch}")
            else:
                print(f"started {run.execution_id}")

            while epoch < args.epochs:
                epoch += 1
                for batch in torch.randperm(len(x), generator=shuffle).split(32):
                    optimizer.zero_grad()
                    nn.functional.mse_loss(model(x[batch]), y[batch]).backward()
                    optimizer.step()
                with torch.no_grad():
                    loss = nn.functional.mse_loss(model(x), y).item()
                best = min(best, loss)
                release = run.save(
                    epoch,
                    {
                        "model": model,
                        "optim": optimizer,
                        "shuffle": shuffle.get_state(),
                        "epoch": epoch,
                        "best": best,
                    },
                )
                if loss == best:
                    run.set_head(HEAD, release.release_id)
                print(f"epoch {epoch}: loss {loss:.4f}, release {release.release_id}")
                if args.keep:
                    ledger.collect(keep=args.keep)
                if epoch == args.crash_at:
                    # Leaving the block closes the run: it keeps its checkpoints
                    # and drops its lease, so the next command resumes at once.
                    raise SystemExit(f"stopped after epoch {epoch}")

            batch_dim = torch.export.Dim("batch")
            exported = torch.export.export(model, (x[:2],), dynamic_shapes=({0: batch_dim},))
            run.observe(
                exported, scope="model_forward", coverage="complete_under_declared_constraints"
            )
            run.finish()
            if args.keep:
                ledger.collect(keep=args.keep)  # the checkpoint it resumed from goes too
            final = ledger.final_checkpoint(run.execution_id)
            assert final is not None
            print(f"finished {run.execution_id} on release {final.release_id}")

            # Computed after the checkpoint was accepted, so a release of its own.
            with torch.no_grad():
                metrics = {"mse": nn.functional.mse_loss(model(x), y).item()}
            evaluation = ledger.derive(
                final.release_id, "eval", {"metrics": json.dumps(metrics).encode()}
            )
            print(f"derived eval {metrics} as release {evaluation.release.release_id}")


if __name__ == "__main__":
    main()
