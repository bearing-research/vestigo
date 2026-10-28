"""Score the best checkpoint before it ships, loading only what the trainer signed.

    uv run python examples/torch/evaluate.py ledger --verify-key keys/verify.key \
        --out model.safetensors

A promotion gate: a job of its own, holding the public key only, that saves no
checkpoint and holds no lease. It loads the weights ``regression/best`` points at
through the ledger's load-only storage, which checks every file against the trainer's
signed manifest before torch deserializes it. It scores them on held-out data, and
only if they pass does it write them for serving, as safetensors, which a server can
load without unpickling anything. A checkpoint that is unsigned, signed with another
key, or changed since it was signed fails the load, and nothing is written.

Without ``--verify-key`` the gate checks each file against the digest the database
records, which is what ``tamper.py`` shows is not enough.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from train import HEAD

from vestigo.torch import Ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--verify-key", type=Path, help="a hex Ed25519 public key file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-mse", type=float, default=0.1)
    args = parser.parse_args()
    verify_key = bytes.fromhex(args.verify_key.read_text()) if args.verify_key else None

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    with Ledger(args.ledger, verify_key=verify_key) as ledger:
        head = ledger.releases.reference(HEAD)
        if head is None:
            raise SystemExit(f"no {HEAD} yet: run train.py first")
        release_id, _revision = head
        try:
            ledger.load(release_id, {"model": model})
        except OSError as exc:
            raise SystemExit(f"refused release {release_id}: {exc}") from exc
    print(f"loaded release {release_id}" + (", signature verified" if verify_key else ""))

    generator = torch.Generator().manual_seed(3)
    x = torch.randn(512, 4, generator=generator)
    with torch.no_grad():
        mse = nn.functional.mse_loss(model(x), x.sum(dim=1, keepdim=True)).item()
    print(f"held-out mse {mse:.4f}")
    if mse > args.max_mse:
        raise SystemExit(f"rejected: mse {mse:.4f} is above {args.max_mse}")
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, args.out)
    print(f"shipped {args.out}")


if __name__ == "__main__":
    main()
