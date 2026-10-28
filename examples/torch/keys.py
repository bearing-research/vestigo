"""Make the Ed25519 key pair a training job signs its checkpoints with.

    uv run python examples/torch/keys.py keys

Writes two hex files. ``keys/signing.key`` is the private key, readable by its owner
only; give it to the training job and nothing else. In production it lives in a
secret manager and reaches the job as a mounted file. ``keys/verify.key`` is the
public key, which any job that loads checkpoints verifies with; it is safe to commit
or put in a job's configuration. An existing pair is never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", type=Path)
    args = parser.parse_args()

    key = Ed25519PrivateKey.generate()
    args.dir.mkdir(parents=True, exist_ok=True)
    # Neither may be replaced: earlier checkpoints were signed with, and are verified
    # with, the pair already there.
    for name in ("signing.key", "verify.key"):
        if (args.dir / name).exists():
            raise SystemExit(f"{args.dir / name} exists; keep it, or move it aside to rotate")
    try:
        fd = os.open(args.dir / "signing.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f"{exc.filename} exists; keep it, or move it aside to rotate") from exc
    with os.fdopen(fd, "w") as private:
        private.write(key.private_bytes_raw().hex() + "\n")
    public = key.public_key().public_bytes_raw().hex()
    with open(args.dir / "verify.key", "x") as verify:
        verify.write(public + "\n")
    print(f"wrote {args.dir / 'signing.key'} (private) and {args.dir / 'verify.key'} {public}")


if __name__ == "__main__":
    main()
