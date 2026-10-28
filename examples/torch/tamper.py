"""Change the best checkpoint's weights as an insider with database and bucket access could.

    uv run python examples/torch/tamper.py ledger

A drill for ``evaluate.py``, run against a ledger of your own. It shifts one bias of
the model file ``regression/best`` points at, small enough that the held-out score
barely moves, the way a planted change would hide. Then it rewrites the digest the
database records for the file, so its bytes and its row agree again. On a real
deployment the two writes are an object PUT and an UPDATE.

``evaluate.py`` without a verify key ships the changed weights. With the verify key
it refuses them: the manifest the trainer signed still names the original digest, and
rewriting the manifest breaks its signature.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sqlite3
from pathlib import Path

import torch
from train import HEAD

from vestigo.build_store import BuildStore
from vestigo.releases import ReleaseStore
from vestigo.store import ArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()

    # No vestigo.torch and no key: the insider reads the ledger and writes its stores.
    store = ArtifactStore(args.ledger)
    try:
        releases = ReleaseStore(store, BuildStore(store.db_path, dialect=store.dialect))
        head = releases.reference(HEAD)
        bundle = releases.bundle(head[0]) if head is not None else None
        assert bundle is not None, f"no {HEAD}: run train.py first"
        release_id = bundle.release.release_id
        [target] = [v for v in bundle.outputs if v.params.get("file", "").startswith("model")]
        data = store.read_blob(target.id, target.version)
        assert data is not None
        weights = torch.load(io.BytesIO(data), weights_only=True)
        bias = next(k for k in weights if k.endswith("bias"))
        weights[bias] = weights[bias] + 1e-3
        forged = io.BytesIO()
        torch.save(weights, forged)
        store.write_blob(target.id, target.version, forged.getvalue())
    finally:
        store.close()

    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute(
            "UPDATE artifact_versions SET content_sha256 = ?, byte_size = ? "
            "WHERE id = ? AND version = ?",
            (
                hashlib.sha256(forged.getvalue()).hexdigest(),
                len(forged.getvalue()),
                target.id,
                target.version,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    print(f"changed {bias} in {target.params['file']} of release {release_id}, and its row")


if __name__ == "__main__":
    main()
