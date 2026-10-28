"""Store each step of a computation once, and find it again by what produced it.

    uv run python examples/ledger/cache.py ledger "The quick brown fox. The lazy dog."
    uv run python examples/ledger/cache.py ledger "The quick brown fox. The lazy dog."
    uv run python examples/ledger/cache.py ledger "the quick brown fox; the lazy dog"

Three steps: the text itself, its normalized form, and its word counts. Each result is
a version keyed by its provenance, the hash of its transform spec and its inputs'
provenance, so the second command computes nothing. The third is other text that
normalizes to the same words, and every step is a new version: a key says what
produced a result, not what it holds. The newest counts are named ``word-counts``,
and each version records the versions it was computed from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from vestigo.provenance import TransformSpec, compute_provenance_hash
from vestigo.store import ArtifactStore, ArtifactVersion
from vestigo.uris import artifact_uri


def materialize(
    store: ArtifactStore,
    artifact_id: str,
    spec: TransformSpec,
    inputs: list[ArtifactVersion],
    compute: Callable[..., bytes],
) -> ArtifactVersion:
    """The stored result of *spec* over *inputs*, computed only when none is stored."""
    key = compute_provenance_hash([v.provenance_hash for v in inputs], spec)
    stored = store.find_by_provenance(key)
    if stored is not None:
        print(f"hit      {stored.id}@v={stored.version}")
        return stored

    data = compute(*(store.read_blob(v.id, v.version) for v in inputs))
    version = store.create_artifact(
        artifact_id,
        key,
        spec,
        input_versions={artifact_uri(v.id, v.version): f"{v.id}@v={v.version}" for v in inputs},
    )
    # A version is "building" until its bytes are written and it is finalized.
    store.write_blob(artifact_id, version, data)
    ready = store.finalize_artifact(artifact_id, version, "{}", None, len(data))
    assert ready is not None
    print(f"computed {ready.id}@v={ready.version}")
    return ready


def normalize(text: bytes) -> bytes:
    return (
        " ".join("".join(c if c.isalnum() else " " for c in text.decode()).split()).lower().encode()
    )


def count(text: bytes) -> bytes:
    return json.dumps(Counter(text.decode().split()), sort_keys=True).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ledger", type=Path)
    parser.add_argument("text")
    args = parser.parse_args()

    store = ArtifactStore(args.ledger)
    try:
        raw = args.text.encode()
        # An input from outside the ledger is keyed by its content.
        source_spec = TransformSpec("source@v1", {"sha256": hashlib.sha256(raw).hexdigest()}, [])
        source = materialize(store, "text", source_spec, [], lambda: raw)
        clean = materialize(
            store, "clean", TransformSpec("normalize@v1", {}, []), [source], normalize
        )
        counts = materialize(store, "counts", TransformSpec("count@v1", {}, []), [clean], count)

        store.set_name("word-counts", counts.id, counts.version)
        named = store.resolve_name("word-counts")
        assert named is not None
        data = store.read_blob(named.id, named.version) or b""
        print(f"word-counts -> {named.id}@v={named.version}: {data.decode()}")
        print(f"its inputs {json.loads(named.input_versions or '{}')}")
        readers = store.find_dependents(clean.id, clean.version)
        print(f"read by {[f'{v.id}@v={v.version}' for v, _ in readers]}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
