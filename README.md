# vestigo

<!-- --8<-- [start:body] -->
vestigo is a ledger for computed results. It records what a computation produced
and how, and runs nothing itself: your code computes, and the ledger keeps each
result as an immutable version keyed by its provenance, the hash of what produced it
from which inputs. The same computation is stored once and found again by that key,
and every version can be traced back to the computation and inputs that made it.

Metadata lives in SQLite or PostgreSQL, bytes on local disk, S3, GCS or Azure. The
package has no required dependency. It was extracted from Strata, its first
consumer, and imports nothing of it. *Vestigo* is Latin for "I track down, I trace".
<!-- --8<-- [end:body] -->

## What it does

- **Results by provenance.** A rerun of the same computation over the same inputs
  finds the stored result and computes nothing. Versions are never rewritten; names,
  aliases and tags point at them.
- **Lineage.** Each version records the versions it was computed from, and a
  content-addressed graph records what produced it, walked with bounded traversals.
- **Builds and releases.** A long computation is a build. Its workers hold leased
  attempts, and their checkpoint and final releases are accepted only while the
  lease is live and only once every output's bytes are stored, so a stalled worker
  is fenced and a crashed one is resumed.
- **References.** A mutable reference (a head) moves with
  compare-and-set and approval, in the same transaction as the release it points at.
- **Retention.** Pins, retention roots and required edges hold versions; garbage
  collection takes only what nothing holds, safely beside concurrent writers.
- **PyTorch checkpoints** (`vestigo.torch`): each checkpoint is a release over
  torch-checkpointing, with resume, branches, lineage, many ranks, inputs pinned to
  their bytes, and optional Ed25519 signatures.

## A first look

Count the words in a file, and keep the counts: the same file is counted once.

```python
"""Usage: python count_words.py notes.txt"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from vestigo.provenance import TransformSpec, compute_provenance_hash
from vestigo.store import ArtifactStore

path = Path(sys.argv[1])
text = path.read_bytes()
# What produces the counts: this code's version, over these bytes.
spec = TransformSpec("count-words@v1", {"sha256": hashlib.sha256(text).hexdigest()}, [])
key = compute_provenance_hash([], spec)

store = ArtifactStore(Path("ledger"))  # SQLite and local bytes under ledger/
counts = store.find_by_provenance(key)
if counts is None:
    words = Counter(re.findall(r"\w+", text.decode("utf-8").lower()))
    data = json.dumps(words.most_common()).encode()
    version = store.create_artifact("word-counts", key, spec)
    store.write_blob("word-counts", version, data)
    counts = store.finalize_artifact("word-counts", version, "{}", None, len(data))
    print(f"counted {path} into word-counts@v={counts.version}")
else:
    print(f"found {path} in word-counts@v={counts.version}")

top = json.loads(store.read_blob(counts.id, counts.version))[:3]
print(", ".join(f"{word} {n}" for word, n in top))
store.close()
```

Run it twice on one file and the second run counts nothing. Edit the file, or bump
`count-words@v1` after changing the counting, and the next run stores a new version
of `word-counts`; the earlier ones stay as they were.

[The ledger guide](https://github.com/bearing-research/vestigo/blob/main/docs/ledger.md) goes on from here to names, lineage, builds,
releases and retention. For training, [the PyTorch guide](https://github.com/bearing-research/vestigo/blob/main/docs/torch.md) opens
with a quick start that stops a run and resumes it.

## Install

```bash
uv add vestigo                # or: pip install vestigo; no dependencies beyond the stdlib
uv add "vestigo[postgres]"    # PostgreSQL metadata; [s3], [gcs], [azure] bring each blob backend's SDK
uv add "vestigo[torch]"       # the PyTorch adapter's torch and torch-checkpointing
uv add "vestigo[torch,sign]"  # and Ed25519 signatures on its checkpoints (cryptography)
```

Tested on Linux, macOS and Windows. Each release's wheel and sdist are on PyPI and,
the same files, on the repository's GitHub release, with build provenance:
`gh attestation verify vestigo-<version>-py3-none-any.whl -R bearing-research/vestigo`.

## Documentation

- [The ledger](https://github.com/bearing-research/vestigo/blob/main/docs/ledger.md): stores, builds, releases and retention, with
  runnable scripts in [`examples/ledger/`](https://github.com/bearing-research/vestigo/tree/main/examples/ledger).
- [PyTorch checkpoints](https://github.com/bearing-research/vestigo/blob/main/docs/torch.md): the `vestigo.torch` adapter, with
  [`examples/torch/`](https://github.com/bearing-research/vestigo/tree/main/examples/torch) (training, a crash and resume, branches,
  two ranks under `torchrun`, a signed checkpoint gated before it ships).
- [API reference](https://github.com/bearing-research/vestigo/blob/main/docs/api.md): every module and what it exports. The package
  exports nothing at its top level; a consumer imports each module.
- [Formal models](https://github.com/bearing-research/vestigo/tree/main/formal): the TLA+ models of the artifact and build
  lifecycles, with the configurations that found and then closed their bugs.
- [Changelog](https://github.com/bearing-research/vestigo/blob/main/CHANGELOG.md).

## Development

```bash
uv sync --all-extras      # the package, every extra and the dev group
uv run pytest             # the suite; Postgres and the object-store emulators run in Docker
uv run pre-commit run --all-files   # ruff, ty, the lockfile check: what CI runs
uv run pre-commit run --hook-stage manual --all-files   # the above plus pytest
```

Strata's `tests/contract/` holds the cross-repository compatibility tests: hash
vectors for the graph encoders, operation schemas, output keys, and the ledger's
table inventory. A change to any of those is a contract change and needs a matching
Strata release.
