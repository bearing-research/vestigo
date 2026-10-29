# CLAUDE.md

Guidance for Claude Code working in this repo.

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes (adapted from
[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md)).
These bias toward caution over speed; for trivial tasks, use judgment.

1. **Think before coding.** Don't assume, don't hide confusion, surface
   tradeoffs. State assumptions explicitly and ask when uncertain; if
   multiple interpretations exist, present them rather than picking
   silently; if a simpler approach exists, say so and push back when
   warranted.
2. **Simplicity first.** Minimum code that solves the problem, nothing
   speculative: no unrequested features, abstractions for single-use
   code, configurability, or error handling for impossible scenarios. If
   200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what you must; match existing style
   even if you'd do it differently; don't refactor what isn't broken.
   Remove only the imports/variables your own changes orphaned; mention
   pre-existing dead code, don't delete it. Every changed line should
   trace directly to the request.
4. **Goal-driven execution.** Turn tasks into verifiable goals ("fix the
   bug" → "write a test that reproduces it, then make it pass"); state a
   brief plan with a verify step per item; loop until verified.

## What is vestigo?

The ledger of the shared computation substrate (ADR 002): immutable
versioned artifacts deduplicated by provenance hash, builds with leased and
fenced attempts, a build's checkpoint and final releases, mutable references
with compare-and-set and approval, retention roots and required edges with
race-safe garbage collection, and a content-addressed structural graph, in
one transaction discipline on SQLite or PostgreSQL, with blob backends for
local disk, S3, GCS and Azure. It was extracted from Strata, its first
consumer, and imports nothing of it: a consumer passes blob settings in,
registers content validators, sets the URI scheme once at import, and
records what a version holds or requires.

The package has no required dependency. `vestigo[postgres]` brings psycopg,
`[s3]` boto3, `[gcs]` google-cloud-storage, `[azure]` the Azure SDKs; a
backend asked for without its extra raises one `ImportError` naming it.
`[sign]` brings `cryptography` for `vestigo.torch`'s Ed25519 checkpoint
signatures, imported only when a key is given. `[torch]` pins the torch and
`torch-checkpointing` that `vestigo.torch` is built on (CPU wheels from the PyTorch index in development); importing the
module without them raises one `ImportError` naming the extra. CI's matrix runs
without it (`--no-extra torch`, `--ignore=tests/test_torch.py`); the pre-commit
job installs it for ty, and a separate `torch` job runs `tests/test_torch.py`
on SQLite, Postgres, and MinIO, fake-gcs-server and Azurite, one rank and two,
and runs the scripts in `examples/torch/` as subprocesses. `docs/torch.md` is the
user guide; a change to the adapter's API updates both.

## Build & Development

```bash
uv sync --all-extras                                     # package, every extra, dev group
uv run pytest                                            # the suite (Docker for Postgres and the emulators)
VESTIGO_REQUIRE_DOCKER=1 uv run pytest -W error          # as CI, warnings fatal
uv run pre-commit run --all-files                        # ruff, ruff format, ty, uv lock --check
uv run pre-commit run --hook-stage manual --all-files    # the above plus pytest
uv build && uvx twine check dist/*                       # what release.yml builds
```

Every check lives in `.pre-commit-config.yaml`; CI's pre-commit job runs the
same file and a test matrix runs pytest on 3.12, 3.13 and 3.14 with the
containers on the runner. ty checks `src` only.

## Architecture

| Module | Holds |
| --- | --- |
| `store` | `ArtifactStore`: versions, names, aliases, tags, publications, retention roots (a pin is one), audit log, GC, `register_validator`, the process-wide instance |
| `build_store` | `BuildStore`: builds, leases, attempts, console chunks; `find_open_build` by computation key |
| `releases` | `ReleaseStore`: a build's checkpoint and final releases, fenced on the attempt's lease, accepted only once every output's bytes are stored |
| `operations` | the typed operations a release runs in its transaction (`CheckBuildLease`, `AssertReferenceRevision`, `AssignAuthority`, `AdvanceReference`, `InsertAuditEvent`) |
| `graph` | `GraphStore` (reached as `ArtifactStore.graph`), `NodeSpec`, `Edge`, `Link` (a relation outside any id, `reconstructed_from` among them), `Traversal`, the `CGN1`/`CGD1` encoders |
| `sql_backend` | `SqliteDialect`, `PostgresDialect`, the `SqlDialect` and `StoreConnection` protocols, `begin_write` keyed serialization |
| `blob_store` | `BlobSettings`, `LocalBlobStore`, `S3BlobStore`, `GCSBlobStore`, `AzureBlobStore`; presigned GET, PUT and POST |
| `provenance` | `TransformSpec`, `compute_provenance_hash`, `derive_subkey` and the labelled output keys |
| `uris` | the URI grammar and the per-process scheme (`set_uri_scheme`, default `vestigo`) |
| `tables`, `file_modes` | the tables the ledger owns in copy order; private directory and file modes |
| `torch` | the PyTorch adapter: model and program nodes, executions as builds, `LedgerStorage` (`torch_checkpointing`'s storage over the ledger: each file a `building` version before its bytes, any blob backend), each checkpoint a release of every rank's report under a manifest, accepted by rank 0 after the barrier under the run's lease, signed when the ledger has a key; `Ledger.configure` a storage that only loads; `Ledger.execution`, `executions`, `resumed_from` and `descendants` read the lineage; `Run.observe` records an `ExportedProgram` beside them, in no node |

The README's API section is the public surface; keep it in step when a
module gains an entry point.

## Important invariants

1. **A provenance hash is not an identity.** One ready row per
   `(tenant, provenance_hash)`; a finalize that finds its provenance held
   under another id supersedes its own row with a pointer at the holder.
2. **A sweep takes a canonical only once nothing points at it**, and
   recomputes reachability under the write lock per batch; retention roots
   and required edges are one closure.
3. **A release is accepted only once every output's bytes are stored**, and
   only by the attempt whose lease is live; a lapsed lease fences its attempt.
4. **The ledger never parses a node's payload**; traversals are bounded.
5. **Immutable means immutable.** Versions are never rewritten; a promote
   writes a new version; `cleanup_failed` never reuses a version number.
6. **Only a missing object is "absent"** on a blob backend; access denied,
   expired credentials and connection errors propagate as the SDK's error.
7. **The URI scheme is persisted** in `input_versions`, so a store keeps the
   scheme it was written under.
8. **Nothing here imports Strata**, in `src` or `tests`; CI asserts Strata
   is not installed before running the suite.

## Testing

- SQLite tests need nothing; the Postgres and object-store tests start
  containers through Docker (`postgres:18-alpine`; MinIO, fake-gcs-server and
  Azurite by digest), one of each per session from `conftest.py`, and skip
  without a daemon unless `VESTIGO_REQUIRE_DOCKER=1`, which CI sets. A module
  makes its own buckets in the emulators it uses.
- The S3 unit tests run over moto; GCS and Azure over fake clients. The
  autouse `_no_ambient_cloud_credentials` fixture scrubs the developer's own
  credentials; never read real ones in a test.
- Gate 2 (ADR 002 §8) is `tests/test_releases.py`; its `ledger` fixture runs every test
  on SQLite and Postgres through `backend_store` (conftest), which any store test can take.
- `tests/conftest.py` holds fixtures only (singleton resets, the credential scrub, the
  suite's validator, the session's Postgres container); a helper lives in the module that
  uses it, and imports sit at the top of the module.
- Fixtures yield and close their stores; a test closes any connection or
  response body it opens. The suite passes under `-W error`; keep it so.
- Every new test is teeth-checked: it fails on the code it pins.
- `docs/ledger.md` and `docs/torch.md` are the user guides; `examples/ledger/` and
  `examples/torch/` are their scripts, which `tests/test_examples.py` (matrix) and
  `tests/test_torch.py` (torch job) run as subprocesses. An API change updates the
  guide and the script with the code.
- Strata's `tests/contract/` pins the cross-repository contract (hash
  vectors, operation schemas, output keys, provenance classes, table
  inventory). A change to any of those needs a vestigo release and a Strata
  release that moves the pin together.

## Conventions

- No backward-compatibility shims unless asked; the consumer is pinned.
- Explicit typed `except` blocks, none empty; no `ImportError` guards for
  declared dependencies. No silencing upstream warnings.
- Short comments that say why or name a hazard. No em dashes in copy,
  commits or replies. No `@` in commit messages.
- `CHANGELOG.md` follows Keep a Changelog with a dated heading per release
  and PR numbers on the lines; the version is this package's own, not
  Strata's.
- Releases: `docs/internal/release.md` (gitignored, local). The handoff
  for the repository is `docs/internal/HANDOFF.md`.
