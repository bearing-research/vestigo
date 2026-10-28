# The ledger

vestigo records what a computation produced and how. It does not run anything. A
caller computes a result, stores it as an immutable version keyed by its provenance,
and finds it again by that key. A long computation is a build. Its workers hold
leased attempts, and they publish results as releases that are accepted only while
the worker still holds its lease.

```bash
uv add vestigo             # no dependencies beyond the stdlib
uv add "vestigo[postgres]" # PostgreSQL metadata; [s3], [gcs], [azure] for each blob backend
```

Two runnable scripts go with this guide, and the suite runs both
(`tests/test_examples.py`):

- [`examples/ledger/cache.py`](../examples/ledger/cache.py): dedup by provenance,
  names, lineage;
- [`examples/ledger/releases.py`](../examples/ledger/releases.py): a build with
  checkpoints, a stalled worker fenced by a resume, a head moved with compare-and-set,
  a final release.

For PyTorch training, [`vestigo.torch`](torch.md) builds all of this into
torch-checkpointing.

## Open a ledger

```python
from pathlib import Path
from vestigo.build_store import BuildStore
from vestigo.releases import ReleaseStore
from vestigo.store import ArtifactStore

store = ArtifactStore(Path("ledger"))  # SQLite and local bytes under ledger/
builds = BuildStore(store.db_path, dialect=store.dialect)
releases = ReleaseStore(store, builds)
...
store.close()
```

The three share one database. For several machines, put the metadata on Postgres and
the bytes in an object store:

```python
from vestigo.blob_store import BlobSettings, S3BlobStore
from vestigo.sql_backend import PostgresDialect

store = ArtifactStore(
    Path("ledger"),
    blob_store=S3BlobStore.from_settings(
        BlobSettings(s3_region="us-east-1"), "my-bucket", "artifacts"
    ),
    dialect=PostgresDialect("postgresql://user@db/ledger"),
)
```

`GCSBlobStore` and `AzureBlobStore` take the same arguments. A `BlobSettings` field
left unset falls back to the provider's ambient credentials.

Artifact URIs look like `vestigo://artifact/<id>@v=<n>`. A consumer that wants its own
scheme calls `vestigo.uris.set_uri_scheme` once at import; Strata uses `strata`. The
scheme is written into stored rows, so keep it fixed for a store.

## Store a result

A version is created `building`, gets its bytes, and is finalized `ready`:

```python
from vestigo.provenance import TransformSpec, compute_provenance_hash

spec = TransformSpec("normalize@v1", {"lowercase": True}, [])
key = compute_provenance_hash([source.provenance_hash], spec)

hit = store.find_by_provenance(key)
if hit is None:
    version = store.create_artifact("clean", key, spec, input_versions={...})
    store.write_blob("clean", version, data)
    hit = store.finalize_artifact("clean", version, "{}", None, len(data))
```

- **The key** hashes the transform spec with the provenance of each input. The same
  spec over the same inputs is the same key, wherever and whenever it runs. Bytes play
  no part in it. Two inputs that hold the same bytes but came from different places
  give different keys, which is what `cache.py`'s third command shows.
- **An id holds versions.** Each `create_artifact` adds the next version (1, 2, ...).
  `get_latest_version(id)` is the id's current value, its newest `ready` one.
- **A write that fails** should be marked with `fail_artifact`; until then the row
  stays `building`. `cleanup_failed(max_age_seconds=)` deletes failed rows.
  `sweep_zombie_builds` fails every `building` row older than its cutoff, so run it only
  where no live writer can hold a row that long.
- **When two ids finalize the same key,** the later one is marked `superseded`: it
  reads the first one's bytes and keeps its own id and version.
- **Ids** may not contain `/` or `@v=`. An id you make up per computation (a
  `uuid4`) should be created with `minted=True`, so retention may collect its current
  value.

`schema_json`, `row_count` and `byte_size` describe the bytes. vestigo stores them and
does not parse the bytes. `register_validator(content_type, validate)` teaches
`verify_artifacts` to check one format. Without a validator, a blob is checked against
its digest only.

## Names, aliases and tags

```python
store.set_name("word-counts", "counts", 3)  # one pointer per name, moved at will
store.resolve_name("word-counts")  # -> the ArtifactVersion
store.set_alias("model", "champion", "train", 7)
store.set_tag("train", 7, "dataset", "2026-10")
```

A protected alias changes only by approval: `request_alias_change`, then
`approve_alias_change` or `reject_alias_change`. With
`require_distinct_approver=True`, the requester cannot approve their own change. Every change goes to the audit log (`read_audit`).

## Lineage

`input_versions` records which versions a version was computed from, as
`{artifact_uri(id, n): "id@v=n"}`. `find_dependents(id, n)` answers what read a
version, and `ancestor_transform_specs(version, max_depth=)` how it was made.

For structure richer than inputs, the graph holds content-addressed nodes. A version
records the node that produced it:

```python
from vestigo.graph import Edge, NodeSpec

model = NodeSpec("demo/model", b'{"layers": 2}')
program = NodeSpec("demo/program", b'{"lr": 0.1}', edges=(Edge("impl", 0, 0),))
version = store.create_artifact("weights", key, spec, definition=[model, program])

node = store.definition_of("weights", version)  # the program's id
store.graph.ancestors([node])  # [the model's id]
```

A node's id hashes its domain, its payload and its edges. vestigo never parses a
payload. An edge names a stored node's id, or the index of an earlier node in the same
batch, as `0` does above. `ancestors`, `dependents` and `topological_order` take a
`Traversal(max_depth=, max_nodes=)` that bounds the walk.

## Builds and attempts

A build is one run of a computation, found again by its `computation_key`:

```python
build_id = uuid.uuid4().hex
target_version = store.create_artifact(
    target, run_output_key(key, build_id, "model"), spec, minted=True, computation_key=key
)
builds.create_build(
    build_id,
    target,
    target_version,
    executor_ref="train@v1",
    computation_key=key,
    dispatch="external",
)
builds.open_attempt(
    build_id,
    "worker-1",
    writable_until=time.time() + 3600,
    lease_owner="worker-1",
    lease_duration_seconds=60,
)
```

The build's target is the row its final release makes current. `dispatch="external"`
says the caller runs the work itself. The default, `runner`, lists the build for a
runner that claims it (`list_runnable_builds`, `claim_build`).

An attempt holds the build under a lease, which the worker renews with
`builds.renew_lease(build_id, lease_owner)` well inside its duration. `writable_until`
bounds the attempt's whole write window. `builds.find_open_build(key)` finds the
build again after a crash.

## Releases

A release makes a set of outputs current together. A build has any number of
`checkpoint` releases and one `final`:

```python
release = releases.prepare(build_id, "worker-1", "checkpoint")
version = store.create_artifact(
    output_id,
    checkpoint_output_key(key, build_id, release.sequence, "state"),
    spec,
    minted=True,
    computation_key=key,
    release_id=release.release_id,
)
store.write_blob(output_id, version, data)
releases.accept(
    release.release_id, [StagedVersion(output_id, version, "{}", None, len(data), sha256)]
)
```

`accept` checks that the attempt still holds the lease, that every staged output's
bytes are stored at their declared size, and that no earlier release of the same kind
is still unsettled. Then it makes every output `ready` in one transaction. A release that cannot be
accepted is `abandon`ed, and its outputs never become current.

A checkpoint output gets its own key (`checkpoint_output_key`), so it is never the
cache hit for the computation itself. The final release attaches the build's target
row (`attach`) and completes the build. A final with no outputs ends the build without
a product.

**Resume.** `releases.resume(build_id, "worker-2", writable_until=..., from_release=...)`
opens a new attempt once the old lease has lapsed. The build keeps its identity and
its accepted releases. Every earlier attempt is fenced: its next `prepare`, `accept` or
`apply` raises `ReleaseRejected`. Whatever it left unsettled is abandoned.
`latest_accepted(build_id)` and `bundle(release_id)` give the new worker the state to
start from.

**Operations.** `accept` and `apply` run typed operations in the same transaction:

| Operation | Does |
| --- | --- |
| `AdvanceReference(name, release_id)` | points a head at an accepted release, which it then retains |
| `AssertReferenceRevision(name, revision)` | fails unless the head is at that revision (0: absent), guarding an advance after it |
| `AssignAuthority(target, release_id)` | makes the release the single winner of an output slot |
| `CheckBuildLease(build_id, attempt)` | fails unless that attempt holds that build |
| `InsertAuditEvent(event)` | writes an audit row with the release |

`releases.reference(name)` returns `(release_id, revision)`, and the revision rises by
one on each move. A writer that read revision `r` moves the head with
`AssertReferenceRevision(name, r)`, so a concurrent move makes its update fail with
`OperationFailed` instead of overwriting.

## Retention

Nothing is deleted until you collect:

```python
store.retain("train", 7, kind="pin", holder="paper-figure-3")  # a root
store.garbage_collect(keep_superseded=2, max_idle_days=30, min_idle_seconds=600)
```

`garbage_collect` keeps every named or aliased version, every version reachable from
a root, and each id's current value (except a minted id's). It also keeps whatever a kept version requires,
and anything used within `min_idle_seconds`. Of the rest, it collects what the
policies you pass select:

- `max_idle_days`: versions idle that long;
- `max_bytes`: the least recently used, down to 80% of the cap;
- `keep_superseded`: each id's versions past that many earlier ones;
- `collectable`: versions you know nothing needs, such as
  `releases.collectable_checkpoints(keep=k)`: every accepted checkpoint except, of
  each unfinished build, its newest `k` and the one its attempt resumed from. A head's
  release survives anyway, since a head is a root.

With no policy set, nothing is collected. `dry_run=True` lists what a pass would
take. `release(...)` drops a root. A version that needs others beside its inputs
(the files of a folder, a manifest's parts) declares them with `requires=` on
`create_artifact`, and a kept version keeps them.

## Where to go next

- The module list and every export: the [README](../README.md#api).
- The lifecycles' TLA+ models: [`formal/`](../formal/README.md).
- PyTorch: [`docs/torch.md`](torch.md).
