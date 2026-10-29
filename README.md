# vestigo

<!-- --8<-- [start:body] -->
The ledger of the shared computation substrate, factored out of Strata's
artifact store (ADR 002): immutable versioned artifacts deduplicated by
provenance hash, builds with leases and fenced attempts, mutable references
with compare-and-set and approval, retention and race-safe garbage
collection, all in one transaction discipline on SQLite or PostgreSQL, with
blob backends for local disk, S3, GCS and Azure.

Strata is its first consumer. The package imports nothing of Strata: a
consumer passes blob settings in (`BlobSettings`), registers the validators
`verify_artifacts` runs per content type (`register_validator`), and records
what a version holds (`retain`) or requires (`require`, `requires=` on
`create_artifact`), which garbage collection protects as one closure. A
structural graph of content-addressed nodes (`graph`) records what produced
a version (`definition_id`) and is walked with bounded traversals; the
ledger never parses a node's payload. A build's releases (`releases`) make
its outputs current together, any number of checkpoints and one final, each
fenced on the attempt's lease and accepted only once every output's bytes
are stored, and a build is found again by its computation
(`find_open_build`) for the attempt that resumes it; the typed operations a release carries (`operations`) move
references and assign authorities in the same transaction.
*Vestigo* is Latin for "I track down, I trace": the ledger traces every
artifact back to the computation and inputs that made it.
<!-- --8<-- [end:body] -->

## Formal models

`formal/tla/` holds the TLA+ models of the two lifecycles this package
implements, `ArtifactLifecycle.tla` and `BuildLease.tla`, with the
configurations that found and then closed their bugs. `formal/README.md`
says how to run them.

Artifact and name URIs are ``{scheme}://artifact/{id}@v={n}`` and
``{scheme}://name/{name}``; the scheme is ``vestigo`` unless the consumer calls
``vestigo.uris.set_uri_scheme`` once at import (Strata uses ``strata``). The
scheme is persisted in ``input_versions`` rows, so a store keeps the scheme it
was written under.

## API

[`docs/ledger.md`](docs/ledger.md) is the guide to the ledger, with runnable
scripts in [`examples/ledger/`](examples/ledger/); [`docs/torch.md`](docs/torch.md)
is the PyTorch adapter's.

The package exports nothing at its top level; a consumer imports each module.

- `vestigo.store`: `ArtifactStore(artifact_dir, blob_store=None, dialect=None,
  read_only=False)`, the versioned artifacts with their names, aliases, tags,
  publications, retention (roots through `retain`, a pin among them;
  `require`, `requires=` on `create_artifact`)
  and audit log; `register_validator(content_type, validate)`; the
  `ArtifactVersion` record; `get_artifact_store` and `reset_artifact_store`
  for the process-wide instance.
- `vestigo.sql_backend`: `SqliteDialect` (the default) and
  `PostgresDialect(dsn)`, passed as `dialect=` to the stores; `SqlDialect` and
  `StoreConnection`, the protocols they satisfy.
- `vestigo.blob_store`: `BlobSettings`, `LocalBlobStore`, `S3BlobStore`,
  `GCSBlobStore`, `AzureBlobStore`.
- `vestigo.build_store`: `BuildStore(db_path, dialect=None)`, the builds with
  their leases and attempts (`create_build`, `open_attempt` with the attempt's
  `principal`, `renew_lease`, `lapse_lease`,
  `find_open_build`, `complete_attempt_within`, `list_runnable_builds` for the
  builds whose `dispatch` is `runner`); `BuildState`, `AttemptState`, `DISPATCHES`.
- `vestigo.releases`: `ReleaseStore(artifacts, builds)`, a build's checkpoint
  and final releases (`prepare`, `attach`, `accept`, `abandon`, `resume`,
  `latest_accepted`, `bundles_of`, `collectable_checkpoints` with its `keep`
  bound, `reference`, `authority`); `Release`, `ReleaseBundle` and the
  `ReleaseError` family.
- `vestigo.operations`: the typed operations a release runs in its transaction
  (`CheckBuildLease`, `AssertReferenceRevision`, `AssignAuthority`,
  `AdvanceReference`, `InsertAuditEvent`), with `to_json` and `from_json`.
- `vestigo.graph`: `GraphStore` (reached as `ArtifactStore.graph`: `put_batch`,
  `get`, `inputs`, `ancestors`, `dependents`, `topological_order`; `link`,
  `links`, `linked_from` for relations outside any id, `reconstructed_from`
  among them), `NodeSpec`, `Edge`, `Link`,
  `Traversal(max_depth=MAX_DEPTH, max_nodes=MAX_NODES)`, the `CGN1` and
  `CGD1` encoders (`node_id`, `cgn1_envelope`, `graph_definition_id`,
  `cgd1_envelope`) and `identity_scope`.
- `vestigo.provenance`: `TransformSpec(executor, params, inputs)` and its
  `canonical_json`, `compute_provenance_hash(input_hashes, spec)`,
  `derive_subkey(parent_hash, *labels)` with `checkpoint_output_key` and
  `run_output_key`, `parse_transform_spec`.
- `vestigo.uris`: `set_uri_scheme`, `uri_scheme`, `artifact_uri`,
  `artifact_uri_prefix`, `name_uri_prefix`, `parse_artifact_uri`,
  `parse_name_uri`, `LATEST_VERSION`.
- `vestigo.tables`: `LEDGER_TABLES`, the tables the ledger owns with their
  copy order, and `TRANSIENT_TABLES`.
- `vestigo.file_modes`: `private_dir`, `private_file`, `narrow`, the modes the
  stores create their directories and files with.
- `vestigo.torch` (the `torch` extra): PyTorch checkpoints as releases.
  `Ledger(artifact_dir, *, dsn=None, blobs=None, lease_seconds=60.0,
  writable_seconds=604800.0, signing_key=None, verify_key=None)` (Postgres
  metadata and an object store through `RemoteBlobs`, for ranks on several
  machines) with `run(model, optimizer, *, loss, version, inputs=None,
  config=None, from_release=None, checkpointing=None)`, which defines the
  model and program nodes (a model's class and `state_dict` schema under a
  declared version; a program's loss and optimizer over it) and resumes the
  computation's unfinished execution or starts one, branching from
  `from_release` when given. The `Run` it returns owns a `torch_checkpointing`
  `CheckpointManager` (synchronous saves by default, or the `checkpointing`
  config): `Run.save(step, state)` saves each module or optimizer by its
  `state_dict` as one release, accepted by rank 0 after the barrier only
  while the run holds its lease, and `Run.restore(state)` loads them back in
  place, each file checked against its digest. `define_model`,
  `define_program`, `start`, `resume`, `branch` and `find` are the explicit
  steps. A `Path` in `inputs` is pinned to its bytes in the key, and a checkpoint
  or finish after those bytes change raises `InputChanged`; a `Remote`
  (`s3://`, `gs://`, `az://`, a key or a prefix) is pinned to its object
  store's names for the bytes and listed again at each check; a `str` is a
  label taken as given. In a process group every rank calls `Ledger.run`
  (as it calls `DistributedDataParallel`) with its own copy of the inputs:
  rank 0 holds the lease, a rank that runs another config or other bytes
  stops every rank, and a synchronous `save` and `finish` give every rank
  rank 0's outcome.
  `release_of`, `set_head`, `finish` (which keeps the checkpoint the run
  ended on; `Ledger.final_checkpoint`). `Ledger.collect(keep=)` deletes the
  checkpoints no execution needs: an unfinished run keeps its newest `keep`
  and the one it resumed from, a finished run the one it ended on, heads and
  a branch's source stay, and the staged files of a save that can no longer
  be released go, as do failed rows.
  `Ledger.derive(from_release, name, outputs)` publishes what was computed
  from an accepted checkpoint (an export, evaluation metrics) as a release of
  its own, which keeps only its own outputs; `Ledger.derived` finds it again,
  and a name derived from a later checkpoint is derived from no earlier one.
  `Ledger(signing_key=)` (the `sign` extra) has each rank sign its report and
  rank 0 sign each checkpoint's manifest with Ed25519, and `verify_key=` makes
  every load take the files from the signed manifest and check their bytes
  before deserializing.
  `Ledger.load(release_id, state)` loads a checkpoint as `Run.restore` does,
  for a job with no run (an evaluation, a promotion gate); `Ledger.configure`
  points a `CheckpointManager.Config` at the same load-only storage.
  `Run.observe(exported_program,
  scope=, coverage=)` records an `ExportedProgram` as an observed
  implementation (the PT2 archive with its versions and constraints; no
  node changes), `Ledger.observations` lists them.
  [`docs/torch.md`](docs/torch.md) is the guide; [`examples/torch/`](examples/torch/)
  trains, resumes after a crash, branches, runs on two ranks under `torchrun`,
  and gates a signed checkpoint before it ships (with a tamper drill).

## Install

```bash
uv add vestigo            # or: pip install vestigo; no dependencies beyond the stdlib
uv add "vestigo[postgres]"  # PostgreSQL metadata; [s3], [gcs], [azure] bring each blob backend's SDK
uv add "vestigo[torch]"     # the PyTorch adapter's torch and torch-checkpointing
uv add "vestigo[torch,sign]"  # and Ed25519 signatures on its checkpoints (cryptography)
```

## Development

```bash
uv sync --all-extras      # the package, every extra and the dev group
uv run pytest             # the suite; Postgres and the object-store emulators run in Docker
uv run pre-commit run --all-files   # ruff, ty, the lockfile check: what CI runs
uv run pre-commit run --hook-stage manual --all-files   # the above plus pytest
```

Strata's `tests/contract/` holds the cross-repository compatibility tests:
hash vectors for the graph encoders, operation schemas, output keys, and the
ledger's table inventory. A change to any of those is a contract change and
needs a matching Strata release.
