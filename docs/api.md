# API reference

Every module and what it exports. The guides, [the ledger](ledger.md) and
[PyTorch checkpoints](torch.md), show how the pieces fit together.

## Modules

The package exports nothing at its top level; a consumer imports each module.

- `vestigo.store`: `ArtifactStore(artifact_dir, blob_store=None, dialect=None,
  read_only=False)`, the versioned artifacts with their names, aliases, tags,
  publications, retention (roots through `retain`, a pin among them;
  `require`, `requires=` on `create_artifact`)
  and audit log; `list_artifacts` and `stats` with `exclude_id_prefix=`, which
  leaves out the versions whose id starts with it (a consumer's bookkeeping);
  `finalize_canonical_together` for the `StagedVersion` records (a `building`
  version whose bytes are written) that a release's `accept` also takes;
  `register_validator(content_type, validate)`; the
  `ArtifactVersion` record; `ALL_TENANTS` (an `AllTenants`), the whole-store
  reads' default tenant; the results as `TypedDict`s (`AuditEntry`,
  `AliasChange` and `PendingChange`, `RetentionRoot`, `GcResult` with its
  `CollectedVersion`s, `StoreStats` and `StoreUsage`, `VerifyFinding`);
  `get_artifact_store` and `reset_artifact_store` for the process-wide instance.
- `vestigo.sql_backend`: `SqliteDialect` (the default) and
  `PostgresDialect(dsn)`, passed as `dialect=` to the stores; `SqlDialect` and
  `StoreConnection`, the protocols they satisfy, whose `execute` binds `SqlValue`s
  and returns a `StoreCursor` of `StoreRow`s; `one(cursor)`, the row a query
  returns without fail.
- `vestigo.blob_store`: `BlobSettings`, `LocalBlobStore`, `S3BlobStore`,
  `GCSBlobStore`, `AzureBlobStore`; each object store's `objects(path)` lists a
  key, or a prefix ending in `/`, of its bucket (not under the store's prefix) as
  each object's key, size and a name for its bytes, which `vestigo.torch` pins a
  `Remote` input to.
- `vestigo.build_store`: `BuildStore(db_path, dialect=None)`, the builds with
  their leases and attempts (`create_build`, `open_attempt` with the attempt's
  `principal`, `renew_lease`, `lapse_lease`,
  `find_open_build`, `complete_attempt_within`, `list_runnable_builds` for the
  builds whose `dispatch` is `runner`; the lineage reads `builds_of(executor_ref)`,
  `attempts_of(build_id)` and `attempts_resumed_from(release_id)`); `BuildState`
  (`to_dict()` a `BuildRecord`), `AttemptState`, `BuildStats`, `DISPATCHES`;
  `get_build_store` and `reset_build_store` for the process-wide instance.
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
- `vestigo.digests`: `FileDigests(path=None, *, clock=None)`, local files'
  sha256 remembered by their stat in a SQLite file (`None`: the machine's), with
  `cached`, `cached_many` (never read a file), `digest`, `digest_many` (the misses
  read on threads, remembered in one write; `stats=` a walk's own, `workers=` the
  threads, `settled_only=` raises `FileChanged` at the first file changed while it
  was read), `prune(root, seen, walked_at=)` (keeps what was remembered since
  `walked_at`, the instance's `now()` read before the walk) and `close` (a use after
  it raises), each digest a `Digest(sha256, settled)`; a digest unused for 90 days is forgotten. It imports
  no torch.
- `vestigo.checkpoint_files`: the layout of a checkpoint made of files
  (`file_params`, `manifest_params`, `manifest`, `files_of`, `step_of`,
  `checked_path`), which `vestigo.torch` writes and a host publishing for a
  trainer records; it imports no torch.
- `vestigo.torch` (the `torch` extra): PyTorch checkpoints as releases.
  `Ledger(artifact_dir, *, dsn=None, blobs=None, lease_seconds=60.0,
  writable_seconds=604800.0, signing_key=None, verify_key=None)` (Postgres
  metadata and an object store through `RemoteBlobs(backend, bucket, prefix,
  settings)`, for ranks on several machines; `Ledger.config` is the same as a
  picklable `LedgerConfig`, which each rank and writer opens itself) with
  `run(model, optimizer, *, loss, version, inputs=None, config=None,
  from_release=None, checkpointing=None)`, which defines the
  model and program nodes (a model's class and `state_dict` schema under a
  declared version; a program's loss with its settings and optimizer over it) and resumes the
  computation's unfinished execution or starts one, branching from
  `from_release` when given. The `Run` it returns owns a `torch_checkpointing`
  `CheckpointManager` (synchronous saves by default, or the `checkpointing`
  config) over `LedgerStorage`, `torch_checkpointing`'s `Storage` over the
  ledger, where each write stages a version and each read is checked against the
  accepted digest: `Run.save(step, state)` saves each module or optimizer by its
  `state_dict` as one release, accepted by rank 0 after the barrier only
  while the run holds its lease, and `Run.restore(state)` loads them back in
  place, each file checked against its digest. `define_model`,
  `define_program` (each returning a `Definition`), `start`, `resume`, `branch`
  and `find` are the explicit steps. A `Path` in `inputs` is pinned to its bytes in the key, and a checkpoint
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
  be released go, as do failed rows; it returns a `CollectResult`, the store's
  `GcResult` with the orphans it found.
  `Ledger.derive(from_release, name, outputs)` publishes what was computed
  from an accepted checkpoint (an export, evaluation metrics) as a release of
  its own, which keeps only its own outputs; `Ledger.derived` finds it again,
  and a name derived from a later checkpoint is derived from no earlier one.
  `Ledger.execution` returns an execution's record, an `Execution` (program,
  input pins, the checkpoint it branched from, its `Attempt`s and checkpoints);
  `Ledger.executions(of, reading=)` finds those of a program or a model and
  those that read an input's bytes, `Ledger.resumed_from` the `Attempt`s that
  continued from a release, `Ledger.descendants` a run's branches at any
  depth.
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
  `HostedRun(host)` saves and restores for a trainer whose run another system
  holds (a Strata cell): the `Host` (`checkpoint_files`, `restore_files`)
  publishes each checkpoint's files in the layout `Ledger.load` reads.
  [`docs/torch.md`](torch.md) is the guide, opening with a
  quick start (`examples/torch/quickstart.py`);
  [`examples/torch/`](../examples/torch/) also
  trains, resumes after a crash, branches, runs on two ranks under `torchrun`,
  and gates a signed checkpoint before it ships (with a tamper drill).

## URIs

Artifact and name URIs are `{scheme}://artifact/{id}@v={n}` and
`{scheme}://name/{name}`; the scheme is `vestigo` unless the consumer calls
`vestigo.uris.set_uri_scheme` once at import (Strata uses `strata`). The
scheme is persisted in `input_versions` rows, so a store keeps the scheme it
was written under.
