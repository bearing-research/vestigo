# Changelog

All notable changes to vestigo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `vestigo.torch`, behind a new `torch` extra: PyTorch checkpoints as ledger
  releases (Investigation 09, stages T0 to T2 and T5). A model (class and
  `state_dict` schema under a declared version) and a training program (its
  loss and optimizer, an `impl` edge to the model) are structural nodes; an
  execution is a build keyed by the program, its inputs and the release it
  starts from. `LedgerStorage` is `torch_checkpointing`'s storage over the
  ledger: every file a rank writes is a `building` version created before
  its bytes, on local disk, S3, GCS or Azure, so a crash leaves rows garbage
  collection reaps; each rank reports what it wrote before the barrier, and
  rank 0 makes the step one release under a manifest that requires the
  files, accepted only while the run's attempt holds the lease. Saves are
  synchronous or asynchronous; metadata on SQLite or Postgres. `resume`
  continues an execution in a new attempt and fences the old one, `branch`
  opens a new execution from any accepted release, and loads read accepted
  releases only, each file checked against its digest. A resumed run and a
  branch end on the weights of an uninterrupted one; a crash before
  acceptance leaves an orphan, never an authority; a rank that dies before
  writing leaves no release. The extra pins `torch-checkpointing` 0.1.1
  (#29, #30).
- `Run.observe` records an `ExportedProgram` the run captured as an observed
  implementation (Investigation 09, T6): the PT2 archive as a version of the
  execution, with its archive, schema and opset versions, torch version and
  range constraints, and a declared scope and coverage. No node changes, so
  no definition moves. A stale or finished attempt records nothing, an
  archive that fails to store is failed for cleanup, and the archive goes
  through a temporary file rather than memory. The same program exported
  again under one scope and coverage is one version (graph, weights and
  constants compared, not the example inputs). `Ledger.observations` lists
  an execution's own records (#31, #32).
- `Ledger.collect(keep=)`, the latest-k policy for checkpoints
  (Investigation 09 §5.6): of each unfinished execution its newest `keep`
  checkpoints and the one its attempt resumed from stay, as do every head,
  the release each unfinished branch starts from, and the checkpoint a
  finished execution ended on, which `finish` now keeps under the reference
  `vestigo-torch/<execution>` (`Ledger.final_checkpoint`). Every other
  checkpoint goes, and so do the files a fenced or ended attempt staged and
  never released; a live attempt's rows stay. `branch` records a use of its
  source, so `min_idle_seconds` spares it until the branch loads (#34).

### Documentation

- `docs/torch.md`, the guide to `vestigo.torch`, and `examples/torch/`: a
  run that stops, resumes in the same execution and records its exported
  forward pass; a branch that fine-tunes from the best epoch; two ranks under
  `torchrun`. The torch job runs the examples (#33).

## 0.2.1 - 2026-10-27

### Fixed

- An artifact id may hold `@` again, as a consumer's suffix such as
  `x@iter=3` or `x@variant=a` does; only an empty id or one holding `@v=`
  is refused, by `create_artifact`, `import_artifact` and `artifact_uri`.
  0.2.0 refused every `@` (audit G-5), which broke Strata's loop, fan-out,
  checkpoint and import ids. `parse_artifact_uri` now ends the id at the
  last `@v=`, so such ids read back from a URI, pinned or not (#28).

## 0.2.0 - 2026-10-27

Schema changes, all applied on open: `artifact_builds` gains `dispatch`
(backfilled from the old params mark), `build_attempts` gains `principal`,
and `node_links` is a new table (migration 15), so the table inventory
Strata's contract tests pin grows by one. Migration 16 rebuilds
`idx_node_links_target` and migration 17 keys `retention_roots` by tenant.
Strata moves its pin to this version with its own dispatch, principal, link,
import and tenant changes.

The release also carries every fix from an audit of the whole package at this
version (#23 to #26).

### Added

- `ReleaseStore.bundles_of(build_id, kind=None)`: a build's releases with
  their outputs in one read, what a resume or a prune did as `releases_of`
  plus `bundle` per release (#13, friction log 6).
- `node_links` and `GraphStore.link(node_id, relation, target)`, `links`,
  `linked_from`: a relation between two stored nodes that is inside neither
  identity. `reconstructed_from` (ADR 002 §6.3) is the first: a complete node
  names the historical node rebuilt for rows from before roles were recorded
  (#18, friction log 8).
- `build_attempts.principal`: `open_attempt(principal=)` records who runs an
  attempt, which a resume may make someone other than the build's initiator;
  `AttemptState.principal` reads it back (#17, friction log 7).
- `collectable_checkpoints(keep=k)` spares the newest `k` accepted checkpoints
  of each building build, so a latest-k policy is the same read with a deeper
  bound; `accept` states that an empty `final` release ends a build without a
  product (#15, friction log 5 and 2).
- `ArtifactImportInProgress`, an `ArtifactImportConflict` raised when an import
  meets a version still building at its `id@v` (#26).

### Changed

- **A build's `dispatch` is a column.** `artifact_builds.dispatch` is
  `runner` (offered by `list_runnable_builds`) or `external` (the consumer
  runs it itself); `create_build` takes `dispatch=` and refuses any other
  value. The ledger no longer reads the `_dispatch_mode` mark in `params`;
  a database from before the column backfills it from the mark once, on
  either backend (#12, friction log 11).
- One column list and one row mapper build every `ArtifactVersion` the store
  reads; a new column is one entry in `_VERSION_FIELDS` and one dataclass
  field, pinned equal by a test (#10, friction log 1).
  The build store reads a `BuildState` the same way (#11).
- `versions_lacking_definition` documents that it lists every tenant's
  versions, the read of a store-wide upgrade, with each version's tenant for
  the `set_definitions` call that follows; a test pins it (#9).
- `list_expired_leases` and `reclaim_expired_build` take only
  `dispatch = 'runner'` builds unless passed `dispatch=` (`None` for any) (#23).
- A build row's deletion takes its console chunks; its attempts and releases
  outlive it, and `collectable_checkpoints` still reports its checkpoints
  (#23, #26).
- Object-store writers stage under `BlobStore.staging_dir` (a constructor and
  `from_settings` argument), which `remove_stale_temp_files` sweeps; a store
  over a remote blob store sets it to `artifact_dir/staging` when unset
  (#24, #26).
- The object-store error metric drops the never-recorded `list` operation (#24).
- `idx_node_links_target` covers `(scope, target_node_id, relation)`, so
  `linked_from` without a relation uses it (#25).
- `GraphStore(connect)` no longer takes a dialect (#25).
- `retention_roots` is keyed by tenant: two tenants holding one shared version
  under one holder hold two roots, each its own to list and release (#26).
- `fail_artifact`, `retire_ready` and `record_use` take `tenant` and raise
  `ArtifactNotFoundError` for a missing version or another tenant's (#26).
- `reset_artifact_store` closes the store, and `get_artifact_store` raises
  `ValueError` for a different `artifact_dir` (#26).
- `create_artifact` and `import_artifact` refuse an empty id or one containing
  `@` (#26).
- `set_definitions` returns the number of versions it recorded (#26).
- The suite: Gate 2 is `tests/test_releases.py`, its tests grouped by what they
  prove and run on both backends through a `backend_store` fixture; one
  Postgres container serves the session instead of one per module; helpers live
  in the module that uses them and `conftest.py` holds fixtures only; every
  import is at module level (#21). A test pins the table inventory against a
  fresh database, and the release workflow runs the suite on the tagged commit
  before it builds (#22).

### Fixed

- A missing Azure container is an error on every call, as a missing S3 bucket
  is on a read; it read as an absent blob before, and `blob_exists` used the
  SDK's `exists()`, which answers False to any 404 (#19). The emulator suite
  now asserts on the wire what the fakes asserted: a missing bucket, a refused
  credential and an unreachable endpoint are errors, never absence, on MinIO,
  Azurite and fake-gcs-server (where a missing bucket is the one documented
  exception, the GCS API answering one `NotFound` for both).
- The PostgreSQL row shim accepts a slice, as `sqlite3.Row` does (#9).

From the audit at this version:

Builds and releases (#23):

- A runner can open an attempt on the build it claimed or reclaimed: the
  claim's own `lease_owner` gets through its live lease.
- A new attempt replaces the lease owner, NULL when none is given, so a runner
  that lost the build can no longer renew, fail or complete it.
- Opening a new attempt abandons the other attempts' unsettled releases, so a
  dead attempt's release no longer blocks every later checkpoint.
- `create_build` returns the dispatch it stored.
- `AdvanceReference` to a release that is not an accepted release of the
  tenant fails with `OperationFailed` instead of moving the reference and
  dropping the old head's roots.
- A re-opened attempt keeps the later lease deadline and records a new
  `resumed_from`.
- `attach` refuses a settled release and runs under the build's key.
- `collectable_checkpoints` keeps the checkpoint the current attempt resumed
  from, and reports the checkpoints of a build whose row is gone.
- A rejection after a rollback no longer overwrites a retry's acceptance.
- `accept` sizes outputs before taking the build's key, which on SQLite is the
  database's write lock.
- `formal/README.md` lists only the invariant each unpatched BuildLease
  configuration violates, with counts from a fresh TLC run.

SQL and blob backends (#24):

- A nested Postgres connection is refused while the outer transaction holds an
  advisory lock (`begin_write`), a row lock or an uncommitted write, including
  a write in a data-modifying CTE, so an inner commit or rollback can no longer
  release what the outer holds.
- SQLite connections can be closed or used from another thread, as Postgres
  ones can; a closed Postgres connection raises `RuntimeError` on use, and a
  cross-thread close racing the owner's no longer leaks a pooled connection.
- Postgres rows hash, and compare only to rows, like `sqlite3.Row`.
- Postgres `column_exists` and `schema_exists` look only at the current schema.
- `create_tables` runs its DDL once per database on SQLite too.
- `LocalBlobStore.open_blob_reader` returns None when the blob is deleted
  between its check and its open.
- GCS presigning returns None for workload-identity credentials without
  impersonation instead of raising `AttributeError`.
- S3 and GCS POST policies accept an empty upload.
- A read that fails mid-stream counts as a failed `get` in the object-store
  error metric.

Graph and provenance (#25):

- `parse_transform_spec` reads a stored spec it cannot rebuild (NaN or
  Infinity in params) as unrecorded instead of raising.
- Graph walks on SQLite probe the index once per reached node instead of
  scanning the scope's edges at every step.
- `GraphStore.inputs` reads back a stored role that is not an identifier.
- `put_batch`, `topological_order` and the walks bind at most 500 ids per
  query, so wide graphs no longer hit SQLite's variable limit.
- `artifact_uri` refuses an empty id or one containing `@`, which
  `parse_artifact_uri` would not read back.
- `NodeSpec` refuses a non-`bytes` payload with `ValueError`, and `Traversal`
  refuses `max_nodes` below 0.

Artifact store (#26):

- A duplicate's finalize locks its canonical before pointing at it, and
  `delete_artifact` re-reads pointers under its row lock, so no superseded
  version is left without bytes.
- A hold or use landing on a lower version mid-sweep keeps the id's top
  version, so a version number is never reused.
- On Postgres a pin or publication landing on a version a sweep has claimed
  waits, or is refused, instead of being deleted with it.
- `import_artifact` refuses a ready or superseded record without bytes unless
  the version is already here with its bytes; it fills in a failed row at the
  same `id@v` (`written=True`) and raises `ArtifactImportInProgress` for a
  building one; it checks `state` and that `input_versions` is a JSON object.
- `promote_version` no longer copies `release_id`.
- `set_alias`, `set_tag`, `request_alias_change` and `set_name` lock their
  target, so a concurrent delete cannot leave a dangling pointer; `set_name`
  raises `ArtifactNotFoundError` instead of the driver's foreign-key error.
- A finalize retry on a version a refresh superseded returns it, and two racing
  builds of one id and computation leave the newer one ready.
- On Postgres a use note skips a locked row instead of waiting.
- `cleanup_failed` spares failed versions held by a root or a publication.
- `verify_artifacts` streams the digest of blobs no validator reads, and a spec
  that does not parse no longer aborts it.
- A read-only open creates no directory.

### Verified

- Every TLA+ configuration re-run on the rewritten models; counts and the
  expected violations in `formal/README.md` (#16).

## 0.1.0 - 2026-10-26

First release from its own repository. The package was extracted from
[Strata](https://github.com/bearing-research/strata) under ADR 002
(source-first ledger extraction, Strata #1294 to #1299) and carries that
history. It is versioned on its own from here; Strata pins the version it
was built against.

- **Store.** Immutable versioned artifacts deduplicated by provenance hash,
  with mutable references (compare-and-set, approval), retention roots and
  required edges, and race-safe garbage collection, on SQLite or PostgreSQL.
- **Builds.** A build is an execution with leased, fenced attempts; its
  checkpoint and final releases make a set of staged outputs current
  together, each accepted only once every output's bytes are stored.
- **Graph.** Content-addressed structural nodes (`CGN1`, `CGD1`) record what
  produced a version; traversals are bounded and the ledger never parses a
  node's payload.
- **Blobs.** Local disk, S3, GCS and Azure backends behind one interface,
  with presigned URLs.
- **URIs.** Artifact and name URIs carry a per-process scheme
  (`vestigo.uris.set_uri_scheme`, default `vestigo`); a consumer sets its own
  once at import.
- **Formal models.** TLA+ models of the artifact and build lifecycles with
  the configurations that found and then closed their bugs.

### Fixed before the first release (audit of 2026-10-26)

Store:
- `delete_artifact` of a dedup canonical hands its bytes to every version reading them through `superseded_by`, not only to published or retained ones; a rooted (pinned) version is refused like a published one.
- `promote_version` carries the source's retention edges, `minted`, `computation_key` and `release_id`, and takes a `tenant`; another tenant's id raises `ArtifactIdTaken`.
- A sweep recomputes reachability under the write lock per batch and refuses a candidate with a requirement edge into it, so a hold landing mid-sweep keeps what it reads.
- Two fresh first boots of one SQLite store both succeed: the WAL pragma is retried while another process creates the schema and the fresh stamp runs under the schema lock.
- `import_artifact` losing the provenance race on PostgreSQL returns the holder and drops its attempt blob instead of raising the driver's error.
- Use notes never wait behind a SQLite writer: a lookup returns while a sweep or a long finalize holds the lock.
- `finalize_canonical_together` promotes in a fixed order, so two runs finalizing one computation's outputs in opposite orders both finish on PostgreSQL.
- A retried finalize of a row another id overtook returns that canonical; `set_name` accepts a superseded version; `cleanup_failed` never reuses a version number; `retain` without a tenant records the version's; a read-only open of a missing store raises `StoreSchemaMismatch` and creates nothing; a sweep tolerates an id another sweep removed.
- `set_definitions` takes a `tenant`, checks every node against that scope and writes nothing on an unknown one.
- `promote_version` closes the source blob's reader when it refuses the id or its transaction fails; on an object store that reader is an open response body.

Builds and releases:
- `AssertReferenceRevision` guards the `AdvanceReference` of the same name in the same transaction, so two builds asserting one revision have one winner on PostgreSQL as on SQLite; a first advance is one upsert.
- `accept` rejects a release whose build stopped building under it; `fail_build`, `complete_build`, `start_build`, `update_build_output`, `claim_build` and `reclaim_expired_build` serialize on the build's write key.
- `claim_build` and `reclaim_expired_build` clear the build's attempt, and a lapsed lease fences its attempt until it is renewed, so a reclaimed build's old attempt releases nothing.
- `open_attempt` and `resume` refuse while another attempt's or a runner's lease is live; `accept` raises `ReleaseIncomplete` for a building row staged under the release but not declared; a release's tenant defaults to its build's and a mismatch is refused; `abandon`, `attach` and `mark_prepared` take the calling attempt; `prepare` raises `ReleaseError` for a taken sequence; an `InsertAuditEvent` inside a release holds the audit lock; the release columns are added under the schema lock.

Graph and provenance:
- `GraphStore.inputs` returns ports in the encoders' byte order on every collation; nodes are inserted in id order so two batches sharing nodes cannot deadlock on PostgreSQL.
- `Traversal` refuses `max_depth` below one; `topological_order` refuses a root the store lacks; a non-ASCII port name, a role that is not an identifier after NFKC, a label containing `:`, a negative version in `artifact_uri`, and a `TransformSpec` with non-string keys, non-JSON values or non-finite floats are refused with `ValueError`. No stored hash or encoding changes.

Backends:
- S3 and GCS readers treat only a missing object as absent; access denied, expired credentials and connection errors propagate on every backend's read, exists, size and delete.
- A PostgreSQL connection wrapper is a context manager, releases itself with a warning when dropped unclosed, refuses a nested acquisition over uncommitted writes or a failed transaction, and releases from any thread; pooled connections are checked before they are handed out.
- The local backend fsyncs the directory after placing a blob and reports a lost delete race as absent; the inline GCS key file is written without following a planted symlink; half an S3 key pair is refused.

### Changed
- The S3 and GCS backends run on their clouds' own SDKs, boto3 and google-cloud-storage, in place of PyArrow's filesystems, and the package has no required dependency: `vestigo[s3]` and `vestigo[gcs]` bring each backend's SDK. One client per backend reads, writes and signs, so a role resolves once and its session token signs the URLs, and GCS credentials of any kind `GOOGLE_APPLICATION_CREDENTIALS` accepts (a service-account key, workload identity federation, an authorized user) reach the client directly instead of through that variable. A GCS reader fetches 8 MiB at a time, whatever the caller reads. An upload lands whole or not at all, so a failed overwrite keeps the previous object. A missing object is still absent; any other error is the SDK's (`botocore.exceptions.ClientError`, `google.api_core.exceptions.GoogleAPICallError`) rather than an `OSError`. `GCSBlobStore` without a key, not anonymous and with no Application Default Credentials fails at construction. A custom endpoint is addressed path-style and may be given as `host:port`. `BlobSettings.gcs_default_bucket_location` is gone: the store never creates a bucket.
- `find_dependents` and `list_name_reads` read the whole store when `tenant` is omitted; `tenant=None` is the default tenant, as everywhere else.
- Migration 11, which repaired the first consumer's embedded builds, is a no-op here; the repair is the consumer's.
- The TLA+ models describe the code as it is: the lifecycle model has the `superseded_by` pointer path and a `PointersResolve` invariant, the lease model an `OpenAttempt` action and a fence on the attempt and its lease.

### Added
- `ArtifactStore.versions_of_computation(computation_key, tenant=None)`.
- A README section naming every public entry point; the package docstring lists every module.
- Tests for every fix above, the S3 backend over moto and the GCS and Azure backends over fake clients, every object-store backend against its emulator (MinIO, fake-gcs-server, Azurite) started through Docker like the PostgreSQL suite, the pooled connection lifecycle on a live PostgreSQL, and the build store's attempts, leases and console chunks. `VESTIGO_REQUIRE_DOCKER=1` turns a missing Docker daemon into a failure, as CI sets it.
