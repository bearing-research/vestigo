# Changelog

All notable changes to vestigo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `ReleaseStore.bundles_of(build_id, kind=None)`: a build's releases with
  their outputs in one read, what a resume or a prune did as `releases_of`
  plus `bundle` per release (#13, friction log 6).

### Changed

- **A build's `dispatch` is a column.** `artifact_builds.dispatch` is
  `runner` (offered by `list_runnable_builds`) or `external` (the consumer
  runs it itself); `create_build` takes `dispatch=` and refuses any other
  value. The ledger no longer reads the `_dispatch_mode` mark in `params`;
  a database from before the column backfills it from the mark once, on
  either backend (#12, friction log 11). Schema change: the next release
  bumps the minor.
- One column list and one row mapper build every `ArtifactVersion` the store
  reads; a new column is one entry in `_VERSION_FIELDS` and one dataclass
  field, pinned equal by a test (#10, friction log 1).
  The build store reads a `BuildState` the same way (#11).
- `versions_lacking_definition` documents that it lists every tenant's
  versions, the read of a store-wide upgrade, with each version's tenant for
  the `set_definitions` call that follows; a test pins it (#9).

### Fixed

- The PostgreSQL row shim accepts a slice, as `sqlite3.Row` does (#9).

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
