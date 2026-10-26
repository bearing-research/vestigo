# Changelog

All notable changes to vestigo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

## 0.11.0

First release from its own repository. The package was extracted from
[Strata](https://github.com/bearing-research/strata) under ADR 002
(source-first ledger extraction, Strata #1294 to #1299) and carries that
history; its version matches the Strata release that pins it.

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
