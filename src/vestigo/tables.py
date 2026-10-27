"""The tables the ledger owns, for tools that copy or audit a database.

``LEDGER_TABLES`` is in foreign-key order (``artifact_versions`` first) with
each table's primary key, so a row-by-row copy is idempotent. A consumer adds
its own tables after these when it migrates a store.

Three tables are deliberately absent. ``schema_version`` is created by the
target store itself. ``build_console_chunks`` holds only live rows that do not
outlive the process that wrote them. ``import_staging`` rows live up to
``IMPORT_STAGING_TTL_SECONDS`` (a day) and name bytes already in the source's
blob store: a copy taken while an import is in flight leaves that upload
unreachable in the target, where the importer must stage it again.
"""

from __future__ import annotations

LEDGER_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("artifact_versions", ("id", "version")),
    ("artifact_names", ("tenant", "name")),
    ("artifact_aliases", ("tenant", "name", "alias")),
    ("artifact_tags", ("tenant", "artifact_id", "version", "key")),
    ("registry_audit", ("seq",)),
    ("registry_pending", ("tenant", "name", "alias")),
    ("artifact_publications", ("token",)),
    ("retention_roots", ("tenant", "kind", "holder", "artifact_id", "version")),
    ("retention_edges", ("from_artifact_id", "from_version", "to_artifact_id", "to_version")),
    ("store_upgrades", ("name",)),
    ("nodes", ("scope", "node_id")),
    ("node_inputs", ("scope", "node_id", "role", "position")),
    ("node_links", ("scope", "node_id", "relation", "target_node_id")),
    ("build_releases", ("release_id",)),
    ("release_references", ("tenant", "name")),
    ("release_authorities", ("tenant", "target")),
    ("artifact_builds", ("build_id",)),
    ("build_attempts", ("build_id", "attempt")),
)

TRANSIENT_TABLES: tuple[str, ...] = ("schema_version", "import_staging", "build_console_chunks")
"""Created by the ledger but never copied between databases."""
