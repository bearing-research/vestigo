"""The exported table inventory is the schema a store creates, on both backends."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from vestigo.build_store import BuildStore
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore
from vestigo.tables import LEDGER_TABLES, TRANSIENT_TABLES


def _tables(store: ArtifactStore) -> dict[str, tuple[str, ...]]:
    """Each table a fresh store holds, with its primary key in key order."""
    conn = store._get_connection()
    try:
        if store.dialect.name == "sqlite":
            names = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            keys = {}
            for name in names:
                columns = conn.execute(f"PRAGMA table_info({name})").fetchall()
                keys[name] = tuple(
                    c["name"] for c in sorted(columns, key=lambda c: c["pk"]) if c["pk"]
                )
            return keys
        rows = conn.execute(
            """
            SELECT t.table_name, k.column_name, k.ordinal_position
            FROM information_schema.tables t
            LEFT JOIN information_schema.table_constraints c
              ON c.table_schema = t.table_schema AND c.table_name = t.table_name
             AND c.constraint_type = 'PRIMARY KEY'
            LEFT JOIN information_schema.key_column_usage k
              ON k.constraint_name = c.constraint_name AND k.table_schema = c.table_schema
            WHERE t.table_schema = current_schema() AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name, k.ordinal_position
            """
        ).fetchall()
        keys: dict[str, tuple[str, ...]] = {}
        for row in rows:
            keys.setdefault(row["table_name"], ())
            if row["column_name"] is not None:
                keys[row["table_name"]] += (row["column_name"],)
        return keys
    finally:
        conn.close()


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgres", marks=[pytest.mark.integration, pytest.mark.slow]),
    ]
)
def fresh_store(request, tmp_path) -> Iterator[ArtifactStore]:
    """A store on a database nothing else has written to: the session's Postgres holds
    other tests' tables, so this one gets a database of its own."""
    if request.param == "sqlite":
        store = ArtifactStore(tmp_path / "artifacts")
        yield store
        store.close()
        return
    dsn = request.getfixturevalue("postgres_dsn")
    name = f"inventory_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    dialect = PostgresDialect(dsn.rsplit("/", 1)[0] + f"/{name}")
    try:
        yield ArtifactStore(tmp_path / "artifacts", dialect=dialect)
    finally:
        dialect.close()
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {name}")


def test_the_inventory_names_every_table_with_its_primary_key(fresh_store):
    # The build tables are the build store's; a consumer opens both on one database.
    BuildStore(fresh_store.db_path, dialect=fresh_store.dialect)
    found = _tables(fresh_store)

    assert set(found) == {name for name, _ in LEDGER_TABLES} | set(TRANSIENT_TABLES)
    for name, key in LEDGER_TABLES:
        assert found[name] == key, name


def test_the_inventory_is_in_foreign_key_order():
    """A row-by-row copy in this order never inserts a row before what it references."""
    names = [name for name, _ in LEDGER_TABLES]
    assert names[0] == "artifact_versions"
    assert names.index("artifact_builds") < names.index("build_attempts")
