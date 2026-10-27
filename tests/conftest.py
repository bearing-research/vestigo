"""Fixtures for the ledger's own suite.

Nothing here imports Strata: the suite must run from a checkout of this package alone.
Helpers that one module uses live in that module; this file holds what every module
shares: the singleton resets, the credential scrub, the validator the suite registers,
and the Docker-backed Postgres that any store test can run on.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import docker
import pyarrow as pa
import pytest
from testcontainers.community.postgres import PostgresContainer

from vestigo.build_store import reset_build_store
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore, register_validator, reset_artifact_store
from vestigo.uris import DEFAULT_URI_SCHEME, set_uri_scheme, uri_scheme


def _arrow_rows(data: bytes) -> int:
    """Rows in exactly one Arrow IPC stream; trailing bytes (a second stream) are an error."""
    if not data:
        return 0
    buf = pa.BufferReader(pa.py_buffer(data))
    rows = sum(batch.num_rows for batch in pa.ipc.open_stream(buf))
    if buf.tell() != len(data):
        raise ValueError(f"{len(data) - buf.tell()} bytes after the stream's end")
    return rows


# The application registers its formats at startup; this suite stands in for one.
register_validator("arrow/ipc", _arrow_rows)
register_validator("", _arrow_rows)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """The store singletons are process globals shared by every test on a worker.

    The URI scheme is one too: a consumer collected in the same session sets its
    own at import, and this suite writes the default's, then hands the consumer's
    back so its tests still parse their own URIs afterwards.
    """
    before = uri_scheme()
    set_uri_scheme(DEFAULT_URI_SCHEME)
    yield
    set_uri_scheme(before)
    reset_artifact_store()
    reset_build_store()


@pytest.fixture(scope="session")
def _no_cloud_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("no-cloud-config")


@pytest.fixture(autouse=True)
def _no_ambient_cloud_credentials(monkeypatch: pytest.MonkeyPatch, _no_cloud_config_dir: Path):
    """The SDKs read the developer's own credentials; a test must see none.

    A store built without credentials is what several tests assert on, and boto3 resolves
    its chain when the client is built, through whatever ``~/.aws`` configures.
    """
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_STORAGE_CONNECTION_STRING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(_no_cloud_config_dir / "no-aws-config"))
    monkeypatch.setenv(
        "AWS_SHARED_CREDENTIALS_FILE", str(_no_cloud_config_dir / "no-aws-credentials")
    )
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.fixture(scope="session")
def docker_daemon() -> None:
    """Skip what needs a container when the Docker daemon is unreachable; fail where required.

    Only possible locally: CI sets ``VESTIGO_REQUIRE_DOCKER`` so a missing daemon cannot
    skip a container-backed test to green.
    """
    try:
        docker.from_env().ping()
    except Exception:
        if os.environ.get("VESTIGO_REQUIRE_DOCKER"):
            raise RuntimeError("VESTIGO_REQUIRE_DOCKER is set and the Docker daemon is unreachable")
        pytest.skip("Docker daemon is not running")


@pytest.fixture(scope="session")
def postgres_dsn(docker_daemon) -> Iterator[str]:
    """One live Postgres for the session; each store fixture starts from an empty schema."""
    with PostgresContainer("postgres:18-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


LEDGER_TABLES = (
    "artifact_versions, artifact_names, artifact_aliases, artifact_tags, registry_audit, "
    "registry_pending, artifact_publications, retention_roots, retention_edges, "
    "store_upgrades, nodes, node_inputs, node_links, build_releases, release_references, "
    "release_authorities, import_staging, artifact_builds, build_attempts, "
    "build_console_chunks, schema_version"
)


@pytest.fixture
def postgres_store(postgres_dsn, tmp_path) -> Iterator[ArtifactStore]:
    """A store with metadata on Postgres and local blobs, on a schema this test creates."""
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript(f"DROP TABLE IF EXISTS {LEDGER_TABLES} CASCADE;")
        conn.commit()
    finally:
        conn.close()

    yield ArtifactStore(tmp_path / "artifacts", dialect=dialect)

    # Each test builds its own pool against the shared container. Left open
    # they accumulate toward Postgres's default max_connections of 100 and
    # later tests start failing for reasons that have nothing to do with them.
    dialect.close()


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgres", marks=[pytest.mark.integration, pytest.mark.slow]),
    ]
)
def backend_store(request, tmp_path) -> Iterator[ArtifactStore]:
    """A store on each metadata backend: a test that takes it runs once per backend."""
    if request.param == "postgres":
        yield request.getfixturevalue("postgres_store")
        return
    store = ArtifactStore(tmp_path / "artifacts")
    yield store
    store.close()
