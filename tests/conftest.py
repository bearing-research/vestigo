"""Fixtures for the ledger's own suite.

Nothing here imports Strata: the suite must run from a checkout of this package alone.
Helpers that one module uses live in that module; this file holds what every module
shares: the singleton resets, the credential scrub, the validator the suite registers,
and the Docker-backed Postgres and object-store emulators that any test can run on.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path

import docker
import pyarrow as pa
import pytest
from testcontainers.community.azurite import AzuriteContainer
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient
from testcontainers.core.wait_strategies import HttpWaitStrategy, LogMessageWaitStrategy

from vestigo.build_store import reset_build_store
from vestigo.sql_backend import PostgresDialect
from vestigo.store import ArtifactStore, register_validator, reset_artifact_store
from vestigo.tables import LEDGER_TABLES, TRANSIENT_TABLES
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


_EVERY_TABLE = ", ".join([name for name, _ in LEDGER_TABLES] + list(TRANSIENT_TABLES))


@pytest.fixture
def postgres_store(postgres_dsn, tmp_path) -> Iterator[ArtifactStore]:
    """A store with metadata on Postgres and local blobs, on a schema this test creates."""
    dialect = PostgresDialect(postgres_dsn)
    conn = dialect.connect()
    try:
        conn.executescript(f"DROP TABLE IF EXISTS {_EVERY_TABLE} CASCADE;")
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


# --- Object-store emulators: one of each per session; a module makes its own buckets ---

MINIO_IMAGE = (
    "cgr.dev/chainguard/minio"
    "@sha256:bd014394a80898e68c149f2311fdf8d5a2c2f3bb2c33b9327ae6d02b4b065ae1"
)
FAKE_GCS_IMAGE = (
    "fsouza/fake-gcs-server@sha256:797ce226d62f947c009dc40246b30cfb456b8473d8241407f9d6f2c04e4d69ef"
)
AZURITE_IMAGE = (
    "mcr.microsoft.com/azure-storage/azurite"
    "@sha256:830430c1da1a2d537e08f3e6764dd1f5ae00cf0346bcaf625b968ec3f0971fd5"
)


@pytest.fixture(scope="session")
def minio_endpoint(docker_daemon) -> Iterator[str]:
    """MinIO's URL; its root user and password are both ``minioadmin``."""
    # Not testcontainers' MinioContainer: that imports the minio client package.
    container = (
        DockerContainer(MINIO_IMAGE)
        .with_env("MINIO_ROOT_USER", "minioadmin")
        .with_env("MINIO_ROOT_PASSWORD", "minioadmin")
        .with_command("server /data --address :9000")
        .with_exposed_ports(9000)
        .waiting_for(HttpWaitStrategy(9000, "/minio/health/live"))
    )
    with container:
        yield f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9000)}"


@pytest.fixture(scope="session")
def fake_gcs_endpoint(docker_daemon) -> Iterator[str]:
    """fake-gcs-server's URL; it checks no credential."""
    # A resumable upload's session URL is built from the external URL, so the port is
    # chosen here and bound explicitly rather than mapped at random.
    host = DockerClient().host()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    container = (
        DockerContainer(FAKE_GCS_IMAGE)
        .with_bind_ports(4443, port)
        .with_command(
            f"-scheme http -port 4443 -external-url http://{host}:{port} -public-host {host}:{port}"
        )
        .waiting_for(LogMessageWaitStrategy("server started"))
    )
    with container:
        yield f"http://{host}:{port}"


@pytest.fixture(scope="session")
def azurite_container(docker_daemon) -> Iterator[AzuriteContainer]:
    """Azurite, for its connection string, account name and key."""
    # The SDK speaks a newer API version than the emulator admits to; the check is skipped.
    container = AzuriteContainer(AZURITE_IMAGE).with_command(
        "azurite --skipApiVersionCheck --blobHost 0.0.0.0 --queueHost 0.0.0.0 --tableHost 0.0.0.0"
    )
    with container:
        yield container
