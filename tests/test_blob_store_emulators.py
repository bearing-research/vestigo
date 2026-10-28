"""Each object-store backend against its emulator, started through Docker.

MinIO checks SigV4 signatures and POST policies as S3 does; fake-gcs-server and Azurite
answer the real SDKs. The unit tests in test_blob_store.py cover the backends' logic over
fakes; these cover the wire. Needs Docker (CI requires it, locally the module skips).
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError, ServiceRequestError
from botocore.exceptions import ClientError, EndpointConnectionError

from vestigo.blob_store import (
    AzureBlobStore,
    GCSBlobStore,
    S3BlobStore,
    get_object_store_errors,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BUCKET = "vestigo-blobs"
LARGE = os.urandom(1024) * (9 * 1024)  # 9 MiB: past the S3 multipart and GCS resumable thresholds


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _minio_store(endpoint: str, *, bucket: str = BUCKET, secret_key: str = "minioadmin"):
    return S3BlobStore(
        bucket=bucket,
        prefix="artifacts",
        region="us-east-1",
        endpoint_url=endpoint,
        access_key="minioadmin",
        secret_key=secret_key,
    )


def _gcs_store(endpoint: str, *, bucket: str = BUCKET):
    return GCSBlobStore(
        bucket=bucket, prefix="artifacts", anonymous=True, endpoint_override=endpoint
    )


def _azure_store(container, *, bucket: str = BUCKET, endpoint: str | None = None, key=None):
    if endpoint is None and key is None:
        return AzureBlobStore(
            account_name=container.account_name,
            container_name=bucket,
            prefix="artifacts",
            connection_string=container.get_connection_string(),
        )
    return AzureBlobStore(
        account_name=container.account_name,
        container_name=bucket,
        prefix="artifacts",
        account_key=key or container.account_key,
        endpoint_url=endpoint or _azure_endpoint(container),
    )


def _azure_endpoint(container) -> str:
    host = container.get_container_host_ip()
    return f"http://{host}:{container.get_exposed_port(container.blob_service_port)}"


# The containers are conftest's, one per session; this module makes its bucket in each.


@pytest.fixture(scope="module")
def minio(minio_endpoint):
    store = _minio_store(minio_endpoint)
    store._client.create_bucket(Bucket=BUCKET)
    return store


@pytest.fixture(scope="module")
def fake_gcs(fake_gcs_endpoint):
    store = _gcs_store(fake_gcs_endpoint)
    store._client.create_bucket(BUCKET)
    return store


@pytest.fixture(scope="module")
def azurite(azurite_container):
    store = _azure_store(azurite_container)
    store._client.create_container()
    return store


@pytest.fixture(params=["minio", "fake_gcs", "azurite"])
def store(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["minio", "fake_gcs", "azurite"])
def backend(request):
    """``(name, what a store of it is built from)`` for the stores that must fail."""
    if request.param == "minio":
        return "minio", request.getfixturevalue("minio_endpoint")
    if request.param == "fake_gcs":
        return "fake_gcs", request.getfixturevalue("fake_gcs_endpoint")
    return "azurite", request.getfixturevalue("azurite_container")


def _errors() -> int:
    return sum(get_object_store_errors().snapshot().values())


def test_a_blob_round_trips_and_is_gone_after_delete(store, tmp_path: Path):
    artifact = f"art-{uuid.uuid4().hex[:8]}"

    assert store.read_blob(artifact, 1) is None
    assert store.blob_exists(artifact, 1) is False
    assert store.blob_size(artifact, 1) is None
    assert store.delete_blob(artifact, 1) is False

    store.write_blob(artifact, 1, b"streamed")
    assert store.read_blob(artifact, 1) == b"streamed"
    assert store.blob_exists(artifact, 1) is True
    assert store.blob_size(artifact, 1) == len(b"streamed")

    source = tmp_path / "incoming.bin"
    source.write_bytes(b"published")
    store.publish_blob_from_path(artifact, 1, source)
    assert store.read_blob(artifact, 1) == b"published"

    assert store.delete_blob(artifact, 1) is True
    assert store.blob_exists(artifact, 1) is False
    assert store.delete_blob(artifact, 1) is False


def test_a_large_blob_round_trips_in_parts(store, tmp_path: Path):
    artifact = f"big-{uuid.uuid4().hex[:8]}"
    source = tmp_path / "big.bin"
    source.write_bytes(LARGE)

    store.publish_blob_from_path(artifact, 1, source)

    assert store.blob_size(artifact, 1) == len(LARGE)
    reader = store.open_blob_reader(artifact, 1)
    assert reader is not None
    with reader as f:
        chunks = iter(lambda: f.read(1 << 20), b"")
        assert b"".join(chunks) == LARGE


def test_a_missing_bucket_is_an_error_where_the_service_tells_it_apart(backend):
    """S3 says ``NoSuchBucket`` on a GET and Azure ``ContainerNotFound`` on every call;
    neither reads as an absent blob. GCS answers the one ``NotFound`` for both, so a
    misnamed bucket there is the documented exception; a HEAD on S3 is the other."""
    name, source = backend
    missing = f"no-such-{uuid.uuid4().hex[:8]}"
    before = _errors()
    if name == "minio":
        store = _minio_store(source, bucket=missing)
        with pytest.raises(ClientError, match="NoSuchBucket"):
            store.read_blob("art", 1)
        assert _errors() == before + 1
    elif name == "fake_gcs":
        store = _gcs_store(source, bucket=missing)
        assert store.read_blob("art", 1) is None
        assert store.blob_exists("art", 1) is False
        assert _errors() == before
    else:
        store = _azure_store(source, bucket=missing)
        for call in (store.read_blob, store.blob_exists, store.blob_size, store.delete_blob):
            with pytest.raises(ResourceNotFoundError, match="ContainerNotFound"):
                call("art", 1)
        assert _errors() == before + 4


def test_a_refused_credential_is_an_error_on_every_call(backend):
    """A wrong key is never "not there": read, exists, size and delete all raise."""
    name, source = backend
    if name == "minio":
        store, error, text = (
            _minio_store(source, secret_key="wrong"),
            ClientError,
            "403|SignatureDoesNotMatch",  # a HEAD carries no body, so no code
        )
    elif name == "fake_gcs":
        pytest.skip("fake-gcs-server checks no credential")
    else:
        # Azure answers ``AuthenticationFailed`` (a ClientAuthenticationError), Azurite a
        # plain ``AuthorizationFailure``; both are the one HttpResponseError family.
        store = _azure_store(source, key="d29uZw==")
        error, text = HttpResponseError, "Authentication|Authorization"
    before = _errors()

    for call in (store.read_blob, store.blob_exists, store.blob_size, store.delete_blob):
        with pytest.raises(error, match=text):
            call("art", 1)

    assert _errors() == before + 4


def test_an_unreachable_endpoint_is_an_error(backend, monkeypatch):
    """A connection failure propagates as the SDK's error and counts for the metric."""
    name, source = backend
    nowhere = f"http://127.0.0.1:{_free_port()}"
    if name == "minio":
        monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")  # boto3 would retry for seconds otherwise
        store, error = _minio_store(nowhere), EndpointConnectionError
    elif name == "fake_gcs":
        pytest.skip("google-cloud-storage retries a connection error for 120 s by design")
    else:
        store, error = _azure_store(source, endpoint=nowhere), ServiceRequestError
    before = _errors()

    with pytest.raises(error):
        store.read_blob("art", 1)

    assert _errors() == before + 1


def test_a_presigned_get_reads_the_blob_from_minio(minio):
    minio.write_blob("fig", 1, b"bytes of the figure")

    url = minio.presign_get("fig", 1, ttl_seconds=60)

    assert url is not None
    with urllib.request.urlopen(url) as response:
        assert response.read() == b"bytes of the figure"


def test_a_presigned_post_uploads_within_the_bound_and_minio_refuses_beyond_it(minio):
    signed = minio.presign_post("out", 1, max_bytes=16, ttl_seconds=60)
    assert signed is not None
    url, fields = signed

    status, _ = _post_form(url, fields, b"0123456789")
    assert status == 204
    assert minio.read_blob("out", 1) == b"0123456789"

    status, body = _post_form(url, fields, b"x" * 17)
    assert status == 400
    assert b"EntityTooLarge" in body


def test_a_presigned_post_uploads_an_empty_blob(minio):
    signed = minio.presign_post("empty", 1, max_bytes=16, ttl_seconds=60)
    assert signed is not None

    status, _ = _post_form(*signed, b"")

    assert status == 204
    assert minio.read_blob("empty", 1) == b""


def test_a_presigned_put_and_get_go_through_azurite(azurite):
    """Azurite checks an account-key SAS as Azure does, its permissions included."""
    artifact = f"sas-{uuid.uuid4().hex[:8]}"
    put = azurite.presign_put(artifact, 1, ttl_seconds=60)
    get_url = azurite.presign_get(artifact, 1, ttl_seconds=60)
    assert put is not None and get_url is not None
    put_url, headers = put

    assert _put(put_url, headers, b"put through a SAS") == 201
    assert azurite.read_blob(artifact, 1) == b"put through a SAS"
    with urllib.request.urlopen(get_url) as response:
        assert response.read() == b"put through a SAS"
    # The read SAS grants no write.
    assert _put(get_url, headers, b"overwritten") == 403
    assert azurite.read_blob(artifact, 1) == b"put through a SAS"


def _put(url: str, headers: dict[str, str], payload: bytes) -> int:
    request = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as error:
        with error:
            return error.code


def _post_form(url: str, fields: dict[str, str], payload: bytes) -> tuple[int, bytes]:
    """POST ``fields`` and ``payload`` as the ``file`` part of a multipart form."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="f"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        # The error owns the response body; unclosed it warns at collection.
        with error:
            return error.code, error.read()
