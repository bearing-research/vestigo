"""The ledger examples (``examples/ledger/``), run as a reader would run them.

Each runs in its own process on a fresh ledger; the assertions read what it left.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from vestigo.build_store import BuildStore
from vestigo.releases import ReleaseStore
from vestigo.store import ArtifactStore

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "ledger"


def _example(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _verbs(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.split()[0] in ("hit", "computed")
    ]


def test_the_cache_example_computes_each_step_once(tmp_path: Path):
    root = str(tmp_path / "ledger")
    text = "The quick brown fox. The lazy dog."

    assert _verbs(_example("cache.py", root, text)) == ["computed"] * 3
    assert _verbs(_example("cache.py", root, text)) == ["hit"] * 3
    assert (
        _verbs(_example("cache.py", root, "the quick brown fox; the lazy dog")) == ["computed"] * 3
    )

    store = ArtifactStore(Path(root))
    try:
        named = store.resolve_name("word-counts")
        assert named is not None and (named.id, named.version) == ("counts", 2)
        assert [(v.id, v.version) for v, _ in store.find_dependents("clean", 2)] == [("counts", 2)]
    finally:
        store.close()


def test_the_releases_example_fences_the_stalled_worker(tmp_path: Path):
    root = tmp_path / "ledger"
    result = _example("releases.py", str(root))

    assert "first: rejected" in result.stdout
    assert "second: a stale head update failed" in result.stdout
    match = re.search(r"^build (\w+): ready$", result.stdout, re.MULTILINE)
    assert match is not None, result.stdout
    build_id = match.group(1)
    store = ArtifactStore(root)
    try:
        releases = ReleaseStore(store, BuildStore(store.db_path, dialect=store.dialect))
        checkpoints = releases.releases_of(build_id, "checkpoint")
        assert [(r.sequence, r.attempt, r.state) for r in checkpoints] == [
            (1, "first", "accepted"),
            (2, "first", "accepted"),
            (3, "second", "accepted"),
            (4, "second", "accepted"),
        ]
        assert releases.reference("example/best") == (checkpoints[3].release_id, 1)
        final = store.get_latest_version(f"run_{build_id}")
        assert final is not None and final.state == "ready"
    finally:
        store.close()
