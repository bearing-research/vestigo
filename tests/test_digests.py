"""The file digest cache: content keys, reused only while the stat holds."""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

import vestigo.digests
from vestigo.digests import Digest, FileChanged, FileDigests

_LATER_NS = 10_000_000_000
_DAY_NS = 86_400 * 10**9
# The fixture below replaces os.stat itself; helpers read the filesystem through this.
_REAL_STAT = os.stat


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every file the cache reads, in order."""
    seen: list[Path] = []
    real = vestigo.digests._read_digest

    def counted(path: Path) -> str:
        seen.append(Path(path))
        return real(path)

    monkeypatch.setattr(vestigo.digests, "_read_digest", counted)
    return seen


@pytest.fixture
def stats(monkeypatch: pytest.MonkeyPatch) -> dict[Path, SimpleNamespace]:
    """Stats to report in place of the filesystem's, by path: ctime can't be set."""
    fixed: dict[Path, SimpleNamespace] = {}

    def stat(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) in fixed:
            return fixed[Path(path)]
        return _REAL_STAT(path, *args, **kwargs)

    monkeypatch.setattr(vestigo.digests.os, "stat", stat)
    return fixed


def _stat_of(path: Path, **changed: int) -> SimpleNamespace:
    st = _REAL_STAT(path)
    fields = ("st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_ino", "st_dev")
    return SimpleNamespace(**{f: getattr(st, f) for f in fields} | changed)


def _cache(path: Path | None, offset_ns: int = _LATER_NS) -> FileDigests:
    """A cache in *path* (``None``: the machine's) whose clock is *offset_ns* past now,
    so what it reads is that old."""
    return FileDigests(path, clock=lambda: time.time_ns() + offset_ns)


@pytest.fixture
def cache(tmp_path: Path):
    digests = _cache(tmp_path / "state" / "digests.sqlite")
    yield digests
    digests.close()


@pytest.fixture
def racy(tmp_path: Path):
    """A cache whose clock is now: what it reads was just written."""
    digests = _cache(tmp_path / "state" / "digests.sqlite", offset_ns=0)
    yield digests
    digests.close()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _writes_into(folder: Path) -> bool:
    probe = folder / "probe"
    try:
        probe.touch()
    except PermissionError:
        return False
    probe.unlink()
    return True


def _warnings(caplog: pytest.LogCaptureFixture, text: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "vestigo.digests" and text in r.getMessage()]


def test_a_files_key_is_its_content(tmp_path: Path, cache: FileDigests, reads: list[Path]):
    data = tmp_path / "a.csv"
    data.write_bytes(b"x,y\n1,2\n")

    assert cache.digest(data).sha256 == _sha(b"x,y\n1,2\n")
    assert cache.digest(data).sha256 == _sha(b"x,y\n1,2\n")
    assert cache.cached(data) == _sha(b"x,y\n1,2\n")
    assert reads == [data]


def test_a_rewrite_that_keeps_size_and_mtime_is_read_again(
    tmp_path: Path, cache: FileDigests, reads: list[Path], stats: dict
):
    """Only the ctime says so: a write moves it, and nothing can set it back."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    before = _stat_of(data)
    stats[data] = before
    cache.digest(data)

    data.write_bytes(b"3,4\n")
    stats[data] = _stat_of(data, st_mtime_ns=before.st_mtime_ns, st_ctime_ns=before.st_ctime_ns + 1)

    assert cache.cached(data) is None
    assert cache.digest(data).sha256 == _sha(b"3,4\n")
    assert len(reads) == 2


def test_a_metadata_change_costs_a_read_and_keeps_the_digest(
    tmp_path: Path, cache: FileDigests, reads: list[Path]
):
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    first = cache.digest(data).sha256

    os.utime(data, ns=(0, 0))
    os.chmod(data, 0o600)

    assert cache.digest(data).sha256 == first
    assert len(reads) == 2


def test_a_file_read_as_it_changed_is_not_remembered(
    tmp_path: Path, racy: FileDigests, stats: dict
):
    """The stat cannot see a write in the same tick as the read, so such a digest is not
    kept."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data)
    assert racy.digest(data).settled
    assert racy.cached(data) is None

    data.write_bytes(b"3,4\n")  # in the same tick: every stat field as before

    assert racy.digest(data).sha256 == _sha(b"3,4\n")


def test_an_old_mtime_does_not_make_a_fresh_write_trusted(
    tmp_path: Path, racy: FileDigests, stats: dict
):
    """An mtime set back (rsync -t, cp -p) says nothing; the ctime is when it changed."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data, st_mtime_ns=0)
    racy.digest(data)

    data.write_bytes(b"3,4\n")  # in the same tick, the mtime set back again

    assert racy.digest(data).sha256 == _sha(b"3,4\n")


def test_a_file_read_well_after_its_change_is_remembered(
    tmp_path: Path, cache: FileDigests, reads: list[Path], stats: dict
):
    """The counterpart: what the racy rule refuses above, it trusts here."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data)
    cache.digest(data)

    data.write_bytes(b"3,4\n")

    assert cache.digest(data).sha256 == _sha(b"1,2\n")
    assert reads == [data]


def test_a_file_written_while_it_is_read_is_unsettled_and_not_kept(
    tmp_path: Path, cache: FileDigests, monkeypatch: pytest.MonkeyPatch, stats: dict
):
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    before = _stat_of(data)
    stats[data] = before
    real = vestigo.digests._read_digest
    reads: list[Path] = []

    def written_during(path: Path) -> str:
        reads.append(path)
        digest = real(path)
        if len(reads) == 1:
            with open(path, "ab") as f:
                f.write(b"3,4\n")
            stats[data] = _stat_of(data, st_ctime_ns=before.st_ctime_ns + 1)
        return digest

    monkeypatch.setattr(vestigo.digests, "_read_digest", written_during)

    first = cache.digest(data)
    assert not first.settled
    # Not kept even under the stat it was read at, should a stat report that again (an
    # NFS client's attribute cache, until it expires).
    stats[data] = before
    assert cache.cached(data) is None

    del stats[data]
    second = cache.digest(data)
    assert second.settled
    assert second.sha256 == _sha(b"1,2\n3,4\n")
    assert cache.cached(data) == second.sha256
    assert len(reads) == 2


def test_a_file_is_judged_settled_by_the_clock_before_its_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A long read: a file written just before it is not settled by the time it ends."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    now = [time.time_ns()]
    real = vestigo.digests._read_digest

    def slow(path: Path) -> str:
        now[0] += _LATER_NS
        return real(path)

    monkeypatch.setattr(vestigo.digests, "_read_digest", slow)
    cache = FileDigests(tmp_path / "digests.sqlite", clock=lambda: now[0])
    try:
        assert cache.digest(data).settled
        assert cache.cached(data) is None
    finally:
        cache.close()


def _after_the_stop(reads: list[Path]) -> None:
    # A read after the third is slow, as a real one is: the one worker would otherwise
    # race through tiny files before the caller cancels them.
    if len(reads) > 3:
        time.sleep(0.2)


def test_settled_only_stops_at_the_first_file_that_changed(
    tmp_path: Path, cache: FileDigests, monkeypatch: pytest.MonkeyPatch
):
    """A caller that refuses a changed file has no use for the reads after it."""
    files = [tmp_path / f"{i:02}.csv" for i in range(20)]
    for i, path in enumerate(files):
        path.write_bytes(b"%d\n" % i)
    real = vestigo.digests._read_digest
    reads: list[Path] = []

    def written_during(path: Path) -> str:
        reads.append(Path(path))
        _after_the_stop(reads)
        digest = real(path)
        if len(reads) == 3:
            with open(path, "ab") as f:
                f.write(b"more\n")
        return digest

    monkeypatch.setattr(vestigo.digests, "_read_digest", written_during)

    with pytest.raises(FileChanged) as excinfo:
        cache.digest_many(files, workers=1, settled_only=True)

    assert Path(excinfo.value.args[0]) == reads[2]
    # One worker: the one that changed, and at most the one it took up meanwhile.
    assert len(reads) <= 4
    # What was read before it is not read again.
    assert cache.cached_many(reads[:2]) == [real(path) for path in reads[:2]]


def test_what_was_read_before_a_failed_read_is_remembered(
    tmp_path: Path, cache: FileDigests, monkeypatch: pytest.MonkeyPatch
):
    files = [tmp_path / f"{i:02}.csv" for i in range(20)]
    for i, path in enumerate(files):
        path.write_bytes(b"%d\n" % i)
    real = vestigo.digests._read_digest
    reads: list[Path] = []

    def failing_third(path: Path) -> str:
        reads.append(Path(path))
        _after_the_stop(reads)
        if len(reads) == 3:
            raise PermissionError(path)
        return real(path)

    monkeypatch.setattr(vestigo.digests, "_read_digest", failing_third)

    with pytest.raises(PermissionError):
        cache.digest_many(files, workers=1)

    assert len(reads) <= 4
    # A relaunch after the fix reads what failed and what was not started, no more.
    assert cache.cached_many(reads[:2]) == [real(path) for path in reads[:2]]


@pytest.mark.parametrize("where", ["explicit", "machine"])
def test_the_cache_outlives_its_process(tmp_path: Path, reads: list[Path], where: str):
    path = tmp_path / "state" / "digests.sqlite" if where == "explicit" else None
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    first = _cache(path)
    first.digest(data)
    first.close()

    again = _cache(path)
    try:
        assert again.digest(data).sha256 == _sha(b"1,2\n")
    finally:
        again.close()
    assert reads == [data]
    if path is None:
        assert (Path(os.environ["VESTIGO_CACHE_DIR"]) / "digests.sqlite").is_file()


@pytest.mark.parametrize("where", ["machine", "explicit"])
def test_the_machine_cache_s_table_that_keyed_paths_as_text_is_dropped(tmp_path: Path, where: str):
    """A cache file the caller names is its own: a table of that name there is kept."""
    path = Path(os.environ["VESTIGO_CACHE_DIR"]) / "digests.sqlite"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE digests (path TEXT PRIMARY KEY, stat TEXT, sha256 TEXT)")
    _cache(None if where == "machine" else path).close()

    with closing(sqlite3.connect(path)) as db:
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master")}
    assert "file_digests" in tables
    assert ("digests" in tables) is (where == "explicit")


def test_the_old_table_is_dropped_once_so_a_0_4_process_beside_keeps_it():
    path = Path(os.environ["VESTIGO_CACHE_DIR"]) / "digests.sqlite"
    _cache(None).close()
    # A 0.4 process on the same machine makes its table again, and goes on using it.
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE digests (path TEXT PRIMARY KEY, stat TEXT, sha256 TEXT)")
    _cache(None).close()

    with closing(sqlite3.connect(path)) as db:
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master")}
    assert "digests" in tables


def test_prune_forgets_only_the_files_a_walk_of_its_root_did_not_find(
    tmp_path: Path, cache: FileDigests
):
    root = tmp_path / "data"
    sibling = tmp_path / "data2"
    for folder in (root, sibling):
        folder.mkdir()
    kept, gone, nested, beside = (
        root / "kept.csv",
        root / "gone.csv",
        root / "sub" / "gone.csv",
        sibling / "beside.csv",
    )
    nested.parent.mkdir()
    for path in (kept, gone, nested, beside):
        path.write_bytes(path.name.encode())
        cache.digest(path)

    assert cache.prune(root, [kept]) == 2

    assert cache.cached(kept) is not None
    assert cache.cached(beside) is not None
    # Still on disk, unchanged: only the forgotten row makes them unknown.
    assert cache.cached(gone) is None
    assert cache.cached(nested) is None


def test_prune_waits_for_another_writer_rather_than_failing(
    tmp_path: Path, cache: FileDigests, caplog: pytest.LogCaptureFixture
):
    """Ranks share the machine's cache. A read lock raised to a write one, as another
    connection commits, fails at once and unwaited: prune takes the write lock first."""
    root = tmp_path / "data"
    root.mkdir()
    files = [root / "a.csv", root / "b.csv"]
    for path in files:
        path.write_bytes(path.name.encode())
    cache.digest_many(files)
    other = threading.Thread(target=_write_beside, args=(tmp_path / "state" / "digests.sqlite",))
    started: list[str] = []

    def deleting(statement: str) -> None:
        if statement.startswith("DELETE") and not started:
            started.append(statement)
            other.start()
            time.sleep(0.3)  # for the other writer to come as far as it can

    assert cache._conn is not None
    cache._conn.set_trace_callback(deleting)
    with caplog.at_level(logging.WARNING, logger="vestigo.digests"):
        assert cache.prune(root, files[:1]) == 1
    other.join()

    assert not _warnings(caplog, "No digest cache")
    assert cache.cached_many(files) == [_sha(b"a.csv"), None]


def test_prune_that_finds_nothing_gone_takes_no_write_lock(
    tmp_path: Path, cache: FileDigests, caplog: pytest.LogCaptureFixture
):
    """Launches on the same data at once need not wait on each other for nothing."""
    root = tmp_path / "data"
    root.mkdir()
    kept = root / "kept.csv"
    kept.write_bytes(b"1,2\n")
    cache.digest(kept)

    with closing(sqlite3.connect(tmp_path / "state" / "digests.sqlite")) as other:
        other.execute("BEGIN IMMEDIATE")  # another process's write, under way
        with caplog.at_level(logging.WARNING, logger="vestigo.digests"):
            assert cache.prune(root, [kept]) == 0
        other.rollback()

    assert not _warnings(caplog, "No digest cache")
    assert cache.cached(kept) == _sha(b"1,2\n")


def _write_beside(path: Path) -> None:
    with closing(sqlite3.connect(path, timeout=30, isolation_level=None)) as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT INTO file_digests VALUES (x'00', 'stat', 'sha', 0)")
        db.execute("COMMIT")


def test_a_missing_file_raises(tmp_path: Path, cache: FileDigests):
    with pytest.raises(FileNotFoundError):
        cache.digest(tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        cache.cached(tmp_path / "absent.csv")


def test_an_inode_past_sqlites_integers_is_kept(
    tmp_path: Path, cache: FileDigests, stats: dict, reads: list[Path]
):
    """NFS fileids, overlayfs xino and ReFS's 128-bit ids pass a signed 64-bit integer."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    stats[data] = _stat_of(data, st_ino=2**64 - 1)

    assert cache.digest(data).sha256 == _sha(b"1,2\n")
    assert cache.cached(data) == _sha(b"1,2\n")
    assert reads == [data]


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows takes `..` off a path's spelling before any link"
)
def test_a_path_through_a_link_and_up_is_the_file_it_names(
    tmp_path: Path, cache: FileDigests, reads: list[Path]
):
    """``link/..`` is the link target's parent, not the link's. Normalized, the two paths
    below would share a row and evict each other: read on every call, never reused."""
    (tmp_path / "other" / "dir").mkdir(parents=True)
    (tmp_path / "other" / "data.csv").write_bytes(b"there\n")
    (tmp_path / "mnt").mkdir()
    (tmp_path / "mnt" / "data.csv").write_bytes(b"here\n")
    try:
        (tmp_path / "mnt" / "link").symlink_to(tmp_path / "other" / "dir")
    except OSError:
        pytest.skip("no symlinks here")
    through = tmp_path / "mnt" / "link" / ".." / "data.csv"
    here = tmp_path / "mnt" / "data.csv"

    for _ in range(2):
        assert cache.digest(through).sha256 == _sha(b"there\n")
        assert cache.digest(here).sha256 == _sha(b"here\n")
    assert len(reads) == 2


def test_a_name_that_is_not_utf8_is_read_remembered_and_pruned(
    tmp_path: Path, cache: FileDigests, reads: list[Path]
):
    root = tmp_path / "data"
    root.mkdir()
    try:
        # Windows's names are UTF-16: no bytes name to decode, so none that is not UTF-8.
        odd = root / os.fsdecode(b"caf\xe9.csv")
        odd.write_bytes(b"1,2\n")
    except (OSError, UnicodeError):
        pytest.skip("this filesystem only takes valid names")

    assert cache.digest(odd).sha256 == _sha(b"1,2\n")
    assert cache.cached(odd) == _sha(b"1,2\n")
    assert cache.prune(root, []) == 1
    assert cache.cached(odd) is None
    assert reads == [odd]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
@pytest.mark.timeout(10)
def test_a_fifo_is_refused_not_read(tmp_path: Path, cache: FileDigests):
    """Opening a FIFO blocks until a writer comes; a device's read never ends."""
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)

    with pytest.raises(OSError) as excinfo:
        cache.digest(pipe)
    assert excinfo.value.errno == errno.EINVAL
    assert cache.cached(pipe) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no read-only folders")
@pytest.mark.parametrize("where", ["explicit", "machine"])
def test_a_cache_that_cannot_be_opened_remembers_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reads: list[Path],
    where: str,
):
    """A long-lived caller asks again and again: remembering nothing would read each time,
    and a notebook's staleness check would never see a cell as fresh."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    if _writes_into(locked):
        pytest.skip("this account writes into a read-only folder")
    monkeypatch.setenv("VESTIGO_CACHE_DIR", str(locked))
    path = locked / "digests.sqlite" if where == "explicit" else None
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")

    with caplog.at_level(logging.WARNING, logger="vestigo.digests"):
        cache = _cache(path)
        try:
            assert cache.digest(data).sha256 == _sha(b"1,2\n")
            assert cache.cached(data) == _sha(b"1,2\n")
            assert cache.digest(data).sha256 == _sha(b"1,2\n")
        finally:
            cache.close()
    assert reads == [data]
    assert len(_warnings(caplog, "No digest cache")) == 1
    assert not (locked / "digests.sqlite").exists()


def test_many_files_are_read_together_and_remembered_in_one_write(
    tmp_path: Path, cache: FileDigests
):
    files = [tmp_path / f"{i:02}.csv" for i in range(12)]
    for i, path in enumerate(files):
        path.write_bytes(b"%d\n" % i)
    one_by_one = _cache(tmp_path / "other.sqlite")
    try:
        expected = [one_by_one.digest(path) for path in files]
    finally:
        one_by_one.close()
    statements: list[str] = []
    assert cache._conn is not None
    cache._conn.set_trace_callback(statements.append)

    assert cache.digest_many(files, workers=4) == expected

    # Each statement as SQLite ran it: the lookup's transaction, then the write's.
    transactions = " ".join(statements).split("BEGIN")
    writes = [t for t in transactions if "INSERT" in t]
    assert len(writes) == 1 and writes[0].count("INSERT") == len(files)
    assert cache.cached_many(files) == [d.sha256 for d in expected]


def _remembered(path: Path) -> set[Path]:
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute("SELECT path FROM file_digests").fetchall()
    return {Path(os.fsdecode(where)) for (where,) in rows}


def test_a_digest_unused_for_ninety_days_is_forgotten(tmp_path: Path):
    """Rows of files deleted, moved or no longer read do not pile up."""
    old, used, new = (tmp_path / name for name in ("old.csv", "used.csv", "new.csv"))
    for path in (old, used, new):
        path.write_bytes(path.name.encode())
    now = [time.time_ns() + _LATER_NS]
    path = tmp_path / "digests.sqlite"
    cache = FileDigests(path, clock=lambda: now[0])
    try:
        cache.digest_many([old, used])
        now[0] += 60 * _DAY_NS
        assert cache.cached(used) is not None
        now[0] += 31 * _DAY_NS
        cache.digest(new)  # a write: 91 days since old's last use, 31 since used's
    finally:
        cache.close()

    assert _remembered(path) == {used.absolute(), new.absolute()}


def test_a_look_marks_its_rows_used_at_most_once_a_day(tmp_path: Path):
    """So a look stays a read, and takes no write lock, on most days."""
    data = tmp_path / "a.csv"
    data.write_bytes(b"1,2\n")
    now = [time.time_ns() + _LATER_NS]
    cache = FileDigests(tmp_path / "digests.sqlite", clock=lambda: now[0])
    try:
        cache.digest(data)
        statements: list[str] = []
        assert cache._conn is not None
        cache._conn.set_trace_callback(statements.append)
        now[0] += 3600 * 10**9
        cache.cached(data)
        cache.digest(data)
        assert not [s for s in statements if s.startswith(("BEGIN IMMEDIATE", "UPDATE"))]

        now[0] += 2 * _DAY_NS
        cache.cached(data)
        assert len([s for s in statements if s.startswith("UPDATE")]) == 1
    finally:
        cache.close()


def test_a_walk_s_stats_are_not_taken_again(
    tmp_path: Path, cache: FileDigests, monkeypatch: pytest.MonkeyPatch
):
    """A caller that just walked has each file's stat: a hit costs no other, and a file
    changed since the walk comes back unsettled."""
    kept, moved = tmp_path / "kept.csv", tmp_path / "moved.csv"
    for path in (kept, moved):
        path.write_bytes(b"1,2\n")
    cache.digest(kept)
    walked = [_REAL_STAT(kept), _REAL_STAT(moved)]
    with open(moved, "ab") as f:
        f.write(b"3,4\n")
    taken: list[Path] = []

    def counted(path, *args, **kwargs):
        taken.append(Path(path))
        return _REAL_STAT(path, *args, **kwargs)

    monkeypatch.setattr(vestigo.digests.os, "stat", counted)
    hit, changed = cache.digest_many([kept, moved], stats=walked)

    assert hit == Digest(_sha(b"1,2\n"), settled=True)
    assert not changed.settled
    assert taken == [moved]


def test_a_file_stamped_past_the_clock_is_not_remembered_and_said_once(
    tmp_path: Path, cache: FileDigests, caplog: pytest.LogCaptureFixture
):
    """A share whose server clock runs ahead: each look reads it, so say why, once."""
    ahead = time.time_ns() + 3600 * 10**9
    files = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for path in files:
        path.write_bytes(path.name.encode())
        os.utime(path, ns=(ahead, ahead))

    with caplog.at_level(logging.WARNING, logger="vestigo.digests"):
        cache.digest_many(files)
        cache.digest(files[0])

    assert cache.cached_many(files) == [None, None]
    [said] = _warnings(caplog, "later than this machine's clock")
    assert str(files[0]) in said.getMessage()
