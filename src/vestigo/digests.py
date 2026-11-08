"""The sha256 of local files, remembered by their stat so an unchanged file is not read.

A remembered digest is reused while the file's stat holds: its device, inode, size,
mtime and ctime must all be as they were. ctime too, since nothing sets it back, so a
rewrite that keeps the size and restores the mtime still shows (on Windows ``st_ctime``
is the creation time, so there it does not). A stat that moves for no reason (a
``chmod``, a hard link, an xattr) costs a read, never a wrong digest.

A digest is remembered only for a file last changed :data:`_RACY_NS` before its read
began (git's rule for its index): a write in the same timestamp tick as the read could
leave every stat field as it was. So every remembered digest is trusted.

The cache is a SQLite file readable by its owner only, since its rows name every file
read. ``vestigo.torch`` keeps the machine's (``path=None``); a caller with a life of
its own, such as a notebook, passes its own file. Imports no torch.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Collection, Generator, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vestigo.file_modes import private_dir, private_file

__all__ = ["Digest", "FileChanged", "FileDigests"]

logger = logging.getLogger(__name__)

# Wider than the coarsest mtime in use (FAT: 2 s).
_RACY_NS = 2_000_000_000
_CHUNK = 8 * 1024 * 1024
_DAY_NS = 86_400 * 10**9
# A row is marked used at most once a day, so a look stays a read on most days.
_TOUCH_NS = _DAY_NS
# A row unused this long goes: the file was deleted, moved, or is no longer read.
_UNUSED_NS = 90 * _DAY_NS

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS file_digests (path BLOB PRIMARY KEY, stat TEXT NOT NULL,"
    " sha256 TEXT NOT NULL, used_at INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS file_digests_used_at ON file_digests (used_at)",
)

_Row = tuple[bytes, str, str]


@dataclass(frozen=True)
class Digest:
    """A file's sha256. ``settled`` is false when the file changed while it was read:
    the digest may be of bytes the file never held whole, and is not remembered."""

    sha256: str
    settled: bool


class FileChanged(Exception):
    """A file changed while it was read, from ``digest_many(settled_only=True)``. Its
    argument is the file's path."""


def _cache_path() -> Path:
    if root := os.environ.get("VESTIGO_CACHE_DIR"):
        return Path(root) / "digests.sqlite"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "vestigo" / "digests.sqlite"


def _row_path(path: str | os.PathLike[str]) -> bytes:
    # Absolute, never normalized: ``link/..`` is the link target's parent, not the
    # link's. As the OS's bytes: a name that is not UTF-8 has no str SQLite can store.
    return os.fsencode(Path(path).absolute())


def _stat_key(st: os.stat_result) -> str:
    # As text: an inode can pass SQLite's signed 64 bits (NFS, overlayfs xino, ReFS).
    return str((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))


def _read_digest(path: str | os.PathLike[str]) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _open(path: Path, *, own_folder: bool) -> sqlite3.Connection:
    if own_folder:
        private_dir(path.parent)
    else:
        # The caller's folder: made owner-only if missing, never narrowed.
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # The rollback journal, not WAL: WAL needs shared memory, which an NFS home lacks.
    conn = sqlite3.connect(
        private_file(path), timeout=30, check_same_thread=False, isolation_level=None
    )
    try:
        if own_folder and conn.execute("PRAGMA user_version").fetchone()[0] < 1:
            # The machine cache's table of 0.4 and before keyed paths as str; a cache, so
            # dropped, not moved. Once: a 0.4 process beside this one makes it again, and
            # uses it. A caller's own file may hold a table of that name.
            conn.execute("DROP TABLE IF EXISTS digests")
            conn.execute("PRAGMA user_version = 1")
        for statement in _SCHEMA:
            conn.execute(statement)
    except sqlite3.Error:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection, begin: str = "BEGIN") -> Generator[sqlite3.Connection]:
    conn.execute(begin)
    try:
        yield conn
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


class FileDigests:
    """Local files' sha256, read only where the cache holds none for the file's stat.

    *path* is the cache file; ``None`` is the machine's, in ``$VESTIGO_CACHE_DIR``, else
    ``$XDG_CACHE_HOME/vestigo``, else ``~/.cache/vestigo``. When it cannot be opened (no
    home, a read-only one, SQLite on NFS) one warning is logged and digests are
    remembered in memory for the life of the instance instead. A file's digest unused
    for 90 days is forgotten (a look marks it used at most once a day).

    A file is named by its path made absolute, never resolved or normalized: a link and
    its target each get a row, both keyed on the target's stat. Safe to share between
    threads.

    *clock* is the wall clock in ns, ``time.time_ns`` by default. A local file's times
    come from the same clock; a network share's server clock can drift from it, and a
    file stamped ahead of *clock* is never remembered (logged once per instance).
    """

    def __init__(self, path: Path | None = None, *, clock: Callable[[], int] | None = None):
        self._clock = clock or time.time_ns
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._memory: dict[bytes, tuple[str, str]] = {}
        self._where: Path | None = None
        self._skew_logged = False
        try:
            self._where = _cache_path() if path is None else Path(path)
            self._conn = _open(self._where, own_folder=path is None)
        except (OSError, RuntimeError, sqlite3.Error):
            self._degraded()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def cached(self, path: str | os.PathLike[str]) -> str | None:
        """*path*'s sha256 if the remembered one still holds, else ``None``. Never reads
        the file.

        Raises:
            OSError: *path* cannot be stat'ed.
        """
        return self.cached_many([path])[0]

    def cached_many(self, paths: Iterable[str | os.PathLike[str]]) -> list[str | None]:
        """:meth:`cached` of each of *paths*, in one read of the cache."""
        now = self._clock()
        known, used = self._lookup([(_row_path(p), _stat_key(os.stat(p))) for p in paths], now)
        self._write([], used, now)
        return known

    def digest(self, path: str | os.PathLike[str]) -> Digest:
        """*path*'s sha256: the remembered one if it still holds, else read.

        Raises:
            OSError: *path* cannot be stat'ed or read, or is not a regular file (a FIFO
                would block the read, a device never end it).
        """
        return self.digest_many([path])[0]

    def digest_many(
        self,
        paths: Iterable[str | os.PathLike[str]],
        *,
        stats: Iterable[os.stat_result] | None = None,
        workers: int | None = None,
        settled_only: bool = False,
    ) -> list[Digest]:
        """:meth:`digest` of each of *paths*: one read of the cache, the misses read on
        *workers* threads (``min(32, cpu_count)`` by default), and what they found
        remembered in one write. A read that fails stops the reads not yet started, and
        what was read before it is remembered; a file that changed while it was read
        does not stop them, and comes back unsettled.

        *stats* are the paths' stats as the caller just took them, in a walk: none is
        taken again before the reads, and a file whose stat has moved since comes back
        unsettled.

        *settled_only* stops at the first file that changed while it was read, as a
        failed read does: what was read before it is remembered, and it raises.

        Raises:
            OSError: as :meth:`digest`, for the first of *paths* that fails.
            FileChanged: with *settled_only*, a file changed while it was read.
        """
        paths = list(paths)
        taken = [os.stat(p) for p in paths] if stats is None else list(stats)
        stats_of = []
        for path, st in zip(paths, taken, strict=True):
            if not stat.S_ISREG(st.st_mode):
                raise OSError(errno.EINVAL, "not a regular file", str(path))
            stats_of.append((_row_path(path), _stat_key(st), max(st.st_mtime_ns, st.st_ctime_ns)))
        # Before any file is read: a write after its stat was taken shows in the stat
        # only if that write's timestamp tick is later than the one the stat holds.
        now = self._clock()
        known, used = self._lookup([(where, key) for where, key, _ in stats_of], now)
        found = [None if sha is None else Digest(sha, settled=True) for sha in known]
        missing = [i for i, sha in enumerate(known) if sha is None]
        if not missing:
            self._write([], used, now)
            return cast(list[Digest], found)

        def read(i: int) -> Digest:
            sha = _read_digest(paths[i])
            return Digest(sha, settled=_stat_key(os.stat(paths[i])) == stats_of[i][1])

        stopped: Exception | None = None
        pool = ThreadPoolExecutor(max_workers=workers or min(32, os.cpu_count() or 1))
        try:
            pending = {pool.submit(read, i): i for i in missing}
            for future in as_completed(pending):
                i = pending[future]
                try:
                    found[i] = digest = future.result()
                except OSError as exc:
                    stopped = exc
                    break
                if settled_only and not digest.settled:
                    stopped = FileChanged(paths[i])
                    break
        finally:
            # The reads not yet started are not started at all.
            pool.shutdown(cancel_futures=True)
        read_now = [(i, d) for i in missing if (d := found[i]) is not None]
        self._skewed([paths[i] for i, _ in read_now if stats_of[i][2] > now])
        rows = [
            (stats_of[i][0], stats_of[i][1], d.sha256)
            for i, d in read_now
            if d.settled and stats_of[i][2] < now - _RACY_NS
        ]
        # What was read before a stop is remembered too.
        self._write(rows, used, now)
        if stopped is not None:
            raise stopped
        return cast(list[Digest], found)

    def prune(self, root: str | os.PathLike[str], seen: Collection[str | os.PathLike[str]]) -> int:
        """Forget the files under *root* not in *seen*, what a walk of it just found.

        *root* is spelled as the walk spelled the paths it read. Returns how many were
        forgotten.
        """
        # The separator: ``data2`` is not under ``data``.
        prefix = os.fsencode(os.path.join(Path(root).absolute(), ""))
        below = prefix[:-1] + bytes([prefix[-1] + 1])
        keep = {_row_path(p) for p in seen}
        with self._lock:
            if self._conn is not None:
                try:
                    # A look first: most launches find nothing gone, and need no write lock.
                    with _transaction(self._conn) as conn:
                        gone = self._gone(conn, prefix, below, keep)
                    if not gone:
                        return 0
                    # IMMEDIATE, so looked at again: a read lock raised to a write one
                    # fails at once, unwaited, when another connection is writing.
                    with _transaction(self._conn, "BEGIN IMMEDIATE") as conn:
                        gone = self._gone(conn, prefix, below, keep)
                        conn.executemany("DELETE FROM file_digests WHERE path = ?", gone)
                    return len(gone)
                except sqlite3.Error:
                    self._degraded()
            forgotten = [w for w in self._memory if w.startswith(prefix) and w not in keep]
            for where in forgotten:
                del self._memory[where]
            return len(forgotten)

    @staticmethod
    def _gone(
        conn: sqlite3.Connection, prefix: bytes, below: bytes, keep: set[bytes]
    ) -> list[tuple[bytes]]:
        rows = conn.execute(
            "SELECT path FROM file_digests WHERE path >= ? AND path < ?", (prefix, below)
        ).fetchall()
        return [(where,) for (where,) in rows if where not in keep]

    def _lookup(
        self, keys: Sequence[tuple[bytes, str]], now: int
    ) -> tuple[list[str | None], list[bytes]]:
        """Each key's remembered sha256, and the paths of the hits last marked used over
        a day before *now*."""
        with self._lock:
            rows = self._rows([where for where, _ in keys], now)
        known: list[str | None] = []
        used = []
        for row, (where, key) in zip(rows, keys, strict=True):
            if row is None or row[0] != key:
                known.append(None)
                continue
            known.append(row[1])
            if row[2] < now - _TOUCH_NS:
                used.append(where)
        return known, used

    def _rows(self, paths: Sequence[bytes], now: int) -> list[tuple[str, str, int] | None]:
        if self._conn is not None:
            try:
                with _transaction(self._conn) as conn:
                    return [
                        conn.execute(
                            "SELECT stat, sha256, used_at FROM file_digests WHERE path = ?",
                            (where,),
                        ).fetchone()
                        for where in paths
                    ]
            except sqlite3.Error:
                self._degraded()
        # Memory lasts the instance's life: never marked used, never forgotten.
        return [
            None if (row := self._memory.get(where)) is None else (*row, now) for where in paths
        ]

    def _write(self, rows: Sequence[_Row], used: Sequence[bytes], now: int) -> None:
        """*rows* remembered and *used* marked used at *now*, in one write, which also
        forgets the rows unused for :data:`_UNUSED_NS`."""
        if not rows and not used:
            return
        with self._lock:
            if self._conn is not None:
                try:
                    with _transaction(self._conn, "BEGIN IMMEDIATE") as conn:
                        conn.executemany(
                            "INSERT OR REPLACE INTO file_digests VALUES (?, ?, ?, ?)",
                            [(*row, now) for row in rows],
                        )
                        conn.executemany(
                            "UPDATE file_digests SET used_at = ? WHERE path = ?",
                            [(now, where) for where in used],
                        )
                        conn.execute(
                            "DELETE FROM file_digests WHERE used_at < ?", (now - _UNUSED_NS,)
                        )
                    return
                except sqlite3.Error:
                    self._degraded()
            self._memory.update({where: (key, sha) for where, key, sha in rows})

    def _degraded(self) -> None:
        """From the cache file to memory, for the rest of this instance's life. Called in
        an ``except`` block, whose error the warning carries."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        logger.warning(
            "No digest cache at %s; remembering digests in memory", self._where, exc_info=True
        )

    def _skewed(self, ahead: Sequence[str | os.PathLike[str]]) -> None:
        if ahead and not self._skew_logged:
            self._skew_logged = True
            logger.warning(
                "%s is stamped later than this machine's clock (a share with a skewed "
                "server clock?); files like it are read at every look until it is past",
                ahead[0],
            )
