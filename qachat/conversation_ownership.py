"""Provide exclusive write ownership for one conversation JSON path.

``ConversationFileLock`` protects an authoritative conversation record using a
companion ``.json.lock`` file. Holding an operating-system advisory lock on the
companion, rather than on the JSON inode, survives atomic JSON replacement.
A process-local registry also rejects duplicate owners before opening another
file descriptor. Canonical paths make aliases such as ``/var`` and
``/private/var`` refer to the same writer on macOS.

The lock is nonblocking: contention raises ``ConversationLockedError`` without
reading or modifying the conversation JSON. Forked child processes close
inherited lock descriptors so they do not prolong the parent's ownership.
On POSIX the implementation uses
``fcntl.flock``; on Windows it uses ``msvcrt.locking``. The operating system
releases locks when the owning process exits. The companion file is retained;
its presence alone does not indicate active ownership.

This protocol coordinates cooperating QAChat writers, not external programs
that bypass the lock. It does not provide a multi-file transaction, durable
sync on power loss, or a guarantee against crashes during separate writes.

Key Features
------------

- **Single-writer ownership**

    Exclude simultaneous writers across threads, logger instances, and
    independent Python processes without waiting for another writer.

- **Atomic-replacement compatibility**

    Keep the lock attached to a stable sidecar as the JSON is replaced.

- **Explicit borrowing**

    Permit code already holding an ownership token to perform a coordinated
    operation without acquiring a second, competing lock.

Module Structure
----------------

- ``ConversationLockedError``

    Raised when the target conversation already has a writer.

- ``ConversationFileLock``

    Acquire, inspect, and release one conversation's write ownership.

- ``conversation_write_ownership``

    Context manager using either a caller-held token or a short-lived lock.

Examples
--------
>>> from qachat.conversation_ownership import ConversationFileLock
>>> lock = ConversationFileLock("/path/to/conversation_example.json")
>>> with lock:
...     # Read, modify, and replace the conversation while owning its path.
...     pass

Close long-lived owners explicitly. Do not remove a companion lock file while
another process might be using it: doing so can split the locking protocol.

"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
from threading import Lock
from typing import Iterator
from weakref import WeakSet

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = [
    "ConversationFileLock",
    "ConversationLockedError",
    "conversation_write_ownership",
]


class ConversationLockedError(RuntimeError):
    """Indicate that another owner is writing this conversation."""


_REGISTRY_LOCK = Lock()
_ACTIVE_PATHS: set[Path] = set()
_OWNERS: WeakSet[ConversationFileLock] = WeakSet()


def _after_fork_in_child() -> None:
    """Drop inherited lock descriptors without unlocking the parent.

    Notes
    -----
    POSIX flock locks may share an open-file description across fork. Closing
    only the child's inherited descriptor ensures an unrelated child process
    cannot keep its parent's ownership alive after the parent exits. Never
    issue LOCK_UN here: it could also unlock the parent's shared description.
    The inherited registry mutex is replaced because its owning thread may
    no longer exist in the child.

    """

    global _REGISTRY_LOCK
    for owner in tuple(_OWNERS):
        descriptor = owner._descriptor
        if descriptor is not None:
            owner._descriptor = None
            try:
                os.close(descriptor)
            except OSError:
                pass
    _OWNERS.clear()
    _ACTIVE_PATHS.clear()
    _REGISTRY_LOCK = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


class ConversationFileLock:
    """Own the exclusive write lease for one canonical conversation path.

    Parameters
    ----------
    path: str | pathlib.Path
        Conversation JSON path; the sidecar is ``<path>.lock``.

    Attributes
    ----------
    path: pathlib.Path
        Canonical conversation JSON path.
    lock_path: pathlib.Path
        Persistent companion file used for the operating-system lock.

    Methods
    -------
    acquire()
        Acquire without waiting; raise if another writer owns the path.
    holds(path)
        Check whether this token currently owns the specified path.
    release()
        Release the OS lock and process-local registration, idempotently.

    Notes
    -----
    An owner must stay alive for the entire read/modify/write lifetime.
    Another component may borrow this token but must not release it.

    """

    def __init__(self, path: str | Path) -> None:
        """Remember the canonical path without acquiring or writing.

        Parameters
        ----------
        path: str | pathlib.Path
            Conversation JSON whose companion will be locked.

        """

        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._descriptor: int | None = None

    def acquire(self) -> None:
        """Acquire exclusive ownership without waiting for other writers.

        Raises
        ------
        ConversationLockedError
            If the conversation is already owned in this or another process.
        OSError
            If the sidecar cannot be created, opened, or locked for another
            reason (including unsupported filesystems).
        RuntimeError
            If this token has already acquired its lock.

        Notes
        -----
        Registration is rolled back on every unsuccessful acquisition.
        A symlinked sidecar is rejected where ``O_NOFOLLOW`` is available.

        """

        if self._descriptor is not None:
            raise RuntimeError("This conversation lock is already held.")
        with _REGISTRY_LOCK:
            if self.path in _ACTIVE_PATHS:
                raise ConversationLockedError(
                    f"Conversation already has a writer: {self.path}"
                )
            _ACTIVE_PATHS.add(self.path)

        descriptor: int | None = None
        stage = "open"
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(self.lock_path, flags, 0o600)
            stage = "lock"
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._descriptor = descriptor
            _OWNERS.add(self)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            with _REGISTRY_LOCK:
                _ACTIVE_PATHS.discard(self.path)
            if stage == "lock" and (
                error.errno in (errno.EACCES, errno.EAGAIN)
                or getattr(error, "winerror", None) in (32, 33)
            ):
                raise ConversationLockedError(
                    f"Conversation already has a writer: {self.path}"
                ) from error
            raise
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            with _REGISTRY_LOCK:
                _ACTIVE_PATHS.discard(self.path)
            raise

    def holds(self, path: str | Path) -> bool:
        """Return whether this token currently owns the canonical path.

        Parameters
        ----------
        path: str | pathlib.Path
            Conversation JSON path to compare with the owned path.

        Returns
        -------
        bool
            True only for a held token with the same canonical path.

        """

        return (
            self._descriptor is not None
            and self.path == Path(path).expanduser().resolve()
        )

    def release(self) -> None:
        """Release the held lock without deleting its companion file.

        Notes
        -----
        Repeated calls are harmless. The in-process reservation is released
        even if the operating-system unlock or descriptor close fails.

        """

        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        _OWNERS.discard(self)
        try:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            try:
                os.close(descriptor)
            finally:
                with _REGISTRY_LOCK:
                    _ACTIVE_PATHS.discard(self.path)

    def __enter__(self) -> ConversationFileLock:
        """Acquire and return this ownership token."""

        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        exc_traceback: object,
    ) -> None:
        """Release ownership regardless of the context's outcome.

        Parameters
        ----------
        exc_type: object
            Exception type from the context, or None.
        exc_value: object
            Exception instance from the context, or None.
        exc_traceback: object
            Exception traceback from the context, or None.

        """

        self.release()


@contextmanager
def conversation_write_ownership(
    path: str | Path,
    *,
    ownership: ConversationFileLock | None = None,
) -> Iterator[None]:
    """Hold a short-lived lock or validate a borrowed existing owner.

    Parameters
    ----------
    path: str | pathlib.Path
        Conversation file whose contents may be modified.
    ownership: ConversationFileLock | None, optional
        Caller-held token for this exact path. The borrower never releases
        it. ``None`` acquires an independent lease for this operation.

    Yields
    ------
    None
        Control while exclusive write ownership is guaranteed.

    Raises
    ------
    ValueError
        If the supplied token does not currently own this path.
    ConversationLockedError
        If acquiring an independent lease would conflict with a writer.

    """

    if ownership is not None:
        if not ownership.holds(path):
            raise ValueError("The supplied conversation ownership is invalid.")
        yield
    else:
        with ConversationFileLock(path):
            yield
