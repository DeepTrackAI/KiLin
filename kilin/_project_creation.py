"""Reserve empty destinations for exclusive project creation.

Creation never adopts a nonempty folder, follows a workspace-root symlink,
or reuses a saved project. An exclusive marker coordinates cooperating
creators, including independent processes. A crashed attempt leaves its
marker and partial files protected rather than being silently recovered.

Key Features
------------

- **Create-only ownership**

    Reject existing projects and unrelated files before resource writes.

- **Deterministic allocation**

    Allocate the preferred default name, then numbered siblings, atomically.

Module Structure
----------------

- ``ProjectAlreadyExistsError`` and ``WorkspaceOccupiedError``

    Report protected destinations through the public project API.

- ``claim_workspace``

    Reserve an explicit empty destination or a fresh default sibling.

Examples
--------
>>> import kilin
>>> project = kilin.initialize_project("/path/to/new_project", "code")
>>> # Calling the same creation function again raises an ownership error.

"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

__all__ = ["ProjectAlreadyExistsError", "WorkspaceOccupiedError"]

_MARKER = ".kilin-creating"


class WorkspaceOccupiedError(ValueError):
    """Report a destination that cannot be claimed for a new project.

    Parameters
    ----------
    directory: pathlib.Path
        Protected destination or containing project.
    reason: str
        Explanation suitable for an interface to display.

    Attributes
    ----------
    directory: pathlib.Path
        Path associated with the conflict.

    """

    def __init__(self, directory: Path, reason: str) -> None:
        """Store the protected destination and explanatory message.

        Parameters
        ----------
        directory: pathlib.Path
            Protected destination or containing project.
        reason: str
            Human-readable explanation of the conflict.

        """

        self.directory = directory
        super().__init__(f"{reason}: {directory}")


class ProjectAlreadyExistsError(WorkspaceOccupiedError):
    """Reject implicit reuse of an existing project.

    Parameters
    ----------
    directory: pathlib.Path
        Existing project that must be opened explicitly.

    """

    def __init__(self, directory: Path) -> None:
        """Identify the project without opening or modifying it.

        Parameters
        ----------
        directory: pathlib.Path
            Project containing a profile entry, including an invalid one.

        """

        super().__init__(
            directory,
            "A project already exists; open it explicitly or choose "
            "another workspace",
        )


def destination_path(
    directory: str | Path, *, allow_occupied_link: bool = False
) -> Path:
    """Resolve a destination without creating any filesystem entries.

    Parameters
    ----------
    directory: str | pathlib.Path
        Requested explicit workspace or preferred default path.
    allow_occupied_link: bool, optional
        Preserve a final linked name as an occupied allocation candidate,
        never following it. Defaults to False for explicit destinations.

    Returns
    -------
    pathlib.Path
        Canonical destination, excluding home and filesystem roots.

    Raises
    ------
    ValueError
        If the path is blank, a home directory, or a filesystem root.
    WorkspaceOccupiedError
        If the final path is a symbolic link or lies inside a project.

    """

    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise ValueError("Choose a workspace folder.")
    requested = Path(directory).expanduser()
    if requested.is_symlink() and not allow_occupied_link:
        raise WorkspaceOccupiedError(requested, "Workspace must not be a link")
    root = requested.parent.resolve() / requested.name
    root = Path(os.path.abspath(root))
    if root in {Path.home().resolve(), Path(root.anchor)}:
        raise ValueError(
            "Choose a project workspace, not a home or drive root."
        )
    for parent in root.parents:
        profile = parent / "profile.ini"
        if profile.exists() or profile.is_symlink():
            raise ProjectAlreadyExistsError(parent)
    return root


def check_available(root: Path) -> None:
    """Check explicit creation preconditions without writing or reserving.

    Parameters
    ----------
    root: pathlib.Path
        Canonical workspace destination.

    Raises
    ------
    WorkspaceOccupiedError
        If the destination is nonempty or a symbolic link.
    ProjectAlreadyExistsError
        If a profile entry already exists, even if malformed.
    NotADirectoryError
        If the destination is an ordinary file.

    Notes
    -----
    This is a preliminary check only. ``claim_workspace`` repeats it while
    obtaining exclusive ownership; callers must not use a check as a lease.

    """

    if root.is_symlink():
        raise WorkspaceOccupiedError(root, "Workspace must not be a link")
    profile = root / "profile.ini"
    if profile.exists() or profile.is_symlink():
        raise ProjectAlreadyExistsError(root)
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(f"Choose a workspace folder: {root}")
        if any(root.iterdir()):
            raise WorkspaceOccupiedError(
                root, "New projects require an empty workspace"
            )


@contextmanager
def claim_workspace(
    preferred: str | Path, *, allocate: bool = False
) -> Iterator[Path]:
    """Reserve a creation destination for the entire initialization.

    Parameters
    ----------
    preferred: str | pathlib.Path
        Explicit destination or preferred default name.
    allocate: bool, optional
        When true, reserve a nonexistent sibling, trying ``-2``, ``-3``,
        and subsequent numbers after the preferred name. Existing empty
        folders are also skipped. Defaults to ``False``.

    Yields
    ------
    pathlib.Path
        Exclusively claimed canonical destination.

    Raises
    ------
    WorkspaceOccupiedError
        If an explicit destination is occupied or another creator owns it.
    OSError
        If reservation fails for a reason other than a name collision.

    Notes
    -----
    The exclusive marker coordinates cooperating creators without guessing
    whether a crashed process is still alive. Stale markers remain protected.
    Cleanup removes only this attempt's marker and, when created here, its
    empty root. Initialization owns cleanup of its other resources. This is
    not protection against hostile processes changing the filesystem.

    """

    root = destination_path(preferred, allow_occupied_link=allocate)
    base = root
    number = 1
    created = False
    while True:
        if not allocate:
            check_available(root)
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
            created = True
        except FileExistsError:
            if allocate:
                number += 1
                root = base.with_name(f"{base.name}-{number}")
                continue
            check_available(root)
        break

    marker = root / _MARKER
    descriptor = None
    identity = None
    try:
        try:
            descriptor = os.open(
                marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise WorkspaceOccupiedError(
                root, "Another creation owns this workspace"
            ) from error
        identity = os.fstat(descriptor)
        # A creator may have lost the race between the first empty check
        # and the exclusive marker. Never initialize over its contents.
        if any(path.name != _MARKER for path in root.iterdir()):
            profile = root / "profile.ini"
            if profile.exists() or profile.is_symlink():
                raise ProjectAlreadyExistsError(root)
            raise WorkspaceOccupiedError(root, "Workspace is no longer empty")
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        yield root
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if identity is not None:
            try:
                current = marker.lstat()
                if (current.st_dev, current.st_ino) == (
                    identity.st_dev, identity.st_ino
                ):
                    marker.unlink()
            except FileNotFoundError:
                pass
        if created:
            try:
                root.rmdir()
            except OSError:
                pass
