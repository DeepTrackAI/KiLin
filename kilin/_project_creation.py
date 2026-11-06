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

- **Conservative Stop cleanup**

    Remove empty creation claims without erasing any project files.

Module Structure
----------------

- ``ProjectAlreadyExistsError`` and ``WorkspaceOccupiedError``

    Report protected destinations through the public project API.

- ``claim_workspace``

    Reserve an explicit empty destination or a fresh default sibling.

- ``cleanup_stopped_creation``

    Safely remove empty claims after a stopped engine; retain project data.

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


def cleanup_stopped_creation(
    projects_folder: Path,
    reported_workspace: Path | None,
    engine_pid: int,
) -> tuple[list[str], list[str]]:
    """Clean only an empty creation claim after its engine was stopped.

    A ``project_created`` event means the profile has been published, even
    if the GUI has not yet received ``opened``. Such a workspace is always
    preserved. A matching process-ID marker identifies a possible unfinished
    claim, but is not proof that its contents can be discarded: process IDs
    can be reused, and files may have been added or edited meanwhile.

    Only a directory containing its creation marker and nothing else may be
    removed, using ``rmdir`` rather than recursive deletion. The marker must
    also record that the engine created the directory, rather than claiming
    an existing empty folder. Older markers without this distinction remain
    untouched. Populated directories require deliberate recovery.

    Parameters
    ----------
    projects_folder: pathlib.Path
        Parent of the GUI's automatically allocated project directories.
    reported_workspace: pathlib.Path | None
        Workspace reported by ``project_created`` but not yet acknowledged
        as opened by the Qt window, or None.
    engine_pid: int
        Process ID of the stopped engine, or 0 when unknown.

    Returns
    -------
    tuple[list[str], list[str]]
        Removed empty claim directories and known preserved workspaces.
        Preserved paths include the reported workspace and matching claims
        containing files; they are not an exhaustive search for leftovers.

    Notes
    -----
    This is conservative crash cleanup, not a secure filesystem transaction
    against hostile concurrent modifications. Never infer that a populated
    project belongs to a stopped process solely from a PID marker.

    """

    parent = projects_folder.resolve()
    preserved: list[str] = []
    protected = None
    if reported_workspace is not None:
        workspace = Path(reported_workspace)
        if not workspace.is_symlink() and workspace.is_dir():
            protected = workspace.resolve()
            preserved.append(str(protected))
    if engine_pid <= 0 or not parent.is_dir():
        return [], preserved

    removed: list[str] = []
    owner = f"pid={engine_pid}\ncreated=1\n"
    old_owner = f"pid={engine_pid}\n"
    existing_owner = f"{old_owner}created=0\n"
    for folder in parent.iterdir():
        if folder.is_symlink() or not folder.is_dir():
            continue
        root = folder.resolve()
        if root.parent != parent:
            continue
        marker = folder / _MARKER
        if marker.is_symlink():
            continue
        try:
            claim = marker.read_text(encoding="ascii")
        except (OSError, UnicodeError):
            continue
        if claim in {old_owner, existing_owner}:
            if root != protected:
                preserved.append(str(root))
            continue
        if claim != owner or root == protected:
            continue
        try:
            entries = list(folder.iterdir())
        except OSError:
            preserved.append(str(root))
            continue
        if entries != [marker]:
            preserved.append(str(root))
            continue
        try:
            marker.unlink()
            folder.rmdir()
        except OSError:
            # Another entry or a permission change must never trigger
            # recursive deletion, even if the marker was already removed.
            preserved.append(str(root))
            continue
        removed.append(str(root))
    return removed, preserved


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
    The exclusive marker records its PID and whether this attempt created
    the directory. Cooperating creators never guess if a crashed process
    remains alive. Populated stale claims remain protected. Normal cleanup
    removes only this attempt's marker and, when created here, its
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
        claim = f"pid={os.getpid()}\ncreated={int(created)}\n"
        os.write(descriptor, claim.encode("ascii"))
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
