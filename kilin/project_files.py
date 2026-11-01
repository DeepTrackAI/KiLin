"""Manage project-owned outputs and templates independently of the GUI.

Only files beneath a project's ``outputs/`` and ``templates/`` folders may
be added, replaced, or removed. Every operation checks the requested paths
and refuses symbolic-link traversal. The project file inventory includes
empty managed directories for any frontend or Python API consumer.

Key Features
------------

- **Explicit operations**

    Copy new files, replace existing files, and remove project-owned entries.

- **Constrained destinations**

    Restrict writes to the two user-editable folders, never to source copies,
    profiles, indexes, or logs.

- **GUI-independent inventory**

    List project-owned files and directories without presentation logic.

Module Structure
----------------

- ``add_project_files``
- ``replace_project_file``
- ``remove_project_files``
- ``list_project_files``

Examples
--------
>>> from kilin.project_files import list_project_files
>>> isinstance(list_project_files, object)
True

"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from tempfile import NamedTemporaryFile
from typing import Sequence

from kilin.project import QAProject
from kilin.workspace_resources import (
    OUTPUTS_FOLDER,
    PROJECT_SOURCES_DIRECTORY,
    TEMPLATES_FOLDER,
)

__all__ = [
    "add_project_files",
    "replace_project_file",
    "remove_project_files",
    "list_project_files",
]

_EDITABLE_FOLDERS = frozenset((OUTPUTS_FOLDER, TEMPLATES_FOLDER))
_INVENTORY_LIMIT = 10_000


def _project_root(project: QAProject) -> Path:
    """Return the resolved, existing project directory.

    Parameters
    ----------
    project: QAProject
        Project whose workspace is being managed.

    Returns
    -------
    pathlib.Path
        Canonical project root.

    """

    if not isinstance(project, QAProject):
        raise TypeError("Expected an opened QAProject.")
    return project.root.resolve(strict=True)


def _managed_directory(root: Path, folder: str) -> Path:
    """Require one of the two real, editable project directories.

    Parameters
    ----------
    root: pathlib.Path
        Canonical project root.
    folder: str
        User-editable folder name.

    Returns
    -------
    pathlib.Path
        The folder, without following a final symbolic link.

    """

    if folder not in _EDITABLE_FOLDERS:
        raise ValueError("Only outputs/ and templates/ can be edited.")
    directory = root / folder
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Invalid project folder: {directory}")
    return directory


def _relative_path(value: str) -> PurePosixPath:
    """Validate a normalized workspace-relative POSIX path.

    Parameters
    ----------
    value: str
        Requested destination relative to the project root.

    Returns
    -------
    pathlib.PurePosixPath
        The checked relative path.

    """

    if not isinstance(value, str) or not value:
        raise ValueError("Choose a workspace-relative path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Invalid workspace path: {value}")
    return path


def _target(root: Path, value: str) -> Path:
    """Validate a path strictly inside a user-editable folder.

    Parameters
    ----------
    root: pathlib.Path
        Canonical project root.
    value: str
        Requested workspace-relative filename or subdirectory.

    Returns
    -------
    pathlib.Path
        Checked path without resolving its final component.

    """

    relative = _relative_path(value)
    if len(relative.parts) < 2:
        raise ValueError("Select an item inside outputs/ or templates/.")
    directory = _managed_directory(root, relative.parts[0])
    current = directory
    for part in relative.parts[1:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"Invalid project subfolder: {current}")
    destination = current / relative.parts[-1]
    if destination.is_symlink():
        raise ValueError(f"Do not follow a project link: {destination}")
    return destination


def _inputs(paths: Sequence[str | Path]) -> list[Path]:
    """Validate a nonempty collection of external copy sources.

    Parameters
    ----------
    paths: Sequence[str | pathlib.Path]
        Selected local source files or folders.

    Returns
    -------
    list[pathlib.Path]
        Resolved ordinary source paths.

    """

    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("Choose files or folders.")
    result = []
    for value in paths:
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Choose files or folders.")
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ValueError(f"Do not copy a symbolic link: {path}")
        path = path.resolve(strict=True)
        if not (path.is_file() or path.is_dir()):
            raise ValueError(f"Choose a regular file or folder: {path}")
        if path.is_dir() and any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError(f"Do not copy a folder containing links: {path}")
        result.append(path)
    return result


def _free_path(directory: Path, name: str) -> Path:
    """Return a noncolliding destination in one managed folder.

    Parameters
    ----------
    directory: pathlib.Path
        Managed destination directory.
    name: str
        Original basename.

    Returns
    -------
    pathlib.Path
        Available destination with a numeric suffix when needed.

    """

    candidate = directory / name
    stem, suffix = candidate.stem, candidate.suffix
    number = 2
    while candidate.exists() or candidate.is_symlink():
        candidate = directory / f"{stem}-{number}{suffix}"
        number += 1
    return candidate


def _copy_regular_file(source: Path, destination: Path) -> None:
    """Copy one file through an isolated temporary destination.

    Parameters
    ----------
    source: pathlib.Path
        Existing ordinary file.
    destination: pathlib.Path
        Validated destination path.

    """

    stream = NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=".kilin-copy-",
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream, source.open("rb") as original:
            shutil.copyfileobj(original, stream)
        shutil.copystat(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def add_project_files(
    project: QAProject,
    folder: str,
    paths: Sequence[str | Path],
    *,
    replace: bool = False,
) -> list[str]:
    """Copy selected files or folders into outputs/ or templates/.

    Parameters
    ----------
    project: QAProject
        Project owning the destination.
    folder: str
        Either ``outputs`` or ``templates``.
    paths: Sequence[str | pathlib.Path]
        External file or directory paths to copy.
    replace: bool, optional
        Replace an existing item of the same name. Otherwise append a
        numeric suffix. Defaults to False.

    Returns
    -------
    list[str]
        Workspace-relative destinations actually written.

    Notes
    -----
    No indexing takes place. An existing destination is never followed
    through a symlink. Directory replacement removes its prior contents,
    matching the desktop interface's existing behavior.

    """

    root = _project_root(project)
    directory = _managed_directory(root, folder)
    if type(replace) is not bool:
        raise ValueError("replace must be true or false.")
    sources = _inputs(paths)
    for source in sources:
        if source == directory or source.is_relative_to(directory):
            raise ValueError(f"{source.name} is already in {folder}/.")
    changed = []
    for source in sources:
        destination = directory / source.name
        if destination.is_symlink():
            raise ValueError(f"Do not overwrite a link: {destination}")
        if destination.exists():
            if replace:
                _delete_path(destination)
            else:
                destination = _free_path(directory, source.name)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=False)
        else:
            _copy_regular_file(source, destination)
        changed.append(destination.relative_to(root).as_posix())
    return changed


def replace_project_file(
    project: QAProject, path: str, source: str | Path
) -> str:
    """Replace an existing template or output file with local content.

    Parameters
    ----------
    project: QAProject
        Project owning the destination.
    path: str
        Workspace-relative destination file.
    source: str | pathlib.Path
        Local file used as the replacement content.

    Returns
    -------
    str
        Workspace-relative file path replaced.

    """

    root = _project_root(project)
    destination = _target(root, path)
    if not destination.is_file():
        raise ValueError("Choose an existing file to replace.")
    sources = _inputs([source])
    origin = sources[0]
    if not origin.is_file():
        raise ValueError("Choose a file to replace it with.")
    if origin != destination:
        _copy_regular_file(origin, destination)
    return destination.relative_to(root).as_posix()


def _delete_path(path: Path) -> None:
    """Remove one validated ordinary project file or directory.

    Parameters
    ----------
    path: pathlib.Path
        Checked destination without symbolic links.

    """

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def remove_project_files(
    project: QAProject, paths: Sequence[str]
) -> list[str]:
    """Remove selected files or directories from outputs/ and templates/.

    Parameters
    ----------
    project: QAProject
        Project owning the destinations.
    paths: Sequence[str]
        Workspace-relative entries inside outputs/ or templates/.

    Returns
    -------
    list[str]
        Workspace-relative entries removed.

    Notes
    -----
    All paths are validated before deletion. The project folders themselves
    cannot be removed, and symlinks are rejected.

    """

    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("Choose files or folders to delete.")
    root = _project_root(project)
    targets = [_target(root, item) for item in paths]
    if any(not target.exists() for target in targets):
        raise ValueError("A selected project file does not exist.")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate paths in removal request.")
    for index, target in enumerate(targets):
        if any(
            target.is_relative_to(other) for j, other in enumerate(targets)
            if index != j
        ):
            raise ValueError("Cannot remove a folder and its child together.")
    for target in targets:
        _delete_path(target)
    return [target.relative_to(root).as_posix() for target in targets]


def list_project_files(project: QAProject) -> list[str]:
    """Return a GUI-independent inventory of managed project entries.

    Parameters
    ----------
    project: QAProject
        Project whose saved files are listed.

    Returns
    -------
    list[str]
        Workspace-relative paths. Directory entries end with ``/``, so
        empty folders remain visible. Inventory is capped at 10,000 entries.

    """

    root = _project_root(project)
    names = {PROJECT_SOURCES_DIRECTORY, "indexes", "logs", OUTPUTS_FOLDER,
             TEMPLATES_FOLDER}
    if project.source.is_relative_to(root) and project.source != root:
        names.add(project.source.relative_to(root).parts[0])
    found = ["profile.ini"]
    for name in sorted(names):
        directory = root / name
        if not directory.is_dir() or directory.is_symlink():
            continue
        found.append(f"{name}/")
        for entry in directory.rglob("*"):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                found.append(entry.relative_to(root).as_posix() + "/")
            elif entry.is_file():
                found.append(entry.relative_to(root).as_posix())
            if len(found) >= _INVENTORY_LIMIT:
                break
    return sorted(set(found))[:_INVENTORY_LIMIT]
