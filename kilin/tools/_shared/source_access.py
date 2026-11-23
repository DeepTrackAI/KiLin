"""Resolve source files only from an authorized project selection.

Tools may inspect a selected source's original text, but a filename in a
source document does not grant permission to open another file. This module
uses the active source manifest and the project's configured source root to
find only selected regular files inside the workspace.

The checks are best-effort filesystem containment, not a race-free sandbox.
The caller must obtain selections from KíLín's validated source inventory.

Key Features
------------

- **Manifest authorization**

    Only paths selected for the active project can be opened.

- **Workspace containment**

    Reject files outside the project, traversal and descendant symlinks.

Module Structure
----------------

- ``selected_paths``

    Obtain authorized relative names from the project source inventory.

- ``selected_source_file``

    Resolve one authorized source without searching elsewhere.

Examples
--------
>>> from pathlib import Path
>>> from kilin.tools._shared.source_access import selected_source_file
>>> selected_source_file(Path("/missing"), Path("/missing/sources"),
...                      Path("paper.tex"), {Path("paper.tex")}) is None
True

"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from kilin.tools._contract import ToolContext

__all__ = [
    "selected_paths",
    "selected_source_file",
]


def selected_paths(context: ToolContext) -> frozenset[Path]:
    """Return the active source inventory's relative selected file paths.

    Parameters
    ----------
    context: ToolContext
        Tool services and selected source inventory. Legacy mock contexts
        without ``source_files`` use their indexed paths.

    Returns
    -------
    frozenset[pathlib.Path]
        Names authorized for the current project. An available but empty
        inventory remains empty and is not replaced by stale index data.

    """

    inventory = getattr(context, "source_files", None)
    if callable(inventory):
        return frozenset(inventory())
    return frozenset(chunk.path for chunk in context.chunks)


def selected_source_file(
    project_root: Path,
    source_root: Path,
    path: Path,
    selected: Collection[Path],
) -> Path | None:
    """Locate one selected regular file under the project source root.

    Parameters
    ----------
    project_root: pathlib.Path
        Workspace root, containing project state and managed sources.
    source_root: pathlib.Path
        Saved profile's source directory, which must be in the workspace.
    path: pathlib.Path
        Source-relative file path, as recorded by the source inventory.
    selected: Collection[pathlib.Path]
        Explicitly selected source-relative file paths.

    Returns
    -------
    pathlib.Path | None
        Validated source file path, or ``None`` if the selection, file,
        source root, containment or symlink conditions fail.

    Notes
    -----
    No fallback searches the workspace root or an original import folder.
    Paths inherited from legacy profiles outside the workspace are not
    read by tools through this helper. The checks do not prevent an
    uncooperative process from changing files after validation.

    """

    if not isinstance(path, Path) or path.is_absolute():
        return None
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        return None
    if path not in selected:
        return None
    try:
        root = project_root.resolve(strict=True)
        source = source_root.resolve(strict=True)
        if (
            source_root.is_symlink()
            or not source.is_relative_to(root)
            or not source.is_dir()
        ):
            return None
        candidate = source
        for part in path.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                return None
        target = candidate.resolve(strict=True)
        if target.is_relative_to(source) and target.is_file():
            return candidate
    except (OSError, ValueError):
        pass
    return None
