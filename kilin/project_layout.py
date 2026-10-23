"""Plan and materialize a project's core folders and tool resources.

Every project, including one created without the desktop interface, owns
``indexes/``, ``logs/``, ``outputs/``, and ``templates/``. Project types
select tools through their intents; each registered tool may declare
packaged resources for its validated settings. Only those resources are
installed when a new profile is created. Opening an existing project never
reinstalls deleted or edited templates.

The planner performs no writes. Materialization validates paths, preserves
existing project files, and does not execute tool code. A project workspace
is not an execution sandbox, and preparation is not a filesystem transaction.

Key Features
------------

- **GUI-independent layout**

    The core creates every project's shared folders.

- **Tool-selected resources**

    Deduplicate required resources across project-type intents.

- **Safe installation**

    Reject invalid destinations, symbolic links, and conflicting resources;
    preserve existing files instead of overwriting them.

- **Project-owned files**

    Reopening never relocates or deletes unrelated workspace files.

Module Structure
----------------

- ``ProjectLayout``

    Immutable plan containing a project's required packaged resources.

- ``plan_project_layout``

    Resolve the resources of a validated project type without writing files.

- ``ensure_project_folders``

    Prepare shared folders without copying packaged resources.

- ``install_project_resources``

    Install only missing files listed in a validated project layout.

- ``source_scan_exclusions``

    Exclude managed folders only when the workspace root is also its source.

Examples
--------
>>> from kilin.project_layout import plan_project_layout
>>> plan = plan_project_layout("code")
>>> plan.resources
()

"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
from typing import Final

from kilin.project_type import project_type
from kilin.tools._registry import TOOL_REGISTRY
from kilin.workspace_resources import (
    OUTPUTS_FOLDER,
    TEMPLATES_FOLDER,
    WorkspaceResource,
)

__all__ = [
    "ProjectLayout",
    "ensure_project_folders",
    "install_project_resources",
    "plan_project_layout",
    "source_scan_exclusions",
]

_CORE_FOLDERS: Final[tuple[str, ...]] = (
    "indexes", "logs", OUTPUTS_FOLDER, TEMPLATES_FOLDER
)
_RESERVED_RESOURCE_ROOTS: Final[frozenset[str]] = frozenset(
    ("sources", "indexes", "logs", "outputs")
)
_PROJECT_SOURCE_EXCLUDES: Final[tuple[str, ...]] = (
    *_CORE_FOLDERS,
    "profile.ini",
    "sources.json",
    "source_snapshot.json",
)


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    """Required packaged resources for one project's selected tools.

    Attributes
    ----------
    resources: tuple[WorkspaceResource, ...]
        Distinct packaged files to install on first profile creation.

    """

    resources: tuple[WorkspaceResource, ...]


def _safe_destination(destination: str) -> PurePosixPath:
    """Validate and return one workspace-relative resource destination.

    Parameters
    ----------
    destination: str
        POSIX-style relative file path from a registered tool.

    Returns
    -------
    pathlib.PurePosixPath
        Validated destination below a tool-owned resource directory.

    Raises
    ------
    ValueError
        If the path could escape or overwrite a managed workspace area.

    """

    if not isinstance(destination, str) or not destination:
        raise ValueError("A tool resource needs a relative destination.")
    parts = destination.split("/")
    path = PurePosixPath(destination)
    if (
        path.is_absolute()
        or PureWindowsPath(destination).is_absolute()
        or "\\" in destination
        or len(parts) < 2
        or any(part in ("", ".", "..") for part in parts)
        or parts[0] in _RESERVED_RESOURCE_ROOTS
        or parts[0] == "profile.ini"
        or ":" in destination
    ):
        raise ValueError(f"Unsafe tool resource destination: {destination}")
    return path


def _validated_source(source: Path) -> Path:
    """Require a real, readable packaged resource file.

    Parameters
    ----------
    source: pathlib.Path
        Tool-provided packaged resource path.

    Returns
    -------
    pathlib.Path
        Resolved regular file to be copied.

    Raises
    ------
    ValueError
        If the resource is missing, linked, or not a regular file.

    """

    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or invalid packaged resource: {path}")
    return path.resolve(strict=True)


def _tool_owned_source(source: Path, tool_name: str) -> Path:
    """Require a declared file to belong to its registered tool package.

    Parameters
    ----------
    source: pathlib.Path
        Packaged file declared by the tool.
    tool_name: str
        Registered name of the declaring executable tool.

    Returns
    -------
    pathlib.Path
        Validated, resolved file in the declaring package.

    Raises
    ------
    ValueError
        If the file is missing or outside the declaring tool package.

    """

    path = _validated_source(source)
    package = Path(files(f"kilin.tools.{tool_name}")).resolve()
    if not path.is_relative_to(package):
        raise ValueError(
            f"Tool {tool_name!r} declared a resource outside "
            f"its package: {source}"
        )
    return path


def plan_project_layout(type_id: str) -> ProjectLayout:
    """Build a project resource plan from its configured tool intents.

    Parameters
    ----------
    type_id: str
        Registered project type identifier.

    Returns
    -------
    ProjectLayout
        Required packaged files, deduplicated and ordered by destination.

    Raises
    ------
    ValueError
        If a resource is outside its tool package, a destination is
        unsafe, or two tools claim conflicting destination files.

    """

    definition = project_type(type_id)
    resources: dict[str, WorkspaceResource] = {}
    for intent in definition.intents.values():
        use = intent.tool
        if use is None:
            continue
        declaration = TOOL_REGISTRY[use.name].workspace_resources
        if declaration is None:
            continue
        for resource in declaration(use.settings):
            destination = _safe_destination(resource.destination)
            source = _tool_owned_source(resource.source, use.name)
            key = destination.as_posix()
            old = resources.get(key)
            if old is not None and old.source != source:
                raise ValueError(
                    f"Conflicting project resources for {key}."
                )
            resources[key] = WorkspaceResource(source, key)
    return ProjectLayout(tuple(resources[key] for key in sorted(resources)))


def _safe_directory(path: Path) -> None:
    """Create an ordinary directory without following a final symlink.

    Parameters
    ----------
    path: pathlib.Path
        Destination directory within a validated workspace.

    Raises
    ------
    ValueError
        If a directory is a symbolic link or a non-directory path.

    """

    if path.is_symlink():
        raise ValueError(f"Workspace directory cannot be a link: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Workspace directory is not a folder: {path}")
    path.mkdir(exist_ok=True)


def ensure_project_folders(root: Path) -> None:
    """Create shared core folders without installing any tool resources.

    Parameters
    ----------
    root: pathlib.Path
        Existing project root containing a validated profile.

    Raises
    ------
    ValueError
        If a managed folder is a symlink or not a directory.

    Notes
    -----
    ``templates/`` is always present, including for code projects with no
    template-using tools. Existing files are not moved or deleted.

    """

    root = Path(root)
    for name in _CORE_FOLDERS:
        _safe_directory(root / name)


def install_project_resources(root: Path, layout: ProjectLayout) -> None:
    """Copy missing resources into a new or explicitly repaired project.

    Parameters
    ----------
    root: pathlib.Path
        Project root, with its shared folders already initialized.
    layout: ProjectLayout
        Prevalidated resources selected by the project's tool intents.

    Raises
    ------
    ValueError
        If a destination contains a link, a conflicting directory, or an
        invalid path.

    Notes
    -----
    This is called automatically only for a newly generated profile. It
    never overwrites existing resources; restoring deleted files in an
    existing project requires explicitly calling this operation.

    """

    root = Path(root)
    for resource in layout.resources:
        relative = _safe_destination(resource.destination)
        source = _validated_source(resource.source)
        target = root.joinpath(*relative.parts)
        parent = root
        for folder in relative.parts[:-1]:
            parent = parent / folder
            _safe_directory(parent)
        if target.is_symlink():
            raise ValueError(f"Project resource cannot be a link: {target}")
        if target.exists():
            if not target.is_file():
                raise ValueError(
                    f"Project resource is not a file: {target}"
                )
            continue
        # Exclusive creation avoids overwriting user-owned files, including
        # files created after checking whether the destination exists.
        with source.open("rb") as original:
            try:
                with target.open("xb") as installed:
                    shutil.copyfileobj(original, installed)
            except FileExistsError:
                if target.is_symlink() or not target.is_file():
                    raise ValueError(
                        f"Project resource is not a file: {target}"
                    ) from None


def source_scan_exclusions(root: Path, source: Path) -> tuple[str, ...]:
    """Exclude managed project folders when scanning the project root.

    Parameters
    ----------
    root: pathlib.Path
        Absolute root of an opened KíLín project.
    source: pathlib.Path
        Resolved source folder recorded in the project profile.

    Returns
    -------
    tuple[str, ...]
        Managed top-level folders to exclude when source is the project
        root, or an empty tuple when sources live in a separate directory.

    Notes
    -----
    Source folders of unrelated repositories named ``templates`` or
    ``outputs`` are not excluded globally; only KíLín-owned root folders
    are skipped in the root-as-source profile layout.

    """

    return (
        _PROJECT_SOURCE_EXCLUDES
        if Path(root).resolve() == Path(source).resolve()
        else ()
    )
