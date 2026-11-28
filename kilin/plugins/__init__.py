"""Discover and inspect trusted KíLín plugins independently of the GUI.

The runtime catalogs initialize lazily and use one user storage directory.
Placing Python code in its tools folder is an explicit trust decision.
Candidate validation never imports replacement code in the active runtime.
Installation, updating and removal are not managed by this module yet.

Key Features
------------
- Distribution and user-level discovery, with independent diagnostics.
- Structural checks without executing candidate code.
- Optional runtime validation in a fresh interpreter, not a sandbox.

Module Structure
----------------
- ``plugin_directory`` resolves the configured user directory.
- ``list_tools`` and ``list_project_types`` inspect the runtime catalogs.
- ``validate_tool`` and ``validate_project_type`` check candidates.
- ``ToolPluginInfo`` describes an individual tool candidate.

Examples
--------
>>> from kilin.plugins import plugin_directory
>>> plugin_directory().is_absolute()
True

"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from kilin.plugins._discovery import ToolPluginInfo
from kilin.plugins._paths import plugin_directory

if TYPE_CHECKING:
    from kilin.project_type import ProjectTypeInfo

__all__ = [
    "ToolPluginInfo", "plugin_directory", "list_tools",
    "list_project_types", "validate_tool", "validate_project_type",
]


def list_tools() -> tuple[ToolPluginInfo, ...]:
    """Return immutable information about available and rejected tools.

    Returns
    -------
    tuple[ToolPluginInfo, ...]
        Registered entries followed by rejected user candidates, including
        ID conflicts. Initializes the runtime tool catalog if needed.

    """

    from kilin.tools._registry import _tool_plugin_infos

    return _tool_plugin_infos()


def list_project_types() -> tuple[ProjectTypeInfo, ...]:
    """Return available and unavailable project types, including conflicts.

    Returns
    -------
    tuple[ProjectTypeInfo, ...]
        Immutable diagnostics for all discovered definitions, including
        user definitions whose identifiers conflict with bundled ones.

    """

    from kilin.project_type import _project_type_infos

    return _project_type_infos()


def validate_tool(path: Path, *, trusted: bool = False) -> ToolPluginInfo:
    """Inspect a tool candidate without changing the active registry.

    Parameters
    ----------
    path: pathlib.Path
        Candidate tool source folder, named after its descriptor ID.
    trusted: bool, optional
        Allow importing the candidate in a fresh interpreter. False only
        checks layout, syntax and JSON; it never executes candidate code.

    Returns
    -------
    ToolPluginInfo
        Structural or runtime validation result. The runtime_checked flag
        distinguishes an inspected folder from a verified descriptor.

    Notes
    -----
    A successful check is not installation, security auditing, or proof
    of correct tool execution. Runtime validation executes trusted code
    with the user's permissions.

    """

    from kilin.plugins._validation import _validate_tool

    return _validate_tool(Path(path), trusted=trusted)


def validate_project_type(
    path: Path, *, bind_tools: bool = True,
) -> ProjectTypeInfo:
    """Validate one JSON candidate without adding it to the runtime catalog.

    Parameters
    ----------
    path: pathlib.Path
        Candidate project-type JSON file.
    bind_tools: bool, optional
        Resolve required tools against the runtime registry. False checks
        only the JSON structure, even if a required tool is absent.

    Returns
    -------
    ProjectTypeInfo
        Captured schema information, required tools and any error. The
        bound flag distinguishes structural checks from fully bound types.

    """

    from kilin.project_type import PROJECT_TYPES_DIR, _type_record

    if type(bind_tools) is not bool:
        raise TypeError("bind_tools must be a boolean.")
    path = Path(path)
    if (PROJECT_TYPES_DIR / path.name).is_file():
        record = _type_record(path, "user", bind_tools=False)
        return replace(
            record.info,
            error=f"Project type ID {path.stem!r} is already registered.",
        )
    return _type_record(path, "user", bind_tools=bind_tools).info


def _project_plugin_snapshot(type_id: str) -> dict:
    """Capture the identities of a type and its configured tool packages.

    Parameters
    ----------
    type_id: str
        Selected project type in the current runtime.

    Returns
    -------
    dict
        Portable project-type identity and tool identities keyed by ID.
        Unknown identities are None, including synthetic test descriptors.
        No source files are rehashed and no absolute paths are recorded.

    """

    from kilin.project_type import project_type_catalog
    from kilin.tools._registry import _tool_identity

    info = project_type_catalog().get(type_id)
    if info is None or not info.available:
        return {"project_type": None, "tools": {}}
    return {
        "project_type": info.identity(),
        "tools": {name: _tool_identity(name) for name in info.required_tools},
    }
