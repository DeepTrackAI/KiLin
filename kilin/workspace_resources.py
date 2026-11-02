"""Represent packaged resources installed into a KíLín project workspace.

A tool may declare files that a new project needs, independently of the
GUI. This module holds the resource contract and common folder names, so tool
modules can import them without importing project creation or the GUI.

Key Features
------------

- **Declarative resources**

    Identify a packaged source file and its workspace-relative destination.

- **No side effects**

    Declaring resources never creates or copies project files.

Module Structure
----------------

- ``WorkspaceResource``

    One source file and its destination in a new project.

- ``OUTPUTS_FOLDER`` and ``TEMPLATES_FOLDER``

    Names of the shared user-editable workspace directories.

- ``PROJECT_SOURCES_DIRECTORY``

    Name of the source directory every new project creates, whatever its
    project type.

- ``templates_folder``

    Return the project-owned templates path without creating files.

Examples
--------
>>> from pathlib import Path
from typing import Final
>>> from kilin.workspace_resources import WorkspaceResource
>>> resource = WorkspaceResource(Path("format.md"), "templates/format.md")
>>> resource.destination
'templates/format.md'

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "OUTPUTS_FOLDER",
    "PROJECT_SOURCES_DIRECTORY",
    "TEMPLATES_FOLDER",
    "WorkspaceResource",
    "templates_folder",
]

OUTPUTS_FOLDER: Final[str] = "outputs"
TEMPLATES_FOLDER: Final[str] = "templates"
# Every new project keeps its sources here. Existing projects keep the
# source folder recorded in their profile.ini.
PROJECT_SOURCES_DIRECTORY: Final[str] = "sources"


def templates_folder(project_root: Path) -> Path:
    """Return the user-editable template folder without creating it.

    Parameters
    ----------
    project_root: pathlib.Path
        Root of the initialized project workspace.

    Returns
    -------
    pathlib.Path
        Project-owned template directory.

    """

    return Path(project_root) / TEMPLATES_FOLDER


@dataclass(frozen=True, slots=True)
class WorkspaceResource:
    """One packaged resource required by a configured tool.

    Attributes
    ----------
    source: pathlib.Path
        File supplied by the installed tool package.
    destination: str
        Relative POSIX path in the workspace; the core validates it.

    """

    source: Path
    destination: str
