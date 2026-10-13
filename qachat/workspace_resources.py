"""Represent packaged resources installed into a QAChat project workspace.

A tool may declare files that a new project needs, independently of the
GUI. This module holds only the resource contract, so tool modules can
import it without importing project creation or other tools.

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

Examples
--------
>>> from pathlib import Path
>>> from qachat.workspace_resources import WorkspaceResource
>>> resource = WorkspaceResource(Path("format.md"), "templates/format.md")
>>> resource.destination
'templates/format.md'

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["WorkspaceResource"]


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
