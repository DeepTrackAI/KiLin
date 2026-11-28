"""Expose the registered ``assistant_guide`` tool.

Its implementation and configuration are local to this package.
Common algorithms are provided by private ``kilin.tools._shared``
modules, not by other registered tools.

Key Features
------------

- **Tool execution**

    Parse validated settings and execute the named tool.

- **Independent defaults**

    Read this package's ``defaults.json`` configuration.

Module Structure
----------------

- ``PLUGIN``

    Versioned descriptor discovered by the tool registry.

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

Examples
--------
>>> from kilin.tools.assistant_guide import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    GuideSettings,
    parse_settings,
    run,
)

__all__ = [
    "PLUGIN",
    "GuideSettings",
    "parse_settings",
    "run",
]


PLUGIN = Tool(
    name="assistant_guide",
    summary="Answer what KíLín can do with one ready-to-send example question "
    "per kind of request, about the project's own files.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
