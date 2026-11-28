"""Expose the registered ``show_source`` tool.

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
>>> from kilin.tools.show_source import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from kilin.tools._shared.earlier_turns import newest_listed
from .tool import (
    SourceSettings,
    explain_prompt,
    parse_settings,
    run,
)

__all__ = [
    "PLUGIN",
    "SourceSettings",
    "explain_prompt",
    "newest_listed",
    "parse_settings",
    "run",
]


PLUGIN = Tool(
    name="show_source",
    summary="Show the sentence an earlier answer cited, with the sentences "
    "around it, and explain it without searching again.",
    parse_settings=parse_settings,
    run=run,
    follows_up=True,
    version="1.0.0",
    api_version=1,
)
