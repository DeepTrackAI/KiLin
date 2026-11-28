"""Expose the registered ``filter_results`` tool.

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
>>> from kilin.tools.filter_results import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    FilterSettings,
    filter_prompt,
    parse_settings,
    run,
)

__all__ = [
    "PLUGIN",
    "FilterSettings",
    "filter_prompt",
    "parse_settings",
    "run",
]


PLUGIN = Tool(
    name="filter_results",
    summary="Judge each item an earlier answer listed and keep those that "
    "match a follow-up.",
    parse_settings=parse_settings,
    run=run,
    follows_up=True,
    version="1.0.0",
    api_version=1,
)
