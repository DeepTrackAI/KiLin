"""Expose the registered ``find_fact`` tool.

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
>>> from kilin.tools.find_fact import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    FactSettings,
    fact_prompt,
    parse_settings,
    run,
)

__all__ = [
    "PLUGIN",
    "FactSettings",
    "fact_prompt",
    "parse_settings",
    "run",
]


PLUGIN = Tool(
    name="find_fact",
    summary="Find a specific fact in one paper or every paper by reading "
    "every passage; the user can choose a quick answer instead.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
