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

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

Examples
--------
>>> from kilin.tools.find_fact import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools.find_fact.tool import (
    FactSettings,
    fact_prompt,
    parse_settings,
    run,
)

__all__ = [
    "FactSettings",
    "fact_prompt",
    "parse_settings",
    "run",
]
