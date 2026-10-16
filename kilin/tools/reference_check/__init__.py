"""Expose the registered ``reference_check`` tool.

Its implementation, configuration, and its private citation finder and
reference matcher are local to this package. The reference-list
extractor is shared with other tools through ``kilin.tools._shared``.

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

- ``check_references``, ``report_markdown`` and ``chat_answer``

    The model-free check and its two write-ups.

Examples
--------
>>> from kilin.tools.reference_check import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools.reference_check.tool import (
    ReferenceCheckSettings,
    ReferenceReport,
    chat_answer,
    check_references,
    parse_settings,
    report_markdown,
    run,
)

__all__ = [
    "ReferenceCheckSettings",
    "ReferenceReport",
    "chat_answer",
    "check_references",
    "parse_settings",
    "report_markdown",
    "run",
]
