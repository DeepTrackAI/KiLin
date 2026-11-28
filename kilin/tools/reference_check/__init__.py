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

- ``PLUGIN``

    Versioned descriptor discovered by the tool registry.

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

- ``check_references``, ``report_claims``, ``report_markdown`` and
  ``chat_answer``

    The rule-based check, its claims, and its two write-ups.

Examples
--------
>>> from kilin.tools.reference_check import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    ReferenceCheckSettings,
    ReferenceReport,
    chat_answer,
    check_references,
    parse_settings,
    report_claims,
    report_markdown,
    run,
)

__all__ = [
    "PLUGIN",
    "ReferenceCheckSettings",
    "ReferenceReport",
    "chat_answer",
    "check_references",
    "parse_settings",
    "report_claims",
    "report_markdown",
    "run",
]


PLUGIN = Tool(
    name="reference_check",
    summary="Check one paper's reference list against the project's files, "
    "find where each reference is cited, optionally check claims, "
    "and give offline or explicitly authorized online links for "
    "missing works; save the report to outputs/.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
