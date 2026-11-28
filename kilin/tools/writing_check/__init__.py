"""Expose the registered ``writing_check`` tool.

It checks the grammar and style of one document of the project, or a part
of it, and suggests changes without making them. Its implementation and
configuration are local to this package.

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

- ``WritingCheckSettings``, ``report_markdown`` and ``chat_answer``

    The settings, the saved report and the chat answer.

- ``asks_rewrite``

    Whether a question asks to rewrite a line: a follow-up of a check,
    so routing passes the earlier turn.

- ``_document``, ``_scope``, ``_checks``, ``_review`` and ``_followup``

    Sections and paragraphs of a source, the part a question names, the
    checks done in code, the paragraph checks done by the model, and line
    rewrites after a check.

Examples
--------
>>> from kilin.tools.writing_check import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from ._followup import asks_rewrite
from .tool import (
    WritingCheckSettings,
    chat_answer,
    parse_settings,
    report_markdown,
    run,
)

__all__ = [
    "PLUGIN",
    "WritingCheckSettings",
    "asks_rewrite",
    "chat_answer",
    "parse_settings",
    "report_markdown",
    "run",
]


PLUGIN = Tool(
    name="writing_check",
    summary="Check the grammar and style of one document or part of it, "
    "paragraph by paragraph, and suggest changes without making them; "
    "save the report to outputs/.",
    parse_settings=parse_settings,
    run=run,
    follows_up_when=asks_rewrite,
    version="1.0.0",
    api_version=1,
)
