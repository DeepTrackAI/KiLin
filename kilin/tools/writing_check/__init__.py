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

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

- ``WritingCheckSettings``, ``report_markdown`` and ``chat_answer``

    The settings, the saved report and the chat answer.

- ``_document``, ``_scope``, ``_checks`` and ``_review``

    Sections and paragraphs of a source, the part a question names, the
    checks done in code, and the paragraph checks done by the model.

Examples
--------
>>> from kilin.tools.writing_check import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools.writing_check.tool import (
    WritingCheckSettings,
    chat_answer,
    parse_settings,
    report_markdown,
    run,
)

__all__ = [
    "WritingCheckSettings",
    "chat_answer",
    "parse_settings",
    "report_markdown",
    "run",
]
