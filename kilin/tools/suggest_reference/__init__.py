"""Expose the registered ``suggest_reference`` tool.

Its implementation and configuration are local to this package; the
judgement of a work against a sentence is shared with ``reference_check``
through ``kilin.tools._shared.claims``.

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

- ``request_of``, ``statement_of``, ``candidate_files`` and
  ``chat_answer``

    The statement in a question, the papers worth checking, and the
    answer.

- ``named_subject`` and ``name_statement``

    The name a question asks the introducing paper for ("cite for
    BRAPH2"), and the statement that a paper introduces it.

Examples
--------
>>> from kilin.tools.suggest_reference import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    SuggestReferenceSettings,
    candidate_files,
    chat_answer,
    name_statement,
    named_subject,
    parse_settings,
    request_of,
    run,
    statement_of,
)

__all__ = [
    "PLUGIN",
    "SuggestReferenceSettings",
    "candidate_files",
    "chat_answer",
    "name_statement",
    "named_subject",
    "parse_settings",
    "request_of",
    "run",
    "statement_of",
]


PLUGIN = Tool(
    name="suggest_reference",
    summary="Find the papers in the project that could be cited for a "
    "statement, judging each with a quote from its own passages.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
