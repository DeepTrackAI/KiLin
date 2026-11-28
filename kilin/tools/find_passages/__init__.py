"""Expose the registered ``find_passages`` tool.

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
>>> from kilin.tools.find_passages import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from kilin.tools._shared.sentences import Sentence, split_sentences
from .tool import (
    PassageSettings,
    judge_prompt,
    parse_settings,
    run,
    topic_prompt,
)

__all__ = [
    "PLUGIN",
    "PassageSettings",
    "Sentence",
    "judge_prompt",
    "parse_settings",
    "run",
    "split_sentences",
    "topic_prompt",
]


PLUGIN = Tool(
    name="find_passages",
    summary="Read every sentence in scope and list the ones about a topic, "
    "copied word for word.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
