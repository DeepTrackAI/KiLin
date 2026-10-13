"""Expose the registered ``find_passages`` tool.

Its implementation and configuration are local to this package.
Common algorithms are provided by private ``qachat.tools._shared``
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
>>> from qachat.tools.find_passages import parse_settings
>>> callable(parse_settings)
True

"""

from qachat.tools._shared.sentences import Sentence, split_sentences
from qachat.tools.find_passages.tool import (
    PassageSettings,
    judge_prompt,
    parse_settings,
    run,
    topic_prompt,
)

__all__ = [
    "PassageSettings",
    "Sentence",
    "judge_prompt",
    "parse_settings",
    "run",
    "split_sentences",
    "topic_prompt",
]
