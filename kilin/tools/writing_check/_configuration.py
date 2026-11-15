"""Validate the defaults owned by the writing_check tool.

Unknown configuration keys and incorrect types are rejected at import.
The ``CHECKS`` section of ``checks.json`` holds the word lists and limits
of the checks done in code, so they can be extended without changing
code. (It becomes the tool's ``defaults.json`` when the tool is
registered: only registered tools have one.)

Key Features
------------

- **Isolated configuration**

    The tool's own ``checks.json``, validated once.

Module Structure
----------------

- ``LONG_SENTENCE_WORDS``

    Sentence length, in words, reported as long.

- ``PROMOTIONAL_WORDS``

    Words that praise the work instead of describing it.

- ``SPELLING_PAIRS``

    US and UK spellings of the same word.

- ``KNOWN_ACRONYMS``

    Acronyms readers know without a definition, in lower case.

Examples
--------
>>> from kilin.tools.writing_check import _configuration
>>> _configuration.LONG_SENTENCE_WORDS > 0
True

"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from kilin.tools._defaults import (
    load_tool_defaults,
    require_int,
    require_keys,
    require_section,
    require_words,
)

__all__ = [
    "KNOWN_ACRONYMS",
    "LONG_SENTENCE_WORDS",
    "PROMOTIONAL_WORDS",
    "SPELLING_PAIRS",
]

_TOOL = "writing_check"
_DEFAULTS = load_tool_defaults(Path(__file__).with_name("checks.json"))
require_keys(_DEFAULTS, {"CHECKS"}, _TOOL)
_CHECKS = require_section(_DEFAULTS, "CHECKS", _TOOL)
require_keys(
    _CHECKS,
    {
        "LONG_SENTENCE_WORDS", "PROMOTIONAL_WORDS", "SPELLING_PAIRS",
        "KNOWN_ACRONYMS",
    },
    f"{_TOOL}.CHECKS",
)
_WHERE = f"{_TOOL}.CHECKS"

LONG_SENTENCE_WORDS: Final[int] = require_int(
    _CHECKS, "LONG_SENTENCE_WORDS", _WHERE, 10, 200
)
PROMOTIONAL_WORDS: Final[tuple[str, ...]] = require_words(
    _CHECKS, "PROMOTIONAL_WORDS", _WHERE
)
KNOWN_ACRONYMS: Final[frozenset[str]] = frozenset(
    require_words(_CHECKS, "KNOWN_ACRONYMS", _WHERE)
)


def _pairs(words: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Return ``us:uk`` entries as pairs.

    Parameters
    ----------
    words: tuple[str, ...]
        Entries such as ``"color:colour"``.

    Returns
    -------
    tuple[tuple[str, str], ...]
        The US and UK spelling of each entry.

    Raises
    ------
    ValueError
        If an entry is not two different words joined by one colon.

    """

    pairs = []
    for word in words:
        us, colon, uk = word.partition(":")
        if not colon or not us or not uk or ":" in uk or us == uk:
            raise ValueError(
                f"{_WHERE}.SPELLING_PAIRS entries are us:uk, got {word!r}."
            )
        pairs.append((us, uk))
    return tuple(pairs)


SPELLING_PAIRS: Final[tuple[tuple[str, str], ...]] = _pairs(
    require_words(_CHECKS, "SPELLING_PAIRS", _WHERE)
)
