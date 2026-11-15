"""Validate the defaults owned by the writing_check tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool ``SETTINGS`` defaults. The
other sections belong to the tool: the word lists and limits of the checks
done in code (``CHECKS``), the planning estimates (``ESTIMATES``) and the
policy of its model calls (``MODEL_CALLS``).

Key Features
------------

- **Isolated configuration**

    The tool's own ``defaults.json``, validated once.

Module Structure
----------------

- ``DEFAULT_LEVEL``, ``DEFAULT_CONFIRMATION``, ``DEFAULT_LARGE_BATCHES``,
  ``DEFAULT_QUICK_PARAGRAPHS`` and ``DEFAULT_CHAT_FINDINGS``

    Packaged settings, overridable by a project type.

- ``LONG_SENTENCE_WORDS``, ``PROMOTIONAL_WORDS``, ``SPELLING_PAIRS`` and
  ``KNOWN_ACRONYMS``

    The checks done in code.

- ``PROMPT_TOKENS_PER_SECOND`` and the other estimates

    Planning assumptions for the time one model call takes.

- ``MODEL_CALLS``

    The thinking policy of the tool's model calls.

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
    require_number,
    require_section,
    require_string,
    require_words,
)
from kilin.tools._model_calls import (
    ModelCallSettings,
    load_model_call_settings,
)
from kilin.tools._shared.planning import CONFIRMATION_CHOICES

__all__ = [
    "CALL_SECONDS",
    "DEFAULT_CHAT_FINDINGS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "DEFAULT_LEVEL",
    "DEFAULT_QUICK_PARAGRAPHS",
    "KNOWN_ACRONYMS",
    "LEVELS",
    "LONG_SENTENCE_WORDS",
    "MODEL_CALLS",
    "PROMOTIONAL_WORDS",
    "PROMPT_OVERHEAD_TOKENS",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_TOKENS",
    "REPLY_TOKENS_PER_SECOND",
    "SPELLING_PAIRS",
]

_TOOL = "writing_check"
_PATH = Path(__file__).with_name("defaults.json")
_DEFAULTS = load_tool_defaults(_PATH)
_GROUPS = {
    "SETTINGS": {
        "LEVEL", "CONFIRMATION", "LARGE_BATCHES", "QUICK_PARAGRAPHS",
        "CHAT_FINDINGS",
    },
    "CHECKS": {
        "LONG_SENTENCE_WORDS", "PROMOTIONAL_WORDS", "SPELLING_PAIRS",
        "KNOWN_ACRONYMS",
    },
    "ESTIMATES": {
        "PROMPT_TOKENS_PER_SECOND", "REPLY_TOKENS_PER_SECOND",
        "CALL_SECONDS", "PROMPT_OVERHEAD_TOKENS", "REPLY_TOKENS",
    },
    "MODEL_CALLS": {"THINKING"},
}
require_keys(_DEFAULTS, set(_GROUPS), _TOOL)
for _group, _keys in _GROUPS.items():
    require_keys(
        require_section(_DEFAULTS, _group, _TOOL), _keys, f"{_TOOL}.{_group}"
    )

# Which model passes run: grammar only, style only, both, or grammar and,
# when the question asks for it, style ("auto").
LEVELS: Final[tuple[str, ...]] = ("auto", "grammar", "style", "all")

_SETTINGS = require_section(_DEFAULTS, "SETTINGS", _TOOL)
_WHERE = f"{_TOOL}.SETTINGS"
DEFAULT_LEVEL: Final[str] = require_string(_SETTINGS, "LEVEL", _WHERE)
if DEFAULT_LEVEL not in LEVELS:
    raise ValueError(f"{_WHERE}.LEVEL must be one of {LEVELS}.")
DEFAULT_CONFIRMATION: Final[str] = require_string(
    _SETTINGS, "CONFIRMATION", _WHERE
)
if DEFAULT_CONFIRMATION not in CONFIRMATION_CHOICES:
    raise ValueError(f"{_WHERE}.CONFIRMATION is not a supported policy.")
DEFAULT_LARGE_BATCHES: Final[int] = require_int(
    _SETTINGS, "LARGE_BATCHES", _WHERE, 0, 1000
)
DEFAULT_QUICK_PARAGRAPHS: Final[int] = require_int(
    _SETTINGS, "QUICK_PARAGRAPHS", _WHERE, 1, 1000
)
DEFAULT_CHAT_FINDINGS: Final[int] = require_int(
    _SETTINGS, "CHAT_FINDINGS", _WHERE, 0, 200
)

_CHECKS = require_section(_DEFAULTS, "CHECKS", _TOOL)
_CHECKS_WHERE = f"{_TOOL}.CHECKS"
LONG_SENTENCE_WORDS: Final[int] = require_int(
    _CHECKS, "LONG_SENTENCE_WORDS", _CHECKS_WHERE, 10, 200
)
PROMOTIONAL_WORDS: Final[tuple[str, ...]] = require_words(
    _CHECKS, "PROMOTIONAL_WORDS", _CHECKS_WHERE
)
KNOWN_ACRONYMS: Final[frozenset[str]] = frozenset(
    require_words(_CHECKS, "KNOWN_ACRONYMS", _CHECKS_WHERE)
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
                f"{_CHECKS_WHERE}.SPELLING_PAIRS entries are us:uk, got "
                f"{word!r}."
            )
        pairs.append((us, uk))
    return tuple(pairs)


SPELLING_PAIRS: Final[tuple[tuple[str, str], ...]] = _pairs(
    require_words(_CHECKS, "SPELLING_PAIRS", _CHECKS_WHERE)
)

_ESTIMATES = require_section(_DEFAULTS, "ESTIMATES", _TOOL)
PROMPT_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "PROMPT_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
REPLY_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "REPLY_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
CALL_SECONDS: Final[float] = require_number(
    _ESTIMATES, "CALL_SECONDS", "ESTIMATES", 0.0, 1e9
)
PROMPT_OVERHEAD_TOKENS: Final[int] = require_int(
    _ESTIMATES, "PROMPT_OVERHEAD_TOKENS", "ESTIMATES", 0, 100000
)
REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "REPLY_TOKENS", "ESTIMATES", 1, 131072
)
MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(_PATH)
