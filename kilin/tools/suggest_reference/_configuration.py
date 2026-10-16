"""Validate the defaults owned by the suggest_reference tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool SETTINGS defaults; the
other sections are how the statement is read from the question, and the
planning estimates.

Key Features
------------

- **Isolated configuration**

    Read and validate the tool-local JSON file.

Module Structure
----------------

- ``DEFAULT_*``

    Validated project-type settings.

- ``MIN_WORDS``, ``ROUTER_TOKENS`` and ``ROUTER_WORD_SHARE``

    How the statement is taken from the question.

- ``PROMPT_TOKENS_PER_SECOND`` and the other estimates

    Planning assumptions for the time one paper's check takes.

Examples
--------
>>> from kilin.tools.suggest_reference import _configuration
>>> _configuration.MIN_WORDS > 0
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
)
from kilin.tools._shared.planning import CONFIRMATION_CHOICES

__all__ = [
    "CALL_SECONDS",
    "DEFAULT_ANSWER_TOKENS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_FILES",
    "DEFAULT_LARGE_BATCHES",
    "DEFAULT_PASSAGES",
    "DEFAULT_PASSAGE_TOKENS",
    "DEFAULT_QUICK_FILES",
    "MIN_WORDS",
    "PROMPT_OVERHEAD_TOKENS",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_SHARE",
    "REPLY_TOKENS_PER_SECOND",
    "ROUTER_TOKENS",
    "ROUTER_WORD_SHARE",
]

_TOOL = "suggest_reference"
_DEFAULTS = load_tool_defaults(Path(__file__).with_name("defaults.json"))
_GROUPS = {
    "SETTINGS": {
        "FILES", "PASSAGES", "PASSAGE_TOKENS", "ANSWER_TOKENS",
        "QUICK_FILES", "CONFIRMATION", "LARGE_BATCHES",
    },
    "STATEMENT": {"MIN_WORDS", "ROUTER_TOKENS", "ROUTER_WORD_SHARE"},
    "ESTIMATES": {
        "PROMPT_TOKENS_PER_SECOND", "REPLY_TOKENS_PER_SECOND",
        "CALL_SECONDS", "PROMPT_OVERHEAD_TOKENS", "REPLY_SHARE",
    },
}
require_keys(_DEFAULTS, set(_GROUPS), _TOOL)
for _group, _keys in _GROUPS.items():
    require_keys(
        require_section(_DEFAULTS, _group, _TOOL), _keys, f"{_TOOL}.{_group}"
    )

_SETTINGS = require_section(_DEFAULTS, "SETTINGS", _TOOL)
_STATEMENT = require_section(_DEFAULTS, "STATEMENT", _TOOL)
_ESTIMATES = require_section(_DEFAULTS, "ESTIMATES", _TOOL)
_WHERE = f"{_TOOL}.SETTINGS"

DEFAULT_FILES: Final[int] = require_int(_SETTINGS, "FILES", _WHERE, 1, 20)
DEFAULT_PASSAGES: Final[int] = require_int(
    _SETTINGS, "PASSAGES", _WHERE, 1, 20
)
DEFAULT_PASSAGE_TOKENS: Final[int] = require_int(
    _SETTINGS, "PASSAGE_TOKENS", _WHERE, 300, 16000
)
DEFAULT_ANSWER_TOKENS: Final[int] = require_int(
    _SETTINGS, "ANSWER_TOKENS", _WHERE, 100, 1000
)
DEFAULT_QUICK_FILES: Final[int] = require_int(
    _SETTINGS, "QUICK_FILES", _WHERE, 1, 20
)
DEFAULT_CONFIRMATION: Final[str] = require_string(
    _SETTINGS, "CONFIRMATION", _WHERE
)
if DEFAULT_CONFIRMATION not in CONFIRMATION_CHOICES:
    raise ValueError(f"{_WHERE}.CONFIRMATION is not a supported policy.")
DEFAULT_LARGE_BATCHES: Final[int] = require_int(
    _SETTINGS, "LARGE_BATCHES", _WHERE, 0, 1000
)

MIN_WORDS: Final[int] = require_int(
    _STATEMENT, "MIN_WORDS", "STATEMENT", 1, 100
)
ROUTER_TOKENS: Final[int] = require_int(
    _STATEMENT, "ROUTER_TOKENS", "STATEMENT", 20, 2000
)
ROUTER_WORD_SHARE: Final[float] = require_number(
    _STATEMENT, "ROUTER_WORD_SHARE", "STATEMENT", 0.0, 1.0
)

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
REPLY_SHARE: Final[float] = require_number(
    _ESTIMATES, "REPLY_SHARE", "ESTIMATES", 0.0, 1.0
)
