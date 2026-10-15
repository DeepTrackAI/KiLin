"""Validate the defaults owned by the find_passages tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool SETTINGS defaults.

Key Features
------------

- **Isolated configuration**

    Read and validate the tool-local JSON file.

Module Structure
----------------

- ``DEFAULT_*`` and internal constants

    Validated settings and algorithm parameters.

Examples
--------
>>> from qachat.tools.find_passages import _configuration
>>> bool(_configuration._DEFAULTS)
True

"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from qachat.tools._defaults import (
    load_tool_defaults,
    require_int,
    require_keys,
    require_number,
    require_section,
    require_string,
)

__all__ = [
    "DEFAULT_BATCH_TOKENS",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_QUICK_PASSAGES",
    "DEFAULT_MAX_LISTED",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "MIN_WORDS",
    "TOPIC_REPLY_TOKENS",
    "JUDGE_REPLY_TOKENS",
    "SENTENCE_NUMBER_TOKENS",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_TOKENS_PER_SECOND",
    "JUDGE_PROMPT_TOKENS",
    "JUDGE_REPLY_ESTIMATE_TOKENS",
    "CALL_SECONDS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "SETTINGS",
    "HEURISTICS",
    "ESTIMATES",
}, "find_passages")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "find_passages"
)
require_keys(
    _SETTINGS, {
        "BATCH_TOKENS",
        "MAX_MODEL_CALLS",
        "QUICK_PASSAGES",
        "MAX_LISTED",
        "CONFIRMATION",
        "LARGE_BATCHES",
    }, "find_passages.SETTINGS"
)
DEFAULT_BATCH_TOKENS: Final[int] = (
    require_int(_SETTINGS, "BATCH_TOKENS", "find_passages.SETTINGS", 500, 8000)
)
DEFAULT_MAX_MODEL_CALLS: Final[int] = (
    require_int(_SETTINGS, "MAX_MODEL_CALLS", "find_passages.SETTINGS", 1, 500)
)
DEFAULT_QUICK_PASSAGES: Final[int] = (
    require_int(_SETTINGS, "QUICK_PASSAGES", "find_passages.SETTINGS", 5, 200)
)
DEFAULT_MAX_LISTED: Final[int] = (
    require_int(_SETTINGS, "MAX_LISTED", "find_passages.SETTINGS", 10, 1000)
)
DEFAULT_CONFIRMATION: Final[str] = (
    require_string(_SETTINGS, "CONFIRMATION", "find_passages.SETTINGS")
)
if DEFAULT_CONFIRMATION not in ('always', 'when_large', 'never'):
    raise ValueError("find_passages.SETTINGS.CONFIRMATION is not supported.")
DEFAULT_LARGE_BATCHES: Final[int] = (
    require_int(_SETTINGS, "LARGE_BATCHES", "find_passages.SETTINGS", 0, 100)
)
_HEURISTICS = require_section(
    _DEFAULTS, "HEURISTICS", "find_passages"
)
require_keys(
    _HEURISTICS, {
        "MIN_WORDS",
    }, "find_passages.HEURISTICS"
)
MIN_WORDS: Final[int] = (
    require_int(_HEURISTICS, "MIN_WORDS", "find_passages.HEURISTICS", 1, 100)
)
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "find_passages"
)
require_keys(
    _ESTIMATES, {
        "TOPIC_REPLY_TOKENS",
        "JUDGE_REPLY_TOKENS",
        "SENTENCE_NUMBER_TOKENS",
        "PROMPT_TOKENS_PER_SECOND",
        "REPLY_TOKENS_PER_SECOND",
        "JUDGE_PROMPT_TOKENS",
        "JUDGE_REPLY_ESTIMATE_TOKENS",
        "CALL_SECONDS",
    }, "find_passages.ESTIMATES"
)
TOPIC_REPLY_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "TOPIC_REPLY_TOKENS",
        "find_passages.ESTIMATES",
        1,
        100000,
    )
)
JUDGE_REPLY_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "JUDGE_REPLY_TOKENS",
        "find_passages.ESTIMATES",
        1,
        100000,
    )
)
SENTENCE_NUMBER_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "SENTENCE_NUMBER_TOKENS",
        "find_passages.ESTIMATES",
        0,
        100000,
    )
)
PROMPT_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "PROMPT_TOKENS_PER_SECOND",
        "find_passages.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
REPLY_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "REPLY_TOKENS_PER_SECOND",
        "find_passages.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
JUDGE_PROMPT_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "JUDGE_PROMPT_TOKENS",
        "find_passages.ESTIMATES",
        0,
        100000,
    )
)
JUDGE_REPLY_ESTIMATE_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "JUDGE_REPLY_ESTIMATE_TOKENS",
        "find_passages.ESTIMATES",
        0,
        100000,
    )
)
CALL_SECONDS: Final[float] = (
    require_number(
        _ESTIMATES,
        "CALL_SECONDS",
        "find_passages.ESTIMATES",
        0,
        1000000000.0,
    )
)
