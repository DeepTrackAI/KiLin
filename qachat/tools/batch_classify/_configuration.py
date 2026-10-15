"""Validate the defaults owned by the batch_classify tool.

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
>>> from qachat.tools.batch_classify import _configuration
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
    "DEFAULT_CASE_SENSITIVE",
    "DEFAULT_WHOLE_WORDS",
    "DEFAULT_CONTEXT_LINES",
    "DEFAULT_MAX_LISTED",
    "DEFAULT_CANDIDATES",
    "DEFAULT_CATEGORIES",
    "DEFAULT_BATCH_TOKENS",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_TOKENS_PER_SECOND",
    "REPLY_TOKENS_PER_ITEM",
    "PROMPT_OVERHEAD_TOKENS",
    "REPLY_TOKEN_LIMIT_PER_ITEM",
    "RETRY_GROUP",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "SETTINGS",
    "ESTIMATES",
}, "batch_classify")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "batch_classify"
)
require_keys(
    _SETTINGS, {
        "CASE_SENSITIVE",
        "WHOLE_WORDS",
        "CONTEXT_LINES",
        "MAX_LISTED",
        "CANDIDATES",
        "CATEGORIES",
        "BATCH_TOKENS",
        "MAX_MODEL_CALLS",
        "CONFIRMATION",
        "LARGE_BATCHES",
    }, "batch_classify.SETTINGS"
)
DEFAULT_CASE_SENSITIVE: Final[bool] = _SETTINGS["CASE_SENSITIVE"]
if type(DEFAULT_CASE_SENSITIVE) is not bool:
    raise ValueError("batch_classify.SETTINGS.CASE_SENSITIVE must be boolean.")
DEFAULT_WHOLE_WORDS: Final[bool] = _SETTINGS["WHOLE_WORDS"]
if type(DEFAULT_WHOLE_WORDS) is not bool:
    raise ValueError("batch_classify.SETTINGS.WHOLE_WORDS must be boolean.")
DEFAULT_CONTEXT_LINES: Final[int] = (
    require_int(_SETTINGS, "CONTEXT_LINES", "batch_classify.SETTINGS", 0, 5)
)
DEFAULT_MAX_LISTED: Final[int] = (
    require_int(_SETTINGS, "MAX_LISTED", "batch_classify.SETTINGS", 1, 1000)
)
DEFAULT_CANDIDATES: Final[str] = (
    require_string(_SETTINGS, "CANDIDATES", "batch_classify.SETTINGS")
)
if DEFAULT_CANDIDATES not in ('mentions',):
    raise ValueError("batch_classify.SETTINGS.CANDIDATES is not supported.")
_CATEGORIES = _SETTINGS["CATEGORIES"]
if (
    not isinstance(_CATEGORIES, list)
    or len(_CATEGORIES) == 1
    or len(_CATEGORIES) > 8
    or not all(
        isinstance(item, str) and 0 < len(item.strip()) <= 40
        for item in _CATEGORIES
    )
    or len({item.strip().casefold() for item in _CATEGORIES})
    != len(_CATEGORIES)
):
    raise ValueError(
        "batch_classify.SETTINGS.CATEGORIES must contain 2-8 "
        "distinct labels of at most 40 characters, or be empty."
    )
DEFAULT_CATEGORIES: Final[tuple[str, ...]] = tuple(
    item.strip() for item in _CATEGORIES
)
DEFAULT_BATCH_TOKENS: Final[int] = (
    require_int(
        _SETTINGS,
        "BATCH_TOKENS",
        "batch_classify.SETTINGS",
        500,
        8000,
    )
)
DEFAULT_MAX_MODEL_CALLS: Final[int] = (
    require_int(
        _SETTINGS,
        "MAX_MODEL_CALLS",
        "batch_classify.SETTINGS",
        1,
        200,
    )
)
DEFAULT_CONFIRMATION: Final[str] = (
    require_string(_SETTINGS, "CONFIRMATION", "batch_classify.SETTINGS")
)
if DEFAULT_CONFIRMATION not in ('always', 'when_large', 'never'):
    raise ValueError("batch_classify.SETTINGS.CONFIRMATION is not supported.")
DEFAULT_LARGE_BATCHES: Final[int] = (
    require_int(_SETTINGS, "LARGE_BATCHES", "batch_classify.SETTINGS", 0, 100)
)
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "batch_classify"
)
require_keys(
    _ESTIMATES, {
        "PROMPT_TOKENS_PER_SECOND",
        "REPLY_TOKENS_PER_SECOND",
        "REPLY_TOKENS_PER_ITEM",
        "PROMPT_OVERHEAD_TOKENS",
        "REPLY_TOKEN_LIMIT_PER_ITEM",
        "RETRY_GROUP",
    }, "batch_classify.ESTIMATES"
)
PROMPT_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "PROMPT_TOKENS_PER_SECOND",
        "batch_classify.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
REPLY_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "REPLY_TOKENS_PER_SECOND",
        "batch_classify.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
REPLY_TOKENS_PER_ITEM: Final[int] = (
    require_int(
        _ESTIMATES,
        "REPLY_TOKENS_PER_ITEM",
        "batch_classify.ESTIMATES",
        0,
        100000,
    )
)
PROMPT_OVERHEAD_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "PROMPT_OVERHEAD_TOKENS",
        "batch_classify.ESTIMATES",
        0,
        100000,
    )
)
REPLY_TOKEN_LIMIT_PER_ITEM: Final[int] = (
    require_int(
        _ESTIMATES,
        "REPLY_TOKEN_LIMIT_PER_ITEM",
        "batch_classify.ESTIMATES",
        1,
        100000,
    )
)
RETRY_GROUP: Final[int] = (
    require_int(
        _ESTIMATES,
        "RETRY_GROUP",
        "batch_classify.ESTIMATES",
        1,
        10000,
    )
)
