"""Validate the defaults owned by the find_fact tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool SETTINGS defaults.
MODEL_CALLS owns the internal thinking policy, independently of the GUI.

Key Features
------------

- **Isolated configuration**

    Read and validate the tool-local JSON file.

Module Structure
----------------

- ``MODEL_CALLS``

    Validated policy for internal answering-model requests.

- ``DEFAULT_*`` and internal constants

    Validated settings and algorithm parameters.

Examples
--------
>>> from kilin.tools.find_fact import _configuration
>>> bool(_configuration._DEFAULTS)
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

from kilin.tools._model_calls import (
    ModelCallSettings,
    load_model_call_settings,
)

__all__ = [
    "ANSWER_REPLY_TOKENS",
    "NOTE_REPLY_TOKENS",
    "MODEL_CALLS",
    "DEFAULT_PART_TOKENS",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "DEFAULT_EXTRA_SEARCHES",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_TOKENS_PER_SECOND",
    "NOTES_PROMPT_TOKENS",
    "CALL_SECONDS",
    "QUICK_ANSWER_SECONDS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
    "ESTIMATES",
}, "find_fact")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "find_fact"
)
require_keys(
    _SETTINGS, {
        "PART_TOKENS",
        "MAX_MODEL_CALLS",
        "CONFIRMATION",
        "LARGE_BATCHES",
        "EXTRA_SEARCHES",
    }, "find_fact.SETTINGS"
)
DEFAULT_PART_TOKENS: Final[int] = (
    require_int(_SETTINGS, "PART_TOKENS", "find_fact.SETTINGS", 1000, 16000)
)
DEFAULT_MAX_MODEL_CALLS: Final[int] = (
    require_int(_SETTINGS, "MAX_MODEL_CALLS", "find_fact.SETTINGS", 1, 500)
)
DEFAULT_CONFIRMATION: Final[str] = (
    require_string(_SETTINGS, "CONFIRMATION", "find_fact.SETTINGS")
)
if DEFAULT_CONFIRMATION not in ('always', 'when_large', 'never'):
    raise ValueError("find_fact.SETTINGS.CONFIRMATION is not supported.")
DEFAULT_LARGE_BATCHES: Final[int] = (
    require_int(_SETTINGS, "LARGE_BATCHES", "find_fact.SETTINGS", 0, 100)
)
DEFAULT_EXTRA_SEARCHES: Final[int] = (
    require_int(_SETTINGS, "EXTRA_SEARCHES", "find_fact.SETTINGS", 0, 5)
)
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "find_fact"
)
require_keys(
    _ESTIMATES, {
        "ANSWER_REPLY_TOKENS",
        "NOTE_REPLY_TOKENS",
        "PROMPT_TOKENS_PER_SECOND",
        "REPLY_TOKENS_PER_SECOND",
        "NOTES_PROMPT_TOKENS",
        "CALL_SECONDS",
        "QUICK_ANSWER_SECONDS",
    }, "find_fact.ESTIMATES"
)
NOTE_REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "NOTE_REPLY_TOKENS", "find_fact.ESTIMATES", 1, 131072
)
ANSWER_REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "ANSWER_REPLY_TOKENS", "find_fact.ESTIMATES", 1, 131072
)
PROMPT_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "PROMPT_TOKENS_PER_SECOND",
        "find_fact.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
REPLY_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "REPLY_TOKENS_PER_SECOND",
        "find_fact.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
NOTES_PROMPT_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "NOTES_PROMPT_TOKENS",
        "find_fact.ESTIMATES",
        0,
        100000,
    )
)
CALL_SECONDS: Final[float] = (
    require_number(
        _ESTIMATES,
        "CALL_SECONDS",
        "find_fact.ESTIMATES",
        0,
        1000000000.0,
    )
)
QUICK_ANSWER_SECONDS: Final[float] = (
    require_number(
        _ESTIMATES,
        "QUICK_ANSWER_SECONDS",
        "find_fact.ESTIMATES",
        0,
        1000000000.0,
    )
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
