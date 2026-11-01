"""Validate the defaults owned by the batch_review tool.

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
>>> from kilin.tools.batch_review import _configuration
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
    "MODEL_CALLS",
    "DEFAULT_PART_TOKENS",
    "DEFAULT_NOTE_TOKENS",
    "DEFAULT_ANSWER_TOKENS",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "PROMPT_TOKENS_PER_SECOND",
    "REPLY_TOKENS_PER_SECOND",
    "PROMPT_OVERHEAD_TOKENS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
    "ESTIMATES",
}, "batch_review")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "batch_review"
)
require_keys(
    _SETTINGS, {
        "PART_TOKENS",
        "NOTE_TOKENS",
        "ANSWER_TOKENS",
        "MAX_MODEL_CALLS",
        "CONFIRMATION",
        "LARGE_BATCHES",
    }, "batch_review.SETTINGS"
)
DEFAULT_PART_TOKENS: Final[int] = (
    require_int(_SETTINGS, "PART_TOKENS", "batch_review.SETTINGS", 1000, 16000)
)
DEFAULT_NOTE_TOKENS: Final[int] = (
    require_int(_SETTINGS, "NOTE_TOKENS", "batch_review.SETTINGS", 100, 1500)
)
DEFAULT_ANSWER_TOKENS: Final[int] = (
    require_int(_SETTINGS, "ANSWER_TOKENS", "batch_review.SETTINGS", 200, 4000)
)
DEFAULT_MAX_MODEL_CALLS: Final[int] = (
    require_int(_SETTINGS, "MAX_MODEL_CALLS", "batch_review.SETTINGS", 1, 200)
)
DEFAULT_CONFIRMATION: Final[str] = (
    require_string(_SETTINGS, "CONFIRMATION", "batch_review.SETTINGS")
)
if DEFAULT_CONFIRMATION not in ('always', 'when_large', 'never'):
    raise ValueError("batch_review.SETTINGS.CONFIRMATION is not supported.")
DEFAULT_LARGE_BATCHES: Final[int] = (
    require_int(_SETTINGS, "LARGE_BATCHES", "batch_review.SETTINGS", 0, 100)
)
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "batch_review"
)
require_keys(
    _ESTIMATES, {
        "PROMPT_TOKENS_PER_SECOND",
        "REPLY_TOKENS_PER_SECOND",
        "PROMPT_OVERHEAD_TOKENS",
    }, "batch_review.ESTIMATES"
)
PROMPT_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "PROMPT_TOKENS_PER_SECOND",
        "batch_review.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
REPLY_TOKENS_PER_SECOND: Final[float] = (
    require_number(
        _ESTIMATES,
        "REPLY_TOKENS_PER_SECOND",
        "batch_review.ESTIMATES",
        0.001,
        1000000000.0,
    )
)
PROMPT_OVERHEAD_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "PROMPT_OVERHEAD_TOKENS",
        "batch_review.ESTIMATES",
        0,
        100000,
    )
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
