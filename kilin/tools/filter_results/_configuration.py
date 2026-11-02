"""Validate the defaults owned by the filter_results tool.

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
>>> from kilin.tools.filter_results import _configuration
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
    "DEFAULT_BATCH_TOKENS",
    "DEFAULT_ITEM_CHARACTERS",
    "ITEM_NUMBER_TOKENS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
    "ESTIMATES",
}, "filter_results")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "filter_results"
)
require_keys(
    _SETTINGS, {
        "BATCH_TOKENS",
        "ITEM_CHARACTERS",
    }, "filter_results.SETTINGS"
)
DEFAULT_BATCH_TOKENS: Final[int] = (
    require_int(
        _SETTINGS,
        "BATCH_TOKENS",
        "filter_results.SETTINGS",
        500,
        8000,
    )
)
DEFAULT_ITEM_CHARACTERS: Final[int] = (
    require_int(
        _SETTINGS,
        "ITEM_CHARACTERS",
        "filter_results.SETTINGS",
        100,
        2000,
    )
)
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "filter_results"
)
require_keys(
    _ESTIMATES, {
        "ITEM_NUMBER_TOKENS",
    }, "filter_results.ESTIMATES"
)
ITEM_NUMBER_TOKENS: Final[int] = (
    require_int(
        _ESTIMATES,
        "ITEM_NUMBER_TOKENS",
        "filter_results.ESTIMATES",
        0,
        100000,
    )
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
