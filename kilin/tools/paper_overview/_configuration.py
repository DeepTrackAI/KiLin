"""Validate the defaults owned by the paper_overview tool.

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
>>> from kilin.tools.paper_overview import _configuration
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
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_ANSWER_TOKENS",
    "OPENING_PASSAGES",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
    "HEURISTICS",
}, "paper_overview")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "paper_overview"
)
require_keys(
    _SETTINGS, {
        "MAX_TOKENS",
        "ANSWER_TOKENS",
    }, "paper_overview.SETTINGS"
)
DEFAULT_MAX_TOKENS: Final[int] = (
    require_int(
        _SETTINGS,
        "MAX_TOKENS",
        "paper_overview.SETTINGS",
        1000,
        20000,
    )
)
DEFAULT_ANSWER_TOKENS: Final[int] = (
    require_int(
        _SETTINGS,
        "ANSWER_TOKENS",
        "paper_overview.SETTINGS",
        100,
        2000,
    )
)
_HEURISTICS = require_section(
    _DEFAULTS, "HEURISTICS", "paper_overview"
)
require_keys(
    _HEURISTICS, {
        "OPENING_PASSAGES",
    }, "paper_overview.HEURISTICS"
)
OPENING_PASSAGES: Final[int] = (
    require_int(
        _HEURISTICS,
        "OPENING_PASSAGES",
        "paper_overview.HEURISTICS",
        1,
        100,
    )
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
