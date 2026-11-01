"""Validate the defaults owned by the scan_and_list tool.

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
>>> from kilin.tools.scan_and_list import _configuration
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
    "DEFAULT_CASE_SENSITIVE",
    "DEFAULT_WHOLE_WORDS",
    "DEFAULT_CONTEXT_LINES",
    "DEFAULT_MAX_LISTED",
    "UNSEARCHABLE_EXAMPLES",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
    "HEURISTICS",
}, "scan_and_list")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "scan_and_list"
)
require_keys(
    _SETTINGS, {
        "CASE_SENSITIVE",
        "WHOLE_WORDS",
        "CONTEXT_LINES",
        "MAX_LISTED",
    }, "scan_and_list.SETTINGS"
)
DEFAULT_CASE_SENSITIVE: Final[bool] = _SETTINGS["CASE_SENSITIVE"]
if type(DEFAULT_CASE_SENSITIVE) is not bool:
    raise ValueError("scan_and_list.SETTINGS.CASE_SENSITIVE must be boolean.")
DEFAULT_WHOLE_WORDS: Final[bool] = _SETTINGS["WHOLE_WORDS"]
if type(DEFAULT_WHOLE_WORDS) is not bool:
    raise ValueError("scan_and_list.SETTINGS.WHOLE_WORDS must be boolean.")
DEFAULT_CONTEXT_LINES: Final[int] = (
    require_int(_SETTINGS, "CONTEXT_LINES", "scan_and_list.SETTINGS", 0, 5)
)
DEFAULT_MAX_LISTED: Final[int] = (
    require_int(_SETTINGS, "MAX_LISTED", "scan_and_list.SETTINGS", 1, 1000)
)
_HEURISTICS = require_section(
    _DEFAULTS, "HEURISTICS", "scan_and_list"
)
require_keys(
    _HEURISTICS, {
        "UNSEARCHABLE_EXAMPLES",
    }, "scan_and_list.HEURISTICS"
)
UNSEARCHABLE_EXAMPLES: Final[int] = (
    require_int(
        _HEURISTICS,
        "UNSEARCHABLE_EXAMPLES",
        "scan_and_list.HEURISTICS",
        0,
        100,
    )
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
