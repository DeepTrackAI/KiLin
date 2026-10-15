"""Validate the defaults owned by the show_source tool.

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
>>> from qachat.tools.show_source import _configuration
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
    "DEFAULT_CONTEXT_SENTENCES",
    "DEFAULT_MAX_PASSAGES",
    "MIN_WORDS",
    "DEFAULT_ANSWER_TOKENS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "SETTINGS",
    "HEURISTICS",
}, "show_source")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "show_source"
)
require_keys(
    _SETTINGS, {
        "CONTEXT_SENTENCES",
        "MAX_PASSAGES",
        "ANSWER_TOKENS",
    }, "show_source.SETTINGS"
)
DEFAULT_CONTEXT_SENTENCES: Final[int] = (
    require_int(_SETTINGS, "CONTEXT_SENTENCES", "show_source.SETTINGS", 0, 10)
)
DEFAULT_MAX_PASSAGES: Final[int] = (
    require_int(_SETTINGS, "MAX_PASSAGES", "show_source.SETTINGS", 1, 10)
)
DEFAULT_ANSWER_TOKENS: Final[int] = (
    require_int(_SETTINGS, "ANSWER_TOKENS", "show_source.SETTINGS", 100, 2000)
)

_HEURISTICS = require_section(
    _DEFAULTS, "HEURISTICS", "show_source"
)
require_keys(
    _HEURISTICS, {"MIN_WORDS"}, "show_source.HEURISTICS"
)
MIN_WORDS: Final[int] = require_int(
    _HEURISTICS, "MIN_WORDS", "show_source.HEURISTICS", 1, 100
)
