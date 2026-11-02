"""Validate the defaults owned by the project_inventory tool.

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
>>> from kilin.tools.project_inventory import _configuration
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
    "DEFAULT_MAX_LISTED",
    "DEFAULT_MAX_FOLDERS",
]

_DEFAULTS = load_tool_defaults(
    Path(__file__).with_name("defaults.json")
)
require_keys(_DEFAULTS, {
    "MODEL_CALLS",
    "SETTINGS",
}, "project_inventory")
_SETTINGS = require_section(
    _DEFAULTS, "SETTINGS", "project_inventory"
)
require_keys(
    _SETTINGS, {
        "MAX_LISTED",
        "MAX_FOLDERS",
    }, "project_inventory.SETTINGS"
)
DEFAULT_MAX_LISTED: Final[int] = (
    require_int(_SETTINGS, "MAX_LISTED", "project_inventory.SETTINGS", 1, 2000)
)
DEFAULT_MAX_FOLDERS: Final[int] = (
    require_int(_SETTINGS, "MAX_FOLDERS", "project_inventory.SETTINGS", 1, 100)
)

MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(
    Path(__file__).with_name("defaults.json")
)
