"""Validate the defaults owned by the project_summary tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool ``SETTINGS`` defaults. The
other sections belong to the tool: how much is read and grouped
(``READING``), the template checks (``TEMPLATE_VALIDATION``), the planning
estimates (``ESTIMATES``) and the policy of its model calls
(``MODEL_CALLS``).

Key Features
------------

- **Isolated configuration**

    The tool's own ``defaults.json``, validated once.

Module Structure
----------------

- ``DEFAULT_TEMPLATE``, ``DEFAULT_DOCUMENT_TOKENS``,
  ``DEFAULT_MAX_DOCUMENTS``, ``DEFAULT_REPAIR_ATTEMPTS``,
  ``DEFAULT_CONFIRMATION`` and ``DEFAULT_LARGE_BATCHES``

    Packaged settings, overridable by a project type.

- ``QUICK_DOCUMENT_TOKENS`` and ``GROUP_TOKENS``

    The excerpt of the quick version, and the notes written from in one
    call.

- ``TEMPLATE_RULES``

    The tool's limits for templates and reports.

- ``PROMPT_TOKENS_PER_SECOND`` and the other estimates

    Planning assumptions for the time one model call takes.

- ``MODEL_CALLS``

    The thinking policy of the tool's model calls.

Examples
--------
>>> from kilin.tools.project_summary import _configuration
>>> _configuration.DEFAULT_DOCUMENT_TOKENS > 0
True

"""

from __future__ import annotations

from pathlib import Path
import re
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
from kilin.tools._shared.planning import CONFIRMATION_CHOICES
from kilin.tools._shared.report_templates import TemplateRules

__all__ = [
    "CALL_SECONDS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_DOCUMENT_TOKENS",
    "DEFAULT_LARGE_BATCHES",
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_REPAIR_ATTEMPTS",
    "DEFAULT_TEMPLATE",
    "GROUP_TOKENS",
    "MAP_PROMPT_TOKENS",
    "MAP_REPLY_TOKENS",
    "MODEL_CALLS",
    "PROMPT_TOKENS_PER_SECOND",
    "QUICK_DOCUMENT_TOKENS",
    "REPLY_TOKENS_PER_SECOND",
    "SECTION_PROMPT_TOKENS",
    "SECTION_REPLY_TOKENS",
    "TEMPLATE_RULES",
]

_TOOL = "project_summary"
_PATH = Path(__file__).with_name("defaults.json")
_DEFAULTS = load_tool_defaults(_PATH)
_GROUPS = {
    "SETTINGS": {
        "TEMPLATE", "DOCUMENT_TOKENS", "MAX_DOCUMENTS", "REPAIR_ATTEMPTS",
        "CONFIRMATION", "LARGE_BATCHES",
    },
    "READING": {"QUICK_DOCUMENT_TOKENS", "GROUP_TOKENS"},
    "TEMPLATE_VALIDATION": {"LIMIT_TOLERANCE", "MIN_TOTAL_WORDS"},
    "ESTIMATES": {
        "PROMPT_TOKENS_PER_SECOND", "REPLY_TOKENS_PER_SECOND",
        "CALL_SECONDS", "MAP_PROMPT_TOKENS", "MAP_REPLY_TOKENS",
        "SECTION_PROMPT_TOKENS", "SECTION_REPLY_TOKENS",
    },
    "MODEL_CALLS": {"THINKING"},
}
require_keys(_DEFAULTS, set(_GROUPS), _TOOL)
for _group, _keys in _GROUPS.items():
    require_keys(
        require_section(_DEFAULTS, _group, _TOOL), _keys, f"{_TOOL}.{_group}"
    )

_SETTINGS = require_section(_DEFAULTS, "SETTINGS", _TOOL)
_WHERE = f"{_TOOL}.SETTINGS"
DEFAULT_TEMPLATE: Final[str] = require_string(_SETTINGS, "TEMPLATE", _WHERE)
if not re.fullmatch(r"[A-Za-z0-9][\w-]*", DEFAULT_TEMPLATE):
    raise ValueError(f"{_WHERE}.TEMPLATE must be a template file name.")
DEFAULT_DOCUMENT_TOKENS: Final[int] = require_int(
    _SETTINGS, "DOCUMENT_TOKENS", _WHERE, 300, 16000
)
DEFAULT_MAX_DOCUMENTS: Final[int] = require_int(
    _SETTINGS, "MAX_DOCUMENTS", _WHERE, 1, 2000
)
DEFAULT_REPAIR_ATTEMPTS: Final[int] = require_int(
    _SETTINGS, "REPAIR_ATTEMPTS", _WHERE, 0, 4
)
DEFAULT_CONFIRMATION: Final[str] = require_string(
    _SETTINGS, "CONFIRMATION", _WHERE
)
if DEFAULT_CONFIRMATION not in CONFIRMATION_CHOICES:
    raise ValueError(f"{_WHERE}.CONFIRMATION is not a supported policy.")
DEFAULT_LARGE_BATCHES: Final[int] = require_int(
    _SETTINGS, "LARGE_BATCHES", _WHERE, 0, 1000
)

_READING = require_section(_DEFAULTS, "READING", _TOOL)
QUICK_DOCUMENT_TOKENS: Final[int] = require_int(
    _READING, "QUICK_DOCUMENT_TOKENS", f"{_TOOL}.READING", 100, 16000
)
GROUP_TOKENS: Final[int] = require_int(
    _READING, "GROUP_TOKENS", f"{_TOOL}.READING", 1000, 64000
)

_VALIDATION = require_section(_DEFAULTS, "TEMPLATE_VALIDATION", _TOOL)
TEMPLATE_RULES: Final[TemplateRules] = TemplateRules(
    limit_tolerance=require_number(
        _VALIDATION, "LIMIT_TOLERANCE", "TEMPLATE_VALIDATION", 0.0, 1.0
    ),
    min_total_words=require_int(
        _VALIDATION, "MIN_TOTAL_WORDS", "TEMPLATE_VALIDATION", 1, 100000
    ),
)

_ESTIMATES = require_section(_DEFAULTS, "ESTIMATES", _TOOL)
PROMPT_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "PROMPT_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
REPLY_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "REPLY_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
CALL_SECONDS: Final[float] = require_number(
    _ESTIMATES, "CALL_SECONDS", "ESTIMATES", 0.0, 1e9
)
MAP_PROMPT_TOKENS: Final[int] = require_int(
    _ESTIMATES, "MAP_PROMPT_TOKENS", "ESTIMATES", 0, 100000
)
MAP_REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "MAP_REPLY_TOKENS", "ESTIMATES", 1, 131072
)
SECTION_PROMPT_TOKENS: Final[int] = require_int(
    _ESTIMATES, "SECTION_PROMPT_TOKENS", "ESTIMATES", 0, 100000
)
SECTION_REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "SECTION_REPLY_TOKENS", "ESTIMATES", 1, 131072
)
MODEL_CALLS: Final[ModelCallSettings] = load_model_call_settings(_PATH)
