"""Validate the paper-summary tool's packaged configuration defaults.

Only this tool reads its ``defaults.json``. The project-type parser may
override values in ``SETTINGS``; reference analysis, writing estimates,
and template-validation heuristics remain internal tool configuration.
Every value is checked at import time before tool execution can begin.

Key Features
------------

- **Tool ownership**

    Defaults and their validation are colocated with the tool package.

- **Explicit schema**

    Unknown keys, missing fields, and out-of-range values fail promptly.

Module Structure
----------------

- ``DEFAULT_*``

    Values used to initialize settings when project types omit fields.

- ``BEGINNING_*``, ``REFERENCE_*``, and related constants

    Validated reference-selection and report-writing parameters.

- ``LIMIT_TOLERANCE`` and ``MIN_TOTAL_WORDS``

    Template validation parameters consumed by the private parser.

Examples
--------
>>> from kilin.tools.generate_paper_summary._configuration import (
...     DEFAULT_PART_TOKENS,
... )
>>> DEFAULT_PART_TOKENS
6000

"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from kilin.tools._defaults import (
    load_tool_defaults,
    require_int,
    require_int_pair,
    require_keys,
    require_number,
    require_section,
    require_string,
    require_words,
)
from kilin.tools._shared.planning import CONFIRMATION_CHOICES

__all__ = [
    "BEGINNING_CHARACTERS",
    "BEGINNING_PASSAGES",
    "CALL_SECONDS",
    "COMMON_WORDS",
    "DEFAULT_CONFIRMATION",
    "DEFAULT_LARGE_BATCHES",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_NOTE_TOKENS",
    "DEFAULT_PART_TOKENS",
    "DEFAULT_REFERENCE_ITEMS",
    "DEFAULT_REPAIR_ATTEMPTS",
    "DEFAULT_SECTION_TOKENS",
    "DEFAULT_TEMPLATE",
    "EARLIER_SECTION_WORDS",
    "ENTRY_MIN_WORDS",
    "HEADLINE_TOKENS",
    "LIMIT_TOLERANCE",
    "MIN_TOTAL_WORDS",
    "NUMBERED_ENTRIES_MIN",
    "ORGANISATION_WORDS",
    "PART_PROMPT_TOKENS",
    "PROMPT_TOKENS_PER_SECOND",
    "REFERENCES_REPLY_TOKENS",
    "REFERENCES_TOKENS",
    "REFERENCE_CHARACTERS",
    "REPLY_SHARE",
    "REPLY_TOKENS_PER_SECOND",
    "RUNNING_LINE_REPEATS",
    "RUNNING_LINE_WORDS",
    "SECTION_PROMPT_TOKENS",
]

_NAME = re.compile(r"[A-Za-z0-9][\w-]*")
_DEFAULTS = load_tool_defaults(Path(__file__).with_name("defaults.json"))
_GROUPS = {
    "SETTINGS": {
        "TEMPLATE", "PART_TOKENS", "NOTE_TOKENS", "SECTION_TOKENS",
        "REPAIR_ATTEMPTS", "MAX_MODEL_CALLS", "CONFIRMATION",
        "LARGE_BATCHES",
    },
    "PAPER": {
        "BEGINNING_PASSAGES", "BEGINNING_CHARACTERS",
        "ORGANISATION_WORDS",
    },
    "REFERENCES": {
        "REFERENCE_CHARACTERS", "NUMBERED_ENTRIES_MIN",
        "RUNNING_LINE_REPEATS", "RUNNING_LINE_WORDS",
        "ENTRY_MIN_WORDS", "DEFAULT_REFERENCE_ITEMS", "COMMON_WORDS",
    },
    "WRITING": {
        "EARLIER_SECTION_WORDS", "HEADLINE_TOKENS",
        "REFERENCES_TOKENS",
    },
    "ESTIMATES": {
        "PROMPT_TOKENS_PER_SECOND", "REPLY_TOKENS_PER_SECOND",
        "CALL_SECONDS", "PART_PROMPT_TOKENS", "SECTION_PROMPT_TOKENS",
        "REPLY_SHARE", "REFERENCES_REPLY_TOKENS",
    },
    "TEMPLATE_VALIDATION": {"LIMIT_TOLERANCE", "MIN_TOTAL_WORDS"},
}
require_keys(_DEFAULTS, set(_GROUPS), "generate_paper_summary")
for _group, _keys in _GROUPS.items():
    require_keys(
        require_section(_DEFAULTS, _group, "generate_paper_summary"),
        _keys,
        f"generate_paper_summary.{_group}",
    )

_SETTINGS = require_section(_DEFAULTS, "SETTINGS", "generate_paper_summary")
_PAPER = require_section(_DEFAULTS, "PAPER", "generate_paper_summary")
_REFERENCES = require_section(
    _DEFAULTS, "REFERENCES", "generate_paper_summary"
)
_WRITING = require_section(_DEFAULTS, "WRITING", "generate_paper_summary")
_ESTIMATES = require_section(
    _DEFAULTS, "ESTIMATES", "generate_paper_summary"
)
_VALIDATION = require_section(
    _DEFAULTS, "TEMPLATE_VALIDATION", "generate_paper_summary"
)

DEFAULT_TEMPLATE: Final[str] = require_string(
    _SETTINGS, "TEMPLATE", "SETTINGS"
)
if not _NAME.fullmatch(DEFAULT_TEMPLATE):
    raise ValueError("SETTINGS.TEMPLATE must be a plain template name.")
DEFAULT_PART_TOKENS: Final[int] = require_int(
    _SETTINGS, "PART_TOKENS", "SETTINGS", 1000, 16000
)
DEFAULT_NOTE_TOKENS: Final[int] = require_int(
    _SETTINGS, "NOTE_TOKENS", "SETTINGS", 100, 1500
)
DEFAULT_SECTION_TOKENS: Final[int] = require_int(
    _SETTINGS, "SECTION_TOKENS", "SETTINGS", 100, 2000
)
DEFAULT_REPAIR_ATTEMPTS: Final[int] = require_int(
    _SETTINGS, "REPAIR_ATTEMPTS", "SETTINGS", 0, 4
)
DEFAULT_MAX_MODEL_CALLS: Final[int] = require_int(
    _SETTINGS, "MAX_MODEL_CALLS", "SETTINGS", 1, 200
)
DEFAULT_CONFIRMATION: Final[str] = require_string(
    _SETTINGS, "CONFIRMATION", "SETTINGS"
)
if DEFAULT_CONFIRMATION not in CONFIRMATION_CHOICES:
    raise ValueError("SETTINGS.CONFIRMATION is not a supported policy.")
DEFAULT_LARGE_BATCHES: Final[int] = require_int(
    _SETTINGS, "LARGE_BATCHES", "SETTINGS", 0, 100
)

BEGINNING_PASSAGES: Final[int] = require_int(
    _PAPER, "BEGINNING_PASSAGES", "PAPER", 1, 100
)
BEGINNING_CHARACTERS: Final[int] = require_int(
    _PAPER, "BEGINNING_CHARACTERS", "PAPER", 1, 1000000
)
ORGANISATION_WORDS: Final[frozenset[str]] = frozenset(
    require_words(_PAPER, "ORGANISATION_WORDS", "PAPER")
)

REFERENCE_CHARACTERS: Final[int] = require_int(
    _REFERENCES, "REFERENCE_CHARACTERS", "REFERENCES", 1, 1000000
)
NUMBERED_ENTRIES_MIN: Final[int] = require_int(
    _REFERENCES, "NUMBERED_ENTRIES_MIN", "REFERENCES", 1, 1000
)
RUNNING_LINE_REPEATS: Final[int] = require_int(
    _REFERENCES, "RUNNING_LINE_REPEATS", "REFERENCES", 1, 1000
)
RUNNING_LINE_WORDS: Final[int] = require_int(
    _REFERENCES, "RUNNING_LINE_WORDS", "REFERENCES", 1, 1000
)
ENTRY_MIN_WORDS: Final[int] = require_int(
    _REFERENCES, "ENTRY_MIN_WORDS", "REFERENCES", 1, 1000
)
DEFAULT_REFERENCE_ITEMS: Final[tuple[int, int]] = require_int_pair(
    _REFERENCES, "DEFAULT_REFERENCE_ITEMS", "REFERENCES"
)
COMMON_WORDS: Final[tuple[str, ...]] = require_words(
    _REFERENCES, "COMMON_WORDS", "REFERENCES"
)

EARLIER_SECTION_WORDS: Final[int] = require_int(
    _WRITING, "EARLIER_SECTION_WORDS", "WRITING", 1, 100000
)
HEADLINE_TOKENS: Final[int] = require_int(
    _WRITING, "HEADLINE_TOKENS", "WRITING", 1, 100000
)
REFERENCES_TOKENS: Final[int] = require_int(
    _WRITING, "REFERENCES_TOKENS", "WRITING", 1, 100000
)

PROMPT_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "PROMPT_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
REPLY_TOKENS_PER_SECOND: Final[float] = require_number(
    _ESTIMATES, "REPLY_TOKENS_PER_SECOND", "ESTIMATES", 0.001, 1e9
)
CALL_SECONDS: Final[float] = require_number(
    _ESTIMATES, "CALL_SECONDS", "ESTIMATES", 0, 1e9
)
PART_PROMPT_TOKENS: Final[int] = require_int(
    _ESTIMATES, "PART_PROMPT_TOKENS", "ESTIMATES", 0, 1000000
)
SECTION_PROMPT_TOKENS: Final[int] = require_int(
    _ESTIMATES, "SECTION_PROMPT_TOKENS", "ESTIMATES", 0, 1000000
)
REPLY_SHARE: Final[float] = require_number(
    _ESTIMATES, "REPLY_SHARE", "ESTIMATES", 0, 1
)
REFERENCES_REPLY_TOKENS: Final[int] = require_int(
    _ESTIMATES, "REFERENCES_REPLY_TOKENS", "ESTIMATES", 0, 1000000
)

LIMIT_TOLERANCE: Final[float] = require_number(
    _VALIDATION, "LIMIT_TOLERANCE", "TEMPLATE_VALIDATION", 0, 1
)
MIN_TOTAL_WORDS: Final[int] = require_int(
    _VALIDATION, "MIN_TOTAL_WORDS", "TEMPLATE_VALIDATION", 1, 1000000
)
