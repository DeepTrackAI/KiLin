"""Validate the defaults owned by the reference_check tool.

Unknown configuration keys and incorrect types are rejected at import.
Project-type values override only the per-tool SETTINGS defaults; the
other sections are the tool's thresholds for the reference list, the
matching of entries to files, and the finding of in-text citations.

Key Features
------------

- **Isolated configuration**

    Read and validate the tool-local JSON file.

Module Structure
----------------

- ``DEFAULT_MAX_LISTED``

    Validated project-type setting.

- ``REFERENCE_LAYOUT``, ``MATCH_RULES`` and ``CITATION_RULES``

    Validated thresholds, in the form the extractor, the matcher and the
    citation finder take.

Examples
--------
>>> from kilin.tools.reference_check import _configuration
>>> _configuration.REFERENCE_LAYOUT.entry_min_words > 0
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
    require_words,
)
from kilin.tools._shared.references import ReferenceLayout
from kilin.tools.reference_check._citations import CitationRules
from kilin.tools.reference_check._matching import MatchRules

__all__ = [
    "CITATION_RULES",
    "DEFAULT_MAX_LISTED",
    "MATCH_RULES",
    "REFERENCE_LAYOUT",
]

_TOOL = "reference_check"
_DEFAULTS = load_tool_defaults(Path(__file__).with_name("defaults.json"))
_GROUPS = {
    "SETTINGS": {"MAX_LISTED"},
    "REFERENCES": {
        "REFERENCE_CHARACTERS", "NUMBERED_ENTRIES_MIN",
        "RUNNING_LINE_REPEATS", "RUNNING_LINE_WORDS", "ENTRY_MIN_WORDS",
    },
    "MATCHING": {
        "FIRST_CHARACTERS", "TITLE_WORDS_MIN", "TITLE_ON_PAGE",
        "TITLE_FOUND", "TITLE_PROBABLE", "PAGE_FOUND", "PAGE_PROBABLE",
        "ENTRY_WORDS_MIN", "COMMON_WORDS",
    },
    "CITATIONS": {
        "BRACKET_MARKERS_MIN", "SUPERSCRIPT_RATIO", "SIGNATURE_CHARACTERS",
        "MAX_RANGE", "UNIT_WORDS", "BYLINE_SHARE",
    },
}
require_keys(_DEFAULTS, set(_GROUPS), _TOOL)
for _group, _keys in _GROUPS.items():
    require_keys(
        require_section(_DEFAULTS, _group, _TOOL), _keys, f"{_TOOL}.{_group}"
    )

_SETTINGS = require_section(_DEFAULTS, "SETTINGS", _TOOL)
_REFERENCES = require_section(_DEFAULTS, "REFERENCES", _TOOL)
_MATCHING = require_section(_DEFAULTS, "MATCHING", _TOOL)
_CITATIONS = require_section(_DEFAULTS, "CITATIONS", _TOOL)

DEFAULT_MAX_LISTED: Final[int] = require_int(
    _SETTINGS, "MAX_LISTED", f"{_TOOL}.SETTINGS", 1, 1000
)

REFERENCE_LAYOUT: Final[ReferenceLayout] = ReferenceLayout(
    max_characters=require_int(
        _REFERENCES, "REFERENCE_CHARACTERS", "REFERENCES", 1, 1000000
    ),
    numbered_entries_min=require_int(
        _REFERENCES, "NUMBERED_ENTRIES_MIN", "REFERENCES", 1, 1000
    ),
    running_line_repeats=require_int(
        _REFERENCES, "RUNNING_LINE_REPEATS", "REFERENCES", 1, 1000
    ),
    running_line_words=require_int(
        _REFERENCES, "RUNNING_LINE_WORDS", "REFERENCES", 1, 1000
    ),
    entry_min_words=require_int(
        _REFERENCES, "ENTRY_MIN_WORDS", "REFERENCES", 1, 1000
    ),
)

MATCH_RULES: Final[MatchRules] = MatchRules(
    first_characters=require_int(
        _MATCHING, "FIRST_CHARACTERS", "MATCHING", 100, 100000
    ),
    title_words_min=require_int(
        _MATCHING, "TITLE_WORDS_MIN", "MATCHING", 1, 100
    ),
    title_on_page=require_number(
        _MATCHING, "TITLE_ON_PAGE", "MATCHING", 0.0, 1.0
    ),
    title_found=require_number(_MATCHING, "TITLE_FOUND", "MATCHING", 0.0, 1.0),
    title_probable=require_number(
        _MATCHING, "TITLE_PROBABLE", "MATCHING", 0.0, 1.0
    ),
    page_found=require_number(_MATCHING, "PAGE_FOUND", "MATCHING", 0.0, 1.0),
    page_probable=require_number(
        _MATCHING, "PAGE_PROBABLE", "MATCHING", 0.0, 1.0
    ),
    entry_words_min=require_int(
        _MATCHING, "ENTRY_WORDS_MIN", "MATCHING", 1, 100
    ),
    common_words=frozenset(
        require_words(_MATCHING, "COMMON_WORDS", "MATCHING")
    ),
)
if MATCH_RULES.title_probable > MATCH_RULES.title_found:
    raise ValueError("MATCHING.TITLE_PROBABLE must not exceed TITLE_FOUND.")
if MATCH_RULES.page_probable > MATCH_RULES.page_found:
    raise ValueError("MATCHING.PAGE_PROBABLE must not exceed PAGE_FOUND.")

CITATION_RULES: Final[CitationRules] = CitationRules(
    bracket_markers_min=require_int(
        _CITATIONS, "BRACKET_MARKERS_MIN", "CITATIONS", 1, 1000
    ),
    superscript_ratio=require_number(
        _CITATIONS, "SUPERSCRIPT_RATIO", "CITATIONS", 0.1, 1.0
    ),
    signature_characters=require_int(
        _CITATIONS, "SIGNATURE_CHARACTERS", "CITATIONS", 3, 100
    ),
    max_range=require_int(_CITATIONS, "MAX_RANGE", "CITATIONS", 1, 1000),
    unit_words=frozenset(
        require_words(_CITATIONS, "UNIT_WORDS", "CITATIONS")
    ),
    byline_share=require_number(
        _CITATIONS, "BYLINE_SHARE", "CITATIONS", 0.0, 1.0
    ),
)
