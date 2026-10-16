"""Expose the registered ``scan_and_list`` tool.

Its implementation and configuration are local to this package.
Common algorithms are provided by private ``kilin.tools._shared``
modules, not by other registered tools.

Key Features
------------

- **Tool execution**

    Parse validated settings and execute the named tool.

- **Independent defaults**

    Read this package's ``defaults.json`` configuration.

Module Structure
----------------

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

Examples
--------
>>> from kilin.tools.scan_and_list import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools._shared.mentions import (
    Mention,
    MentionCollection,
    MentionScan,
    collect_mentions,
    mention_evidence,
    mention_location,
    mention_snippet,
    scan_mentions,
)
from kilin.tools._shared.paths import short_names
from kilin.tools._shared.scan_settings import SCAN_SETTING_NAMES
from kilin.tools._shared.terms import (
    MAX_TERMS,
    document_phrase,
    quoted_terms,
    term_pattern,
    terms_prompt,
    validated_document,
    validated_terms,
)
from kilin.tools.scan_and_list.tool import (
    ScanSettings,
    mention_report,
    parse_settings,
    run,
    scan_fields,
)

__all__ = [
    "MAX_TERMS",
    "Mention",
    "MentionCollection",
    "MentionScan",
    "SCAN_SETTING_NAMES",
    "ScanSettings",
    "collect_mentions",
    "document_phrase",
    "mention_evidence",
    "mention_location",
    "mention_report",
    "mention_snippet",
    "parse_settings",
    "quoted_terms",
    "run",
    "scan_fields",
    "scan_mentions",
    "short_names",
    "term_pattern",
    "terms_prompt",
    "validated_document",
    "validated_terms",
]
