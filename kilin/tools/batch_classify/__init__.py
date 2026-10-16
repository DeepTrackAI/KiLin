"""Expose the registered ``batch_classify`` tool.

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
>>> from kilin.tools.batch_classify import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools.batch_classify.tool import (
    CANDIDATES_MENTIONS,
    CANDIDATE_CHOICES,
    UNCLEAR,
    CandidateSource,
    ClassifySettings,
    parse_settings,
    run,
    Label,
    classification_evidence,
    classification_report,
    classify_prompt,
    estimate_seconds,
    item_text,
    parse_labels,
    plan_batches,
    quick_sample,
    validated_categories,
)

__all__ = [
    "CANDIDATES_MENTIONS",
    "CANDIDATE_CHOICES",
    "UNCLEAR",
    "CandidateSource",
    "ClassifySettings",
    "parse_settings",
    "run",
    "Label",
    "classification_evidence",
    "classification_report",
    "classify_prompt",
    "estimate_seconds",
    "item_text",
    "parse_labels",
    "plan_batches",
    "quick_sample",
    "validated_categories",
]
