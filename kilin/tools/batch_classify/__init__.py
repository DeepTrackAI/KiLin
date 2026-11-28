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

- ``PLUGIN``

    Versioned descriptor discovered by the tool registry.

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

Examples
--------
>>> from kilin.tools.batch_classify import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
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
    "PLUGIN",
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


PLUGIN = Tool(
    name="batch_classify",
    summary="Sort every exact mention into categories with the answering "
    "model, and count each category.",
    parse_settings=parse_settings,
    run=run,
    version="1.0.0",
    api_version=1,
)
