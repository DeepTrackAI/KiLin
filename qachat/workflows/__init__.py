"""Optional answering workflows that intents can declare.

An intent in ``qachat/project_types.json`` normally retrieves evidence and
asks the answering model. An intent may instead declare a ``workflow``; the
router still selects the intent, and the workflow's configuration decides
how the answer is produced. Intents without a workflow are unchanged.

Key Features
------------

- **scan_and_list**

    List and count every exact mention of a term across the indexed text,
    with source-linked matches and no language-model analysis.

- **batch_classify**

    Classify every exact mention with the answering model in batches,
    validate the labels, and count in Python.

- **Planning**

    Estimate expensive work and let the user continue, run a quick
    sample, or cancel.

Module Structure
----------------

- ``scan_and_list``

    Exhaustive mention listing.

- ``batch_classify``

    Batched classification of mentions.

- ``planning``

    Plans, confirmation policy, and the user's choice.

"""

from qachat.workflows.batch_classify import (
    UNCLEAR,
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
from qachat.workflows.planning import (
    WORKFLOW_CHOICE_CANCEL,
    WORKFLOW_CHOICE_CHOICES,
    WORKFLOW_CHOICE_CONTINUE,
    WORKFLOW_CHOICE_QUICK,
    WorkflowChoice,
    WorkflowPlan,
    needs_confirmation,
)
from qachat.workflows.scan_and_list import (
    MAX_TERMS,
    Mention,
    MentionScan,
    mention_evidence,
    mention_location,
    mention_report,
    mention_snippet,
    quoted_terms,
    scan_mentions,
    short_names,
    terms_prompt,
    validated_document,
    validated_terms,
)

__all__ = [
    "MAX_TERMS",
    "UNCLEAR",
    "WORKFLOW_CHOICE_CANCEL",
    "WORKFLOW_CHOICE_CHOICES",
    "WORKFLOW_CHOICE_CONTINUE",
    "WORKFLOW_CHOICE_QUICK",
    "Label",
    "Mention",
    "MentionScan",
    "WorkflowChoice",
    "WorkflowPlan",
    "classification_evidence",
    "classification_report",
    "classify_prompt",
    "estimate_seconds",
    "item_text",
    "mention_evidence",
    "mention_location",
    "mention_report",
    "mention_snippet",
    "needs_confirmation",
    "parse_labels",
    "plan_batches",
    "quick_sample",
    "quoted_terms",
    "scan_mentions",
    "short_names",
    "terms_prompt",
    "validated_document",
    "validated_categories",
    "validated_terms",
]
