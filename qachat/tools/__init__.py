"""Tools that intents in a project type can use.

An intent normally retrieves evidence and asks the answering model. An
intent may instead name a tool in ``qachat/project_types/<type>.json``:
``"tool": {"name": "scan_and_list", ...settings}``. The router still
selects the intent; the tool then produces the answer. Intents without a
tool are unchanged.

Each tool is one module with a settings dataclass, ``parse_settings``, and
``run``; ``registry`` lists the tools that exist, and ``contract`` defines
what a tool receives and returns.

Key Features
------------

- **project_inventory**

    Count and list the project's files by type and folder, and which have
    searchable text.

- **scan_and_list**

    List and count every exact mention of a term across the indexed text,
    with source-linked matches and no language-model analysis.

- **batch_classify**

    Sort every exact mention into categories with the answering model in
    batches, validate the labels, and count in Python.

- **batch_review**

    Read one whole document in parts with notes, then answer from them.

- **generate_paper_summary**

    Summarize one paper with a report template (for example a research
    briefing), check its structure, and save it to ``outputs/``.

- **Planning**

    Estimate expensive runs and let the user continue, run a quick
    sample, or cancel.

Module Structure
----------------

- ``contract``

    ``ToolRequest``, ``ToolContext``, ``ToolResult``, settings checks.

- ``registry``

    Tool names, ``TOOL_REGISTRY``, ``parse_tool``, ``run_tool``.

- ``project_inventory``

    Exact facts about the project's files.

- ``scan_and_list``

    Exhaustive mention listing.

- ``batch_classify``

    Batched classification of mentions.

- ``batch_review``

    Part-by-part reading of one document.

- ``generate_paper_summary``

    Template-checked paper summaries; the custom-tool reference example.

- ``outputs``

    The never-indexed ``outputs/`` folder.

- ``report_template``

    Markdown report templates and the checks against them.

- ``planning``

    Plans, confirmation policy, and the user's choice.

"""

from qachat.tools.batch_classify import (
    CANDIDATE_CHOICES,
    CANDIDATES_MENTIONS,
    UNCLEAR,
    CandidateSource,
    ClassifySettings,
    Label,
)
from qachat.tools.contract import (
    EarlierTurn,
    ToolContext,
    ToolRequest,
    ToolResult,
)
from qachat.tools.planning import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_CHOICES,
    CONFIRMATION_NEVER,
    CONFIRMATION_WHEN_LARGE,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_CHOICES,
    PLAN_CHOICE_CONTINUE,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    PlanChoice,
    ToolPlan,
    needs_confirmation,
)
from qachat.tools.registry import (
    TOOL_BATCH_CLASSIFY,
    TOOL_BATCH_REVIEW,
    TOOL_CHOICES,
    TOOL_GENERATE_PAPER_SUMMARY,
    TOOL_PROJECT_INVENTORY,
    TOOL_REGISTRY,
    TOOL_SCAN_AND_LIST,
    Tool,
    ToolName,
    ToolUse,
    parse_tool,
    run_tool,
)
from qachat.tools.scan_and_list import (
    MAX_TERMS,
    Mention,
    MentionScan,
    ScanSettings,
    scan_mentions,
)

__all__ = [
    "EarlierTurn",
    "CANDIDATE_CHOICES",
    "CANDIDATES_MENTIONS",
    "CONFIRMATION_ALWAYS",
    "CONFIRMATION_CHOICES",
    "CONFIRMATION_NEVER",
    "CONFIRMATION_WHEN_LARGE",
    "MAX_TERMS",
    "PLAN_CHOICE_CANCEL",
    "PLAN_CHOICE_CHOICES",
    "PLAN_CHOICE_CONTINUE",
    "PLAN_CHOICE_QUICK",
    "TOOL_BATCH_CLASSIFY",
    "TOOL_BATCH_REVIEW",
    "TOOL_CHOICES",
    "TOOL_GENERATE_PAPER_SUMMARY",
    "TOOL_PROJECT_INVENTORY",
    "TOOL_REGISTRY",
    "TOOL_SCAN_AND_LIST",
    "UNCLEAR",
    "CandidateSource",
    "ClassifySettings",
    "ConfirmationPolicy",
    "Label",
    "Mention",
    "MentionScan",
    "PlanChoice",
    "ScanSettings",
    "Tool",
    "ToolContext",
    "ToolName",
    "ToolPlan",
    "ToolRequest",
    "ToolResult",
    "ToolUse",
    "needs_confirmation",
    "parse_tool",
    "run_tool",
    "scan_mentions",
]
