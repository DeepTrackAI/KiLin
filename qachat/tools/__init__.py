"""Tools that intents in a project type can use.

An intent normally retrieves evidence and asks the answering model. An
intent may instead name a tool in ``qachat/project_types/<type>.json``:
``"tool": {"name": "scan_and_list", ...settings}``. The router still
selects the intent; the tool then produces the answer. Intents without a
tool are unchanged.

Registered tools provide a settings dataclass, ``parse_settings``, and
``run``. The private ``_registry`` lists available tools, and ``_contract``
defines their requests and results. Other shared operations reside in
``_shared/``. Tool packages can own their ``defaults.json``, helper
modules, and templates. Every registered executable tool is a package.

This package re-exports the contract, the planning vocabulary, the
registry, and the names other modules use from ``batch_classify`` and
``scan_and_list``. Other tools and helpers are reached through their own
submodules.

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

- **filter_results**

    Judge each item an earlier answer listed and keep those that match a
    follow-up.

- **find_fact**

    Find a fact, or a complete list, by reading every passage of one paper
    or of every paper.

- **find_passages**

    List every sentence in scope about a topic, copied word for word.

- **paper_overview**

    Summarize one paper in the chat, asking which paper when it is
    unclear.

- **show_source**

    Show the sentence an earlier answer cited, with its neighbours, and
    explain it without searching again.

- **generate_paper_summary**

    Summarize one paper with a report template (for example a research
    briefing), check its structure, and save it to ``outputs/``.

- **Planning**

    Estimate expensive runs and let the user continue, run a quick
    sample, or cancel.

Module Structure
----------------

- ``EarlierTurn``, ``ToolRequest``, ``ToolContext``, and ``ToolResult``

    What a tool receives and returns (from ``_contract``).

- ``CONFIRMATION_ALWAYS``, ``CONFIRMATION_WHEN_LARGE``,
  ``CONFIRMATION_NEVER``, ``CONFIRMATION_CHOICES``, and
  ``ConfirmationPolicy``

    When a tool asks the user before running (from ``_shared.planning``).

- ``PLAN_CHOICE_CONTINUE``, ``PLAN_CHOICE_QUICK``, ``PLAN_CHOICE_CANCEL``,
  ``PLAN_CHOICE_CHOICES``, and ``PlanChoice``

    The user's answer to a plan (from ``_shared.planning``).

- ``ToolPlan``

    The work a tool would do, for the user to confirm.

- ``needs_confirmation``

    Tell whether a plan must be confirmed before it runs.

- ``TOOL_BATCH_CLASSIFY``, ``TOOL_BATCH_REVIEW``,
  ``TOOL_GENERATE_PAPER_SUMMARY``, ``TOOL_PROJECT_INVENTORY``,
  ``TOOL_SCAN_AND_LIST``, ``TOOL_CHOICES``, and ``ToolName``

    Registered tool names (from ``_registry``).

- ``TOOL_REGISTRY``

    Map from each tool name to its registered tool.

- ``Tool`` and ``ToolUse``

    One registered tool, and an intent's tool with validated settings.

- ``parse_tool`` and ``run_tool``

    Validate an intent's ``tool`` object, and run it.

- ``CANDIDATES_MENTIONS``, ``CANDIDATE_CHOICES``, and ``CandidateSource``

    How ``batch_classify`` collects the items it labels.

- ``ClassifySettings``

    Settings of ``batch_classify`` in a project type.

- ``UNCLEAR`` and ``Label``

    The always-offered category, and the category given to one mention.

- ``MAX_TERMS``

    Most terms ``scan_and_list`` scans for in one question.

- ``ScanSettings``

    Settings of ``scan_and_list`` in a project type.

- ``Mention`` and ``MentionScan``

    One occurrence of a term, and all mentions found.

- ``scan_mentions``

    Find every occurrence of the terms in the indexed text.

Examples
--------
>>> from qachat.tools import TOOL_REGISTRY, TOOL_SCAN_AND_LIST
>>> print(TOOL_REGISTRY[TOOL_SCAN_AND_LIST].summary)
List and count every exact mention of the question's terms.

"""

from qachat.tools.batch_classify import (
    CANDIDATE_CHOICES,
    CANDIDATES_MENTIONS,
    UNCLEAR,
    CandidateSource,
    ClassifySettings,
    Label,
)
from qachat.tools._contract import (
    EarlierTurn,
    ToolContext,
    ToolRequest,
    ToolResult,
)
from qachat.tools._shared.planning import (
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
from qachat.tools._registry import (
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
    "CANDIDATES_MENTIONS",
    "CANDIDATE_CHOICES",
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
    "EarlierTurn",
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
