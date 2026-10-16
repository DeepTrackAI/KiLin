"""Register KíLín's tools and resolve an intent's tool.

The registry is the controlled boundary between configuration and code. A
project type names a tool (``"tool": {"name": "scan_and_list", ...}``); the
name must be in ``TOOL_REGISTRY``, and the tool's own ``parse_settings``
checks the rest. Configuration never imports arbitrary Python paths, so a
project type file can choose among these tools but cannot add code.
A tool may also declare resources to install in newly created projects;
the core resolves and copies them, not the GUI.

Adding a tool means providing a package with a settings
dataclass, ``parse_settings(data, where)``, and ``run(settings, request,
context)`` (see ``kilin.tools._contract``), then adding it here. A tool
marked ``follows_up`` works on an earlier answer, so routing relates a
question sent to it to an earlier turn even when the router named none.

Key Features
------------

- **Names**

    The tools that exist, as a ``Literal`` type and ``Final`` constants.

- **Registry**

    One entry per tool with its summary, parser, runner, and optional
    packaged workspace resources.

- **Configuration**

    Validate an intent's ``tool`` object and run the tool it names.

Module Structure
----------------

- ``TOOL_PROJECT_INVENTORY``, ``TOOL_SCAN_AND_LIST``,
  ``TOOL_BATCH_CLASSIFY``, ``TOOL_BATCH_REVIEW``, ``TOOL_FILTER_RESULTS``,
  ``TOOL_FIND_FACT``, ``TOOL_FIND_PASSAGES``,
  ``TOOL_GENERATE_PAPER_SUMMARY``, ``TOOL_PAPER_OVERVIEW``,
  ``TOOL_REFERENCE_CHECK``, and ``TOOL_SHOW_SOURCE``

    Canonical string values of the registered tool names.

- ``TOOL_CHOICES``

    Ordered collection of the registered tool names.

- ``ToolName``

    Type alias for the registered tool names.

- ``Tool``

    One registered tool.

- ``ToolUse``

    An intent's tool with its validated settings.

- ``TOOL_REGISTRY``

    Map from each tool name to its registered tool.

- ``parse_tool``

    Validate an intent's ``tool`` object.

- ``run_tool``

    Run an intent's tool.

Examples
--------
>>> from kilin.tools._registry import TOOL_SCAN_AND_LIST, parse_tool
>>> use = parse_tool("code.intents.list", {"name": TOOL_SCAN_AND_LIST})
>>> use.name
'scan_and_list'

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from kilin.tools import (
    batch_classify,
    batch_review,
    filter_results,
    find_fact,
    find_passages,
    generate_paper_summary,
    paper_overview,
    project_inventory,
    reference_check,
    scan_and_list,
    show_source,
)
from kilin.tools._contract import ToolContext, ToolRequest, ToolResult
from kilin.workspace_resources import WorkspaceResource

__all__ = [
    "TOOL_BATCH_CLASSIFY",
    "TOOL_BATCH_REVIEW",
    "TOOL_CHOICES",
    "TOOL_FILTER_RESULTS",
    "TOOL_FIND_FACT",
    "TOOL_FIND_PASSAGES",
    "TOOL_GENERATE_PAPER_SUMMARY",
    "TOOL_PAPER_OVERVIEW",
    "TOOL_PROJECT_INVENTORY",
    "TOOL_REFERENCE_CHECK",
    "TOOL_REGISTRY",
    "TOOL_SCAN_AND_LIST",
    "TOOL_SHOW_SOURCE",
    "Tool",
    "ToolName",
    "ToolUse",
    "parse_tool",
    "run_tool",
]

# The tools that exist. Project type files name tools by these values.
ToolName: TypeAlias = Literal[
    "project_inventory",
    "scan_and_list",
    "batch_classify",
    "batch_review",
    "filter_results",
    "find_fact",
    "find_passages",
    "generate_paper_summary",
    "paper_overview",
    "reference_check",
    "show_source",
]
TOOL_PROJECT_INVENTORY: Final[ToolName] = "project_inventory"
TOOL_SCAN_AND_LIST: Final[ToolName] = "scan_and_list"
TOOL_BATCH_CLASSIFY: Final[ToolName] = "batch_classify"
TOOL_BATCH_REVIEW: Final[ToolName] = "batch_review"
TOOL_FILTER_RESULTS: Final[ToolName] = "filter_results"
TOOL_FIND_FACT: Final[ToolName] = "find_fact"
TOOL_FIND_PASSAGES: Final[ToolName] = "find_passages"
TOOL_GENERATE_PAPER_SUMMARY: Final[ToolName] = "generate_paper_summary"
TOOL_PAPER_OVERVIEW: Final[ToolName] = "paper_overview"
TOOL_REFERENCE_CHECK: Final[ToolName] = "reference_check"
TOOL_SHOW_SOURCE: Final[ToolName] = "show_source"
TOOL_CHOICES: Final[tuple[ToolName, ...]] = (
    TOOL_PROJECT_INVENTORY,
    TOOL_SCAN_AND_LIST,
    TOOL_BATCH_CLASSIFY,
    TOOL_BATCH_REVIEW,
    TOOL_FILTER_RESULTS,
    TOOL_FIND_FACT,
    TOOL_FIND_PASSAGES,
    TOOL_GENERATE_PAPER_SUMMARY,
    TOOL_PAPER_OVERVIEW,
    TOOL_REFERENCE_CHECK,
    TOOL_SHOW_SOURCE,
)


@dataclass(frozen=True, slots=True)
class Tool:
    """One registered tool.

    Attributes
    ----------
    name: ToolName
        Name used in project types.
    summary: str
        What the tool does, for documentation and listings.
    parse_settings: Callable[[dict, str], object]
        Validates the tool's settings from a project type.
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]
        Runs the tool with validated settings.
    follows_up: bool
        True when the tool works on an earlier answer, so a question
        routed to it always relates to an earlier turn.
    workspace_resources: Callable | None
        Optional declaration of packaged resources for a new project.

    """

    name: ToolName
    summary: str
    parse_settings: Callable[[dict, str], object]
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]
    follows_up: bool = False
    workspace_resources: (
        Callable[[object], tuple[WorkspaceResource, ...]] | None
    ) = None


@dataclass(frozen=True, slots=True)
class ToolUse:
    """An intent's tool with its validated settings.

    Attributes
    ----------
    name: ToolName
        The tool.
    settings: object
        The tool's settings dataclass, such as ``ScanSettings``.

    """

    name: ToolName
    settings: object


TOOL_REGISTRY: Final[dict[ToolName, Tool]] = {
    TOOL_PROJECT_INVENTORY: Tool(
        TOOL_PROJECT_INVENTORY,
        "Count and list the project's files by type and folder, with "
        "which have searchable text.",
        project_inventory.parse_settings,
        project_inventory.run,
    ),
    TOOL_SCAN_AND_LIST: Tool(
        TOOL_SCAN_AND_LIST,
        "List and count every exact mention of the question's terms.",
        scan_and_list.parse_settings,
        scan_and_list.run,
    ),
    TOOL_BATCH_CLASSIFY: Tool(
        TOOL_BATCH_CLASSIFY,
        "Sort every exact mention into categories with the answering "
        "model, and count each category.",
        batch_classify.parse_settings,
        batch_classify.run,
    ),
    TOOL_BATCH_REVIEW: Tool(
        TOOL_BATCH_REVIEW,
        "Read one whole document in parts, taking notes with passage "
        "numbers, then answer from the notes.",
        batch_review.parse_settings,
        batch_review.run,
    ),
    TOOL_FILTER_RESULTS: Tool(
        TOOL_FILTER_RESULTS,
        "Judge each item an earlier answer listed and keep those that match "
        "a follow-up.",
        filter_results.parse_settings,
        filter_results.run,
        follows_up=True,
    ),
    TOOL_FIND_FACT: Tool(
        TOOL_FIND_FACT,
        "Find a specific fact in one paper or every paper by reading every "
        "passage; the user can choose a quick answer instead.",
        find_fact.parse_settings,
        find_fact.run,
    ),
    TOOL_FIND_PASSAGES: Tool(
        TOOL_FIND_PASSAGES,
        "Read every sentence in scope and list the ones about a topic, "
        "copied word for word.",
        find_passages.parse_settings,
        find_passages.run,
    ),
    TOOL_PAPER_OVERVIEW: Tool(
        TOOL_PAPER_OVERVIEW,
        "Summarize one paper in the chat; ask which paper when it is "
        "unclear.",
        paper_overview.parse_settings,
        paper_overview.run,
    ),
    TOOL_REFERENCE_CHECK: Tool(
        TOOL_REFERENCE_CHECK,
        "Check one paper's reference list against the project's files, "
        "find where each reference is cited, and judge each claim against "
        "the cited paper; save the full report to outputs/.",
        reference_check.parse_settings,
        reference_check.run,
    ),
    TOOL_SHOW_SOURCE: Tool(
        TOOL_SHOW_SOURCE,
        "Show the sentence an earlier answer cited, with the sentences "
        "around it, and explain it without searching again.",
        show_source.parse_settings,
        show_source.run,
        follows_up=True,
    ),
    TOOL_GENERATE_PAPER_SUMMARY: Tool(
        TOOL_GENERATE_PAPER_SUMMARY,
        "Summarize one paper with a report template, check the structure, "
        "and save the Markdown to outputs/.",
        generate_paper_summary.parse_settings,
        generate_paper_summary.run,
        workspace_resources=generate_paper_summary.workspace_resources,
    ),
}


def parse_tool(where: str, data: object) -> ToolUse:
    """Validate an intent's ``tool`` object.

    Parameters
    ----------
    where: str
        Location for errors, such as ``"code.intents.x"``.
    data: object
        The ``tool`` value from the project type.

    Returns
    -------
    ToolUse
        The tool and its validated settings.

    Raises
    ------
    ValueError
        If the value is not an object, names no registered tool, or has
        invalid settings.

    """

    if not isinstance(data, dict):
        raise ValueError(f"Intent {where!r} tool must be an object.")

    name = data.get("name")
    if name not in TOOL_REGISTRY:
        raise ValueError(
            f"Intent {where!r} tool name must be one of: "
            f"{', '.join(TOOL_CHOICES)}."
        )

    settings = {key: value for key, value in data.items() if key != "name"}
    return ToolUse(
        name, TOOL_REGISTRY[name].parse_settings(settings, f"{where}.tool")
    )


def run_tool(
    use: ToolUse, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Run an intent's tool.

    Parameters
    ----------
    use: ToolUse
        The tool and its settings.
    request: ToolRequest
        The question and the user's words.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The tool's answer, evidence, and summary.

    """

    return TOOL_REGISTRY[use.name].run(use.settings, request, context)
