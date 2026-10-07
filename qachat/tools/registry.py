"""Register QAChat's tools and resolve an intent's tool.

The registry is the controlled boundary between configuration and code. A
project type names a tool (``"tool": {"name": "scan_and_list", ...}``); the
name must be in ``TOOL_REGISTRY``, and the tool's own ``parse_settings``
checks the rest. Configuration never imports arbitrary Python paths.

Adding a tool means writing one module in ``qachat/tools`` with a settings
dataclass, ``parse_settings(data, where)``, and ``run(settings, request,
context)`` (see ``qachat.tools.contract``), then adding it here.

Key Features
------------

- **Names**

    ``ToolName`` and ``TOOL_*`` constants: the tools that exist.

- **Registry**

    ``TOOL_REGISTRY`` maps each name to its settings parser and runner.

- **Configuration**

    ``parse_tool`` validates an intent's ``tool`` object into a
    ``ToolUse``; ``run_tool`` runs it.

Module Structure
----------------

- ``ToolName``
- ``Tool``
- ``ToolUse``
- ``TOOL_REGISTRY``
- ``parse_tool``
- ``run_tool``

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from qachat.tools import (
    batch_classify,
    batch_review,
    find_passages,
    generate_paper_summary,
    paper_overview,
    project_inventory,
    scan_and_list,
)
from qachat.tools.contract import ToolContext, ToolRequest, ToolResult

__all__ = [
    "TOOL_BATCH_CLASSIFY",
    "TOOL_BATCH_REVIEW",
    "TOOL_FIND_PASSAGES",
    "TOOL_CHOICES",
    "TOOL_GENERATE_PAPER_SUMMARY",
    "TOOL_PAPER_OVERVIEW",
    "TOOL_PROJECT_INVENTORY",
    "TOOL_REGISTRY",
    "TOOL_SCAN_AND_LIST",
    "Tool",
    "ToolName",
    "ToolUse",
    "parse_tool",
    "run_tool",
]

# The tools that exist. The interface mirrors these values.
ToolName: TypeAlias = Literal[
    "project_inventory",
    "scan_and_list",
    "batch_classify",
    "batch_review",
    "find_passages",
    "generate_paper_summary",
    "paper_overview",
]
TOOL_PROJECT_INVENTORY: Final[ToolName] = "project_inventory"
TOOL_SCAN_AND_LIST: Final[ToolName] = "scan_and_list"
TOOL_BATCH_CLASSIFY: Final[ToolName] = "batch_classify"
TOOL_BATCH_REVIEW: Final[ToolName] = "batch_review"
TOOL_FIND_PASSAGES: Final[ToolName] = "find_passages"
TOOL_GENERATE_PAPER_SUMMARY: Final[ToolName] = "generate_paper_summary"
TOOL_PAPER_OVERVIEW: Final[ToolName] = "paper_overview"
TOOL_CHOICES: Final[tuple[ToolName, ...]] = (
    TOOL_PROJECT_INVENTORY,
    TOOL_SCAN_AND_LIST,
    TOOL_BATCH_CLASSIFY,
    TOOL_BATCH_REVIEW,
    TOOL_FIND_PASSAGES,
    TOOL_GENERATE_PAPER_SUMMARY,
    TOOL_PAPER_OVERVIEW,
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

    """

    name: ToolName
    summary: str
    parse_settings: Callable[[dict, str], object]
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]


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
    TOOL_GENERATE_PAPER_SUMMARY: Tool(
        TOOL_GENERATE_PAPER_SUMMARY,
        "Summarize one paper with a report template, check the structure, "
        "and save the Markdown to outputs/.",
        generate_paper_summary.parse_settings,
        generate_paper_summary.run,
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
        QAChat's services.

    Returns
    -------
    ToolResult
        The tool's answer, evidence, and summary.

    """

    return TOOL_REGISTRY[use.name].run(use.settings, request, context)
