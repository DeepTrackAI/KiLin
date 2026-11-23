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
question sent to it to an earlier turn even when the router named none;
one with ``follows_up_when`` does so only for the questions it accepts.

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
  ``TOOL_PROJECT_SUMMARY``, ``TOOL_REFERENCE_CHECK``,
  ``TOOL_SHOW_SOURCE``, ``TOOL_SUGGEST_REFERENCE``,
  ``TOOL_WRITING_CHECK``, and ``TOOL_ASSISTANT_GUIDE``

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
from pathlib import Path
from typing import Final, Literal, TypeAlias

from kilin.tools import (
    assistant_guide,
    batch_classify,
    batch_review,
    filter_results,
    find_fact,
    find_passages,
    generate_paper_summary,
    paper_overview,
    project_inventory,
    project_summary,
    reference_check,
    scan_and_list,
    show_source,
    suggest_reference,
    writing_check,
)
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolRouting,
)
from kilin.tools._defaults import load_tool_defaults
from kilin.tools._model_calls import (
    ModelCallSettings,
    load_model_call_settings,
)
from kilin.workspace_resources import WorkspaceResource

__all__ = [
    "TOOL_ASSISTANT_GUIDE",
    "TOOL_BATCH_CLASSIFY",
    "TOOL_BATCH_REVIEW",
    "TOOL_CHOICES",
    "TOOL_FILTER_RESULTS",
    "TOOL_FIND_FACT",
    "TOOL_FIND_PASSAGES",
    "TOOL_GENERATE_PAPER_SUMMARY",
    "TOOL_PAPER_OVERVIEW",
    "TOOL_PROJECT_INVENTORY",
    "TOOL_PROJECT_SUMMARY",
    "TOOL_REFERENCE_CHECK",
    "TOOL_REGISTRY",
    "TOOL_SCAN_AND_LIST",
    "TOOL_SHOW_SOURCE",
    "TOOL_SUGGEST_REFERENCE",
    "TOOL_WRITING_CHECK",
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
    "suggest_reference",
    "writing_check",
    "project_summary",
    "assistant_guide",
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
TOOL_SUGGEST_REFERENCE: Final[ToolName] = "suggest_reference"
TOOL_WRITING_CHECK: Final[ToolName] = "writing_check"
TOOL_PROJECT_SUMMARY: Final[ToolName] = "project_summary"
TOOL_ASSISTANT_GUIDE: Final[ToolName] = "assistant_guide"
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
    TOOL_SUGGEST_REFERENCE,
    TOOL_WRITING_CHECK,
    TOOL_PROJECT_SUMMARY,
    TOOL_ASSISTANT_GUIDE,
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
    follows_up_when: Callable[[str], bool] | None
        For a tool that works on an earlier answer only for some
        questions: True for those questions, which then relate to an
        earlier turn like ``follows_up``. They are not ruled out for
        naming a file.
    routing: Callable[[object, Path, tuple[object, ...]], ToolRouting | None]
        Optional ``routing(settings, project_root, others)``: cues and a
        router description from the project's own files, added to the
        intent before routing (``with_project_routing``); ``others`` are
        the settings of the project type's other intents using the same
        tool, whose cases the routing can leave to them.
    model_calls: ModelCallSettings
        Tool-owned policy for internal answering-model calls.
    workspace_resources: Callable | None
        Optional declaration of packaged resources for a new project.

    """

    name: ToolName
    summary: str
    parse_settings: Callable[[dict, str], object]
    run: Callable[[object, ToolRequest, ToolContext], ToolResult]
    follows_up: bool = False
    follows_up_when: Callable[[str], bool] | None = None
    routing: (
        Callable[[object, Path, tuple[object, ...]], ToolRouting | None]
        | None
    ) = None
    workspace_resources: (
        Callable[[object], tuple[WorkspaceResource, ...]] | None
    ) = None

    @property
    def model_calls(self) -> ModelCallSettings:
        """Return the registered tool's validated internal thinking policy.

        Returns
        -------
        ModelCallSettings
            Cached policy from this tool's own packaged defaults.

        """

        return load_model_call_settings(
            Path(__file__).parent / self.name / "defaults.json"
        )

    @property
    def internal_defaults(self) -> dict[str, object]:
        """Return the tool's packaged defaults other than ``SETTINGS``.

        These sections, such as ``MODEL_CALLS`` or ``ESTIMATES``, belong to
        the tool: project types and users cannot change them.

        Returns
        -------
        dict[str, object]
            The sections by name, empty when the tool has no defaults file.

        """

        path = Path(__file__).parent / self.name / "defaults.json"
        if not path.is_file():
            return {}
        defaults = load_tool_defaults(path)
        defaults.pop("SETTINGS", None)
        return defaults


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
        "find where each reference is cited, optionally check claims, "
        "and give offline or explicitly authorized online links for "
        "missing works; save the report to outputs/.",
        reference_check.parse_settings,
        reference_check.run,
    ),
    TOOL_SUGGEST_REFERENCE: Tool(
        TOOL_SUGGEST_REFERENCE,
        "Find the papers in the project that could be cited for a "
        "statement, judging each with a quote from its own passages.",
        suggest_reference.parse_settings,
        suggest_reference.run,
    ),
    TOOL_WRITING_CHECK: Tool(
        TOOL_WRITING_CHECK,
        "Check the grammar and style of one document or part of it, "
        "paragraph by paragraph, and suggest changes without making them; "
        "save the report to outputs/.",
        writing_check.parse_settings,
        writing_check.run,
        follows_up_when=writing_check.asks_rewrite,
    ),
    TOOL_ASSISTANT_GUIDE: Tool(
        TOOL_ASSISTANT_GUIDE,
        "Answer what KíLín can do with one ready-to-send example question "
        "per kind of request, about the project's own files.",
        assistant_guide.parse_settings,
        assistant_guide.run,
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
        routing=generate_paper_summary.routing,
        workspace_resources=generate_paper_summary.workspace_resources,
    ),
    TOOL_PROJECT_SUMMARY: Tool(
        TOOL_PROJECT_SUMMARY,
        "Write one report on all the documents of a project with a project "
        "template, reading every document, and save it to outputs/.",
        project_summary.parse_settings,
        project_summary.run,
        routing=project_summary.routing,
        workspace_resources=project_summary.workspace_resources,
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
