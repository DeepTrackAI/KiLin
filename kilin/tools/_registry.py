"""Register KíLín's tools and resolve an intent's tool.

The registry is the controlled boundary between configuration and code. A
project type names a tool (``"tool": {"name": "scan_and_list", ...}``); the
name must be in ``TOOL_REGISTRY``, and the tool's own ``parse_settings``
checks the rest. Configuration never imports arbitrary Python paths, so a
project type file can choose among these tools but cannot add code.
The registry is constructed on first access, not during module import.
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

    Bundled identifiers and an open string type for tool names.

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

- ``registered_tools`` and ``tool_choices``

    Explicit accessors for the lazily constructed runtime registry.

- ``TOOL_REGISTRY``

    Deferred legacy export of the same registry.

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

from collections.abc import Mapping
from threading import RLock
from types import MappingProxyType
from typing import Final

from kilin.tools._contract import (
    PLUGIN_API_VERSION,
    Tool,
    ToolContext,
    ToolName,
    ToolRequest,
    ToolResult,
    ToolUse,
)

__all__ = [
    "registered_tools",
    "tool_choices",
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

# Bundled tools retain their identifiers during the migration.
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
_BUILTIN_TOOL_ORDER: Final[tuple[ToolName, ...]] = (
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


# Construct the registry only when a consumer explicitly needs it. The lock
# prevents concurrent first accesses from publishing different registries.
_REGISTRY_LOCK = RLock()
_registry: dict[ToolName, Tool] | None = None
_initializing = False


def _build_tool_registry() -> dict[ToolName, Tool]:
    """Build the existing bundled registry without import-time execution.

    Returns
    -------
    dict[ToolName, Tool]
        Tool definitions in their established registration order.

    """

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

    return {
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
            "Judge each item an earlier answer listed and keep those that "
            "match a follow-up.",
            filter_results.parse_settings,
            filter_results.run,
            follows_up=True,
        ),
        TOOL_FIND_FACT: Tool(
            TOOL_FIND_FACT,
            "Find a specific fact in one paper or every paper by reading "
            "every passage; the user can choose a quick answer instead.",
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
            "Write one report on all the documents of a project with a "
            "project template, reading every document, and save it to "
            "outputs/.",
            project_summary.parse_settings,
            project_summary.run,
            routing=project_summary.routing,
            workspace_resources=project_summary.workspace_resources,
        ),
    }


def _tool_registry() -> dict[ToolName, Tool]:
    """Resolve the single process-local registry on first use.

    Returns
    -------
    dict[ToolName, Tool]
        The registry, constructed once on successful access.

    Raises
    ------
    RuntimeError
        If construction recursively requests the registry.
    ValueError
        If a bundled tool declares an unsupported plugin API version.

    """

    global _registry, _initializing
    with _REGISTRY_LOCK:
        if _registry is None:
            if _initializing:
                raise RuntimeError("Recursive tool registry initialization.")
            _initializing = True
            try:
                entries = _build_tool_registry()
                for name, tool in entries.items():
                    if (
                        type(tool.api_version) is not int
                        or tool.api_version != PLUGIN_API_VERSION
                    ):
                        raise ValueError(
                            f"Tool {name!r} declares plugin API version "
                            f"{tool.api_version!r}; supported version is "
                            f"{PLUGIN_API_VERSION}."
                        )
                _registry = entries
            finally:
                _initializing = False
        return _registry


def registered_tools() -> Mapping[ToolName, Tool]:
    """Return a read-only view of the process-local tool registry.

    Returns
    -------
    Mapping[ToolName, Tool]
        Registered tools in their established registration order.

    """

    return MappingProxyType(_tool_registry())


def tool_choices() -> tuple[ToolName, ...]:
    """Return the available tool identifiers in their established order.

    Returns
    -------
    tuple[ToolName, ...]
        Registered identifiers, with the original bundled ordering.

    """

    entries = _tool_registry()
    return tuple(name for name in _BUILTIN_TOOL_ORDER if name in entries) + (
        tuple(name for name in entries if name not in _BUILTIN_TOOL_ORDER)
    )


def __getattr__(name: str) -> object:
    """Resolve legacy registry exports without initializing on import.

    Parameters
    ----------
    name: str
        Requested module attribute.

    Returns
    -------
    object
        The requested deferred export.

    Raises
    ------
    AttributeError
        If the module does not provide the requested attribute.

    """

    if name == "TOOL_REGISTRY":
        return _tool_registry()
    if name == "TOOL_CHOICES":
        return tool_choices()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    entries = _tool_registry()
    if not isinstance(name, str) or name not in entries:
        raise ValueError(
            f"Intent {where!r} tool name must be one of: "
            f"{', '.join(tool_choices())}."
        )

    settings = {key: value for key, value in data.items() if key != "name"}
    return ToolUse(
        name, entries[name].parse_settings(settings, f"{where}.tool")
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

    return _tool_registry()[use.name].run(use.settings, request, context)
