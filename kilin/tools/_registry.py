"""Register KíLín's tools and resolve an intent's tool.

The registry is the controlled boundary between configuration and code. A
project type names a tool (``"tool": {"name": "scan_and_list", ...}``); the
name must be in ``TOOL_REGISTRY``, and the tool's own ``parse_settings``
checks the rest. Configuration never imports arbitrary Python paths, so a
project type file can choose among these tools but cannot add code.
The registry is constructed on first access, not during module import.
A tool may also declare resources to install in newly created projects;
the core resolves and copies them, not the GUI.

Each bundled tool is a package exposing a ``PLUGIN`` descriptor with
its own settings parser and runner (see ``kilin.tools._contract``).
Discovery binds the descriptor to the package actually loaded. A tool
marked ``follows_up`` works on an earlier answer, so routing relates a
question sent to it to an earlier turn even when the router named none;
one with ``follows_up_when`` does so only for the questions it accepts.

Key Features
------------

- **Names**

    Bundled identifiers and an open string type for tool names.

- **Registry**

    Discover each public package, validate its descriptor and bind its
    observed package root before publishing the registry.

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
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from importlib import import_module
import json
from pathlib import Path
import re
from threading import RLock
from types import MappingProxyType, ModuleType
from typing import Final

from kilin.tools._defaults import load_tool_defaults, require_section
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


# Preserve the historical mapping iteration order separately from chooser
# order. Neither preference list controls which packages are discovered.
_BUILTIN_REGISTRY_ORDER: Final[tuple[ToolName, ...]] = (
    TOOL_PROJECT_INVENTORY,
    TOOL_SCAN_AND_LIST,
    TOOL_BATCH_CLASSIFY,
    TOOL_BATCH_REVIEW,
    TOOL_FILTER_RESULTS,
    TOOL_FIND_FACT,
    TOOL_FIND_PASSAGES,
    TOOL_PAPER_OVERVIEW,
    TOOL_REFERENCE_CHECK,
    TOOL_SUGGEST_REFERENCE,
    TOOL_WRITING_CHECK,
    TOOL_ASSISTANT_GUIDE,
    TOOL_SHOW_SOURCE,
    TOOL_GENERATE_PAPER_SUMMARY,
    TOOL_PROJECT_SUMMARY,
)


# Construct the registry only when a consumer explicitly needs it. The lock
# prevents concurrent first accesses from publishing different registries.
_REGISTRY_LOCK = RLock()
_registry: dict[ToolName, Tool] | None = None
_initializing = False


@dataclass(frozen=True, slots=True)
class _RegisteredTool(Tool):
    """Bind an authoring descriptor to a loader-observed package.

    Attributes
    ----------
    _package_root: pathlib.Path
        Resolved directory of the imported package. It is not supplied
        by the descriptor and is retained when replacing test callbacks.

    """

    _package_root: Path = field(kw_only=True, repr=False)

    @property
    def resource_root(self) -> Path:
        """Return the observed root of the registered package.

        Returns
        -------
        pathlib.Path
            Absolute directory for owned configuration and assets.

        """

        return self._package_root


def _validate_descriptor(name: str, tool: object) -> None:
    """Validate the identity and declared contract of a plugin.

    Parameters
    ----------
    name: str
        Expected tool identifier, from the package directory.
    tool: object
        Candidate descriptor exported by the package.

    Raises
    ------
    ValueError
        If its type, identity, version or callable declarations are invalid.

    """

    if not isinstance(tool, Tool):
        raise ValueError(f"Tool {name!r} must export a Tool as PLUGIN.")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or tool.name != name:
        raise ValueError(f"Tool {name!r} has invalid descriptor identity.")
    if (
        type(tool.api_version) is not int
        or tool.api_version != PLUGIN_API_VERSION
    ):
        raise ValueError(
            f"Tool {name!r} declares plugin API version "
            f"{tool.api_version!r}; supported version is "
            f"{PLUGIN_API_VERSION}."
        )
    for key in ("summary", "version"):
        value = getattr(tool, key)
        if (
            not isinstance(value, str) or not value.strip()
            or value != value.strip()
        ):
            raise ValueError(f"Tool {name!r} has invalid {key}.")
    if type(tool.follows_up) is not bool:
        raise ValueError(f"Tool {name!r} has invalid follows_up.")
    for key in ("parse_settings", "run"):
        if not callable(getattr(tool, key)):
            raise ValueError(f"Tool {name!r} requires callable {key}.")
    for key in ("follows_up_when", "routing", "workspace_resources"):
        value = getattr(tool, key)
        if value is not None and not callable(value):
            raise ValueError(f"Tool {name!r} has invalid {key}.")


def _register_tool(module: ModuleType, directory: Path) -> Tool:
    """Bind and validate a descriptor from an already imported package.

    Parameters
    ----------
    module: types.ModuleType
        Imported trusted Python package exposing ``PLUGIN``.
    directory: pathlib.Path
        Candidate package directory selected by the loader, not the plugin.

    Returns
    -------
    Tool
        A descriptor bound to its actual resource directory.

    Raises
    ------
    ValueError
        If the package origin, descriptor or declared defaults are invalid.

    """

    directory = directory.resolve()
    location = getattr(module, "__file__", None)
    if (
        not location or Path(location).resolve() != directory / "__init__.py"
    ):
        raise ValueError(f"Tool {directory.name!r} has unexpected origin.")
    name = directory.name
    descriptor = getattr(module, "PLUGIN", None)
    _validate_descriptor(name, descriptor)
    # Copy only authoring fields, never ownership supplied by a descriptor.
    tool = _RegisteredTool(
        **{
            item.name: getattr(descriptor, item.name) for item in fields(Tool)
        },
        _package_root=directory,
    )
    # Validate only declared defaults. A tool with required settings need
    # not accept an empty mapping before a project type configures it.
    path = directory / "defaults.json"
    if path.exists():
        defaults = load_tool_defaults(path)
        if "SETTINGS" in defaults:
            data = require_section(defaults, "SETTINGS", name)
            normalized = {key.lower(): value for key, value in data.items()}
            if len(normalized) != len(data):
                raise ValueError(f"Tool {name!r} has duplicate settings.")
            settings = tool.parse_settings(normalized, f"{name}.defaults")
            if not is_dataclass(settings) or isinstance(settings, type):
                raise ValueError(
                    f"Tool {name!r} settings must be a dataclass."
                )
            try:
                json.dumps(asdict(settings), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Tool {name!r} settings must be JSON-compatible."
                ) from error
    # Validate supplied model policies at registration, without requiring
    # a policy or help file for deterministic, settings-free plugins.
    _ = tool.model_calls
    return tool


def _build_tool_registry() -> dict[ToolName, Tool]:
    """Discover the bundled tool packages and register their descriptors.

    Returns
    -------
    dict[ToolName, Tool]
        Tool definitions in their established registration order, followed
        by newly distributed packages in alphabetical order.

    Raises
    ------
    ValueError
        If a public bundled package cannot be registered.

    """

    entries = {}
    for directory in sorted(Path(__file__).parent.iterdir()):
        name = directory.name
        if name.startswith("_") or not (directory / "__init__.py").is_file():
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid bundled tool package: {name!r}.")
        module = import_module(f"kilin.tools.{name}")
        entries[name] = _register_tool(module, directory)
    order = (*_BUILTIN_REGISTRY_ORDER, *entries)
    return {name: entries[name] for name in order if name in entries}


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
                    _validate_descriptor(name, tool)
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
