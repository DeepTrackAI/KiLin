"""List and change the tool settings of one KíLín project.

A project type's intents name the tools they use, with settings in
``kilin/project_types/<type>.json``; each tool's ``defaults.json`` holds the
values it starts from. A project can change a tool's settings for all of its
intents. The changes are saved in the project's ``profile.ini``, one
``[tool.<name>]`` section per tool, holding only values that differ from the
project type's. Every change is validated by the tool's own
``parse_settings`` before it is saved or used.

Key Features
------------

- **Built from the tools**

    The settings shown for a tool are exactly those its ``defaults.json``
    lists under ``SETTINGS``, so a new tool needs no extra code here.

- **Per project**

    Saved changes apply to every intent using the tool in this project only.

- **Validated twice**

    Changes are checked when saved and again before a tool runs.

Module Structure
----------------

- ``project_tool_settings``

    Describe each tool the project type can use, with its current settings.

- ``save_tool_settings``

    Validate and save one tool's settings for a project.

- ``with_project_settings``

    Return an intent's tool with the project's saved settings applied.

Examples
--------
>>> from kilin.tool_settings import project_tool_settings
>>> tools = project_tool_settings("/path/to/project")
>>> [tool["name"] for tool in tools]  # doctest: +SKIP
['find_fact', 'batch_review']

"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
from typing import Final

from kilin.atomic_writes import atomic_write_text
from kilin.project import open_project
from kilin.project_type import ProjectTypeId, project_type
from kilin.tools import ToolUse
from kilin.tools._registry import _parse_settings, registered_tools
from kilin.trace_logging import NullTraceLogger, TraceSink

__all__ = [
    "project_tool_settings",
    "save_tool_settings",
    "with_project_settings",
]

# The profile.ini section of one tool's changes, e.g. [tool.find_fact].
TOOL_SECTION_PREFIX: Final[str] = "tool."


def project_tool_settings(
    directory: str | Path,
) -> list[dict[str, object]]:
    """Describe each tool the project type can use, with its settings.

    Parameters
    ----------
    directory: str | pathlib.Path
        Project folder containing ``profile.ini``.

    Returns
    -------
    list[dict[str, object]]
        One record per tool, in the order intents first use it:
        ``name``; ``summary``; ``intents`` using it; ``settings``, the
        values the project can change, with saved changes applied;
        ``type_settings``, the project type's values for them; ``per_intent``,
        settings each intent sets differently, such as a template, which
        are not changed here.

    Raises
    ------
    ValueError
        If the profile or a saved tool setting is invalid.

    """

    project = open_project(directory)
    saved = _saved_settings(project.root)
    tools = []
    for name, uses in _tool_uses(project.profile.type_id).items():
        _, type_settings, per_intent = _split_settings(uses)
        tools.append({
            "name": name,
            "summary": registered_tools()[name].summary,
            "intents": list(uses),
            "settings": {**type_settings, **saved.get(name, {})},
            "type_settings": type_settings,
            "per_intent": per_intent,
        })
    return tools


def save_tool_settings(
    directory: str | Path,
    name: str,
    settings: dict[str, object],
    *,
    trace: TraceSink | None = None,
) -> list[dict[str, object]]:
    """Validate and save one tool's settings for a project.

    Parameters
    ----------
    directory: str | pathlib.Path
        Project folder containing ``profile.ini``.
    name: str
        Tool the project type uses.
    settings: dict[str, object]
        New values for any of the tool's changeable settings. A value equal
        to the project type's is not saved, so passing the project type's
        values resets the tool.
    trace: TraceSink | None, optional
        Records the saved change. Defaults to ``None``.

    Returns
    -------
    list[dict[str, object]]
        The project's tools as ``project_tool_settings`` returns them.

    Raises
    ------
    ValueError
        If the project type does not use the tool, a setting cannot be
        changed here, or a value is rejected by the tool.
    OSError
        If ``profile.ini`` cannot be read or written.

    """

    active_trace = trace if trace is not None else NullTraceLogger()
    project = open_project(directory)
    uses = _tool_uses(project.profile.type_id).get(name)
    if uses is None:
        raise ValueError(f"This project type does not use the tool {name}.")
    _, type_settings, _ = _split_settings(uses)
    fixed = sorted(set(settings) - set(type_settings))
    if fixed:
        raise ValueError(
            f"{name}: {', '.join(fixed)} cannot be changed for the project."
        )
    values = _json_values(settings)
    for use in uses.values():
        _parse_settings(
            registered_tools()[name],
            {**_json_values(asdict(use.settings)), **values}, name
        )
    changed = {
        key: value
        for key, value in values.items()
        if value != type_settings[key]
    }
    path = project.root / "profile.ini"
    # JSON values may contain "%", so no interpolation.
    parser = ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    section = f"{TOOL_SECTION_PREFIX}{name}"
    before = _saved_settings(project.root).get(name, {})
    kept = {
        key: value
        for key, value in before.items()
        if key not in values
    }
    parser.remove_section(section)
    if kept or changed:
        parser.add_section(section)
        for key, value in {**kept, **changed}.items():
            parser.set(section, key, json.dumps(value))
    stream = StringIO()
    parser.write(stream)
    atomic_write_text(path, stream.getvalue())
    active_trace.emit(
        "project.tool_settings.saved",
        component="project",
        data={
            "tool": name, "changed": {**kept, **changed},
            "requested": values, "before": before,
            "effective": {**type_settings, **kept, **changed},
        },
    )
    return project_tool_settings(project.root)


def with_project_settings(root: Path, use: ToolUse) -> ToolUse:
    """Return an intent's tool with the project's saved settings applied.

    Parameters
    ----------
    root: pathlib.Path
        Project folder containing ``profile.ini``.
    use: ToolUse
        The intent's tool with the project type's settings.

    Returns
    -------
    ToolUse
        ``use`` itself when the project changed nothing, otherwise the tool
        with its settings validated again by the tool.

    Raises
    ------
    ValueError
        If a saved setting is no longer valid for the tool.

    """

    saved = _saved_settings(root).get(use.name)
    if not saved:
        return use
    return ToolUse(
        use.name,
        _parse_settings(
            registered_tools()[use.name],
            {**_json_values(asdict(use.settings)), **saved},
            f"profile.ini [{TOOL_SECTION_PREFIX}{use.name}]",
        ),
    )


def _tool_uses(type_id: ProjectTypeId) -> dict[str, dict[str, ToolUse]]:
    """Group a project type's intents by the tool they use.

    Parameters
    ----------
    type_id: ProjectTypeId
        Registered project type.

    Returns
    -------
    dict[str, dict[str, ToolUse]]
        Tool name to its intents' tools by intent ID, in intent order.

    """

    uses: dict[str, dict[str, ToolUse]] = {}
    for intent_id, intent in project_type(type_id).intents.items():
        if intent.tool is not None:
            uses.setdefault(intent.tool.name, {})[intent_id] = intent.tool
    return uses


def _split_settings(
    uses: dict[str, ToolUse],
) -> tuple[list[str], dict[str, object], dict[str, dict[str, object]]]:
    """Separate settings all intents share from those each intent sets.

    Parameters
    ----------
    uses: dict[str, ToolUse]
        One tool's uses by intent ID.

    Returns
    -------
    tuple[list[str], dict[str, object], dict[str, dict[str, object]]]
        The shared setting names, their project-type values, and the other
        settings by intent ID.

    """

    values = {
        intent_id: _json_values(asdict(use.settings))
        for intent_id, use in uses.items()
    }
    first = next(iter(values.values()))
    shared = [
        key for key in first
        if all(settings[key] == first[key] for settings in values.values())
    ]
    per_intent = {
        intent_id: {
            key: value for key, value in settings.items()
            if key not in shared
        }
        for intent_id, settings in values.items()
    }
    return (
        shared,
        {key: first[key] for key in shared},
        {key: value for key, value in per_intent.items() if value},
    )


def _saved_settings(root: Path) -> dict[str, dict[str, object]]:
    """Read the tool settings a project saved in ``profile.ini``.

    Parameters
    ----------
    root: pathlib.Path
        Project folder containing ``profile.ini``.

    Returns
    -------
    dict[str, dict[str, object]]
        Changed values by tool name and setting name.

    Raises
    ------
    ValueError
        If a saved value is not valid JSON.

    """

    parser = ConfigParser(interpolation=None)
    parser.read(Path(root) / "profile.ini", encoding="utf-8")
    saved: dict[str, dict[str, object]] = {}
    for section in parser.sections():
        if not section.startswith(TOOL_SECTION_PREFIX):
            continue
        name = section.removeprefix(TOOL_SECTION_PREFIX)
        try:
            saved[name] = {
                key: json.loads(value)
                for key, value in parser.items(section)
            }
        except json.JSONDecodeError as error:
            raise ValueError(
                f"profile.ini [{section}] has an unreadable value."
            ) from error
    return saved


def _json_values(settings: dict[str, object]) -> dict[str, object]:
    """Return settings as JSON would store them, with lists for tuples.

    Parameters
    ----------
    settings: dict[str, object]
        Setting names and values.

    Returns
    -------
    dict[str, object]
        An independent copy comparable with values read from JSON.

    """

    return json.loads(json.dumps(settings))

