"""Read explanatory metadata for public KíLín configuration settings.

This module makes packaged setting descriptions available to the core,
command-line consumers, and user interfaces without exposing any GUI
implementation detail. Shared descriptions live in
``kilin/settings_info.json``. Each registered tool supplies its own
``settings_info.json`` beside its ``defaults.json`` file.

Only titles and explanatory text are stored here: defaults, accepted types,
precedence, and runtime validation remain with their existing owners.
Loading the metadata never initializes a project or model, and failures do
not affect answering or tool execution unless a caller explicitly requests
settings information.

Key Features
------------

- **Independent core API**

    Retrieve settings information without creating a workspace or GUI.

- **Tool-owned descriptions**

    Each tool describes its own public ``SETTINGS`` keys.

- **Validated, read-once resources**

    Reject malformed metadata and return defensive copies of cached data.

Module Structure
----------------

- ``get_settings_info``

    Return the combined core and tool catalog using canonical identifiers.

- ``get_tool_settings_info``

    Retrieve the descriptive metadata owned by one registered tool.

- ``get_setting_info``

    Look up an individual setting by its canonical identifier.

Examples
--------
>>> import kilin
>>> info = kilin.get_setting_info("project.profile.temperature")
>>> info["title"]
'Temperature'

"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path

__all__ = [
    "get_setting_info",
    "get_settings_info",
    "get_tool_settings_info",
]

_FORMAT = "settings-info-v1"
_FIELDS = frozenset(("title", "summary", "description"))
_MAX_BYTES = 262_144


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate keys in a packaged metadata document.

    Parameters
    ----------
    pairs: list[tuple[str, object]]
        Keys and values supplied by the JSON parser.

    Returns
    -------
    dict[str, object]
        The parsed mapping with each key occurring exactly once.

    Raises
    ------
    ValueError
        If a JSON object repeats a key.

    """

    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"Duplicate settings metadata key: {key}.")
        parsed[key] = value
    return parsed


def _load_resource(package: str | Path, filename: str) -> dict[str, object]:
    """Parse and strictly validate one packaged settings-information file.

    Parameters
    ----------
    package: str | pathlib.Path
        Installed core package name or observed tool resource directory.
    filename: str
        Name of the JSON resource within that package.

    Returns
    -------
    dict[str, object]
        Validated mapping of canonical identifiers to descriptive records.

    Raises
    ------
    ValueError
        If the file is malformed or its records are incomplete.

    """

    location = (
        package / filename if isinstance(package, Path)
        else files(package).joinpath(filename)
    )
    raw = location.read_bytes()
    if len(raw) > _MAX_BYTES:
        raise ValueError(f"Settings metadata in {package} is too large.")
    try:
        data = json.loads(raw.decode("utf-8"),
                          object_pairs_hook=_unique_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid settings metadata in {package}.") \
            from error
    if not isinstance(data, dict) or set(data) != {"format", "settings"}:
        raise ValueError(f"Invalid settings metadata document in {package}.")
    if data["format"] != _FORMAT:
        raise ValueError(f"Unsupported settings metadata format in {package}.")
    settings = data["settings"]
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f"Settings metadata in {package} is empty.")
    for key, info in settings.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Invalid setting identifier in {package}.")
        if not isinstance(info, dict) or set(info) != _FIELDS:
            raise ValueError(f"Incomplete metadata for {package}:{key}.")
        for field, value in info.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid {field} for setting {package}:{key}."
                )
    return settings


@lru_cache(maxsize=1)
def _core_info() -> dict[str, object]:
    """Read the shared descriptions once for independent use by clients.

    Returns
    -------
    dict[str, object]
        Cached, validated shared setting descriptions.

    """

    return _load_resource("kilin", "settings_info.json")


@lru_cache(maxsize=None)
def _read_tool_info(name: str, package: Path) -> dict[str, object]:
    """Validate optional settings help from a specific package location.

    Parameters
    ----------
    name: str
        Candidate tool identifier.
    package: pathlib.Path
        Observed package directory, independent of registry state.

    Returns
    -------
    dict[str, object]
        Validated descriptions, or an empty mapping when absent.

    """

    if not (package / "settings_info.json").exists():
        return {}
    settings = _load_resource(package, "settings_info.json")
    path = package / "defaults.json"
    if not path.exists():
        return settings
    defaults = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs
    )
    if not isinstance(defaults, dict):
        raise ValueError(f"Invalid defaults for tool {name}.")
    if "SETTINGS" not in defaults:
        return settings
    if not isinstance(defaults["SETTINGS"], dict):
        raise ValueError(f"Invalid defaults for tool {name}.")
    expected = {key.lower() for key in defaults["SETTINGS"]}
    if set(settings) != expected:
        raise ValueError(
            f"Tool metadata keys disagree with {name} defaults: "
            f"missing {sorted(expected - settings)}, "
            f"unexpected {sorted(set(settings) - expected)}."
        )
    return settings



@lru_cache(maxsize=None)
def _tool_info(name: str) -> dict[str, object]:
    """Resolve metadata from the registered package.

    Parameters
    ----------
    name: str
        Registered tool identifier.

    Returns
    -------
    dict[str, object]
        Descriptions, empty for absent or rejected optional user help.

    """

    from kilin.tools._registry import registered_tools

    tools = registered_tools()
    if name not in tools:
        raise ValueError(f"Unknown KíLín tool: {name}.")
    tool = tools[name]
    info = getattr(tool, "_info", None)
    if info is not None and info.origin == "user" and info.warnings:
        return {}
    return _read_tool_info(name, tool.resource_root)


def get_tool_settings_info(name: str) -> dict[str, dict[str, str]]:
    """Return an independent copy of one tool's settings descriptions.

    Parameters
    ----------
    name: str
        Registered tool identifier.

    Returns
    -------
    dict[str, dict[str, str]]
        Metadata indexed by lowercase keys from the tool's ``SETTINGS``.

    Raises
    ------
    ValueError
        If the tool or its packaged metadata is invalid.

    """

    return deepcopy(_tool_info(name))


@lru_cache(maxsize=1)
def _combined_info() -> dict[str, object]:
    """Combine shared settings and registered tools without GUI imports.

    Returns
    -------
    dict[str, object]
        Cached complete settings-information document.

    """

    from kilin.tools._registry import tool_choices

    settings = dict(_core_info())
    for name in tool_choices():
        for key, info in _tool_info(name).items():
            settings[f"tools.{name}.{key}"] = info
    return {"format": _FORMAT, "settings": settings}


def get_settings_info() -> dict[str, object]:
    """Return descriptive metadata for all public configurable settings.

    Returns
    -------
    dict[str, object]
        Versioned catalog of shared and tool-owned setting descriptions.
        Returned data is independent of the cached JSON representation.

    Raises
    ------
    ValueError
        If a requested packaged metadata resource is invalid.

    Notes
    -----
    Calling this function does not affect configuration, models, retrieval,
    or tools. Clients may display the metadata or generate help text from it.

    """

    return deepcopy(_combined_info())


def get_setting_info(key: str) -> dict[str, str] | None:
    """Look up descriptive metadata by a stable dotted setting identifier.

    Parameters
    ----------
    key: str
        Core setting identifier, or ``tools.<tool>.<setting>``.

    Returns
    -------
    dict[str, str] | None
        Independent descriptive record, or None for an unknown identifier.

    """

    if not isinstance(key, str):
        raise TypeError("Setting identifier must be a string.")
    if key.startswith("tools."):
        parts = key.split(".")
        if len(parts) != 3:
            return None
        from kilin.tools._registry import tool_choices

        if parts[1] not in tool_choices():
            return None
        value = _tool_info(parts[1]).get(parts[2])
    else:
        value = _core_info().get(key)
    return deepcopy(value) if value is not None else None
