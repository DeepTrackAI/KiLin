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


def _load_resource(package: str, filename: str) -> dict[str, object]:
    """Parse and strictly validate one packaged settings-information file.

    Parameters
    ----------
    package: str
        Installed Python package owning the metadata.
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

    location = files(package).joinpath(filename)
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
def _tool_info(name: str) -> dict[str, object]:
    """Load one tool's descriptions and check its declared settings.

    Parameters
    ----------
    name: str
        Registered tool name.

    Returns
    -------
    dict[str, object]
        Cached metadata keyed by normalized tool setting names.

    Raises
    ------
    ValueError
        If the tool is unknown or its metadata disagrees with defaults.

    """

    from kilin.tools._registry import TOOL_CHOICES

    if name not in TOOL_CHOICES:
        raise ValueError(f"Unknown KíLín tool: {name}.")
    package = f"kilin.tools.{name}"
    settings = _load_resource(package, "settings_info.json")
    raw = files(package).joinpath("defaults.json").read_text(
        encoding="utf-8"
    )
    defaults = json.loads(raw, object_pairs_hook=_unique_pairs)
    if not isinstance(defaults, dict) or not isinstance(
        defaults.get("SETTINGS"), dict
    ):
        raise ValueError(f"Invalid defaults for tool {name}.")
    expected = {key.lower() for key in defaults["SETTINGS"]}
    if set(settings) != expected:
        raise ValueError(
            f"Tool metadata keys disagree with {name} defaults: "
            f"missing {sorted(expected - settings)}, "
            f"unexpected {sorted(set(settings) - expected)}."
        )
    return settings


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

    from kilin.tools._registry import TOOL_CHOICES

    settings = dict(_core_info())
    for name in TOOL_CHOICES:
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
        from kilin.tools._registry import TOOL_CHOICES

        if parts[1] not in TOOL_CHOICES:
            return None
        value = _tool_info(parts[1]).get(parts[2])
    else:
        value = _core_info().get(key)
    return deepcopy(value) if value is not None else None
