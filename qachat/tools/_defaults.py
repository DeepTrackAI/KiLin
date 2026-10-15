"""Read and validate defaults owned by individual QAChat tools.

Each tool package can contain a ``defaults.json`` next to its Python
implementation. This private reader keeps the package's configuration
independent of the core ``qachat_defaults.json`` and returns defensive
copies so one tool cannot change another tool's defaults at runtime.

Key Features
------------

- **Independent files**

    Read each tool's configuration from the path it supplies.

- **Fail-fast validation**

    Reject missing or unexpected keys, invalid types, and values outside
    the tool's declared ranges.

- **Read-once semantics**

    Cache parsed JSON while giving each consumer an independent copy.

Module Structure
----------------

- ``load_tool_defaults``

    Return one tool's packaged JSON configuration.

- ``require_keys`` and ``require_section``

    Validate complete objects and retrieve nested configuration groups.

- ``require_int``, ``require_number``, ``require_string``,
  ``require_words``, and ``require_int_pair``

    Validate typed settings with clear file and key context.

Examples
--------
>>> from pathlib import Path
>>> from qachat.tools._defaults import require_int
>>> require_int({'size': 3}, 'size', 'example', 1, 10)
3

"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path
from typing import Mapping

__all__ = [
    "load_tool_defaults",
    "require_int",
    "require_int_pair",
    "require_keys",
    "require_number",
    "require_section",
    "require_string",
    "require_words",
]


@lru_cache(maxsize=None)
def _read_json(path: Path) -> dict[str, object]:
    """Read one packaged JSON file and verify its root object.

    Parameters
    ----------
    path: pathlib.Path
        Absolute configuration filename.

    Returns
    -------
    dict[str, object]
        Cached parsed defaults.

    Raises
    ------
    ValueError
        If the file contains invalid JSON or is not an object.
    OSError
        If the file cannot be read.

    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error.msg}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    if any(not isinstance(key, str) for key in data):
        raise ValueError(f"{path} must have string section names.")
    return data


def load_tool_defaults(path: Path) -> dict[str, object]:
    """Return an independent copy of a tool's packaged defaults.

    Parameters
    ----------
    path: pathlib.Path
        Path to the tool-owned ``defaults.json`` file.

    Returns
    -------
    dict[str, object]
        Parsed defaults that callers may modify independently.

    """

    return deepcopy(_read_json(Path(path).resolve()))


def require_keys(
    settings: Mapping[str, object],
    expected: set[str],
    where: str,
) -> None:
    """Reject missing or unknown settings in a configuration mapping.

    Parameters
    ----------
    settings: Mapping[str, object]
        One loaded object.
    expected: set[str]
        Exactly the keys permitted by its schema.
    where: str
        Location displayed in configuration errors.

    Raises
    ------
    ValueError
        If the object has missing or unknown keys.

    """

    actual = set(settings)
    missing, unknown = sorted(expected - actual), sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{where}: missing keys {missing}; unknown keys {unknown}."
        )


def require_section(
    settings: Mapping[str, object], name: str, where: str
) -> dict[str, object]:
    """Return one required nested configuration object.

    Parameters
    ----------
    settings: Mapping[str, object]
        Parent configuration object.
    name: str
        Required section name.
    where: str
        Location displayed in error messages.

    Returns
    -------
    dict[str, object]
        Validated shallow copy of the section.

    Raises
    ------
    ValueError
        If the named section is not an object.

    """

    value = settings.get(name)
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{where}.{name} must be an object.")
    return dict(value)


def require_int(
    settings: Mapping[str, object],
    name: str,
    where: str,
    minimum: int,
    maximum: int,
) -> int:
    """Read a strictly typed integer within an inclusive range.

    Parameters
    ----------
    settings: Mapping[str, object]
        Configuration mapping.
    name: str
        Integer setting name.
    where: str
        Location displayed in errors.
    minimum: int
        Lowest allowed value.
    maximum: int
        Highest allowed value.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    ValueError
        If the value is absent, boolean, or outside the range.

    """

    value = settings.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{where}.{name} must be an integer from "
            f"{minimum} to {maximum}."
        )
    return value


def require_number(
    settings: Mapping[str, object],
    name: str,
    where: str,
    minimum: float,
    maximum: float,
) -> float:
    """Read a finite JSON number within an inclusive range.

    Parameters
    ----------
    settings: Mapping[str, object]
        Configuration mapping.
    name: str
        Numeric setting name.
    where: str
        Location displayed in errors.
    minimum: float
        Lowest allowed value.
    maximum: float
        Highest allowed value.

    Returns
    -------
    float
        Validated numerical setting.

    Raises
    ------
    ValueError
        If the number is absent, nonfinite, or outside the range.

    """

    value = settings.get(name)
    if (
        type(value) not in (float, int)
        or not isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"{where}.{name} must be a finite number from "
            f"{minimum} to {maximum}."
        )
    return float(value)


def require_string(
    settings: Mapping[str, object], name: str, where: str
) -> str:
    """Read a nonempty string without surrounding whitespace.

    Parameters
    ----------
    settings: Mapping[str, object]
        Configuration mapping.
    name: str
        String setting name.
    where: str
        Location displayed in errors.

    Returns
    -------
    str
        Validated string setting.

    Raises
    ------
    ValueError
        If the string is absent, blank, or padded with whitespace.

    """

    value = settings.get(name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{where}.{name} must be a nonempty string.")
    return value


def require_words(
    settings: Mapping[str, object], name: str, where: str
) -> tuple[str, ...]:
    """Read a nonempty sequence of distinct lowercase terms.

    Parameters
    ----------
    settings: Mapping[str, object]
        Configuration mapping.
    name: str
        Vocabulary setting name.
    where: str
        Location displayed in errors.

    Returns
    -------
    tuple[str, ...]
        Validated lowercase words, retaining configured order.

    Raises
    ------
    ValueError
        If the vocabulary is empty, malformed, or duplicated.

    """

    value = settings.get(name)
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or item.lower() != item
            or any(char.isspace() for char in item)
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{where}.{name} must be distinct lowercase words."
        )
    return tuple(value)


def require_int_pair(
    settings: Mapping[str, object], name: str, where: str
) -> tuple[int, int]:
    """Read two positive increasing integer bounds.

    Parameters
    ----------
    settings: Mapping[str, object]
        Configuration mapping.
    name: str
        Pair setting name.
    where: str
        Location displayed in errors.

    Returns
    -------
    tuple[int, int]
        Inclusive lower and upper bounds.

    Raises
    ------
    ValueError
        If the value is not an ordered positive integer pair.

    """

    value = settings.get(name)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
        or value[0] > value[1]
    ):
        raise ValueError(f"{where}.{name} must be two positive bounds.")
    return value[0], value[1]
