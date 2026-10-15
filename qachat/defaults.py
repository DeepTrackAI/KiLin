"""Load and validate QAChat's packaged configuration defaults.

The packaged ``qachat_defaults.json`` is read once per process. Its sections
provide starting values for the modules that own each configurable policy.
Consumers validate domain-specific rules before exposing their constants.
Returned sections are independent copies, so callers cannot mutate the
shared, cached configuration.

Key Features
------------

- **One packaged configuration**

    Resolve the resource relative to this module and parse it once.

- **Explicit errors**

    Report missing sections and invalid types with qualified setting names.

- **Independent consumers**

    Return a defensive copy of each section without importing higher-level
    QAChat modules or optional dependencies.

Module Structure
----------------

- ``load_defaults_section``

    Return one mapping of configured defaults, with its keys validated.

- ``require_default_int``

    Read a positive integer, explicitly rejecting booleans.

- ``require_default_nonnegative_int``

    Read a nonnegative integer, excluding booleans.

- ``require_default_bool``

    Read an exact JSON boolean.

- ``require_default_number``

    Read a finite number with an optional zero-inclusive bound.

- ``require_default_string``

    Read a nonempty configuration string.

- ``require_default_strings``

    Read a nonempty sequence of nonempty strings.

- ``require_default_pattern``

    Compile a configured regular expression with meaningful errors.

- ``require_default_lowercase_words``

    Validate a configured vocabulary used for term filtering.

Examples
--------
>>> from qachat.defaults import load_defaults_section, require_default_int
>>> settings = load_defaults_section("ANSWERING")
>>> require_default_int(settings, "MAX_RELATED_TURNS", "ANSWERING")
3

Notes
-----
Values are loaded at import time by the consuming modules. Editing the
packaged JSON requires restarting QAChat; it does not override settings
already saved in a project profile.

"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path
import re
from typing import Mapping

__all__ = [
    "load_defaults_section",
    "require_default_bool",
    "require_default_int",
    "require_default_nonnegative_int",
    "require_default_number",
    "require_default_lowercase_words",
    "require_default_pattern",
    "require_default_string",
    "require_default_strings",
]

_DEFAULTS_PATH = Path(__file__).with_name("qachat_defaults.json")


@lru_cache(maxsize=1)
def _packaged_defaults() -> dict[str, object]:
    """Parse the packaged configuration once and check its top-level shape.

    Returns
    -------
    dict[str, object]
        Parsed mapping of package-wide default sections.

    Raises
    ------
    ValueError
        If the file is not valid JSON or its root is not an object.
    OSError
        If the packaged resource cannot be read.

    """

    try:
        data = json.loads(_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {_DEFAULTS_PATH.name}: {error.msg}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError("qachat_defaults.json must contain a JSON object.")
    if any(not isinstance(key, str) for key in data):
        raise ValueError("Default section names must be strings.")
    return data


def load_defaults_section(name: str) -> dict[str, object]:
    """Return an independent copy of a named configuration section.

    Parameters
    ----------
    name: str
        Required top-level section in ``qachat_defaults.json``.

    Returns
    -------
    dict[str, object]
        Configured values owned by the corresponding QAChat module.

    Raises
    ------
    ValueError
        If the section is missing, not an object, or contains invalid keys.

    """

    data = _packaged_defaults()
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(
            f"qachat_defaults.json requires an object section {name!r}."
        )
    if any(not isinstance(key, str) for key in section):
        raise ValueError(f"{name} setting names must be strings.")
    return deepcopy(section)


def require_default_int(
    settings: Mapping[str, object], key: str, section: str
) -> int:
    """Read a strictly positive integer from a configuration section.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    int
        Positive integer setting.

    Raises
    ------
    ValueError
        If the setting is missing, boolean, noninteger, or nonpositive.

    """

    value = settings.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{section}.{key} must be a positive integer.")
    return value


def require_default_nonnegative_int(
    settings: Mapping[str, object], key: str, section: str
) -> int:
    """Read a nonnegative exact integer from a configuration section.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    int
        Zero or a positive integer.

    Raises
    ------
    ValueError
        If the setting is missing, boolean, noninteger, or negative.

    """

    value = settings.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{section}.{key} must be a nonnegative integer.")
    return value


def require_default_bool(
    settings: Mapping[str, object], key: str, section: str
) -> bool:
    """Read an exact JSON boolean from a configuration section.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    bool
        Configured boolean value.

    Raises
    ------
    ValueError
        If the setting is missing or not a boolean.

    """

    value = settings.get(key)
    if type(value) is not bool:
        raise ValueError(f"{section}.{key} must be a boolean.")
    return value


def require_default_number(
    settings: Mapping[str, object],
    key: str,
    section: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Read a finite, nonnegative (or positive) numerical default.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.
    allow_zero: bool, optional
        Accept zero when True. Defaults to ``False``.

    Returns
    -------
    float
        Finite setting converted to floating point.

    Raises
    ------
    ValueError
        If the setting is missing, boolean, nonnumeric, nonfinite, or
        outside the supported range.

    """

    value = settings.get(key)
    number: float | None = None
    if type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError:
            pass
    if (
        number is None
        or not isfinite(number)
        or (number < 0 if allow_zero else number <= 0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(
            f"{section}.{key} must be a finite {qualifier} number."
        )
    return number


def require_default_string(
    settings: Mapping[str, object], key: str, section: str
) -> str:
    """Read a nonempty string from a configuration section.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    str
        Nonempty setting, preserved exactly as configured.

    Raises
    ------
    ValueError
        If the setting is missing or blank, or is not a string.

    """

    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a nonempty string.")
    return value


def require_default_strings(
    settings: Mapping[str, object], key: str, section: str
) -> tuple[str, ...]:
    """Read a nonempty list of nonempty strings from a section.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required setting name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    tuple[str, ...]
        Strings in their configured order, without normalization.

    Raises
    ------
    ValueError
        If the setting is not a nonempty list of nonempty strings.

    """

    values = settings.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str) or not value.strip()
            for value in values
        )
    ):
        raise ValueError(
            f"{section}.{key} must be a nonempty list of strings."
        )
    return tuple(values)


def require_default_pattern(
    settings: Mapping[str, object],
    key: str,
    section: str,
    *,
    flags: int = 0,
) -> re.Pattern[str]:
    """Compile a configured regular expression, rejecting invalid syntax.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required pattern name.
    section: str
        Section label used to identify invalid settings.
    flags: int, optional
        Regular-expression flags. Defaults to ``0``.

    Returns
    -------
    re.Pattern[str]
        Validated, compiled regular expression.

    Raises
    ------
    ValueError
        If the setting is missing, blank, or is invalid regex syntax.

    """

    pattern = require_default_string(settings, key, section)
    try:
        return re.compile(pattern, flags=flags)
    except re.error as error:
        raise ValueError(
            f"{section}.{key} must be a valid regular expression."
        ) from error


def require_default_lowercase_words(
    settings: Mapping[str, object], key: str, section: str
) -> frozenset[str]:
    """Read a lowercase vocabulary for text-matching heuristics.

    Parameters
    ----------
    settings: Mapping[str, object]
        Values loaded from one default section.
    key: str
        Required vocabulary name.
    section: str
        Section label used to identify invalid settings.

    Returns
    -------
    frozenset[str]
        Immutable collection of configured lowercase words.

    Raises
    ------
    ValueError
        If a word contains whitespace, is not lowercase, or is invalid.

    """

    words = require_default_strings(settings, key, section)
    if any(
        word != word.strip()
        or not word.islower()
        or any(character.isspace() for character in word)
        for word in words
    ):
        raise ValueError(
            f"{section}.{key} needs lowercase words without whitespace."
        )
    return frozenset(words)
