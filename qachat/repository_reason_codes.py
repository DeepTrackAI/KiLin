"""Load canonical repository-selection reasons shared with the GUI.

The packaged JSON resource owns the wire-format identifiers emitted by the
repository scanner and consumed by the desktop interface. The file is
validated once at import time. The public mapping is immutable, and named
constants provide readable values for Python consumers. The GUI sends the
same mapping to JavaScript through its existing settings event, avoiding a
second hard-coded vocabulary in the frontend.

Key Features
------------

- **Single source of truth**

    Define all selection codes in one packaged JSON resource.

- **Strict resource validation**

    Reject incomplete, malformed, or conflicting definitions explicitly.

Module Structure
----------------

- ``SELECTION_REASON_CODES``

    Read-only mapping from symbolic names to protocol values.

- ``SELECTION_REASON_*``

    Named constants for core and Python GUI consumers.

Examples
--------
>>> from qachat.repository_reason_codes import (
...     SELECTION_REASON_FILE_TOO_LARGE,
... )
>>> SELECTION_REASON_FILE_TOO_LARGE
'file_too_large'

Notes
-----
The JSON resource must be shipped with QAChat. Invalid or missing definitions
are startup errors, not a reason to substitute implicit protocol values.

"""

from __future__ import annotations

from importlib import resources
import json
import re
from types import MappingProxyType
from typing import Final, Mapping

__all__ = [
    "SELECTION_REASON_CODES",
    "SELECTION_REASON_SELECTED",
    "SELECTION_REASON_FILE_TOO_LARGE",
    "SELECTION_REASON_EMPTY_FILE",
    "SELECTION_REASON_FILE_TYPE_NOT_SELECTED",
    "SELECTION_REASON_PROTECTED_FILE",
    "SELECTION_REASON_BLOCKED_FILE_TYPE",
    "SELECTION_REASON_EXCLUDED_PATH",
    "SELECTION_REASON_OUTSIDE_SCOPE",
    "SELECTION_REASON_LINK_NOT_FOLLOWED",
    "SELECTION_REASON_EXCLUDED_DIRECTORY",
    "SELECTION_REASON_UNSUPPORTED_ENTRY",
    "SELECTION_REASON_INSPECTION_FAILED",
]

_REASON_KEYS = (
    "SELECTED",
    "FILE_TOO_LARGE",
    "EMPTY_FILE",
    "FILE_TYPE_NOT_SELECTED",
    "PROTECTED_FILE",
    "BLOCKED_FILE_TYPE",
    "EXCLUDED_PATH",
    "OUTSIDE_SCOPE",
    "LINK_NOT_FOLLOWED",
    "EXCLUDED_DIRECTORY",
    "UNSUPPORTED_ENTRY",
    "INSPECTION_FAILED",
)
_RESOURCE_NAME = "repository_reason_codes.json"


def _validate_reason_codes(data: object) -> Mapping[str, str]:
    """Validate and freeze packaged repository selection identifiers.

    Parameters
    ----------
    data: object
        JSON-decoded resource contents.

    Returns
    -------
    Mapping[str, str]
        Immutable mapping of validated identifiers.

    Raises
    ------
    ValueError
        If keys are missing or unexpected, values are not valid code
        strings, or two symbolic names share a protocol identifier.

    """

    if not isinstance(data, dict):
        raise ValueError(f"{_RESOURCE_NAME} must contain a JSON object.")

    actual = set(data)
    expected = set(_REASON_KEYS)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{_RESOURCE_NAME} has invalid keys: "
            f"missing={missing}, unexpected={unexpected}."
        )

    for key, value in data.items():
        if not isinstance(value, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", value
        ):
            raise ValueError(
                f"{_RESOURCE_NAME}.{key} must be a lowercase identifier."
            )

    if len(set(data.values())) != len(data):
        raise ValueError(f"{_RESOURCE_NAME} contains duplicate codes.")

    return MappingProxyType(dict(data))


def _unique_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Reject repeated names before JSON decoding can discard them.

    Parameters
    ----------
    pairs: list[tuple[str, object]]
        Object members in the order supplied by the JSON decoder.

    Returns
    -------
    dict[str, object]
        Object with unique keys.

    Raises
    ------
    ValueError
        If a key appears more than once in the resource.

    """

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{_RESOURCE_NAME} repeats key {key}.")
        result[key] = value
    return result


def _load_reason_codes() -> Mapping[str, str]:
    """Read and validate the packaged reason-code resource once.

    Returns
    -------
    Mapping[str, str]
        Immutable validated selection identifiers.

    Raises
    ------
    FileNotFoundError
        If the packaged resource is missing.
    ValueError
        If its JSON is invalid or its definitions violate the contract.

    """

    text = resources.files("qachat").joinpath(_RESOURCE_NAME).read_text(
        encoding="utf-8"
    )
    try:
        data = json.loads(text, object_pairs_hook=_unique_json_keys)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {_RESOURCE_NAME}: {error.msg}"
        ) from error
    return _validate_reason_codes(data)


SELECTION_REASON_CODES: Final[Mapping[str, str]] = _load_reason_codes()
SELECTION_REASON_SELECTED: Final[str] = (
    SELECTION_REASON_CODES["SELECTED"]
)
SELECTION_REASON_FILE_TOO_LARGE: Final[str] = (
    SELECTION_REASON_CODES["FILE_TOO_LARGE"]
)
SELECTION_REASON_EMPTY_FILE: Final[str] = (
    SELECTION_REASON_CODES["EMPTY_FILE"]
)
SELECTION_REASON_FILE_TYPE_NOT_SELECTED: Final[str] = (
    SELECTION_REASON_CODES["FILE_TYPE_NOT_SELECTED"]
)
SELECTION_REASON_PROTECTED_FILE: Final[str] = (
    SELECTION_REASON_CODES["PROTECTED_FILE"]
)
SELECTION_REASON_BLOCKED_FILE_TYPE: Final[str] = (
    SELECTION_REASON_CODES["BLOCKED_FILE_TYPE"]
)
SELECTION_REASON_EXCLUDED_PATH: Final[str] = (
    SELECTION_REASON_CODES["EXCLUDED_PATH"]
)
SELECTION_REASON_OUTSIDE_SCOPE: Final[str] = (
    SELECTION_REASON_CODES["OUTSIDE_SCOPE"]
)
SELECTION_REASON_LINK_NOT_FOLLOWED: Final[str] = (
    SELECTION_REASON_CODES["LINK_NOT_FOLLOWED"]
)
SELECTION_REASON_EXCLUDED_DIRECTORY: Final[str] = (
    SELECTION_REASON_CODES["EXCLUDED_DIRECTORY"]
)
SELECTION_REASON_UNSUPPORTED_ENTRY: Final[str] = (
    SELECTION_REASON_CODES["UNSUPPORTED_ENTRY"]
)
SELECTION_REASON_INSPECTION_FAILED: Final[str] = (
    SELECTION_REASON_CODES["INSPECTION_FAILED"]
)
