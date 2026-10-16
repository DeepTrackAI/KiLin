"""Validate shared mention-scanning fields with caller-owned defaults.

The default values belong to each executable tool, not this service.

Key Features
------------

- **Explicit defaults**

    Validate shared scanning settings without importing a registered tool.

Module Structure
----------------

- ``SCAN_SETTING_NAMES`` and ``scan_fields``

    Shared scanning fields and their typed validation.

Examples
--------
>>> from kilin.tools._shared.scan_settings import SCAN_SETTING_NAMES
>>> "max_listed" in SCAN_SETTING_NAMES
True

"""

from __future__ import annotations

from typing import Final, Protocol

from kilin.tools._contract import check_bool, check_int

__all__ = ["SCAN_SETTING_NAMES", "ScanFieldDefaults", "scan_fields"]

SCAN_SETTING_NAMES: Final[tuple[str, ...]] = (
    "case_sensitive",
    "whole_words",
    "context_lines",
    "max_listed",
)


class ScanFieldDefaults(Protocol):
    """Requirements for tool-owned exact-scan defaults.

    Attributes
    ----------
    case_sensitive: bool
        Whether case matters.
    whole_words: bool
        Whether only whole words match.
    context_lines: int
        Number of context lines.
    max_listed: int
        Maximum number of results listed.

    """

    case_sensitive: bool
    whole_words: bool
    context_lines: int
    max_listed: int


def scan_fields(
    data: dict, where: str, defaults: ScanFieldDefaults
) -> dict[str, object]:
    """Validate the settings every mention-collecting tool shares.

    Parameters
    ----------
    data: dict
        Tool settings.
    where: str
        Location for errors.
    defaults: ScanFieldDefaults
        Defaults supplied by the calling tool.

    Returns
    -------
    dict[str, object]
        ``case_sensitive``, ``whole_words``, ``context_lines``, and
        ``max_listed``.

    Raises
    ------
    ValueError
        If a setting has the wrong type or is out of range.

    """

    return {
        "case_sensitive": check_bool(
            where, data, "case_sensitive", defaults.case_sensitive
        ),
        "whole_words": check_bool(
            where, data, "whole_words", defaults.whole_words
        ),
        "context_lines": check_int(
            where, data, "context_lines", defaults.context_lines, 0, 5
        ),
        "max_listed": check_int(
            where, data, "max_listed", defaults.max_listed, 1, 1000
        ),
    }

