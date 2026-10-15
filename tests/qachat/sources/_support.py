"""Shared synthetic fixture paths for QAChat source-extraction tests.

Fixtures live beside the format-specific tests and are never imported or
executed. Unit tests generally use in-memory text instead.

"""

from __future__ import annotations

from pathlib import Path


def fixture_path(name: str) -> Path:
    """Return the path of a committed source-extraction fixture.

    Parameters
    ----------
    name: str
        Fixture filename.

    Returns
    -------
    pathlib.Path
        Path to the fixture inside the test package.

    """

    return Path(__file__).parent / "extensions" / "fixtures" / name


def fixture_text(name: str) -> str:
    """Read one synthetic UTF-8 source fixture without executing it.

    Parameters
    ----------
    name: str
        Fixture filename.

    Returns
    -------
    str
        Unmodified fixture text.

    """

    return fixture_path(name).read_text(encoding="utf-8")
