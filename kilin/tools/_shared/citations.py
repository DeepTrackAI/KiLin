"""Validate numbered source-passage citations in document-tool replies.

Citation groups can contain individual numbers, comma-separated numbers,
and bounded ranges. This service validates them before expansion, so a
malformed or arbitrarily large range cannot exhaust memory or silently
become evidence. It does not assess whether a cited passage supports a claim.

Key Features
------------

- **Bounded parsing**

    Resolve numbers and short ranges within the available passage count.

- **Explicit validation**

    Keep valid numbers and report invalid numbers or malformed range parts.

Module Structure
----------------

- ``CitationReport``

    Valid passage references and two kinds of invalid references.

- ``inspect_citations``

    Check citation groups against a supplied passage count.

Examples
--------
>>> from kilin.tools._shared.citations import inspect_citations
>>> result = inspect_citations("Evidence [1, 3-5]", 3)
>>> result.valid, result.invalid_groups
(frozenset({1}), ('3-5',))

"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final

__all__ = ["CitationReport", "inspect_citations"]

# Bound expansion independently of the document size. Long ranges should
# be stated explicitly, rather than accidentally allocating huge sets.
MAX_CITATION_RANGE: Final[int] = 100
# Avoid integer-conversion limits and enormous diagnostic strings.
_MAX_NUMBER_DIGITS: Final[int] = 15
_MAX_REPORTED_PART: Final[int] = 80
_GROUP: Final[re.Pattern[str]] = re.compile(r"\[([^\[\]\r\n]*)\]")
_NUMBER: Final[re.Pattern[str]] = re.compile(r"\d+")
_RANGE: Final[re.Pattern[str]] = re.compile(
    r"(\d+)\s*[-–]\s*(\d+)"
)


@dataclass(frozen=True, slots=True)
class CitationReport:
    """References checked against a finite passage list.

    Attributes
    ----------
    valid: frozenset[int]
        Existing one-based passage numbers.
    invalid_numbers: tuple[int, ...]
        Missing individual passage numbers, in first-seen order.
    invalid_groups: tuple[str, ...]
        Malformed, oversized, or out-of-range citation parts. Each item
        is bounded in length for safe diagnostics.

    """

    valid: frozenset[int]
    invalid_numbers: tuple[int, ...]
    invalid_groups: tuple[str, ...]


def inspect_citations(text: str, passages: int) -> CitationReport:
    """Check passage citation groups without unbounded range expansion.

    Parameters
    ----------
    text: str
        Generated text containing citations such as ``[2, 4-6]``.
    passages: int
        Available one-based passage count, including zero for no passages.

    Returns
    -------
    CitationReport
        Valid numbers, invalid individual numbers, and invalid parts.
        A mixed group keeps its independent valid numbers. Unrelated
        brackets such as ``[ref]`` are ignored.

    Raises
    ------
    ValueError
        If ``passages`` is negative or not an integer.

    """

    if type(passages) is not int or passages < 0:
        raise ValueError("passages must be a nonnegative integer.")

    valid: set[int] = set()
    invalid_numbers: dict[int, None] = {}
    invalid_groups: dict[str, None] = {}
    for match in _GROUP.finditer(text):
        group = match.group(1).strip()
        if not re.match(r"^\d", group):
            continue
        for raw in group.split(","):
            piece = raw.strip()
            single = _NUMBER.fullmatch(piece)
            pair = _RANGE.fullmatch(piece)
            if single is None and pair is None:
                _record_invalid(invalid_groups, piece)
                continue
            numbers = (piece,) if single else pair.groups()
            if any(len(number) > _MAX_NUMBER_DIGITS for number in numbers):
                _record_invalid(invalid_groups, piece)
                continue
            first = int(numbers[0])
            if single:
                if 1 <= first <= passages:
                    valid.add(first)
                else:
                    invalid_numbers[first] = None
                continue
            last = int(numbers[1])
            if (
                first < 1 or last < first or last > passages
                or last - first + 1 > MAX_CITATION_RANGE
            ):
                _record_invalid(invalid_groups, piece)
                continue
            valid.update(range(first, last + 1))
    return CitationReport(
        frozenset(valid), tuple(invalid_numbers), tuple(invalid_groups)
    )


def _record_invalid(target: dict[str, None], value: str) -> None:
    """Store a bounded diagnostic for an invalid citation part.

    Parameters
    ----------
    target: dict[str, None]
        Ordered set of malformed or invalid citation parts.
    value: str
        Part to report.

    """

    value = value or "(empty)"
    if len(value) > _MAX_REPORTED_PART:
        value = value[:_MAX_REPORTED_PART] + "..."
    target[value] = None
