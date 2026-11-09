"""Recognize questions requiring complete source coverage.

Explicit all-source requests change the source-first evidence policy to
exhaustive for that turn. Merely referring to a numbered subset of papers
must not imply that every source in the project was requested.

Key Features
------------

- **Generic source wording**

    Recognize papers, articles, studies, documents, sources and PDFs.

- **Explicit exhaustive intent**

    Require all, each, every or across, not just a mentioned source count.

Module Structure
----------------

- ``asks_for_all_sources``

    Decide whether a question explicitly requests full source coverage.

Examples
--------
>>> asks_for_all_sources("Compare all five papers")
True
>>> asks_for_all_sources("Compare two papers")
False

"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["asks_for_all_sources"]

_DOCUMENTS: Final[str] = (
    r"(?:papers?|articles?|stud(?:y|ies)|documents?|sources?|pdfs?|works?)"
)
_PLURAL_DOCUMENTS: Final[str] = (
    r"(?:papers|articles|studies|documents|sources|pdfs|works)"
)
_ALL_SOURCES: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:all|each|every)\b(?:\W+\w+){{0,3}}?\W+{_DOCUMENTS}\b"
    r"|\b(?:all|each|every one) of (?:them|these|those)\b"
    rf"|\bacross (?:the |these |those )?{_DOCUMENTS}\b"
    r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|[1-9]\d?)\W+"
    rf"(?:\w+\W+)?{_PLURAL_DOCUMENTS}\b",
    re.IGNORECASE,
)
_EXHAUSTIVE_WORDS: Final[re.Pattern[str]] = re.compile(
    r"\b(?:all|each|every|across)\b", re.IGNORECASE
)


def asks_for_all_sources(question: str) -> bool:
    """Detect explicit requests for every source in a collection.

    Parameters
    ----------
    question: str
        Original question, before query expansion or reformulation.

    Returns
    -------
    bool
        Whether the question asks for complete source coverage. Questions
        about a specific number of papers without an all-source modifier
        are not exhaustive.

    """

    return bool(
        _ALL_SOURCES.search(question) and _EXHAUSTIVE_WORDS.search(question)
    )
