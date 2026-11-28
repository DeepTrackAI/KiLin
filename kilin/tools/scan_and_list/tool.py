"""Tool ``scan_and_list``: every exact mention, without model analysis.

``scan_and_list`` answers questions such as "list all mentions of transfer
learning" or "how many times does the Smith paper say MRI?" exhaustively.
Ranked retrieval returns only the best matches; this tool inspects every
indexed line instead, and Python, not a model, counts and lists the results.
The router model only names the terms and the document to search, and both
are accepted only when they occur in the user's words.

Indexed chunks overlap, so counting inside chunks would count mentions in
the overlaps twice. Every chunk consists of whole source lines with their
line numbers, and line numbers are unique within one natural unit: a file's
section, a PDF page (numbering restarts on each page), or a notebook cell.
The tool therefore reassembles each unit's lines from its chunks once
and scans the reassembled text, so a phrase broken across lines is found and
nothing is counted twice.

Only indexed text is scanned: files without searchable text (for example
scanned PDFs) cannot be read, and a scan of every file names them in the
answer. Matching is literal apart from letter case, spacing, hyphens, and
underscores; synonyms and other wordings are not found.

Key Features
------------

- **Terms in the user's words**

    ``quoted_terms`` takes quoted phrases from the question, and
    ``validated_terms`` accepts a model's suggestions only when they occur
    in the question.

- **Exact scan**

    ``scan_mentions`` matches case-insensitively by default, as whole words,
    with any whitespace (including line breaks) between a phrase's words;
    ``term_pattern`` also accepts spelling variants such as "XYZ2", "XYZ 2",
    and "xyz-2".

- **Scope**

    ``collect_mentions`` scans the files the question names, the document
    it describes, or every indexed file.

- **Report**

    ``mention_report`` writes counts per file, source-linked matches, and
    coverage; ``mention_evidence`` gives each listed match's exact lines.
    A long list can be saved as a Markdown file instead of shown in chat.

Module Structure
----------------

- ``MAX_TERMS``

    Most terms one question may scan for.

- ``SCAN_SETTING_NAMES``

    Names of the settings shared by tools that collect mentions.

- ``ScanSettings``

    Settings of ``scan_and_list`` in a project type.

- ``parse_settings``

    Validate ``scan_and_list`` settings from a project type.

- ``scan_fields``

    Validate the settings every mention-collecting tool shares.

- ``Mention``

    One occurrence of a term.

- ``MentionScan``

    All mentions found, and what was scanned.

- ``MentionCollection``

    Every mention a question asks about, and how they were found.

- ``run``

    List and count every exact mention the question asks about.

- ``collect_mentions``

    Find the terms and scope in the question, then every mention.

- ``quoted_terms``

    Return phrases the user put in double quotes or backticks.

- ``terms_prompt``

    Return the system and user prompts that ask a model for terms.

- ``validated_terms``

    Accept a model's suggested terms only when the user wrote them.

- ``validated_document``

    Accept a model's document phrase only when the user wrote it.

- ``document_phrase``

    Find words naming one document, such as "the Smith paper".

- ``term_pattern``

    Match a term flexibly: "XYZ2", "XYZ 2", "xyz-2" are one term.

- ``scan_mentions``

    Find every occurrence of the terms in the indexed text.

- ``mention_evidence``

    Return evidence fields for the first mentions.

- ``mention_report``

    Write the answer: counts, listed mentions, and coverage.

- ``mention_location``

    Describe where a mention is: file, page or cell, lines, symbol.

- ``mention_snippet``

    Show the occurrence in bold with some text on each side, one line.

- ``short_names``

    Name each file by the shortest path ending that is unique.

Private helpers reassemble units from chunks and read model replies; they
are not part of the public API.

Examples
--------
>>> from kilin.tools.scan_and_list import quoted_terms, term_pattern
>>> quoted_terms('list all mentions of "XYZ 2"')
('XYZ 2',)
>>> bool(term_pattern("XYZ 2", False, True).search("we used xyz-2"))
True

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import _configuration as _config
from kilin.tools._shared.scan_settings import (
    SCAN_SETTING_NAMES,
    scan_fields as _shared_scan_fields,
)
from kilin.tools._shared.mentions import (
    MentionScan,
    collect_mentions,
    mention_evidence,
    mention_location,
    mention_snippet,
)
from kilin.tools._shared.paths import short_names
from kilin.tools import (
    ToolContext,
    ToolRequest,
    ToolResult,
    reject_unknown,
)
from kilin.tools._shared.long_lists import (
    CHAT_LIST_LIMIT,
    file_name,
    offer_export,
)

__all__ = [
    "ScanSettings",
    "mention_report",
    "parse_settings",
    "run",
    "scan_fields",
]

# Settings shared by tools that collect mentions.

# Files without searchable text named in the coverage line.
_UNSEARCHABLE_EXAMPLES: Final[int] = _config.UNSEARCHABLE_EXAMPLES

@dataclass(frozen=True, slots=True)
class ScanSettings:
    """Settings of ``scan_and_list`` in a project type.

    Attributes
    ----------
    case_sensitive: bool
        Whether letter case must match.
    whole_words: bool
        Whether a term must not continue into adjacent letters or digits.
    context_lines: int
        Lines of context kept around each mention (0-5).
    max_listed: int
        Mentions listed with citations (1-1000); counts include all.

    """

    case_sensitive: bool = _config.DEFAULT_CASE_SENSITIVE
    whole_words: bool = _config.DEFAULT_WHOLE_WORDS
    context_lines: int = _config.DEFAULT_CONTEXT_LINES
    max_listed: int = _config.DEFAULT_MAX_LISTED


def parse_settings(data: dict, where: str) -> ScanSettings:
    """Validate ``scan_and_list`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    ScanSettings
        Validated settings; absent ones take their defaults.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, SCAN_SETTING_NAMES)
    return ScanSettings(**scan_fields(data, where))


def scan_fields(data: dict, where: str) -> dict[str, object]:
    """Validate shared scanning fields using this tool's defaults.

    Parameters
    ----------
    data: dict
        Tool-specific settings to validate.
    where: str
        Location for errors.

    Returns
    -------
    dict[str, object]
        Validated shared scanning fields.

    Raises
    ------
    ValueError
        If a field has the wrong type or an invalid value.

    """

    return _shared_scan_fields(data, where, ScanSettings())








def run(
    settings: ScanSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """List and count every exact mention the question asks about.

    Parameters
    ----------
    settings: ScanSettings
        The intent's settings.
    request: ToolRequest
        The question and the user's words.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        Counts per file, listed mentions with exact-line evidence, and
        coverage. No answering-model call is made.

    Notes
    -----
    When the list is long, ``offer_export`` may save the full report as a
    Markdown file; the chat answer then keeps only the counts.

    """

    found = collect_mentions(
        settings, request, context, tool="scan_and_list"
    )
    if isinstance(found, ToolResult):
        return found

    scan, summary = found.scan, found.summary
    terms = " or ".join(f"“{term}”" for term in scan.terms)
    saved = offer_export(
        context,
        count=len(scan.mentions),
        what=f"mentions of {terms}",
        name=file_name("mentions", " ".join(scan.terms)),
        markdown=mention_report(
            scan,
            limit=len(scan.mentions),
            unsearchable=found.unsearchable,
            scope_note=found.scope_note,
        ),
    )
    if saved:
        limit = 0  # The file has the list; the chat keeps the counts.
    elif context.choose is not None:
        limit = min(settings.max_listed, CHAT_LIST_LIMIT)
    else:
        limit = settings.max_listed  # Notebooks: nobody to ask.

    evidence = mention_evidence(scan, limit)
    answer = mention_report(
        scan,
        limit=limit,
        unsearchable=found.unsearchable,
        scope_note=found.scope_note,
    )
    if saved:
        summary["saved"] = saved.as_posix()
        answer += (
            f"\n\nThe full list of {len(scan.mentions)} mentions is saved "
            f"as `{saved.as_posix()}`."
        )

    summary["listed"] = len(evidence)
    context.trace.emit(
        "tool.scan_and_list.result",
        component="tool",
        data={key: value for key, value in summary.items() if key != "name"},
        turn_id=context.trace_turn_id,
    )
    noun = "mention" if len(scan.mentions) == 1 else "mentions"
    context.done(
        f"Found {len(scan.mentions)} {noun} in {len(scan.counts())} files",
        evidence_count=len(evidence),
    )
    return ToolResult(answer, evidence, summary)








def mention_report(
    scan: MentionScan,
    *,
    limit: int,
    unsearchable: Sequence[Path] = (),
    scope_note: str | None = None,
) -> str:
    """Write the answer: counts, listed mentions, and coverage.

    Parameters
    ----------
    scan: MentionScan
        Scan result.
    limit: int
        How many mentions are listed with citations; counts always cover
        every mention.
    unsearchable: Sequence[pathlib.Path], optional
        Source files with no indexed text, which could not be scanned.
        Defaults to ``()``.
    scope_note: str | None, optional
        How the scanned files were chosen, when not all were scanned.
        Defaults to ``None``.

    Returns
    -------
    str
        Markdown. Listed mention ``n`` cites evidence ``[n]``.

    """

    quoted = " or ".join(f"“{term}”" for term in scan.terms)
    total = len(scan.mentions)
    counts = scan.counts()
    files = len(scan.files_scanned)
    rules = ", ".join(
        [
            "case-sensitive" if scan.case_sensitive else "ignoring case",
            "whole words" if scan.whole_words else "also inside words",
        ]
    )
    names = short_names(scan.files_scanned)
    noun = "mention" if total == 1 else "mentions"
    where = (
        f"**{names[scan.files_scanned[0]]}**"
        if files == 1
        else f"**{len(counts)} of {files} files**"
    )
    lines = [f"Found **{total} {noun}** of {quoted} in {where} ({rules})."]
    if scope_note:
        lines += ["", scope_note]
    if len(scan.terms) > 1:
        lines += [
            "",
            *(
                f"- “{term}”: {count}"
                for term, count in scan.term_counts().items()
            ),
        ]

    if counts:
        lines += ["", "**Mentions per file**", ""]
        lines += [
            f"- {names[path]}: {count}"
            for path, count in sorted(
                counts.items(), key=lambda item: (-item[1], str(item[0]))
            )
        ]
    # A limit of zero keeps only the counts (the list is in a file).
    if counts and limit > 0:
        listed = scan.mentions[:limit]
        lines += ["", "**Where**", ""]
        for number, mention in enumerate(listed, start=1):
            snippet = mention_snippet(mention)
            lines.append(
                f"{number}. {mention_location(mention, names[mention.path])}: "
                f"{snippet} [{number}]"
            )
        if len(listed) < total:
            lines += [
                "",
                f"Listed the first {len(listed)} of {total} mentions; the "
                "counts above include all of them.",
            ]

    coverage = (
        f"Scanned all indexed text in {names[scan.files_scanned[0]]}."
        if files == 1
        else f"Scanned all indexed text in {files} files."
    )
    if unsearchable:
        examples = ", ".join(
            path.name for path in unsearchable[:_UNSEARCHABLE_EXAMPLES]
        )
        more = len(unsearchable) - _UNSEARCHABLE_EXAMPLES
        coverage += (
            f" {len(unsearchable)} source files have no searchable text "
            f"(for example scanned PDFs) and were not scanned: {examples}"
            + (f", and {more} more." if more > 0 else ".")
        )

    lines += ["", coverage]
    return "\n".join(lines)
















