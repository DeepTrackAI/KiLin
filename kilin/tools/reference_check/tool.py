"""Tool ``reference_check``: check one paper's references and citations.

"Check the references of the Smith paper" is about one paper (the citing
paper) and the other files of the project (the cited works). The tool:

1. Chooses the citing paper with ``choose_document``: a paper the question
   names, the paper the earlier turns discussed, or the user's choice.
2. Reads its reference list (``numbered_reference_entries``; for a PDF, in
   column order) and numbers each entry as printed.
3. Matches every entry to the project's other files (``_matching``): by
   DOI, title and first author, or first page. Entries without a file are
   reported as missing, so the user knows which cited works to add.
4. Finds the numbered in-text citations (``_citations``) and the sentence
   each appears in, and reports references never cited and citations of
   numbers beyond the list (usually a sign the list was split wrongly).
5. Saves the full report, with every reference and the sentences citing
   it, to ``outputs/<paper>_reference_check.md``, and answers in the chat
   with the counts, the missing and probable references, and problems.

Everything is deterministic: no language model is called, nothing is
invented, and every match states its reason. Checking whether each citing
sentence is supported by the cited work is a separate step.

Limits: author-year citations ("Smith et al., 2020") are not recognised;
superscript citations are recognised only in PDFs (the font sizes confirm
them); an unusual layout can still split or join reference entries.

Key Features
------------

- **Model-free**

    Fast and reproducible; the reasons for every decision are reported.

- **Missing works first**

    The chat lists the cited works the project does not contain.

- **Full report saved**

    Every reference with its file and its citing sentences, as Markdown.

Module Structure
----------------

- ``ReferenceCheckSettings``

    Settings of ``reference_check`` in a project type.

- ``parse_settings``

    Validate ``reference_check`` settings from a project type.

- ``ReferenceReport`` and ``check_references``

    Check one paper's references against the project's files.

- ``report_markdown`` and ``chat_answer``

    Write the full report and the chat answer.

- ``run``

    Check the references of the paper the question means.

Examples
--------
``context`` is the ``ToolContext`` that KíLín passes to a tool:

>>> from kilin.tools import ToolRequest
>>> from kilin.tools.reference_check import ReferenceCheckSettings, run
>>> question = "check the references of smith2020.pdf"
>>> request = ToolRequest(question, question)
>>> result = run(ReferenceCheckSettings(), request, context)

"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kilin.chunking import Chunk
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.choose_document import choose_document
from kilin.tools._shared.outputs import save_output_file
from kilin.tools._shared.references import (
    ReferenceEntry,
    numbered_reference_entries,
    reference_body,
)
from kilin.tools._shared.sentences import Sentence
from kilin.tools.reference_check import _configuration as _config
from kilin.tools.reference_check._citations import (
    STYLE_BRACKETS,
    STYLE_NONE,
    CitationScan,
    find_citations,
    superscript_runs,
)
from kilin.tools.reference_check._matching import (
    MATCH_FOUND,
    MATCH_MISSING,
    MATCH_PROBABLE,
    MATCH_UNREADABLE,
    PaperIdentity,
    ReferenceMatch,
    match_reference,
    paper_identity,
)

__all__ = [
    "ReferenceCheckSettings",
    "ReferenceReport",
    "chat_answer",
    "check_references",
    "parse_settings",
    "report_markdown",
    "run",
]


@dataclass(frozen=True, slots=True)
class ReferenceCheckSettings:
    """Settings of ``reference_check`` in a project type.

    Attributes
    ----------
    max_listed: int
        Most references of each kind (missing, probable) listed in the
        chat (1-1000); the saved report lists all of them.

    """

    max_listed: int = _config.DEFAULT_MAX_LISTED


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    """One paper's references, their files, and their citations.

    Attributes
    ----------
    paper: pathlib.Path
        The citing paper, as indexed.
    entries: tuple[ReferenceEntry, ...]
        Its reference list, in order.
    matches: tuple[ReferenceMatch, ...]
        One per entry, in the same order; ``position`` is the entry's
        reference number.
    scan: CitationScan
        Its in-text citations.
    cited_by: dict[int, tuple[Sentence, ...]]
        For each reference number, the sentences citing it, in order.

    """

    paper: Path
    entries: tuple[ReferenceEntry, ...]
    matches: tuple[ReferenceMatch, ...]
    scan: CitationScan
    cited_by: dict[int, tuple[Sentence, ...]] = field(default_factory=dict)

    def count(self, status: str) -> int:
        """Return how many entries have a match status.

        Parameters
        ----------
        status: str
            ``found``, ``probable``, ``missing``, or ``unreadable``.

        Returns
        -------
        int
            The number of entries with that status.

        """

        return sum(match.status == status for match in self.matches)


def parse_settings(data: dict, where: str) -> ReferenceCheckSettings:
    """Validate ``reference_check`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    ReferenceCheckSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("max_listed",))
    defaults = ReferenceCheckSettings()
    return ReferenceCheckSettings(
        max_listed=check_int(
            where, data, "max_listed", defaults.max_listed, 1, 1000
        ),
    )


def check_references(
    paper: Path,
    passages: Sequence[Chunk],
    file: Path | None,
    others: Sequence[PaperIdentity],
) -> ReferenceReport:
    """Check one paper's references against the project's files.

    Parameters
    ----------
    paper: pathlib.Path
        The citing paper, as indexed.
    passages: Sequence[Chunk]
        Its passages, in order.
    file: pathlib.Path | None
        It on disk, for a PDF's column order and superscripts; None when
        unknown.
    others: Sequence[PaperIdentity]
        The project's other files.

    Returns
    -------
    ReferenceReport
        The entries, a match per entry, and the citations. Superscript
        citations are looked for only when ``file`` is a PDF.

    """

    entries = tuple(
        numbered_reference_entries(passages, _config.REFERENCE_LAYOUT, file)
    )
    matches = tuple(
        match_reference(entry.text, entry.number, others, _config.MATCH_RULES)
        for entry in entries
    )
    superscripts = (
        superscript_runs(file, _config.CITATION_RULES)
        if file is not None and file.suffix.lower() == ".pdf"
        else None
    )
    scan = find_citations(
        reference_body(passages),
        max((entry.number for entry in entries), default=0),
        superscripts,
        _config.CITATION_RULES,
    )

    cited_by: dict[int, list[Sentence]] = defaultdict(list)
    for citation in scan.citations:
        for number in dict.fromkeys(citation.numbers):
            if citation.sentence not in cited_by[number]:
                cited_by[number].append(citation.sentence)

    return ReferenceReport(
        paper=paper,
        entries=entries,
        matches=matches,
        scan=scan,
        cited_by={number: tuple(found) for number, found in cited_by.items()},
    )


def report_markdown(report: ReferenceReport) -> str:
    """Write the full report as Markdown.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    str
        A title, the counts, the missing, probable and unreadable
        references, the citation problems, and every reference with its
        file and the sentences citing it.

    """

    name = report.paper.name
    lines = [f"# Reference check: {name}", "", _overview(report), ""]
    for status, heading in (
        (MATCH_FOUND, "In the project"),
        (MATCH_PROBABLE, "Probably in the project"),
        (MATCH_MISSING, "Missing from the project"),
        (MATCH_UNREADABLE, "Entries too short to identify"),
    ):
        listed = _listed(report, status, None)
        if listed:
            lines += [f"## {heading}", "", *listed, ""]

    problems = _problems(report)
    if problems:
        lines += ["## Citation problems", "", *problems, ""]

    lines += ["## Every reference", ""]
    for entry, match in zip(report.entries, report.matches):
        number = "" if entry.printed else " (number inferred)"
        lines += [f"### [{entry.number}]{number} {entry.text}", ""]
        if match.path is not None:
            lines.append(
                f"- **File:** {match.path.as_posix()} ({match.status}: "
                f"{match.reason})"
            )
        else:
            lines.append(f"- **File:** none ({match.reason})")
        sentences = report.cited_by.get(entry.number, ())
        if not sentences and report.scan.style != STYLE_NONE:
            lines.append("- **Cited:** not found in the text")
        for sentence in sentences:
            lines.append(f"- {_where(sentence)}: “{sentence.text}”")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def chat_answer(
    report: ReferenceReport, max_listed: int, saved: Path | None
) -> str:
    """Write the chat answer.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.
    max_listed: int
        Most references listed for each status.
    saved: pathlib.Path | None
        Where the full report was saved, relative to the project.

    Returns
    -------
    str
        The counts, the found, probable and missing references (at most
        ``max_listed`` each, found and probable with their files), the
        citation problems, and where the full report is.

    """

    name = report.paper.name
    if not report.entries:
        return (
            f"I found no reference list in {name}: no “References” (or "
            "“Bibliography”) heading followed by entries."
        )

    lines = [f"**Reference check of {name}**", "", _overview(report)]
    for status, heading in (
        (MATCH_FOUND, "In the project"),
        (MATCH_PROBABLE, "Probably in the project, please check"),
        (MATCH_MISSING, "Missing from the project"),
    ):
        listed = _listed(report, status, max_listed)
        if listed:
            lines += ["", f"**{heading}:**", "", *listed]

    problems = _problems(report)
    if problems:
        lines += ["", "**Citation problems:**", "", *problems]

    if saved is not None:
        lines += [
            "",
            f"The full report, with every reference and the sentences "
            f"citing it, is in `{saved.as_posix()}`.",
        ]
    return "\n".join(lines)


def run(
    settings: ReferenceCheckSettings,
    request: ToolRequest,
    context: ToolContext,
) -> ToolResult:
    """Check the references of the paper the question means.

    Parameters
    ----------
    settings: ReferenceCheckSettings
        The intent's settings.
    request: ToolRequest
        The question, with the earlier turns it points back to.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        The chat answer, no evidence (the report quotes the sentences),
        and a summary of the counts.

    """

    context.progress("Finding the paper")
    chosen = choose_document(
        request,
        context,
        question="Which paper's references should I check?",
        example="check the references of smith2020.pdf",
        prefer=(".pdf",),
    )
    summary: dict[str, object] = {"name": "reference_check"}
    if isinstance(chosen, str):
        context.done("No paper chosen", evidence_count=0)
        return ToolResult(chosen, (), summary)
    path, reason, guessed = chosen
    summary.update(
        {"paper": path.as_posix(), "reason": reason, "guessed": guessed}
    )

    by_file: dict[Path, list[Chunk]] = defaultdict(list)
    for chunk in context.chunks:
        by_file[chunk.path].append(chunk)

    context.progress("Identifying the project's files")
    others = [
        paper_identity(
            other, passages, _source_file(other, context),
            _config.MATCH_RULES,
        )
        for other, passages in by_file.items()
        if other != path
    ]

    context.progress(f"Reading the references and citations of {path.name}")
    report = check_references(
        path, by_file.get(path, []), _source_file(path, context), others
    )
    saved = (
        save_output_file(
            context.project_root,
            f"{path.stem}_reference_check.md",
            report_markdown(report),
        )
        if report.entries
        else None
    )

    summary.update(
        {
            "references": len(report.entries),
            "numbers_printed": sum(e.printed for e in report.entries),
            "found": report.count(MATCH_FOUND),
            "probable": report.count(MATCH_PROBABLE),
            "missing": report.count(MATCH_MISSING),
            "unreadable": report.count(MATCH_UNREADABLE),
            "citation_style": report.scan.style,
            "citations": len(report.scan.citations),
            "uncited": list(report.scan.uncited),
            "beyond_list": list(report.scan.beyond_list),
            "report": saved.as_posix() if saved else None,
        }
    )
    context.done(
        f"Checked {len(report.entries)} references of {path.name}",
        evidence_count=0,
    )

    answer = chat_answer(report, settings.max_listed, saved)
    if guessed:
        answer += (
            f"\n\n*This is about {path.name}; name the paper if you "
            "meant another one.*"
        )
    return ToolResult(answer, (), summary)


def _overview(report: ReferenceReport) -> str:
    """Return the counts as one paragraph.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    str
        The number of references and how they were numbered, how many are
        in the project, and how many citations were found.

    """

    total = len(report.entries)
    printed = sum(entry.printed for entry in report.entries)
    numbering = (
        "numbered as printed" if printed == total
        else f"{printed} numbered as printed, the others inferred"
    )
    if report.scan.style == STYLE_NONE:
        cited = (
            "In-text citations were not recognised (only numbered "
            "citations are, and superscripts only in PDFs)."
        )
    else:
        style = (
            "bracketed" if report.scan.style == STYLE_BRACKETS
            else "superscript"
        )
        cited = (
            f"{len(report.scan.citations)} {style} citations in the text "
            f"cite {total - len(report.scan.uncited)} of them."
        )
    return (
        f"{total} references ({numbering}). In the project: "
        f"{report.count(MATCH_FOUND)} found, "
        f"{report.count(MATCH_PROBABLE)} probable, "
        f"{report.count(MATCH_MISSING)} missing"
        + (
            f", {report.count(MATCH_UNREADABLE)} too short to identify"
            if report.count(MATCH_UNREADABLE) else ""
        )
        + f". {cited}"
    )


def _listed(
    report: ReferenceReport, status: str, limit: int | None
) -> list[str]:
    """Return the references with one status as Markdown list items.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.
    status: str
        The match status to list.
    limit: int | None
        Most items; None lists all.

    Returns
    -------
    list[str]
        "- [12] entry" items (with the file, and for a probable match the
        reason), then a line saying how many more there are.

    """

    items = [
        (entry, match)
        for entry, match in zip(report.entries, report.matches)
        if match.status == status
    ]
    shown = items if limit is None else items[:limit]
    lines = []
    for entry, match in shown:
        line = f"- [{entry.number}] {entry.text}"
        if status == MATCH_FOUND and match.path is not None:
            line += f" → {match.path.as_posix()}"
        elif status == MATCH_PROBABLE and match.path is not None:
            line += f" → {match.path.as_posix()} ({match.reason})"
        lines.append(line)
    if len(items) > len(shown):
        lines.append(f"- … and {len(items) - len(shown)} more")
    return lines


def _problems(report: ReferenceReport) -> list[str]:
    """Return the citation problems as Markdown list items.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    list[str]
        References never cited in the text, and numbers cited beyond the
        reference list; empty when there are none or citations were not
        recognised.

    """

    if report.scan.style == STYLE_NONE:
        return []

    numbers = {entry.number for entry in report.entries}
    uncited = [number for number in report.scan.uncited if number in numbers]
    problems = []
    if uncited:
        problems.append(
            "- Never cited in the text: "
            + ", ".join(f"[{number}]" for number in uncited)
        )
    if report.scan.beyond_list:
        problems.append(
            "- Cited but not in the reference list: "
            + ", ".join(f"[{number}]" for number in report.scan.beyond_list)
            + " (the list may have been split wrongly)"
        )
    return problems


def _where(sentence: Sentence) -> str:
    """Return where a sentence is, for the report.

    Parameters
    ----------
    sentence: Sentence
        A citing sentence.

    Returns
    -------
    str
        "Page 3, line 12" for PDF pages, whose lines restart on each page;
        "line 12" otherwise.

    """

    section = sentence.chunk.section
    return f"{section}, line {sentence.line}" if section else (
        f"line {sentence.line}"
    )


def _source_file(path: Path, context: ToolContext) -> Path | None:
    """Return a project file on disk.

    Parameters
    ----------
    path: pathlib.Path
        The file, as indexed.
    context: ToolContext
        For the project folder.

    Returns
    -------
    pathlib.Path | None
        The file under the project folder or its ``sources/`` folder; None
        when neither exists.

    """

    for candidate in (
        context.project_root / path,
        context.project_root / "sources" / path,
    ):
        if candidate.is_file():
            return candidate

    return None
