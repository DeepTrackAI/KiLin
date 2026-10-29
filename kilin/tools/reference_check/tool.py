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
5. Checks each *claim* (a citing sentence and one cited work that is in
   the project) with the answering model (``_shared.claims``): the cited
   file's most related passages, a verdict, the passage sentence it rests
   on (checked to be in the passages), and a critical comment. The plan is
   shown first, and the user can run all claims, a few, or none.
6. Saves the full report, with every reference, the sentences citing it
   and their verdicts, to ``outputs/<paper>_reference_check.md``, and
   answers in the chat with the counts, the references, the problems, and
   the claims that are not fully supported, citing the quoted passages.

Steps 1-4 call no language model (only choosing "that paper" may ask the
router to confirm it): they are rules, reproducible and with a stated
reason for every decision. Step 5 uses the answering model, whose verdicts
can be wrong; its quotes are checked, and a reference is called
contradicted only with a confirmed quote.

Limits: author-year citations ("Smith et al., 2020") are not recognised;
superscript citations are recognised only in PDFs (the font sizes and
positions confirm them); an unusual layout can still split or join
reference entries; a claim is judged from a few passages of the cited
work, not all of it.

Key Features
------------

- **Rules before the model**

    References, files and citations are found without a language model.

- **Missing works first**

    The chat lists the cited works the project does not contain.

- **Checked claims**

    Each claim gets a verdict, a quote found in the cited work, and a
    comment; the user confirms the model calls first.

- **Full report saved**

    Every reference with its file, citing sentences and verdicts.

Module Structure
----------------

- ``ReferenceCheckSettings``

    Settings of ``reference_check`` in a project type.

- ``parse_settings``

    Validate ``reference_check`` settings from a project type.

- ``ReferenceReport``, ``check_references`` and ``report_claims``

    Check one paper's references against the project's files, and list
    the claims that can be checked.

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

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Final

from kilin.chunking import Chunk
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_bool,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.choose_document import choose_document
from kilin.tools._shared.outputs import save_output_file
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    CONFIRMATION_ALWAYS,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)
from kilin.tools._shared.references import (
    ReferenceEntry,
    numbered_reference_entries,
    reference_body,
)
from kilin.tools._shared.sentences import Sentence, split_sentences
from kilin.tools.reference_check import _configuration as _config
from kilin.tools.reference_check._resolution import (
    LookupPermission,
    ReferenceLink,
    offline_link,
    linkable_entry,
    resolve_links,
)
from kilin.tools.reference_check._citations import (
    STYLE_BRACKETS,
    STYLE_NONE,
    STYLE_PARENTHESES,
    STYLE_WRITTEN,
    Citation,
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
    settle_duplicates,
)
from kilin.tools._shared.claims import (
    VERDICTS,
    Claim,
    Verdict,
    claim_passages,
    verdict_label,
    verify_claim,
)

__all__ = [
    "ReferenceCheckSettings",
    "ReferenceReport",
    "chat_answer",
    "check_references",
    "parse_settings",
    "report_claims",
    "report_markdown",
    "run",
]

# Order of the checked claims in the chat and the report: those needing
# attention first, most serious first.
_ORDER: Final[dict[str, int]] = {
    verdict: index for index, verdict in enumerate(
        ("contradicted", "not supported", "partly supported", "not found",
         "unclear", "failed", "no text", "supported", "background")
    )
}
# Digits and separators as superscripts, to show citation markers as
# printed ("NN⁸²") and keep "[12]" from reading as a link to evidence 12.
_SUPERSCRIPT: Final[dict[int, str]] = str.maketrans(
    "0123456789–-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻"
)
# Words after which the phrase a reference is cited for starts ("using a
# previously established tracking method⁴⁰").
_CITED_FOR_START: Final[re.Pattern[str]] = re.compile(
    r".*\b(?:using|with|by|via|such as|like|including|from|see|e\.g\.|"
    r"i\.e\.|than|as in|following)\s+",
    flags=re.I | re.S,
)
# A phrase that only attributes the sentence to the reference ("already
# observed in the 1st AnDi Challenge", "as reported in"): the reference is
# cited for the whole sentence.
_ATTRIBUTION: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:already|previously|first|as)\s+)*(?:observed|reported|"
    r"shown|described|found|discussed|demonstrated|noted|seen|proposed|"
    r"suggested|done|introduced)\b",
    flags=re.I,
)
# Most words of the phrase a reference is cited for.
_CITED_FOR_WORDS: Final[int] = 15
# Longest citing sentence shown whole; longer ones (often a table read as
# one sentence) are shown around the marker, in characters.
_SHOWN_CHARACTERS: Final[int] = 320
# A bracketed marker in a claim's marked sentence, removed for ranking.
_UNMARKED: Final[re.Pattern[str]] = re.compile(r"\s?\[[\d,–\- ]+\]")
# Bracketed numbers, which the chat reads as links to evidence.
_NUMBERED: Final[re.Pattern[str]] = re.compile(
    r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]"
)
# A request for bibliographic links must not run the expensive claim audit.
_LINK_REQUEST: Final[re.Pattern[str]] = re.compile(
    r"\b(?:links?|urls?|dois?|online|web)\b",
    flags=re.I,
)
# A link-search request only consults the missing bibliography entries.
_QUICK_LOOKUPS: Final[int] = 5
# A word broken across lines by extraction ("eConnec- tome").
_BROKEN_WORD: Final[re.Pattern[str]] = re.compile(r"(?<=[a-z])- (?=[a-z])")


@dataclass(frozen=True, slots=True)
class ReferenceCheckSettings:
    """Settings of ``reference_check`` in a project type.

    Attributes
    ----------
    max_listed: int
        Most items listed in the chat for each kind (1-1000); the saved
        report lists all of them.
    verify: bool
        Whether to check claims with the answering model.
    passages: int
        Most passages of the cited work read per claim (1-20).
    passage_tokens: int
        Most estimated tokens of those passages (300-16000).
    answer_tokens: int
        Longest verdict answer, in tokens (100-1000).
    max_model_calls: int
        Most claims checked (1-1000); later ones are reported unchecked.
    quick_claims: int
        Claims checked by the quick version (1-100).
    confirmation: ConfirmationPolicy
        When to ask before checking claims: always, when_large, or never.
    large_batches: int
        With ``when_large``, plans with more claims than this ask first
        (0-1000).
    online_lookup: bool
        Offer online reference resolution even without a link request.
        Never grants network permission by itself.
    max_online: int
        Most missing entries offered for resolution in one operation
        (1-1000).

    """

    max_listed: int = _config.DEFAULT_MAX_LISTED
    verify: bool = _config.DEFAULT_VERIFY
    passages: int = _config.DEFAULT_PASSAGES
    passage_tokens: int = _config.DEFAULT_PASSAGE_TOKENS
    answer_tokens: int = _config.DEFAULT_ANSWER_TOKENS
    max_model_calls: int = _config.DEFAULT_MAX_MODEL_CALLS
    quick_claims: int = _config.DEFAULT_QUICK_CLAIMS
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES
    online_lookup: bool = _config.DEFAULT_ONLINE_LOOKUP
    max_online: int = _config.DEFAULT_MAX_ONLINE


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    """One paper's references, their files, citations and verdicts.

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
    before: dict[Sentence, str]
        For each citing sentence, the sentence before it, with its markers
        in brackets.
    verdicts: tuple[Verdict, ...]
        The checked claims, in reading order.
    verification: str
        Why claims were not checked, or not all of them; empty otherwise.
    links: dict[int, ReferenceLink]
        Preferred offline or resolved links for missing entries and short
        entries with usable journal coordinates or a DOI.
    lookup_note: str
        Whether optional online resolution was authorized or declined.

    """

    paper: Path
    entries: tuple[ReferenceEntry, ...]
    matches: tuple[ReferenceMatch, ...]
    scan: CitationScan
    cited_by: dict[int, tuple[Sentence, ...]] = field(default_factory=dict)
    before: dict[Sentence, str] = field(default_factory=dict)
    verdicts: tuple[Verdict, ...] = ()
    verification: str = ""
    links: dict[int, ReferenceLink] = field(default_factory=dict)
    lookup_note: str = ""

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

    reject_unknown(
        where,
        data,
        (
            "max_listed", "verify", "passages", "passage_tokens",
            "answer_tokens", "max_model_calls", "quick_claims",
            "confirmation", "large_batches", "online_lookup",
            "max_online",
        ),
    )
    defaults = ReferenceCheckSettings()
    return ReferenceCheckSettings(
        max_listed=check_int(
            where, data, "max_listed", defaults.max_listed, 1, 1000
        ),
        verify=check_bool(where, data, "verify", defaults.verify),
        passages=check_int(where, data, "passages", defaults.passages, 1, 20),
        passage_tokens=check_int(
            where, data, "passage_tokens", defaults.passage_tokens, 300,
            16000,
        ),
        answer_tokens=check_int(
            where, data, "answer_tokens", defaults.answer_tokens, 100, 1000
        ),
        max_model_calls=check_int(
            where, data, "max_model_calls", defaults.max_model_calls, 1,
            1000,
        ),
        quick_claims=check_int(
            where, data, "quick_claims", defaults.quick_claims, 1, 100
        ),
        confirmation=check_choice(
            where, data, "confirmation", defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 1000
        ),
        online_lookup=check_bool(
            where, data, "online_lookup", defaults.online_lookup
        ),
        max_online=check_int(
            where, data, "max_online", defaults.max_online, 1, 1000
        ),
    )


def check_references(
    paper: Path,
    passages: Sequence[Chunk],
    file: Path | None,
    others: Sequence[PaperIdentity],
) -> ReferenceReport:
    """Check one paper's references against the project's files.

    No language model is called.

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
        The entries, a match per entry, and the citations, without
        verdicts. Superscript citations are looked for only when ``file``
        is a PDF.

    """

    entries = tuple(
        numbered_reference_entries(
            passages, _config.REFERENCE_LAYOUT, file,
            accept_short=linkable_entry,
        )
    )
    matches = settle_duplicates(
        [
            match_reference(
                entry.text, entry.number, others, _config.MATCH_RULES
            )
            for entry in entries
        ]
    )
    superscripts = (
        superscript_runs(file, _config.CITATION_RULES)
        if file is not None and file.suffix.lower() == ".pdf"
        else None
    )
    body = reference_body(passages)
    scan = find_citations(
        body,
        max((entry.number for entry in entries), default=0),
        superscripts,
        _config.CITATION_RULES,
    )

    cited_by: dict[int, list[Sentence]] = defaultdict(list)
    for citation in scan.citations:
        for number in dict.fromkeys(citation.numbers):
            if citation.sentence not in cited_by[number]:
                cited_by[number].append(citation.sentence)

    markers: dict[Sentence, list[Citation]] = defaultdict(list)
    for citation in scan.citations:
        markers[citation.sentence].append(citation)
    sentences = split_sentences(body, min_words=1)
    before = {
        sentence: _marked(previous, markers.get(previous, ()))
        for previous, sentence in zip(sentences, sentences[1:])
        if sentence in markers
    }

    links = {
        entry.number: offline_link(entry.number, entry.text)
        for entry, match in zip(entries, matches)
        if match.status == MATCH_MISSING or (
            match.status == MATCH_UNREADABLE
            and linkable_entry(entry.text)
        )
    }
    return ReferenceReport(
        paper=paper,
        entries=entries,
        matches=matches,
        scan=scan,
        cited_by={number: tuple(found) for number, found in cited_by.items()},
        before=before,
        links=links,
    )


def report_claims(report: ReferenceReport) -> list[Claim]:
    """List the claims that can be checked against a cited file.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    list[Claim]
        For each citation in reading order, one claim per cited reference
        whose file is in the project (found or probable), each sentence
        and reference once; ``also_cited`` holds the sentence's other
        references, ``before`` the sentence before it, ``marked`` the
        sentence with its markers in brackets, and ``cited_for`` the
        phrase the reference is cited for.

    """

    files = {
        entry.number: match.path
        for entry, match in zip(report.entries, report.matches)
        if match.status in (MATCH_FOUND, MATCH_PROBABLE)
        and match.path is not None
    }
    cited: dict[Sentence, list[int]] = defaultdict(list)
    for citation in report.scan.citations:
        for number in citation.numbers:
            if number not in cited[citation.sentence]:
                cited[citation.sentence].append(number)

    markers: dict[Sentence, list[Citation]] = defaultdict(list)
    for citation in report.scan.citations:
        markers[citation.sentence].append(citation)

    claims: list[Claim] = []
    for sentence, numbers in cited.items():
        marked = _marked(sentence, markers[sentence])
        for number in numbers:
            if number in files:
                claims.append(
                    Claim(
                        number, sentence, files[number],
                        tuple(n for n in numbers if n != number),
                        report.before.get(sentence, ""),
                        marked,
                        _cited_for(sentence, markers[sentence], number),
                    )
                )
    return claims


def report_markdown(report: ReferenceReport) -> str:
    """Write the full report as Markdown.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    str
        A title, the counts, the references by status, the citation
        problems, the verdict counts and the claims needing attention, and
        every reference with its file and the sentences citing it, each
        with its verdict.

    """

    name = report.paper.name
    lines = [f"# Reference check: {name}", "", _overview(report), ""]
    for status, heading in (
        (MATCH_FOUND, "In the project"),
        (MATCH_PROBABLE, "Probably in the project"),
        (MATCH_MISSING, "Missing from the project"),
        (MATCH_UNREADABLE, "Entries too short for reliable matching"),
    ):
        listed = _listed(report, status, None)
        if listed:
            lines += [f"## {heading}", "", *listed, ""]

    if report.lookup_note:
        lines += [f"*{report.lookup_note}*", ""]
    problems = _problems(report)
    if problems:
        lines += ["## Citation problems", "", *problems, ""]

    if report.verdicts or report.verification:
        lines += ["## Claims checked against the cited papers", ""]
        lines += [*_verdict_table(report), ""]
        for index, verdict in enumerate(_attention(report, None), start=1):
            lines += [*_claim_block(report, index, verdict, None), ""]

    verdicts = {
        (verdict.claim.number, verdict.claim.sentence): verdict
        for verdict in report.verdicts
    }
    lines += ["## Every reference", ""]
    for entry, match in zip(report.entries, report.matches):
        number = "" if entry.printed else " (number inferred)"
        lines += [f"### {entry.number}.{number} {_escape(entry.text)}", ""]
        link = report.links.get(entry.number)
        if link is not None:
            lines.append(f"**Link:** {_link_markdown(link)}")
        if match.path is not None:
            lines.append(
                f"**File:** `{match.path.as_posix()}` ({match.status}: "
                f"{match.reason})"
            )
        else:
            lines.append(f"**File:** none ({match.reason})")
        sentences = report.cited_by.get(entry.number, ())
        if not sentences and report.scan.style != STYLE_NONE:
            lines += ["", "**Cited:** not found in the text"]
        for sentence in sentences:
            lines += [
                "",
                f"> {_display(report, sentence, entry.number)}",
                f"> — {_where(sentence)}",
            ]
            verdict = verdicts.get((entry.number, sentence))
            if verdict is not None:
                lines += ["", *_judgement(verdict)]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def chat_answer(
    report: ReferenceReport, max_listed: int, saved: Path | None
) -> tuple[str, tuple[Chunk, ...]]:
    """Write the chat answer.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.
    max_listed: int
        Most items listed of each kind.
    saved: pathlib.Path | None
        Where the full report was saved, relative to the project.

    Returns
    -------
    tuple[str, tuple[Chunk, ...]]
        The answer (the counts, the found, probable and missing references
        with their files, the citation problems, the verdict counts, the
        claims that are not fully supported citing their quotes as
        ``[n]``, and where the full report is), and the quoted passages,
        numbered from one in the order cited.

    """

    name = report.paper.name
    if not report.entries:
        return (
            f"I found no reference list in {name}: no “References” (or "
            "“Bibliography”) heading followed by entries.",
            (),
        )

    lines = [f"**Reference check of {name}**", "", _overview(report)]
    for status, heading in (
        (MATCH_FOUND, "In the project"),
        (MATCH_PROBABLE, "Probably in the project, please check"),
        (MATCH_MISSING, "Missing from the project"),
        (MATCH_UNREADABLE, "Too short for local matching; search possible"),
    ):
        listed = _listed(report, status, max_listed)
        if listed:
            lines += ["", f"**{heading}:**", "", *listed]

    problems = _problems(report)
    if problems:
        lines += ["", "**Citation problems:**", "", *problems]

    evidence: list[Chunk] = []
    if report.verdicts or report.verification:
        lines += ["", "**Claims checked against the cited papers**", ""]
        lines += _verdict_table(report)
        attention = _attention(report, max_listed)
        for index, verdict in enumerate(attention, start=1):
            citation = None
            if verdict.quote_chunk is not None:
                if verdict.quote_chunk not in evidence:
                    evidence.append(verdict.quote_chunk)
                citation = evidence.index(verdict.quote_chunk) + 1
            lines += ["", *_claim_block(report, index, verdict, citation)]
        hidden = len(_attention(report, None)) - len(attention)
        if hidden > 0:
            claims = "claim" if hidden == 1 else "claims"
            lines += ["", f"*… and {hidden} more {claims} in the report.*"]

    if report.lookup_note:
        lines += ["", f"*{report.lookup_note}*"]
    if saved is not None:
        lines += [
            "",
            f"The full report, with every reference, the sentences citing "
            f"it and their verdicts, is in `{saved.as_posix()}`.",
        ]
    return "\n".join(lines), tuple(evidence)


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
        The chat answer, the quoted passages of the cited works as
        evidence, and a summary of the counts.

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
    link_request = bool(_LINK_REQUEST.search(request.question))
    if settings.online_lookup or link_request:
        report = _lookup(report, settings, context)
    if settings.verify and not link_request:
        report = _verify(report, by_file, settings, context)

    saved = (
        save_output_file(
            context.project_root,
            f"{path.stem}_reference_check.md",
            report_markdown(report),
        )
        if report.entries
        else None
    )

    verdicts = Counter(verdict.verdict for verdict in report.verdicts)
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
            "claims_checked": len(report.verdicts),
            "verdicts": {v: verdicts[v] for v in VERDICTS if verdicts[v]},
            "verification": report.verification,
            "links": {
                str(number): {
                    "url": link.url, "kind": link.kind,
                    "doi": link.doi, "pdf_url": link.pdf_url,
                    "license": link.license, "version": link.version,
                }
                for number, link in report.links.items()
            },
            "lookup_note": report.lookup_note,
            "report": saved.as_posix() if saved else None,
        }
    )

    answer, quoted = chat_answer(report, settings.max_listed, saved)
    context.done(
        f"Checked {len(report.entries)} references of {path.name}",
        evidence_count=len(quoted),
    )
    if guessed:
        answer += (
            f"\n\n*This is about {path.name}; name the paper if you "
            "meant another one.*"
        )
    evidence = tuple(
        {
            "citation": number,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "text": chunk.text,
            "cell": chunk.cell,
        }
        for number, chunk in enumerate(quoted, start=1)
    )
    return ToolResult(answer, evidence, summary)



def _lookup(
    report: ReferenceReport,
    settings: ReferenceCheckSettings,
    context: ToolContext,
) -> ReferenceReport:
    """Offer online lookup for missing entries, never silently granting it.

    Parameters
    ----------
    report: ReferenceReport
        Local reference check, including offline links.
    settings: ReferenceCheckSettings
        Limits for a single optional lookup operation.
    context: ToolContext
        Provides the interactive approval callback and tracing.

    Returns
    -------
    ReferenceReport
        With resolved links when approved, otherwise offline links.

    """

    missing = [
        entry for entry in report.entries
        if entry.number in report.links
    ][:settings.max_online]
    if not missing:
        return report
    if context.confirm is None:
        return replace(
            report, lookup_note=(
                "Online lookup was not performed: explicit interactive "
                "approval was unavailable. Search links work offline."
            ),
        )

    quick = min(_QUICK_LOOKUPS, len(missing))
    plan = ToolPlan(
        tool="reference_check",
        summary=(
            f"Look up {len(missing)} locally unavailable references "
            "online? "
            "This sends their bibliography entries to Crossref and "
            "their verified DOIs to OpenAlex. No PDFs are downloaded."
        ),
        scope=(
            "Only locally unavailable references of " + report.paper.name +
            "; services: api.crossref.org and api.openalex.org. "
            "Approval applies to this operation only."
        ),
        items=len(missing), batches=len(missing),
        batches_to_run=len(missing), estimated_seconds=0,
        quick_items=quick, quick_seconds=0,
        quick_label=f"Look up first {quick}",
        operation="network",
    )
    choice, asked = confirm_plan(
        plan, CONFIRMATION_ALWAYS, 0, context
    )
    if not asked or choice == PLAN_CHOICE_CANCEL:
        return replace(
            report, lookup_note="Online lookup declined; offline links shown."
        )
    chosen = missing[:quick] if choice == PLAN_CHOICE_QUICK else missing
    records = {entry.number: entry.text for entry in chosen}
    permission = LookupPermission(tuple(records))
    context.progress(f"Resolving {len(records)} references online")
    resolved = resolve_links(records, permission)
    context.trace.emit(
        "tool.reference_check.lookup",
        component="tool",
        data={
            "approved": True, "services": list(permission.services),
            "references": len(records),
            "verified": sum(
                link.kind == "verified_doi" for link in resolved.values()
            ),
        },
        turn_id=context.trace_turn_id,
    )
    note = (
        f"Online lookup approved for {len(records)} references; "
        "links are not proof of download permission."
    )
    remaining = len(report.links) - len(chosen)
    if remaining:
        note += f" {remaining} references retain offline links."
    return replace(
        report, links={**report.links, **resolved}, lookup_note=note
    )


def _link_markdown(link: ReferenceLink) -> str:
    """Show the preferred publication or search link without ambiguity.

    Parameters
    ----------
    link: ReferenceLink
        Publication or search location for a reference.

    Returns
    -------
    str
        Markdown link with a label identifying its confidence.

    """

    label = {
        "search": "Search Google Scholar",
        "printed_doi": "DOI (as printed; unverified)",
        "verified_doi": "Crossref-matched publication",
    }.get(link.kind, "Search publication")
    return f"[{label}]({link.url})"


def _verify(
    report: ReferenceReport,
    by_file: dict[Path, list[Chunk]],
    settings: ReferenceCheckSettings,
    context: ToolContext,
) -> ReferenceReport:
    """Check the report's claims with the answering model, after a plan.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper, without verdicts.
    by_file: dict[pathlib.Path, list[Chunk]]
        Every file's passages.
    settings: ReferenceCheckSettings
        For the passages, limits and confirmation policy.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ReferenceReport
        With the verdicts of the claims checked, and ``verification``
        saying why none or not all were checked.

    """

    claims = report_claims(report)
    if not claims:
        return report

    limit = min(len(claims), settings.max_model_calls)
    quick = min(len(claims), settings.quick_claims)
    files = len({claim.cited for claim in claims})
    plan = ToolPlan(
        tool="reference_check",
        summary=(
            f"Check {len(claims)} claims of {report.paper.name} against "
            f"the {files} cited papers in the project, one model call each."
        ),
        scope=(
            f"Sentences citing a reference whose file is in the project; "
            f"{settings.passages} passages of the cited paper per claim."
        ),
        items=len(claims),
        batches=len(claims),
        batches_to_run=limit,
        estimated_seconds=limit * _claim_seconds(settings),
        quick_items=quick,
        quick_seconds=quick * _claim_seconds(settings),
        quick_label=f"The first {quick} claims",
    )
    choice, _asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    if choice == PLAN_CHOICE_CANCEL:
        return replace(
            report,
            verification=(
                f"The {len(claims)} claims were not checked (cancelled)."
            ),
        )

    to_check = claims[: quick if choice == PLAN_CHOICE_QUICK else limit]
    verdicts: list[Verdict] = []
    for index, claim in enumerate(to_check, start=1):
        context.progress(
            f"Checking claim {index} of {len(to_check)} "
            f"(reference [{claim.number}])"
        )
        plain = _UNMARKED.sub("", claim.marked or claim.sentence.text)
        passages = claim_passages(
            (plain, claim.cited_for),
            by_file.get(claim.cited, []),
            context.rank,
            settings.passages,
            settings.passage_tokens,
        )
        verdict = verify_claim(
            claim, passages, context.ask_model, settings.answer_tokens
        )
        verdicts.append(verdict)
        context.trace.emit(
            "tool.reference_check.claim",
            component="tool",
            data={
                "reference": claim.number,
                "cited": claim.cited.as_posix(),
                "line": claim.sentence.line,
                "verdict": verdict.verdict,
                "quote_found": verdict.quote_chunk is not None,
            },
            turn_id=context.trace_turn_id,
        )

    unchecked = len(claims) - len(to_check)
    note = ""
    if unchecked:
        why = (
            "the quick version" if choice == PLAN_CHOICE_QUICK
            else "the model-call limit"
        )
        note = f"{unchecked} of {len(claims)} claims were not checked ({why})."
    return replace(report, verdicts=tuple(verdicts), verification=note)


def _claim_seconds(settings: ReferenceCheckSettings) -> float:
    """Return the estimated time of one claim check, in seconds.

    Parameters
    ----------
    settings: ReferenceCheckSettings
        For the passage and answer budgets.

    Returns
    -------
    float
        A planning estimate, not a limit.

    """

    prompt = _config.PROMPT_OVERHEAD_TOKENS + settings.passage_tokens
    reply = settings.answer_tokens * _config.REPLY_SHARE
    return (
        _config.CALL_SECONDS
        + prompt / _config.PROMPT_TOKENS_PER_SECOND
        + reply / _config.REPLY_TOKENS_PER_SECOND
    )


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
        style = {
            STYLE_BRACKETS: "bracketed",
            STYLE_PARENTHESES: "parenthesized",
            STYLE_WRITTEN: "written-out",
        }.get(report.scan.style, "superscript")
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
        "- **12.** entry" items (with the file, and for a probable match
        the reason), then a line saying how many more there are. Numbers
        are never written "[12]", which would read as a link to evidence.

    """

    items = [
        (entry, match)
        for entry, match in zip(report.entries, report.matches)
        if match.status == status and (
            status != MATCH_UNREADABLE
            or limit is None or entry.number in report.links
        )
    ]
    shown = items if limit is None else items[:limit]
    lines = []
    for entry, match in shown:
        line = f"- **{entry.number}.** {_escape(entry.text)}"
        if status == MATCH_FOUND and match.path is not None:
            line += f" → `{match.path.as_posix()}`"
        elif status == MATCH_PROBABLE and match.path is not None:
            line += f" → `{match.path.as_posix()}` ({match.reason})"
        if entry.number in report.links:
            line += f" — {_link_markdown(report.links[entry.number])}"
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
            + ", ".join(str(number) for number in uncited)
        )
    if report.scan.beyond_list:
        problems.append(
            "- Cited but not in the reference list: "
            + ", ".join(str(number) for number in report.scan.beyond_list)
            + " (the list may have been split wrongly)"
        )
    return problems


def _verdict_table(report: ReferenceReport) -> list[str]:
    """Return the verdict counts as one line.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.

    Returns
    -------
    list[str]
        "**7 claims checked:** 2 supported · 4 partly supported · 1 not
        found", then why claims were not checked, if any were not.

    """

    counts = Counter(verdict.verdict for verdict in report.verdicts)
    lines = []
    if report.verdicts:
        listed = " · ".join(
            f"{counts[verdict]} {verdict_label(verdict)}"
            for verdict in VERDICTS if counts[verdict]
        )
        claims = "claim" if len(report.verdicts) == 1 else "claims"
        lines.append(f"**{len(report.verdicts)} {claims} checked:** {listed}")
    if report.verification:
        lines += ["", f"*{report.verification}*"] if lines else [
            f"*{report.verification}*"
        ]
    return lines


def _attention(
    report: ReferenceReport, limit: int | None
) -> list[Verdict]:
    """Return the checked claims, those needing attention first.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper.
    limit: int | None
        Most verdicts; None returns all.

    Returns
    -------
    list[Verdict]
        Contradicted, not supported, partly supported, not found, unclear
        and failed claims, then supported and background ones, in reading
        order within each verdict.

    """

    ordered = sorted(
        report.verdicts,
        key=lambda v: _ORDER.get(v.verdict, len(_ORDER)),
    )
    return ordered if limit is None else ordered[:limit]


def _claim_block(
    report: ReferenceReport,
    index: int,
    verdict: Verdict,
    citation: int | None,
) -> list[str]:
    """Return one claim needing attention as a Markdown block.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper, for the sentence's citation markers.
    index: int
        The claim's number in the list.
    verdict: Verdict
        The judged claim.
    citation: int | None
        The evidence number of its quote in the chat; None in the report.

    Returns
    -------
    list[str]
        A heading line (number, verdict, reference and cited file), the
        citing sentence as a quotation with the cited marker in bold and
        its place, the comment, and the cited paper's sentence.

    """

    claim = verdict.claim
    lines = [
        f"**{index}. {verdict_label(verdict.verdict).capitalize()}** · "
        "reference "
        f"{claim.number} → `{claim.cited.name}`",
        f"> {_display(report, claim.sentence, claim.number)}",
        f"> — {report.paper.name}, {_where(claim.sentence)}",
    ]
    if verdict.comment:
        lines += ["", f"*Why:* {_comment(verdict.comment)}"]
    if verdict.quote:
        cited = f" [{citation}]" if citation is not None else ""
        lines += [
            "",
            f"*The cited paper says:* “{_escape(verdict.quote)}”{cited}",
        ]
    return lines


def _judgement(verdict: Verdict) -> list[str]:
    """Return a verdict for the report's reference section.

    Parameters
    ----------
    verdict: Verdict
        The judged claim.

    Returns
    -------
    list[str]
        "**Verdict:** supported — comment", then the cited paper's
        sentence with its file and place.

    """

    line = f"**Verdict:** {verdict_label(verdict.verdict)}"
    if verdict.comment:
        line += f" — {_comment(verdict.comment)}"
    lines = [line]
    if verdict.quote and verdict.quote_chunk is not None:
        chunk = verdict.quote_chunk
        place = (
            f"{chunk.section}, lines {chunk.start_line}-{chunk.end_line}"
            if chunk.section else f"lines {chunk.start_line}-{chunk.end_line}"
        )
        lines += [
            "",
            f"*The cited paper says:* “{_escape(verdict.quote)}” "
            f"({chunk.path.name}, {place})",
        ]
    return lines


def _marked(sentence: Sentence, citations: Sequence[Citation]) -> str:
    """Return a sentence with its markers in brackets, for the model.

    Parameters
    ----------
    sentence: Sentence
        A citing sentence.
    citations: Sequence[Citation]
        Its citations.

    Returns
    -------
    str
        The sentence with each superscript or parenthesized marker written
        " [5,40,88–91]" where it stands (extraction glues superscripts to
        the word: "method5,40,88–91"); bracketed and written-out markers
        are kept as they are.

    """

    text = sentence.text
    parts: list[str] = []
    position = 0
    for citation in sorted(citations, key=lambda c: c.start):
        if citation.start < position or citation.end <= citation.start:
            continue
        shown = text[citation.start:citation.end]
        if shown[:1] != "[" and not shown[:1].isalpha():
            numbers = shown.strip("()").replace(" ", "")
            shown = f" [{numbers}]"
            parts.append(text[position:citation.start].rstrip())
        else:
            parts.append(text[position:citation.start])
        parts.append(shown)
        position = citation.end
    parts.append(text[position:])
    return _BROKEN_WORD.sub("", "".join(parts))


def _cited_for(
    sentence: Sentence, citations: Sequence[Citation], number: int
) -> str:
    """Return the words a reference is cited for.

    Parameters
    ----------
    sentence: Sentence
        A citing sentence.
    citations: Sequence[Citation]
        Its citations.
    number: int
        The reference.

    Returns
    -------
    str
        The text before the reference's first marker, after the previous
        marker and the last comma, semicolon, colon or opening
        parenthesis, after the last word such as "using", "with" or "such
        as", and without a leading "and", "or" or "but"; at most
        ``_CITED_FOR_WORDS`` words. Empty when the reference has no marker
        in the sentence, no words precede it, or they only attribute the
        sentence ("already observed in ..."), so the whole sentence is
        what it is cited for.

    """

    ordered = sorted(citations, key=lambda c: c.start)
    marker = next((c for c in ordered if number in c.numbers), None)
    if marker is None:
        return ""

    start = max(
        (c.end for c in ordered if c.end <= marker.start), default=0
    )
    phrase = sentence.text[start:marker.start]
    clause = re.split(r"[,;:(]", phrase)[-1]
    if _ATTRIBUTION.match(clause.strip()):
        return ""
    if len(clause.split()) >= 2:
        phrase = clause
    phrase = _CITED_FOR_START.sub("", phrase) or phrase
    phrase = re.sub(r"^\s*(?:and|or|but)\s+", "", phrase, flags=re.I)
    words = _BROKEN_WORD.sub("", phrase).split()[-_CITED_FOR_WORDS:]
    return " ".join(words).strip(" .")


def _display(
    report: ReferenceReport, sentence: Sentence, number: int
) -> str:
    """Return a citing sentence for display.

    Parameters
    ----------
    report: ReferenceReport
        The checked paper, for the sentence's citation markers.
    sentence: Sentence
        The citing sentence.
    number: int
        The reference whose marker is shown in bold.

    Returns
    -------
    str
        The sentence with its numeric markers written as superscripts
        ("NN⁸²", "⁽¹³⁾" for a bracketed or parenthesized "[13]"), the
        markers citing ``number`` in bold, words broken across lines
        joined, and Markdown characters escaped. A sentence longer than
        ``_SHOWN_CHARACTERS`` is cut, at spaces, to that length around the
        first bold marker, with "…" where text is left out.

    """

    text = sentence.text
    citations: list[Citation] = sorted(
        (c for c in report.scan.citations if c.sentence == sentence),
        key=lambda citation: citation.start,
    )
    first, last = 0, len(text)
    if len(text) > _SHOWN_CHARACTERS:
        bold = next(
            (c for c in citations if number in c.numbers), None
        )
        centre = bold.start if bold is not None else 0
        first = max(0, centre - _SHOWN_CHARACTERS // 2)
        last = min(len(text), first + _SHOWN_CHARACTERS)
        first = max(0, last - _SHOWN_CHARACTERS)
        if first:
            first = text.find(" ", first) + 1 or first
        if last < len(text):
            last = text.rfind(" ", first, last) if " " in text[
                first:last
            ] else last
    parts: list[str] = ["… "] if first else []
    position = first
    for citation in citations:
        if citation.start < first or citation.end > last:
            continue
        if citation.start < position or citation.end <= citation.start:
            continue
        parts.append(
            _escape(_BROKEN_WORD.sub("", text[position:citation.start]))
        )
        shown = sentence.text[citation.start:citation.end]
        if shown[:1] in "[(":
            inner = shown[1:-1].replace(" ", "")
            shown = f"⁽{inner.translate(_SUPERSCRIPT)}⁾"
        elif shown[:1].isdigit():
            shown = shown.replace(" ", "").translate(_SUPERSCRIPT)
        else:
            shown = _escape(shown)
        parts.append(f"**{shown}**" if number in citation.numbers else shown)
        position = citation.end
    parts.append(_escape(_BROKEN_WORD.sub("", text[position:last])))
    if last < len(text):
        parts.append(" …")
    return "".join(parts)


def _escape(text: str) -> str:
    """Return text that cannot be read as links to evidence.

    The chat reads "[12]" as a link to evidence 12 and shows backslash
    escapes as written, so bracketed numbers are rewritten instead.

    Parameters
    ----------
    text: str
        Any text from a paper.

    Returns
    -------
    str
        With bracketed numbers ("[20, 21]") written as superscripts in
        parentheses ("⁽²⁰,²¹⁾").

    """

    return _NUMBERED.sub(
        lambda match: "⁽"
        + match.group(1).replace(" ", "").translate(_SUPERSCRIPT)
        + "⁾",
        text,
    )


def _comment(text: str) -> str:
    """Return a model comment that cannot be read as links to evidence.

    Parameters
    ----------
    text: str
        The model's comment, which names references like "[40]".

    Returns
    -------
    str
        With bracketed numbers written as "ref. 40" or "refs. 5, 40".

    """

    return _NUMBERED.sub(_reference_words, text)


def _reference_words(match: re.Match[str]) -> str:
    """Return bracketed reference numbers in words.

    Parameters
    ----------
    match: re.Match[str]
        A ``_NUMBERED`` match ("[40]", "[5, 88–91]").

    Returns
    -------
    str
        "ref. 40", or "refs. 5,88–91" for several.

    """

    numbers = match.group(1).replace(" ", "")
    plural = "," in numbers or "–" in numbers or "-" in numbers
    return f"ref{'s' if plural else ''}. {numbers}"


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
