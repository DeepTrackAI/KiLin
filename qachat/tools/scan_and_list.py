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
>>> from qachat.tools.scan_and_list import quoted_terms, term_pattern
>>> quoted_terms('list all mentions of "XYZ 2"')
('XYZ 2',)
>>> bool(term_pattern("XYZ 2", False, True).search("we used xyz-2"))
True

"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final

from qachat.chunking import Chunk
from qachat.retrievals.document_reading import dominant_path
from qachat.tools.choose_document import document_scope
from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_bool,
    check_int,
    reject_unknown,
)
from qachat.tools.long_lists import CHAT_LIST_LIMIT, file_name, offer_export

__all__ = [
    "MAX_TERMS",
    "Mention",
    "MentionCollection",
    "MentionScan",
    "SCAN_SETTING_NAMES",
    "ScanSettings",
    "collect_mentions",
    "document_phrase",
    "mention_evidence",
    "mention_location",
    "mention_report",
    "mention_snippet",
    "parse_settings",
    "quoted_terms",
    "run",
    "scan_fields",
    "scan_mentions",
    "short_names",
    "term_pattern",
    "terms_prompt",
    "validated_document",
    "validated_terms",
]

# More terms would make one answer hard to read; ask again for others.
MAX_TERMS: Final[int] = 5

# Settings shared by tools that collect mentions.
SCAN_SETTING_NAMES: Final[tuple[str, ...]] = (
    "case_sensitive",
    "whole_words",
    "context_lines",
    "max_listed",
)

# Phrases in double quotes, curly quotes, or backticks. Single quotes are
# not used because they also serve as apostrophes.
_QUOTED: Final[re.Pattern[str]] = re.compile(
    r'"([^"\n]{1,80})"|“([^”\n]{1,80})”|`([^`\n]{1,80})`'
)

# "the Smith paper", "the 2020 review", "smith20deep.pdf" style references
# to one document, used when the model names none.
_DOCUMENT_PHRASE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:in|from|of|within|inside)\s+(the\s+(?:[\w.-]+\s+){1,4}?"
    r"(?:paper|article|document|report|review|manuscript|thesis|book|"
    r"chapter|tutorial|guide|file|readme|notebook|script))\b",
    flags=re.I,
)

# Quotes and punctuation a model may leave around a term it copied.
_TERM_EDGES: Final[str] = " '\"`“”?.,;:"

# Router-model reply budgets: terms and a document, or also categories.
_TERMS_REPLY_TOKENS: Final[int] = 96
_CATEGORIES_REPLY_TOKENS: Final[int] = 160

# Files without searchable text named in the coverage line.
_UNSEARCHABLE_EXAMPLES: Final[int] = 5

# Extra field of the terms prompt when ``batch_classify`` needs categories.
_CATEGORY_FIELD: Final[str] = """
"categories": 2 to 6 short labels for the groups the user wants each mention
sorted into, taken from the message (for example "own code" and "citation
of other work")."""


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

    case_sensitive: bool = False
    whole_words: bool = True
    context_lines: int = 1
    max_listed: int = 100


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
    """Validate the settings every mention-collecting tool shares.

    Parameters
    ----------
    data: dict
        Tool settings.
    where: str
        Location for errors.

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

    defaults = ScanSettings()
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


@dataclass(frozen=True, slots=True)
class Mention:
    """One occurrence of a term.

    Attributes
    ----------
    term: str
        The searched term the occurrence matched.
    text: str
        The occurrence as written in the source.
    path: pathlib.Path
        Source file.
    section: str | None
        Section of the unit, such as ``"Page 3"`` for a PDF page.
    cell: int | None
        Notebook cell, when the source is a notebook.
    symbol: str | None
        Code symbol around the occurrence, when known.
    line: int
        First line of the occurrence, in the unit's numbering.
    end_line: int
        Last line of the occurrence (later than ``line`` when a phrase
        continues on the next line).
    context_start: int
        First line of the surrounding context.
    context: str
        Exact source lines from ``context_start`` through the context
        after ``end_line``, joined with newlines.
    offset: int
        Where the occurrence starts within ``context``.

    """

    term: str
    text: str
    path: Path
    section: str | None
    cell: int | None
    symbol: str | None
    line: int
    end_line: int
    context_start: int
    context: str
    offset: int

    @property
    def context_end(self) -> int:
        """Last line of the surrounding context.

        Returns
        -------
        int
            Last line number covered by the stored context text.

        """

        return self.context_start + self.context.count("\n")

    def evidence(self, citation: int) -> dict[str, object]:
        """Return the evidence fields that cite this mention.

        Parameters
        ----------
        citation: int
            The number the answer cites the mention with, from one.

        Returns
        -------
        dict[str, object]
            ``citation``, ``path``, ``start_line``, ``end_line``,
            ``symbol``, ``text`` (the exact context lines, so a preview can
            locate them), and ``cell``.

        """

        return {
            "citation": citation,
            "path": self.path,
            "start_line": self.context_start,
            "end_line": self.context_end,
            "symbol": self.symbol,
            "text": self.context,
            "cell": self.cell,
        }


@dataclass(frozen=True, slots=True)
class MentionScan:
    """All mentions found, and what was scanned.

    Attributes
    ----------
    terms: tuple[str, ...]
        Searched terms.
    mentions: tuple[Mention, ...]
        Every occurrence, in index order (file, then position).
    files_scanned: tuple[pathlib.Path, ...]
        Files whose indexed text was scanned.
    case_sensitive: bool
        Whether letter case had to match.
    whole_words: bool
        Whether matches had to be whole words.

    """

    terms: tuple[str, ...]
    mentions: tuple[Mention, ...]
    files_scanned: tuple[Path, ...]
    case_sensitive: bool
    whole_words: bool

    def counts(self) -> dict[Path, int]:
        """Return mentions per file, in index order, files with any.

        Returns
        -------
        dict[Path, int]
            mentions per file, in index order, files with any.

        """

        totals: dict[Path, int] = {}
        for mention in self.mentions:
            totals[mention.path] = totals.get(mention.path, 0) + 1

        return totals

    def term_counts(self) -> dict[str, int]:
        """Return mentions per term, in the order the terms were given.

        Returns
        -------
        dict[str, int]
            mentions per term, in the order the terms were given.

        """

        totals = {term: 0 for term in self.terms}
        for mention in self.mentions:
            totals[mention.term] += 1

        return totals


@dataclass(frozen=True, slots=True)
class MentionCollection:
    """Every mention a question asks about, and how they were found.

    Attributes
    ----------
    scan: MentionScan
        Every mention in scope.
    scope_note: str | None
        How the scanned files were chosen, when not all were scanned.
    unsearchable: tuple[pathlib.Path, ...]
        Source files without indexed text (only for a full scan).
    summary: dict[str, object]
        Terms, scope, and counts, saved with the turn.
    reply: str
        The router model's extraction reply (it may also carry
        categories).

    """

    scan: MentionScan
    scope_note: str | None
    unsearchable: tuple[Path, ...]
    summary: dict[str, object]
    reply: str


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
        QAChat's services.

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


def collect_mentions(
    settings: ScanSettings,
    request: ToolRequest,
    context: ToolContext,
    *,
    tool: str,
    categories: bool = False,
) -> MentionCollection | ToolResult:
    """Find the terms and scope in the question, then every mention.

    Parameters
    ----------
    settings: ScanSettings
        Matching settings (``batch_classify`` settings work too).
    request: ToolRequest
        The question and the user's words.
    context: ToolContext
        QAChat's services.
    tool: str
        Name of the calling tool, for summaries and progress.
    categories: bool, optional
        Also ask the router model for categories. Defaults to ``False``.

    Returns
    -------
    MentionCollection | ToolResult
        The mentions, or a result that explains why nothing could be
        scanned (no exact term found).

    Notes
    -----
    One router-model call names the terms and any one document to
    search. Quoted terms take precedence, and a term or document phrase is
    used only when it occurs in the user's words. The scope is the files
    the question names; otherwise the file that dominates a search for the
    document phrase; otherwise every indexed file. A term that names a
    file in the question becomes the scope when other terms remain.

    """

    trace = context.trace
    context.progress("Finding the terms to scan")

    # Terms come from the question itself; earlier questions are used only
    # when it names none ("list every mention of it").
    words = request.question
    reply = _ask_terms(context, words, categories)
    quoted = quoted_terms(request.question)
    terms = quoted or validated_terms(reply, words)
    if not terms and request.text != request.question:
        words = request.text
        reply = _ask_terms(context, words, categories)
        terms = validated_terms(reply, words)

    source = "quoted" if quoted else "router"
    document = validated_document(reply, words) or document_phrase(
        request.question
    )
    extracted: dict[str, object] = {
        "terms": list(terms),
        "from": source,
        "document": document,
    }
    if trace.include_content:
        extracted["raw"] = reply
    trace.emit(
        "tool.terms.extracted",
        component="tool",
        data=extracted,
        turn_id=context.trace_turn_id,
    )
    summary: dict[str, object] = {
        "name": tool,
        "terms": list(terms),
        "terms_from": source,
    }
    if not terms:
        context.done("Nothing to scan for", evidence_count=0)
        return ToolResult(
            "I couldn't tell which exact words to look for. Put them in "
            'double quotes, for example: list all mentions of "gradient '
            'descent".',
            (),
            summary,
        )

    indexed = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    named, scoped_by = document_scope(request, context.ask_router, indexed)
    file_terms = [
        term
        for term in terms
        if any(_names_file(term, path) for path in named)
    ]
    if file_terms and len(file_terms) < len(terms):
        # "MRI in smith2020": the file is the scope.
        terms = tuple(term for term in terms if term not in file_terms)
        summary["terms"] = list(terms)
        summary["terms_as_scope"] = file_terms

    # A file named by the only terms is something to find instead.
    scope = [
        path
        for path in named
        if not any(_names_file(term, path) for term in terms)
    ]
    scope_note = None
    if scope:
        scope_note = (
            f"Scanned only {', '.join(path.name for path in scope)}, "
            f"{scoped_by}."
        )
        if scoped_by != "named in the question":
            scope_note += " Name the file to scan a different one."
    elif document:
        # A described document ("the Smith paper") is the file that
        # dominates a search for the description.
        path = dominant_path(
            context.rank(document)[: context.retrieval_count]
        )
        if path is not None:
            scope = [path]
            scope_note = (
                f"Scanned only {path.name}, the best match for "
                f"“{document}”. Name the file to scan a different one."
            )

    listed = " or ".join(f"“{term}”" for term in terms)
    context.progress(
        f"Scanning {scope[0].name} for {listed}"
        if len(scope) == 1
        else f"Scanning {len(scope) or len(indexed)} files for {listed}"
    )
    with trace.span(
        "tool.scan",
        component="tool",
        data={"terms": list(terms), "scope_files": len(scope) or len(indexed)},
        turn_id=context.trace_turn_id,
    ):
        scan = scan_mentions(
            context.chunks,
            terms,
            case_sensitive=settings.case_sensitive,
            whole_words=settings.whole_words,
            context_lines=settings.context_lines,
            paths=scope or None,
        )

    unsearchable = () if scope else tuple(context.unsearchable(indexed))
    summary.update(
        {
            "total_mentions": len(scan.mentions),
            "mentions_per_term": scan.term_counts(),
            "files_with_mentions": len(scan.counts()),
            "files_scanned": len(scan.files_scanned),
            "scope": [path.as_posix() for path in scope],
            "scope_document": document,
            "unsearchable_files": [path.as_posix() for path in unsearchable],
            "case_sensitive": scan.case_sensitive,
            "whole_words": scan.whole_words,
        }
    )
    return MentionCollection(scan, scope_note, unsearchable, summary, reply)


def quoted_terms(text: str) -> tuple[str, ...]:
    """Return phrases the user put in double quotes or backticks.

    Parameters
    ----------
    text: str
        The question.

    Returns
    -------
    tuple[str, ...]
        Distinct quoted phrases, at most ``MAX_TERMS``, in order. Single
        quotes are ignored because they also serve as apostrophes.

    """

    found = [
        " ".join(next(group for group in match.groups() if group).split())
        for match in _QUOTED.finditer(text)
    ]
    return tuple(dict.fromkeys(term for term in found if term))[:MAX_TERMS]


def terms_prompt(
    text: str, *, categories: bool = False
) -> tuple[str, str]:
    """Return the system and user prompts that ask a model for terms.

    Parameters
    ----------
    text: str
        The user's message (with earlier questions it refers to).
    categories: bool, optional
        Also ask for the categories the user wants mentions sorted into
        (for ``batch_classify``). Defaults to ``False``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt. The reply is checked with
        ``validated_terms``, ``validated_document``, and (for categories)
        ``qachat.tools.batch_classify.validated_categories``.

    """

    system = (
        "You pick search terms. You never answer questions. You reply with "
        "one line of JSON only."
    )
    user = f"""Message: {text}

Return one line of JSON with:
"terms": the exact words or phrases (at most {MAX_TERMS}) whose every mention
the user wants listed or counted, copied exactly from the message. Use the
shortest phrase that names the thing; leave out words such as "mentions",
"all", "times", "paper", or "file".
"document": the words naming one particular document to search, copied
exactly from the message (for example "the Smith paper"), or null when all
sources should be searched.{_CATEGORY_FIELD if categories else ""}

Examples:
"list all mentions of gradient descent" -> \
{{"terms": ["gradient descent"], "document": null}}
"how often does the Smith paper say MRI?" -> \
{{"terms": ["MRI"], "document": "the Smith paper"}}
"go through that paper and tell me if MRI is mentioned" -> \
{{"terms": ["MRI"], "document": "that paper"}}"""
    if categories:
        user += """
"which uses of MRI in the Smith paper are clinical and which are research?" \
-> {"terms": ["MRI"], "document": "the Smith paper", "categories": \
["clinical use", "research use"]}"""

    return system, user


def validated_terms(reply: str, text: str) -> tuple[str, ...]:
    """Accept a model's suggested terms only when the user wrote them.

    Parameters
    ----------
    reply: str
        Model reply containing JSON such as ``{"terms": ["MRI"]}``.
    text: str
        The user's text the terms must come from.

    Returns
    -------
    tuple[str, ...]
        Distinct terms that occur in ``text`` (ignoring case and spacing),
        at most ``MAX_TERMS``; empty when the reply is unusable.

    """

    value = _reply_json(reply)
    if not isinstance(value.get("terms"), list):
        return ()

    source = " ".join(text.split()).casefold()
    accepted: list[str] = []
    for item in value["terms"]:
        if not isinstance(item, str):
            continue
        term = " ".join(item.split()).strip(_TERM_EDGES)
        if term and term.casefold() in source and term not in accepted:
            accepted.append(term)

    return tuple(accepted[:MAX_TERMS])


def validated_document(reply: str, text: str) -> str | None:
    """Accept a model's document phrase only when the user wrote it.

    Parameters
    ----------
    reply: str
        Model reply containing JSON such as ``{"document": "the Smith
        paper"}``.
    text: str
        The user's text the phrase must come from.

    Returns
    -------
    str | None
        The phrase, when it occurs in ``text`` (ignoring case and
        spacing); otherwise None.

    """

    phrase = _reply_json(reply).get("document")
    if not isinstance(phrase, str):
        return None

    phrase = " ".join(phrase.split()).strip(_TERM_EDGES)
    source = " ".join(text.split()).casefold()
    return phrase if phrase and phrase.casefold() in source else None


def document_phrase(text: str) -> str | None:
    """Find words naming one document, such as "the Smith paper".

    Parameters
    ----------
    text: str
        The user's words.

    Returns
    -------
    str | None
        The first phrase like "in the Smith paper" (without "in"), or None.
        Used when the model names no document.

    """

    match = _DOCUMENT_PHRASE.search(text)
    return " ".join(match.group(1).split()) if match else None


def term_pattern(
    term: str, case_sensitive: bool, whole_words: bool
) -> re.Pattern[str]:
    """Match a term flexibly: "XYZ2", "XYZ 2", "xyz-2" are one term.

    The term is split into runs of letters, runs of digits, and runs of
    other characters; between these pieces, spaces, hyphens, and
    underscores are optional.

    Parameters
    ----------
    term: str
        The term as the user wrote it.
    case_sensitive: bool
        Whether letter case must match.
    whole_words: bool
        Whether the match must not continue into adjacent word characters
        (checked only at an end of the term that is a word character).

    Returns
    -------
    re.Pattern[str]
        Compiled pattern for the term.

    """

    pieces = re.findall(r"[^\W\d_]+|\d+|[^\w\s-]+", term)
    body = r"[\s_-]*".join(re.escape(piece) for piece in pieces)
    if whole_words:
        if re.match(r"\w", term):
            body = rf"(?<!\w){body}"
        if re.search(r"\w$", term):
            body = rf"{body}(?!\w)"

    return re.compile(body, 0 if case_sensitive else re.IGNORECASE)


def scan_mentions(
    chunks: Sequence[Chunk],
    terms: Sequence[str],
    *,
    case_sensitive: bool = False,
    whole_words: bool = True,
    context_lines: int = 1,
    paths: Sequence[Path] | None = None,
) -> MentionScan:
    """Find every occurrence of the terms in the indexed text.

    Parameters
    ----------
    chunks: Sequence[Chunk]
        All indexed chunks, in index order.
    terms: Sequence[str]
        Terms to find; whitespace inside a term matches any whitespace,
        including line breaks.
    case_sensitive: bool, optional
        Whether letter case must match. Defaults to ``False``.
    whole_words: bool, optional
        Whether a term must not continue into adjacent letters or digits
        (so "MRI" does not match "MRIs"). Defaults to ``True``.
    context_lines: int, optional
        Lines of context kept before and after each occurrence. Defaults
        to ``1``.
    paths: Sequence[pathlib.Path] | None, optional
        Restrict the scan to these files; None scans every file. Defaults
        to ``None``.

    Returns
    -------
    MentionScan
        Every occurrence, in index order. Overlapping matches of different
        terms are all kept; one term's matches never overlap each other.

    Raises
    ------
    ValueError
        If no term is given or ``context_lines`` is negative.

    """

    terms = tuple(
        dict.fromkeys(" ".join(term.split()) for term in terms if term)
    )
    if not terms:
        raise ValueError("At least one term is needed.")

    if context_lines < 0:
        raise ValueError("context_lines must not be negative.")

    patterns = [
        term_pattern(term, case_sensitive, whole_words) for term in terms
    ]
    allowed = set(paths) if paths is not None else None
    units = _units(
        chunk
        for chunk in chunks
        if allowed is None or chunk.path in allowed
    )
    mentions: list[Mention] = []
    files: dict[Path, None] = {}
    for unit in units:
        files[unit.path] = None
        # Sorted by position, then by term order, within the unit.
        found: list[tuple[int, int, Mention]] = []
        for order, (term, pattern) in enumerate(zip(terms, patterns)):
            for match in pattern.finditer(unit.text):
                found.append(
                    (
                        match.start(),
                        order,
                        unit.mention(term, match, context_lines),
                    )
                )
        mentions.extend(item for *_, item in sorted(found, key=_position))

    return MentionScan(
        terms=terms,
        mentions=tuple(mentions),
        files_scanned=tuple(files),
        case_sensitive=case_sensitive,
        whole_words=whole_words,
    )


def mention_evidence(
    scan: MentionScan, limit: int
) -> tuple[dict[str, object], ...]:
    """Return evidence fields for the first ``limit`` mentions.

    Parameters
    ----------
    scan: MentionScan
        Scan result.
    limit: int
        How many mentions to list.

    Returns
    -------
    tuple[dict[str, object], ...]
        ``citation``, ``path``, ``start_line``, ``end_line``, ``symbol``,
        ``text`` (the exact context lines, so a preview can locate them),
        and ``cell``, numbered from one.

    """

    return tuple(
        mention.evidence(number)
        for number, mention in enumerate(scan.mentions[:limit], start=1)
    )


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


def mention_location(mention: Mention, name: str) -> str:
    """Describe where a mention is: file, page or cell, lines, symbol.

    Parameters
    ----------
    mention: Mention
        The mention.
    name: str
        How to name its file (see ``short_names``).

    Returns
    -------
    str
        For example ``"smith2020.pdf, page 3, line 12"``.

    """

    parts = [name]
    if mention.section and mention.section.startswith("Page "):
        parts.append(mention.section.lower())
    if mention.cell is not None:
        parts.append(f"cell {mention.cell}")
    lines = (
        f"line {mention.line}"
        if mention.line == mention.end_line
        else f"lines {mention.line}-{mention.end_line}"
    )
    parts.append(lines)
    if mention.symbol:
        parts.append(f"in {mention.symbol}")

    return ", ".join(parts)


def mention_snippet(mention: Mention, width: int = 60) -> str:
    """Show the occurrence in bold with some text on each side, one line.

    Parameters
    ----------
    mention: Mention
        The mention.
    width: int, optional
        Characters of context kept on each side. Defaults to ``60``.

    Returns
    -------
    str
        Markdown; square brackets become parentheses so source text never
        looks like a citation.

    """

    end = mention.offset + len(mention.text)
    raw_before = mention.context[: mention.offset]
    raw_after = mention.context[end:]
    before = " ".join(raw_before.split())
    after = " ".join(raw_after.split())
    left = "…" if len(before) > width else ""
    right = "…" if len(after) > width else ""
    before, after = before[-width:], after[:width]
    gap_before = " " if before and raw_before[-1:].isspace() else ""
    gap_after = " " if after and raw_after[:1].isspace() else ""
    found = " ".join(mention.text.split())
    text = f"{left}{before}{gap_before}**{found}**{gap_after}{after}{right}"

    # Plain text: brackets would look like citations in the answer.
    return text.replace("[", "(").replace("]", ")")


def short_names(paths: Sequence[Path]) -> dict[Path, str]:
    """Name each file by the shortest path ending that is unique.

    Parameters
    ----------
    paths: Sequence[pathlib.Path]
        Files to name.

    Returns
    -------
    dict[pathlib.Path, str]
        For example ``dev_intro/readme.md`` when several ``readme.md``
        files exist, and ``smith2020.pdf`` otherwise.

    """

    names: dict[Path, str] = {}
    for path in paths:
        parts = path.parts
        for size in range(1, len(parts) + 1):
            ending = parts[-size:]
            if not any(
                other != path and other.parts[-size:] == ending
                for other in paths
            ):
                break
        names[path] = "/".join(ending)

    return names


def _ask_terms(context: ToolContext, text: str, categories: bool) -> str:
    """Ask the router model for terms; a failure becomes the reply text.

    Parameters
    ----------
    context: ToolContext
        QAChat's services (``ask_router``).
    text: str
        The user's words the terms must come from.
    categories: bool
        Also ask for categories (for ``batch_classify``).

    Returns
    -------
    str
        The reply, or ``"failed: ..."`` so that validation finds no terms
        and the answer explains how to quote them instead of failing.

    """

    system, prompt = terms_prompt(text, categories=categories)
    tokens = _CATEGORIES_REPLY_TOKENS if categories else _TERMS_REPLY_TOKENS
    try:
        return context.ask_router(system, prompt, tokens)
    except Exception as error:  # Explain instead of failing.
        return f"failed: {error}"


def _names_file(term: str, path: Path) -> bool:
    """Whether a term is the file's name, with or without its suffix.

    Parameters
    ----------
    term: str
        Term to compare with the filename and stem.
    path: Path
        Path to the target file or directory.

    Returns
    -------
    bool
        True when a term is the file's name, with or without its suffix.

    """

    return term.casefold() in (path.name.casefold(), path.stem.casefold())


def _reply_json(reply: str) -> dict:
    """Return the JSON object in a model reply, or {} when there is none.

    Parameters
    ----------
    reply: str
        Model reply to parse.

    Returns
    -------
    dict
        the JSON object in a model reply, or {} when there is none.

    """

    match = re.search(r"\{.*\}", reply, flags=re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except ValueError:
        value = None

    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class _Unit:
    """One natural unit's lines, reassembled from its chunks.

    Attributes
    ----------
    path: pathlib.Path
        Source file.
    section: str | None
        Section of the unit, such as ``"Page 3"``.
    cell: int | None
        Notebook cell, when the source is a notebook.
    numbers: tuple[int, ...]
        Line numbers, in order.
    lines: tuple[str, ...]
        Line texts, one per number.
    symbols: tuple[str | None, ...]
        Code symbol of each line, from the first chunk that had it.
    starts: tuple[int, ...]
        Where each line starts within ``text``.
    text: str
        The lines joined with newlines; this is what is scanned.

    """

    path: Path
    section: str | None
    cell: int | None
    numbers: tuple[int, ...]
    lines: tuple[str, ...]
    symbols: tuple[str | None, ...]
    starts: tuple[int, ...]
    text: str

    def mention(
        self, term: str, match: re.Match[str], context_lines: int
    ) -> Mention:
        """Turn a match in ``text`` into a mention with its line context.

        Parameters
        ----------
        term: str
            The searched term that matched.
        match: re.Match[str]
            The match in ``text``.
        context_lines: int
            Lines of context kept before and after the match.

        Returns
        -------
        Mention
            The mention, with line numbers in the unit's numbering.

        """

        first = bisect_right(self.starts, match.start()) - 1
        last = bisect_right(self.starts, max(match.end() - 1, 0)) - 1
        low = max(0, first - context_lines)
        high = min(len(self.lines), last + context_lines + 1)
        return Mention(
            term=term,
            text=match.group(0),
            path=self.path,
            section=self.section,
            cell=self.cell,
            symbol=self.symbols[first],
            line=self.numbers[first],
            end_line=self.numbers[last],
            context_start=self.numbers[low],
            context="\n".join(self.lines[low:high]),
            offset=match.start() - self.starts[low],
        )


def _units(chunks: Iterable[Chunk]) -> list[_Unit]:
    """Reassemble each unit's numbered lines once from its chunks.

    Parameters
    ----------
    chunks: Iterable[Chunk]
        Chunks in index order; overlapping lines keep the first chunk's
        text and symbol.

    Returns
    -------
    list[_Unit]
        One unit per (file, section, cell), in the order first seen.

    """

    collected: dict[tuple, dict[int, tuple[str, str | None]]] = {}
    for chunk in chunks:
        key = (chunk.path, chunk.section, chunk.cell)
        lines = collected.setdefault(key, {})
        for offset, line in enumerate(chunk.text.splitlines()):
            lines.setdefault(chunk.start_line + offset, (line, chunk.symbol))

    units = []
    for (path, section, cell), lines in collected.items():
        numbers = tuple(sorted(lines))
        texts = tuple(lines[number][0] for number in numbers)
        starts, offset = [], 0
        for text in texts:
            starts.append(offset)
            offset += len(text) + 1
        units.append(
            _Unit(
                path=path,
                section=section,
                cell=cell,
                numbers=numbers,
                lines=texts,
                symbols=tuple(lines[number][1] for number in numbers),
                starts=tuple(starts),
                text="\n".join(texts),
            )
        )

    return units


def _position(item: tuple[int, int, Mention]) -> tuple[int, int]:
    """Sort key: start in the unit, then term order (never the mention).

    Parameters
    ----------
    item: tuple[int, int, Mention]
        Item to inspect or convert.

    Returns
    -------
    tuple[int, int]
        Start offset and term-order index for sorting mentions.

    """

    return item[0], item[1]
