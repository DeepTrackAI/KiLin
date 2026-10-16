"""Reassemble and scan exact source mentions for multiple tools.

Algorithms shared by registered tools belong here rather than in
another executable tool. The module has no GUI dependency.

Key Features
------------

- **Exact mention scans**

    Deterministic operations reusable by multiple tool packages.

Module Structure
----------------

- ``Mention, MentionScan, collect_mentions``

    Public operations shared within the tools package.

Examples
--------
>>> from kilin.tools._shared.mentions import MentionScan
>>> callable(MentionScan)
True

"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Protocol

from kilin.chunking import Chunk
from kilin.retrievals.document_reading import dominant_path
from kilin.tools._contract import ToolContext, ToolRequest, ToolResult
from kilin.tools._shared.choose_document import document_scope
from kilin.tools._shared.terms import (
    CATEGORIES_REPLY_TOKENS,
    TERMS_REPLY_TOKENS,
    document_phrase,
    quoted_terms,
    term_pattern,
    terms_prompt,
    validated_document,
    validated_terms,
)

__all__ = [
    "Mention",
    "MentionScan",
    "MentionCollection",
    "ScanOptions",
    "collect_mentions",
    "scan_mentions",
    "mention_evidence",
    "mention_location",
    "mention_snippet",
]


class ScanOptions(Protocol):
    """Minimal options required by the shared mention collector.

    Attributes
    ----------
    case_sensitive: bool
        Use case-sensitive matches.
    whole_words: bool
        Require whole words.
    context_lines: int
        Number of surrounding source lines.

    """

    case_sensitive: bool
    whole_words: bool
    context_lines: int


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


def collect_mentions(
    settings: ScanOptions,
    request: ToolRequest,
    context: ToolContext,
    *,
    tool: str,
    categories: bool = False,
) -> MentionCollection | ToolResult:
    """Find the terms and scope in the question, then every mention.

    Parameters
    ----------
    settings: ScanOptions
        Matching settings (``batch_classify`` settings work too).
    request: ToolRequest
        The question and the user's words.
    context: ToolContext
        KíLín's services.
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


def _ask_terms(context: ToolContext, text: str, categories: bool) -> str:
    """Ask the router model for terms; a failure becomes the reply text.

    Parameters
    ----------
    context: ToolContext
        KíLín's services (``ask_router``).
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
    tokens = CATEGORIES_REPLY_TOKENS if categories else TERMS_REPLY_TOKENS
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

