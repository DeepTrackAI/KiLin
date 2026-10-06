"""List every exact mention of a term, without language-model analysis.

``scan_and_list`` answers questions such as "list all mentions of transfer
learning" or "how many times does the Smith paper say MRI?" exhaustively.
Ranked
retrieval returns only the best matches; this workflow inspects every
indexed line instead, and Python, not a model, counts and lists the results.

Indexed chunks overlap, so counting inside chunks would count mentions in
the overlaps twice. Every chunk consists of whole source lines with their
line numbers, and line numbers are unique within one natural unit: a file's
section, a PDF page (numbering restarts on each page), or a notebook cell.
The workflow therefore reassembles each unit's lines from its chunks once
and scans the reassembled text, so a phrase broken across lines is found and
nothing is counted twice.

Key Features
------------

- **Terms in the user's words**

    ``quoted_terms`` takes quoted phrases from the question, and
    ``validated_terms`` accepts a model's suggestions only when they occur
    in the question.

- **Exact scan**

    ``scan_mentions`` matches case-insensitively by default, as whole words,
    with any whitespace (including line breaks) between a phrase's words.

- **Report**

    ``mention_report`` writes counts per file, source-linked matches, and
    coverage; ``mention_evidence`` gives each listed match's exact lines.

Module Structure
----------------

- ``Mention``
- ``MentionScan``
- ``quoted_terms``
- ``terms_prompt``
- ``validated_terms``
- ``validated_document``
- ``scan_mentions``
- ``mention_evidence``
- ``mention_report``
- ``mention_location``
- ``mention_snippet``
- ``short_names``

"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Final

from qachat.chunking import Chunk

__all__ = [
    "MAX_TERMS",
    "Mention",
    "MentionScan",
    "mention_evidence",
    "mention_location",
    "mention_report",
    "mention_snippet",
    "quoted_terms",
    "scan_mentions",
    "short_names",
    "terms_prompt",
    "validated_document",
    "validated_terms",
]

# More terms would make one answer hard to read; ask again for others.
MAX_TERMS: Final[int] = 5
_QUOTED = re.compile(
    r'"([^"\n]{1,80})"|“([^”\n]{1,80})”|`([^`\n]{1,80})`'
)


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
        """Last line of the surrounding context."""

        return self.context_start + self.context.count("\n")


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
        """Return mentions per file, in index order, files with any."""

        totals: dict[Path, int] = {}
        for mention in self.mentions:
            totals[mention.path] = totals.get(mention.path, 0) + 1
        return totals

    def term_counts(self) -> dict[str, int]:
        """Return mentions per term, in the order the terms were given."""

        totals = {term: 0 for term in self.terms}
        for mention in self.mentions:
            totals[mention.term] += 1
        return totals


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
        ``qachat.workflows.batch_classify.validated_categories``.

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
{{"terms": ["MRI"], "document": "the Smith paper"}}"""
    if categories:
        user += """
"which uses of MRI are clinical and which are research?" -> \
{"terms": ["MRI"], "document": null, "categories": ["clinical use", \
"research use"]}"""
    return system, user


_CATEGORY_FIELD = """
"categories": 2 to 6 short labels for the groups the user wants each mention
sorted into, taken from the message (for example "own code" and "citation
of other work")."""


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
        term = " ".join(item.split()).strip(" '\"`“”?.,;:")
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
    phrase = " ".join(phrase.split()).strip(" '\"`“”?.,;:")
    source = " ".join(text.split()).casefold()
    return phrase if phrase and phrase.casefold() in source else None


def _reply_json(reply: str) -> dict:
    match = re.search(r"\{.*\}", reply, flags=re.S)
    try:
        value = json.loads(match.group(0)) if match else None
    except ValueError:
        value = None
    return value if isinstance(value, dict) else {}


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

    terms = tuple(dict.fromkeys(" ".join(t.split()) for t in terms if t))
    if not terms:
        raise ValueError("At least one term is needed.")
    if context_lines < 0:
        raise ValueError("context_lines must not be negative.")
    patterns = [
        (term, _term_pattern(term, case_sensitive, whole_words))
        for term in terms
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
        found: list[tuple[int, int, Mention]] = []
        for term, pattern in patterns:
            for match in pattern.finditer(unit.text):
                found.append(
                    (
                        match.start(),
                        terms.index(term),
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
        {
            "citation": number,
            "path": mention.path,
            "start_line": mention.context_start,
            "end_line": mention.context_end,
            "symbol": mention.symbol,
            "text": mention.context,
            "cell": mention.cell,
        }
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
        names = ", ".join(path.name for path in unsearchable[:5])
        more = len(unsearchable) - 5
        coverage += (
            f" {len(unsearchable)} source files have no searchable text "
            f"(for example scanned PDFs) and were not scanned: {names}"
            + (f", and {more} more." if more > 0 else ".")
        )
    lines += ["", coverage]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _Unit:
    """One natural unit's lines, reassembled from its chunks."""

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


def _units(chunks: object) -> list[_Unit]:
    """Reassemble each unit's numbered lines once from its chunks."""

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


def _term_pattern(
    term: str, case_sensitive: bool, whole_words: bool
) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(word) for word in term.split())
    if whole_words:
        if re.match(r"\w", term):
            body = rf"(?<!\w){body}"
        if re.search(r"\w$", term):
            body = rf"{body}(?!\w)"
    return re.compile(body, 0 if case_sensitive else re.IGNORECASE)


def _position(item: tuple[int, int, Mention]) -> tuple[int, int]:
    return item[0], item[1]


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
