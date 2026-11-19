"""Write a project report from every document's notes and checks.

Private to ``project_summary``. After the map step (``_mapping``) has read
each document, this module writes the report:

1. **Sections from all documents** (``notes``): one model call per
   section, from every document's note for it, citing documents as
   ``[n]``. When the notes do not fit in one call, they are first
   condensed in groups, keeping their citations.
2. **Chosen documents** (``{key_documents}``): the model picks document
   numbers; code writes their titles.
3. **Per-document sections**: assembled by code from the checks, one entry
   or table row per matching document, with how many documents match and
   which are unclear, so no model can drop or invent an entry.
4. **The documents covered** (``{documents}``) and the list closing the
   report, written by code.

Key Features
------------

- **Synthesis, not lists**

    Sections from all documents compare and group them, citing each.

- **Entries by code**

    Per-document sections and counts are never written by the model.

Module Structure
----------------

- ``note_groups``

    Split notes into groups that fit one call.

- ``condense_prompt``, ``section_prompt``, ``fix_prompt``,
  ``headline_prompt`` and ``key_documents_prompt``

    The model calls that write the report.

- ``chosen_numbers``

    The document numbers a reply chooses.

- ``citation_problem``, ``clean_citations`` and ``cite_document_names``

    Whether a section cites documents as it must, its invalid citations
    removed, and documents named in the text ("Document 4") written as
    citations.

- ``support_problem``, ``numeric_support_problem``,
  ``agreement_problem`` and ``coverage_problem``

    Independent source-text checks for cited wording and numeric facts;
    agreement and complete-coverage checks are separate.

- ``headline_fix_prompt`` and ``fitted_headline``

    A headline shortened by the model, or cut at a word.

- ``per_document_markdown``, ``documents_line`` and ``documents_list``

    The parts written by code.

Examples
--------
>>> from kilin.tools.project_summary._reduce import chosen_numbers
>>> chosen_numbers("NUMBERS: 3, 7, 3, 99", 10, 2)
[3, 7]

"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Final

from kilin._model_completion import strip_thinking
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._shared.report_templates import (
    ReportTemplate,
    TemplateSection,
)
from kilin.tools.project_summary._documents import ProjectDocument
from kilin.tools.project_summary._mapping import (
    CHECK_UNCLEAR,
    CHECK_YES,
    DocumentNotes,
)

__all__ = [
    "chosen_numbers",
    "agreement_problem",
    "citation_problem",
    "statement_citation_problem",
    "cite_document_names",
    "clean_citations",
    "coverage_problem",
    "fitted_headline",
    "headline_fix_prompt",
    "support_problem",
    "condense_prompt",
    "documents_line",
    "documents_list",
    "fix_prompt",
    "headline_prompt",
    "key_documents_prompt",
    "note_groups",
    "numeric_support_problem",
    "per_document_markdown",
    "section_prompt",
]

_SYSTEM: Final[str] = (
    "You write one section of a report on a collection of scientific "
    "documents, from notes on each document. Say only what the notes say. "
    "After each statement, cite the documents it rests on by the numbers "
    "the notes give, in brackets, such as [2] or [1, 4]. Never refer to a "
    "document as \"Document 2\" or \"the first document\": cite it. "
    "Compare and group the documents; do not describe them one by one. An "
    "agreement or a disagreement involves at least two different "
    "documents; one document's strengths and limits are not a "
    "disagreement. Do not claim that the documents lack something unless "
    "the notes show it."
)
# A bracketed citation, and one naming a document in prose ("Document 4",
# "Documents 1 and 2").
_BRACKETS = re.compile(r"\[([^\[\]]{1,40})\]")
_DOCUMENT_WORDS = re.compile(r"\bDocuments?\s+\[?\d", re.IGNORECASE)
# Documents named in the text: "Document 4", "Document [4]", "Documents
# 1 and 4", "Documents 1, 2 and [5]".
_DOCUMENT_NAMES = re.compile(
    r"\bDocuments?\s+((?:\[?\d+\]?)(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)"
    r"\[?\d+\]?)*)",
    re.IGNORECASE,
)
# Where the clause a citation supports starts, inside its phrase.
_CLAUSE = re.compile(
    r"[,;:(—–]|\s-\s|\b(?:such\s+as|including|like|e\.g\.)\s",
    re.IGNORECASE,
)
# Words compared between a cited phrase and original source passages:
# four or more letters, compared by their first five ("genomic" and
# "genome-wide" match), without words that say nothing of a topic.
_TOPIC_WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_STEM: Final[int] = 5
_PLAIN_WORDS: Final[frozenset[str]] = frozenset({
    "about", "across", "also", "along", "among", "based", "both", "each",
    "from", "have", "into", "more", "most", "other", "such", "than",
    "that", "their", "them", "these", "they", "this", "those", "through",
    "using", "used", "when", "where", "which", "while", "with", "within",
    "document", "documents", "paper", "papers", "study", "studies",
    "focus", "focuses", "main", "different", "differences", "several",
})
# Sentences saying documents agree or disagree, which need two documents
# in each citation.
_AGREEMENT = re.compile(
    r"\b(?:agree\w*|disagree\w*|concur\w*|consensus|conflict\w*|"
    r"contradict\w*)\b",
    re.IGNORECASE,
)
# Words a cut headline must not end with.
_DANGLING = re.compile(
    r"\s+(?:and|or|of|in|on|for|with|to|the|a|an|by|from|through|via|"
    r"as|at)\s*$",
    re.IGNORECASE,
)
# A cited phrase is checked when it has this many distinctive words; it
# is flagged when another document shares this many of them.
_SUPPORT_WORDS_MIN: Final[int] = 2
_NUMBERS = re.compile(r"\d+")
# Numerical assertions are checked against source passages, not notes.
# Signs and exponents are part of a number: -5%, +5%, and 5e-3 differ.
_NUMBER_TEXT: Final[str] = (
    r"[+\-\u2212\u00b1\u2213]?(?:\d+(?:[.,]\d+)*|\.\d+)"
    r"(?:[eE][+\-\u2212]?\d+)?%?"
)
_FACT_NUMBER = re.compile(rf"(?<![\w.]){_NUMBER_TEXT}(?!\w)")
# Statistical comparisons must preserve the reported inequality direction.
_COMPARISON = re.compile(
    rf"(?<!\w)([a-z][a-z0-9_]*)\s*(<=|>=|<|>|=)\s*"
    rf"({_NUMBER_TEXT})(?!\w)", re.IGNORECASE,
)
# A value alone cannot verify a differently scaled amount (40 vs 40 million).
_QUANTITY = re.compile(
    rf"(?<!\w)({_NUMBER_TEXT})\s+([a-z][a-z0-9-]*)",
    re.IGNORECASE,
)
# Break assertions at paragraph, list-item, sentence and semicolon boundaries.
_STATEMENT_END = re.compile(r"(?<=[.!?])\s+(?=\S)|;\s*(?=\S)")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)")
# Periods in common scientific abbreviations are not sentence boundaries.
_ABBREVIATION = re.compile(
    r"\b(?:e\.g|i\.e|et\s+al|vs|fig|dr|prof|etc)\.",
    re.IGNORECASE,
)
_UNIT_IGNORED: Final[frozenset[str]] = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "was", "were", "with",
})
# Unclear documents named in a per-document section's count line.
_UNCLEAR_NAMED: Final[int] = 10


def _length(section: TemplateSection) -> str:
    """Return how long a section should be, for a prompt.

    Parameters
    ----------
    section: TemplateSection
        The section.

    Returns
    -------
    str
        "about 50 words (40-60)", "one line only", or "a few concise
        sentences".

    """

    if section.inline:
        return "one line only"
    if section.words:
        low, high = section.words
        return f"about {(low + high) // 2} words ({low}-{high})"
    if section.items:
        low, high = section.items
        return f"{low}-{high} items, one per line starting with '- '"
    return "a few concise sentences"


def note_groups(
    notes: Sequence[tuple[int, str]], tokens: int
) -> list[list[tuple[int, str]]]:
    """Split notes into groups that each fit one model call.

    Parameters
    ----------
    notes: Sequence[tuple[int, str]]
        Each document's number and note for one section.
    tokens: int
        Most tokens of notes in one group.

    Returns
    -------
    list[list[tuple[int, str]]]
        The notes in order, in groups of at most ``tokens`` (a note larger
        than that is a group of its own); one group when all fit.

    """

    groups: list[list[tuple[int, str]]] = [[]]
    used = 0
    for number, note in notes:
        cost = estimate_tokens(note) + 4
        if groups[-1] and used + cost > tokens:
            groups.append([])
            used = 0
        groups[-1].append((number, note))
        used += cost
    return [group for group in groups if group]


def _notes_text(notes: Sequence[tuple[int, str]]) -> str:
    """Return notes as prompt lines, each with its document's number.

    Parameters
    ----------
    notes: Sequence[tuple[int, str]]
        Each document's number and note.

    Returns
    -------
    str
        "[3] note" lines.

    """

    return "\n".join(f"[{number}] {note}" for number, note in notes)


def condense_prompt(
    section: TemplateSection, notes: Sequence[tuple[int, str]]
) -> tuple[str, str]:
    """Return the prompts condensing one group of notes for a section.

    Parameters
    ----------
    section: TemplateSection
        The section the notes are for.
    notes: Sequence[tuple[int, str]]
        One group's notes.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the group's facts as short
        lines, each keeping the numbers of the documents it comes from.

    """

    user = (
        f"Notes on several documents for the report section "
        f"“{section.name}” ({section.guidance}):\n\n{_notes_text(notes)}"
        "\n\nCondense them into short lines of facts, grouping documents "
        "that say the same and keeping each line's document numbers, such "
        "as “U-Net models segment nuclei [3, 7].” Keep differences between "
        "documents. Add nothing."
    )
    return _SYSTEM, user


def section_prompt(
    template: ReportTemplate,
    section: TemplateSection,
    notes: str,
    documents: int,
) -> tuple[str, str]:
    """Return the prompts writing one section from all documents' notes.

    Parameters
    ----------
    template: ReportTemplate
        The project template.
    section: TemplateSection
        The section to write.
    notes: str
        The notes for it: "[n] note" lines, or condensed lines citing
        documents.
    documents: int
        How many documents the report covers.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the section's content only, of
        the section's length, citing documents as ``[n]``.

    """

    user = (
        f"Report: {template.title} on {documents} documents"
        + (f" — {template.description}" if template.description else "")
        + f"\n\nSection: {section.name}\nWhat it should say: "
        f"{section.guidance or section.name}\nLength: "
        f"{_length(section)}\n\nNotes:\n{notes or '(no document says '
        'anything about this)'}\n\nWrite only the section's content, "
        "without its heading. When the notes say nothing about it, say so "
        "in one sentence."
        + (
            " Cover every document that has a note here, citing each."
            if section.all_documents else ""
        )
    )
    return _SYSTEM, user


def fix_prompt(
    section: TemplateSection, content: str, problem: str
) -> tuple[str, str]:
    """Return the prompts fixing one section's length or form.

    Parameters
    ----------
    section: TemplateSection
        The section.
    content: str
        Its content as written.
    problem: str
        What is wrong (``section_problem``).

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the content rewritten to fit,
        keeping its facts and citations.

    """

    user = (
        f"This report section has a problem: {problem}\nLength wanted: "
        f"{_length(section)}\n\nSection “{section.name}”:\n{content}\n\n"
        "Rewrite it to fix that, keeping its facts and its citations of "
        "documents, such as [2] or [1, 4], and adding nothing. Write only "
        "the content."
    )
    return _SYSTEM, user


def headline_prompt(
    template: ReportTemplate, sections: Sequence[tuple[str, str]]
) -> tuple[str, str]:
    """Return the prompts writing a project report's headline.

    Parameters
    ----------
    template: ReportTemplate
        The template, with its headline limit and instruction.
    sections: Sequence[tuple[str, str]]
        The written sections' names and content.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for one line of at most
        ``template.headline_characters`` characters.

    """

    written = "\n\n".join(
        f"{name}:\n{content[:600]}" for name, content in sections
    )
    user = (
        f"Write the headline of this report ({template.title}): "
        f"{template.headline_guidance or 'what the documents show'}. At "
        f"most {template.headline_characters} characters, one line, no "
        f"citations, no full stop.\n\n{written}"
    )
    return _SYSTEM, user


def key_documents_prompt(
    section: TemplateSection,
    documents: Sequence[ProjectDocument],
    notes: dict[int, str],
) -> tuple[str, str]:
    """Return the prompts choosing documents for a section.

    Parameters
    ----------
    section: TemplateSection
        A ``{key_documents}`` section, with its item limit.
    documents: Sequence[ProjectDocument]
        The documents.
    notes: dict[int, str]
        A line on each document (its notes), by number.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for ``NUMBERS:`` and the chosen
        documents' numbers, without writing their titles.

    """

    low, high = section.items or (1, 3)
    listed = "\n".join(
        f"[{d.number}] {d.title}: {notes.get(d.number, '')[:300]}"
        for d in documents
    )
    user = (
        f"Choose {low}-{high} documents for the section “{section.name}”: "
        f"{section.guidance}\n\nDocuments:\n{listed}\n\nAnswer with one "
        "line: NUMBERS: <the chosen numbers, separated by commas>"
    )
    return _SYSTEM, user


def chosen_numbers(reply: str, documents: int, most: int) -> list[int]:
    """Return the document numbers a reply chooses.

    Parameters
    ----------
    reply: str
        The model's reply to ``key_documents_prompt``.
    documents: int
        How many documents there are.
    most: int
        Most numbers kept.

    Returns
    -------
    list[int]
        The numbers on the ``NUMBERS:`` line (or the whole reply), in
        order, each once, between 1 and ``documents``.

    """

    text = strip_thinking(reply)
    line = next(
        (line for line in text.splitlines() if "NUMBERS" in line.upper()),
        text,
    )
    chosen: list[int] = []
    for found in _NUMBERS.findall(line):
        number = int(found)
        if 1 <= number <= documents and number not in chosen:
            chosen.append(number)
    return chosen[:most]


def _cited(marker: str, documents: int) -> list[int] | None:
    """Return the document numbers one bracketed citation holds.

    Parameters
    ----------
    marker: str
        The text between the brackets ("1, 4", "2-3", "n1").
    documents: int
        How many documents there are.

    Returns
    -------
    list[int] | None
        The numbers (a range "2-3" gives 2 and 3); None when the marker is
        not a citation of existing documents.

    """

    numbers: list[int] = []
    for part in re.split(r"\s*[,;]\s*", marker.strip()):
        found = re.fullmatch(r"(\d+)(?:\s*[–-]\s*(\d+))?", part)
        if not found:
            return None
        low = int(found.group(1))
        high = int(found.group(2) or low)
        if not 1 <= low <= high <= documents:
            return None
        numbers.extend(range(low, high + 1))
    return numbers or None


def citation_problem(content: str, documents: int, cite: bool) -> str:
    """Return what is wrong with how a section cites documents, or ``""``.

    Parameters
    ----------
    content: str
        A section written from the notes.
    documents: int
        How many documents there are.
    cite: bool
        Whether the section must cite at least one document (it was
        written from notes).

    Returns
    -------
    str
        A problem for the repair prompt: brackets that are not citations
        of existing documents ("[n1]", "[7]" of 5); documents named in
        prose ("Document 4", "Document [4]") instead of cited; or no
        citation at all.
        ``""`` when the citations are valid.

    """

    invalid = [
        f"[{marker}]" for marker in _BRACKETS.findall(content)
        if _cited(marker, documents) is None
    ]
    if invalid:
        return (
            f"It has citations that are not document numbers from 1 to "
            f"{documents}: {', '.join(dict.fromkeys(invalid))}. Cite "
            "documents as [2] or [1, 4]."
        )
    if _DOCUMENT_WORDS.search(content):
        return (
            "It names documents in the text (\"Document 4\", \"Document "
            "[4]\") instead of citing them; cite them as [4] after the "
            "statement, without the word Document."
        )
    if cite and not any(
        _cited(marker, documents) for marker in _BRACKETS.findall(content)
    ):
        return "It cites no document; cite the documents each statement "\
            "rests on, as [2] or [1, 4]."
    return ""


def _statements(content: str) -> list[str]:
    """Separate assertions without splitting wrapped Markdown paragraphs.

    Parameters
    ----------
    content: str
        Model-written section, possibly with lists and wrapped lines.

    Returns
    -------
    list[str]
        Clauses in order. Each list item is separate, and ordinary line
        wrapping is not treated as a new assertion.

    """

    paragraphs: list[str] = []
    current: list[str] = []
    for line in content.splitlines() + [""]:
        line = line.strip()
        if not line or line.startswith("#") or _LIST_LINE.match(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if line and not line.startswith("#"):
                paragraphs.append(line)
        else:
            current.append(line)
    result: list[str] = []
    for paragraph in paragraphs:
        guarded = _ABBREVIATION.sub(
            lambda match: match.group(0).replace(".", "\x00"), paragraph
        )
        result.extend(
            part.replace("\x00", ".").strip()
            for part in _STATEMENT_END.split(guarded)
            if part.strip()
        )
    return result


def _substantive(text: str) -> bool:
    """Whether a clause contains a factual assertion needing a citation.

    Parameters
    ----------
    text: str
        A clause with any citations already removed.

    Returns
    -------
    bool
        Numeric facts and clauses with at least two distinctive topic
        stems are treated as substantive; isolated transition words are
        not. This is a syntactic check, not semantic fact verification.

    """

    text = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|\|)", "", text)
    return bool(_FACT_NUMBER.search(text)) or len(_stems(text)) >= 2


def statement_citation_problem(content: str, documents: int) -> str:
    """Find an uncited substantive assertion in a model-written section.

    Parameters
    ----------
    content: str
        The text to check, with document citations in ``[n]`` form.
    documents: int
        Number of indexed documents included in the report.

    Returns
    -------
    str
        A diagnostic for the first uncited sentence, independent list
        item, or substantive text after the last citation; ``""`` if
        every such assertion has a valid citation. Citation validity
        itself is checked by ``citation_problem`` first.

    """

    for statement in _statements(content):
        citations = [
            match for match in _BRACKETS.finditer(statement)
            if _cited(match.group(1), documents) is not None
        ]
        uncited = _BRACKETS.sub("", statement)
        if not citations and _substantive(uncited):
            shown = " ".join(uncited.split())[:100].rstrip(".!? ")
            return (
                f"Uncited statement: “{shown}”. Cite the source "
                "after each substantive statement, or remove it."
            )
        if citations:
            tail = statement[citations[-1].end():].strip(" .,;:!?)]}")
            if _substantive(tail):
                shown = " ".join(tail.split())[:100]
                return (
                    f"Uncited claim after a citation: “{shown}”. "
                    "Cite that claim explicitly or remove it."
                )
    return ""


def clean_citations(content: str, documents: int) -> tuple[str, int]:
    """Return a section with brackets that cite no document removed.

    Parameters
    ----------
    content: str
        A section.
    documents: int
        How many documents there are.

    Returns
    -------
    tuple[str, int]
        The content without the invalid brackets (and the space before
        them), and how many were removed.

    """

    removed = 0

    def drop(match: re.Match[str]) -> str:
        """Return a citation, or nothing for an invalid one.

        Parameters
        ----------
        match: re.Match[str]
            A bracketed marker with the spaces before it.

        Returns
        -------
        str
            The match when it cites documents, else ``""``.

        """

        nonlocal removed
        if _cited(match.group(1), documents) is not None:
            return match.group(0)
        removed += 1
        return ""

    cleaned = re.sub(r"\s*\[([^\[\]]{1,40})\]", drop, content)
    return cleaned, removed


def _stems(text: str) -> set[str]:
    """Return the topic words of a text, by their first letters.

    Parameters
    ----------
    text: str
        A phrase, or a document's title and notes.

    Returns
    -------
    set[str]
        The first ``_STEM`` letters, in lower case, of its words of four or
        more letters that are not in ``_PLAIN_WORDS``.

    """

    return {
        normalized[:_STEM]
        for word in _TOPIC_WORD.findall(text)
        if (normalized := word.casefold().replace("-", ""))
        not in _PLAIN_WORDS and len(normalized) >= 4
    }


def support_problem(
    content: str, texts: dict[int, str], *, strict: bool = False,
) -> str:
    """Return a citation whose document does not mention what it cites.

    Parameters
    ----------
    content: str
        A section written from the notes.
    texts: dict[int, str]
        Each document's texts by number. The caller must supply original
        indexed source passages for independent validation.
    strict: bool, optional
        Also flag cited phrases with no overlap in their original source
        even when no alternative document is better. Defaults to False
        for the earlier heuristic's compatibility.

    Returns
    -------
    str
        For the first citation whose phrase (the clause before it: the
        words since the previous citation or the start of the sentence,
        cut after the last comma, semicolon, dash or "such as" when that
        clause alone has enough distinctive words; never cut for a
        citation of several documents, which cites a list as a whole, so
        each needs to mention only part of it) has at least
        ``_SUPPORT_WORDS_MIN`` distinctive words, none of them in the cited
        document's text, while another document has at least
        ``_SUPPORT_WORDS_MIN``: a problem naming the phrase, the citation
        and the document that may be meant. Words in more than half the
        documents are not distinctive. In strict mode, a cited phrase
        without a recognizable original-source anchor is reported even
        when no alternative document seems better. ``""`` otherwise.

    """

    stems = {number: _stems(text) for number, text in texts.items()}
    if len(stems) < 2 and not strict:
        return ""
    common = {
        stem for stem in set().union(*stems.values())
        if len(stems) > 1
        and sum(stem in found for found in stems.values()) * 2 > len(stems)
    }
    for sentence in re.split(r"(?<=[.!?])\s+", content):
        start = 0
        for match in _BRACKETS.finditer(sentence):
            phrase = sentence[start:match.start()]
            start = match.end()
            cited = _cited(match.group(1), max(stems, default=0))
            clause = _CLAUSE.split(phrase)[-1]
            threshold = 1 if strict else _SUPPORT_WORDS_MIN
            shared = cited is not None and len(cited) > 1
            if not shared and len(_stems(clause) - common) >= threshold:
                phrase = clause  # "..., such as reservoir computing [1]"
            words = _stems(phrase) - common
            if not cited or len(words) < threshold:
                continue
            for number in cited:
                if words & stems.get(number, set()):
                    continue
                best = max(
                    (n for n in stems if n not in cited),
                    key=lambda n: len(words & stems[n]),
                    default=None,
                )
                shown = " ".join(phrase.strip(" ,;:").split()[-8:])
                if best is not None and len(words & stems[best]) >= (
                    _SUPPORT_WORDS_MIN
                ):
                    return (
                        f"“{shown}” cites [{number}], whose source text "
                        f"does not mention it; it may be [{best}]. Cite "
                        "the document each statement comes from."
                    )
                if strict:
                    return (
                        f"“{shown}” cites [{number}] without a recognizable "
                        "anchor in its original source passages. Check "
                        "the claim against the source."
                    )
    return ""


def _number(value: str) -> str:
    """Normalize one numeric literal without discarding its sign.

    Parameters
    ----------
    value: str
        A signed number, possibly with an exponent or percentage sign.

    Returns
    -------
    str
        Case-folded literal with Unicode minus normalized and grouping
        commas removed. Unwritten signs are never inferred.

    """

    return value.casefold().replace("\u2212", "-").replace(",", "")


def _comparisons(text: str) -> set[tuple[str, str, str]]:
    """Extract comparison operators and values from source text.

    Parameters
    ----------
    text: str
        A claim or original source text.

    Returns
    -------
    set[tuple[str, str, str]]
        Variable, operator, and signed numerical value, with Unicode
        comparison signs and scientific notation normalized.

    """

    normalized = text.replace("≤", "<=").replace("≥", ">=")
    return {
        (key.casefold(), operator, _number(value))
        for key, operator, value in _COMPARISON.findall(normalized)
    }


def numeric_support_problem(
    content: str, texts: dict[int, str]
) -> str:
    """Reject cited numerical values absent from original source text.

    Parameters
    ----------
    content: str
        A model-written report section.
    texts: dict[int, str]
        Original indexed passage text by document citation number.

    Returns
    -------
    str
        A warning identifying an absent signed value, exponent, value-
        unit pair or statistical comparison; ``""`` when the source
        contains the same literal anchors. This is not a semantic proof.

    """

    source_numbers = {
        number: {_number(value) for value in _FACT_NUMBER.findall(text)}
        for number, text in texts.items()
    }
    source_comparisons = {
        number: _comparisons(text) for number, text in texts.items()
    }
    source_quantities = {
        number: {
            (_number(value), unit.casefold())
            for value, unit in _QUANTITY.findall(text)
        }
        for number, text in texts.items()
    }
    for sentence in re.split(r"(?<=[.!?])\s+", content):
        start = 0
        for match in _BRACKETS.finditer(sentence):
            clause = sentence[start:match.start()]
            start = match.end()
            numbers = _FACT_NUMBER.findall(clause)
            cited = _cited(match.group(1), max(texts, default=0))
            if not cited:
                continue
            for number in cited:
                for comparison in _comparisons(clause):
                    if comparison not in source_comparisons.get(
                        number, set()
                    ):
                        key, operator, value = comparison
                        return (
                            f"Comparison {key} {operator} {value} cited "
                            f"as [{number}] was not found in that "
                            "document's original source passages."
                        )
                for value in numbers:
                    if _number(value) not in source_numbers.get(
                        number, set()
                    ):
                        return (
                            f"Value {value} cited as [{number}] was not "
                            "found in that document's original source "
                            "passages. Check or remove the claim."
                        )
                for value, unit in _QUANTITY.findall(clause):
                    if unit.casefold() in _UNIT_IGNORED:
                        continue
                    normalized = (_number(value), unit.casefold())
                    if normalized not in source_quantities.get(number, set()):
                        return (
                            f"Quantity {value} {unit} cited as [{number}] "
                            "was not found in that document's original "
                            "source passages."
                        )
    return ""


def agreement_problem(content: str) -> str:
    """Return an agreement or disagreement citing one document, or ``""``.

    Parameters
    ----------
    content: str
        A section.

    Returns
    -------
    str
        For the first sentence saying documents agree, disagree, concur
        or conflict whose citation names a single document ("agree on X
        [1], and on Y [2]"): a problem asking for two documents in each, or
        for the claim to be dropped. ``""`` otherwise.

    """

    for sentence in re.split(r"(?<=[.!?])\s+", content):
        if not _AGREEMENT.search(sentence):
            continue
        for match in _BRACKETS.finditer(sentence):
            cited = re.findall(r"\d+", match.group(1))
            if len(set(cited)) == 1 and not re.search(
                r"\d\s*[–-]\s*\d", match.group(1)
            ):
                shown = " ".join(sentence.split()[:12])
                return (
                    f"“{shown}…” says documents agree or disagree but cites "
                    f"only [{cited[0]}]. An agreement or disagreement needs "
                    "at least two documents in its citation, such as "
                    "[1, 4]; describe what one document says without "
                    "calling it an agreement, or leave it out."
                )
    return ""


def coverage_problem(content: str, noted: Sequence[int]) -> str:
    """Return the documents a section covering all of them leaves out.

    Parameters
    ----------
    content: str
        A section marked ``all documents``.
    noted: Sequence[int]
        The documents it must cover: every document read.

    Returns
    -------
    str
        A problem naming the documents the section does not cite; ``""``
        when it cites them all.

    """

    cited = {
        int(number)
        for marker in _BRACKETS.findall(content)
        for number in (_cited(marker, max(noted, default=0)) or ())
    }
    missing = [number for number in noted if number not in cited]
    if not missing:
        return ""
    return (
        "It must cover every document, but leaves out "
        + ", ".join(f"[{number}]" for number in missing)
        + "; include what their notes say, citing them."
    )


def headline_fix_prompt(
    template: ReportTemplate, headline: str
) -> tuple[str, str]:
    """Return the prompts shortening a headline that is too long.

    Parameters
    ----------
    template: ReportTemplate
        The template, with its headline limit.
    headline: str
        The headline as written.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for the same headline in at most
        ``template.headline_characters`` characters.

    """

    user = (
        f"This report headline has {len(headline)} characters; it may have "
        f"at most {template.headline_characters}:\n{headline}\n\nWrite a "
        "shorter headline saying the same, on one line, no full stop."
    )
    return _SYSTEM, user


def fitted_headline(text: str, limit: int) -> str:
    """Return a headline cut to its limit at a word.

    Parameters
    ----------
    text: str
        A headline, possibly too long.
    limit: int
        Most characters.

    Returns
    -------
    str
        ``text`` when it fits; otherwise cut at the last space within the
        limit, without a final "and", "of", "the" or similar word, and
        without trailing punctuation.

    """

    text = " ".join(text.split()).rstrip(".")
    if len(text) <= limit:
        return text
    cut = text[:limit + 1].rsplit(" ", 1)[0]
    while True:
        shorter = _DANGLING.sub("", cut)
        if shorter == cut:
            break
        cut = shorter
    return cut.rstrip(" ,;:—–-")


def cite_document_names(content: str, documents: int) -> tuple[str, int]:
    """Return a section with documents named in the text written as citations.

    Parameters
    ----------
    content: str
        A section.
    documents: int
        How many documents there are.

    Returns
    -------
    tuple[str, int]
        The content with "Document 4" and "Document [4]" written "[4]",
        and "Documents 1 and 4" written "[1] and [4]", keeping the words
        between the numbers; a name with a number that is not a document
        is left as it is. Also how many names were rewritten.

    """

    count = 0

    def cite(match: re.Match[str]) -> str:
        """Return one name as citations, or unchanged.

        Parameters
        ----------
        match: re.Match[str]
            "Document" and the numbers after it.

        Returns
        -------
        str
            The numbers in brackets with the words between them, or the
            match when a number is not a document.

        """

        nonlocal count
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if not all(1 <= number <= documents for number in numbers):
            return match.group(0)
        count += 1
        return re.sub(
            r"\[?(\d+)\]?", lambda n: f"[{n.group(1)}]", match.group(1)
        )

    return _DOCUMENT_NAMES.sub(cite, content), count


def per_document_markdown(
    section: TemplateSection,
    documents: Sequence[ProjectDocument],
    notes: dict[int, DocumentNotes],
) -> str:
    """Return a per-document section, assembled from the checks.

    Parameters
    ----------
    section: TemplateSection
        A per-document section.
    documents: Sequence[ProjectDocument]
        The documents, in order.
    notes: dict[int, DocumentNotes]
        Each read document's notes and checks, by number.

    Returns
    -------
    str
        A count line ("3 of 12 documents."), with the unclear ones and why
        ("Unclear: [4] (the quoted words are not in the document).") and
        those not read; then one "- **Title**: entry [n]" line per matching
        document, or for a table a Markdown table with a Document column.
        "None of the documents" only when every document was checked;
        otherwise "No confirmed matches" when none are established.

    """

    matching: list[tuple[ProjectDocument, object]] = []
    unclear: list[str] = []
    unread: list[int] = []
    for document in documents:
        found = notes.get(document.number)
        if found is None or found.failure:
            unread.append(document.number)
            continue
        check = found.checks.get(section.name)
        if check is None:
            continue
        if check.answer == CHECK_YES:
            matching.append((document, check))
        elif check.answer == CHECK_UNCLEAR:
            unclear.append(
                f"[{document.number}]"
                + (f" ({check.why})" if check.why else "")
            )
    count = (
        f"{len(matching)} of {len(documents)} documents."
        if matching else
        ("No confirmed matches in 1 document." if len(documents) == 1
         else f"No confirmed matches among {len(documents)} documents.")
        if unclear or unread else
        f"None of the {len(documents)} documents."
    )
    if unclear:
        named = unclear[:_UNCLEAR_NAMED]
        more = len(unclear) - len(named)
        count += " Unclear: " + ", ".join(named) + (
            f" and {more} more" if more else ""
        ) + "."
    if unread:
        count += " Not read: " + ", ".join(f"[{n}]" for n in unread) + "."
    lines = [count]
    if not matching:
        return "\n".join(lines)
    lines.append("")
    if section.columns:
        def cell(text: str) -> str:
            """Return a table cell's text.

            Parameters
            ----------
            text: str
                A value.

            Returns
            -------
            str
                On one line, with ``|`` escaped.

            """

            return " ".join(text.split()).replace("|", "\\|")

        lines.append("| Document | " + " | ".join(section.columns) + " |")
        lines.append("| --- |" + " --- |" * len(section.columns))
        for document, check in matching:
            lines.append(
                f"| {cell(document.title)} [{document.number}] | "
                + " | ".join(cell(value) for value in check.cells) + " |"
            )
    else:
        for document, check in matching:
            lines.append(
                f"- **{document.title}**: {check.entry} [{document.number}]"
            )
    return "\n".join(lines)


def documents_line(documents: Sequence[ProjectDocument]) -> str:
    """Return the content of a ``{documents}`` line.

    Parameters
    ----------
    documents: Sequence[ProjectDocument]
        The documents covered.

    Returns
    -------
    str
        "12 documents, listed at the end of the report [1–12]."

    """

    count = len(documents)
    return (
        f"{count} document{'s' if count != 1 else ''}, listed at the end of "
        f"the report [1{f'–{count}' if count > 1 else ''}]."
    )


def documents_list(documents: Sequence[ProjectDocument]) -> str:
    """Return the numbered list of documents closing a project report.

    Parameters
    ----------
    documents: Sequence[ProjectDocument]
        The documents covered.

    Returns
    -------
    str
        "**Documents**" and one "[n] Title (`path`)" line per document, so
        each ``[n]`` of the report can be traced.

    """

    lines = ["**Documents**", ""]
    for document in documents:
        path = document.path.as_posix()
        lines.append(
            f"[{document.number}] {document.title}"
            + (f" (`{path}`)" if document.title != document.path.name
               else "")
            + "  "
        )
    return "\n".join(lines).rstrip()
