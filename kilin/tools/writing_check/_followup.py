"""Rewrite the sentence a writing check flagged, when the user asks.

Private to ``writing_check``. After a check, "suggest a better phrasing
for line 75", "split the sentence at line 75" or a pasted finding with
"rephrase it" asks for a rewrite of one flagged sentence. The general
intents cannot do it: they see the earlier answer, where a finding shows
only a few words. This module:

1. Recognizes the request in code: words asking for a rewrite and a line
   number (``rewrite_request``).
2. Finds what to rewrite: the finding the earlier check listed at that
   line (its whole sentence, from the document), or the paragraph at that
   line when no finding is listed there.
3. Asks the model, in one call, for two or three complete rewrites that
   say the same thing, split when the finding is a long sentence.
4. Keeps only rewrites that pass conservative wording safeguards
   (``meaning_change``), introduce few new words, and are not cut short
   with an ellipsis. Human review is still required.

Key Features
------------

- **The whole sentence**

    Taken from the document, not from the few words the answer showed.

- **Checked rewrites**

    The same conservative screens as the paragraph suggestions; these
    cannot guarantee unchanged scientific meaning.

Module Structure
----------------

- ``ShownFinding`` and ``shown_findings``

    The findings an earlier check's answer listed.

- ``asks_rewrite``, ``rewrite_request`` and ``target_text``

    Whether a question asks for a rewrite of a line, and the text to
    rewrite.

- ``rewrite_prompt``, ``parse_options`` and ``option_rejection``

    The model call, its options, and why an option cannot be kept.

Examples
--------
>>> from kilin.tools.writing_check._followup import rewrite_request
>>> rewrite_request("suggest a better phrasing for line 75")
75

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Final

from kilin._model_completion import strip_thinking
from kilin.tools.writing_check._checks import Glossary, sentences
from kilin.tools.writing_check._document import Document
from kilin.tools.writing_check._review import locate, meaning_change

__all__ = [
    "ShownFinding",
    "asks_rewrite",
    "option_rejection",
    "parse_options",
    "rewrite_prompt",
    "rewrite_request",
    "shown_findings",
    "target_text",
]

# Words asking for a rewrite, and a line number, in either order.
_REWRITE_WORDS: Final[str] = (
    r"\b(?:rephras\w*|rewrit\w*|reword\w*|split\w*|shorten\w*|simplif\w*|"
    r"phrasing|alternatives?|better\s+(?:version|wording))\b"
)
_LINE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:extracted-text\s+)?line\s+(\d+)\b", re.I
)
_REWRITE: Final[re.Pattern[str]] = re.compile(_REWRITE_WORDS, re.I)
# Patterns for the project types' cues, matching such a request.
REWRITE_CUES: Final[tuple[str, ...]] = (
    _REWRITE_WORDS + r"[\s\S]{0,400}?\bline\s+\d+\b",
    r"\bline\s+\d+\b[\s\S]{0,400}?" + _REWRITE_WORDS,
)
# One finding line of a writing check's answer (``chat_answer``).
_SHOWN: Final[re.Pattern[str]] = re.compile(
    r"^- (?:extracted-text )?line (?P<line>\d+) · "
    r"(?P<level>\w+): “(?P<original>.+?)”"
    r"(?: → “(?P<suggestion>.*?)”)? \((?P<reason>.*)\)(?: \[\d+\])?\s*$",
    re.MULTILINE,
)
_CHECK_ANSWER: Final[str] = "**Writing check of "
_LONG: Final[re.Pattern[str]] = re.compile(r"\d+ words; consider splitting")
_OPTION: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:OPTION\s*\d*\s*:|\d+[.)])\s*(?P<text>.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_ELLIPSIS: Final[re.Pattern[str]] = re.compile(r"\.\.\.|…")
_CONTENT_WORD: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]{4,}")
# Words a rewrite may add: what joins split sentences, not new content.
_JOINING: Final[frozenset[str]] = frozenset({
    "this", "these", "that", "those", "they", "them", "their", "which",
    "with", "while", "where", "whereas", "thus", "therefore", "also",
    "both", "each", "such", "here", "there", "then", "when", "into",
    "from", "than", "allows", "allowing", "enables", "enabling",
})
# New content words a rewrite may bring before it counts as a new claim.
_NEW_WORDS_MAX: Final[int] = 2


@dataclass(frozen=True, slots=True)
class ShownFinding:
    """A finding an earlier writing check listed in its answer.

    Attributes
    ----------
    line: int
        Its source line.
    level: str
        ``grammar``, ``style`` or ``consistency``.
    original: str
        The words the answer quoted (for a long sentence, its start).
    suggestion: str
        The suggested words, or ``""``.
    reason: str
        The reason shown.

    """

    line: int
    level: str
    original: str
    suggestion: str
    reason: str

    @property
    def long_sentence(self) -> bool:
        """Whether the finding is a sentence too long.

        Returns
        -------
        bool
            True for the long-sentence check's findings.

        """

        return bool(_LONG.search(self.reason))


def rewrite_request(question: str) -> int | None:
    """Return the line a question asks to rewrite, if it asks for one.

    Parameters
    ----------
    question: str
        The question as typed, possibly with a pasted finding.

    Returns
    -------
    int | None
        The first line number of a question that asks for a rewrite
        (rephrase, rewrite, reword, split, shorten, simplify, a better
        phrasing, alternatives); None otherwise.

    """

    line = _LINE.search(question)
    if line is None or not _REWRITE.search(question):
        return None
    return int(line.group(1))


def asks_rewrite(question: str) -> bool:
    """Tell whether a question asks to rewrite a line.

    The registry uses it as ``follows_up_when``: such a question follows
    up a check, so routing passes it the earlier turn.

    Parameters
    ----------
    question: str
        The question as typed.

    Returns
    -------
    bool
        True when ``rewrite_request`` finds a line.

    """

    return rewrite_request(question) is not None


def shown_findings(answer: str) -> list[ShownFinding]:
    """Return the findings an earlier writing check's answer listed.

    Parameters
    ----------
    answer: str
        An earlier answer.

    Returns
    -------
    list[ShownFinding]
        Its finding lines, in order; empty when it is not a writing
        check's answer.

    """

    if _CHECK_ANSWER not in answer:
        return []
    return [
        ShownFinding(
            int(match["line"]), match["level"], match["original"],
            match["suggestion"] or "", match["reason"],
        )
        for match in _SHOWN.finditer(answer)
    ]


def target_text(
    document: Document,
    line: int,
    findings: Sequence[ShownFinding],
    question: str,
) -> tuple[int, str, ShownFinding | None] | None:
    """Return the text a rewrite request is about.

    Parameters
    ----------
    document: Document
        The checked document.
    line: int
        The line the question names.
    findings: Sequence[ShownFinding]
        The findings the earlier check listed.
    question: str
        The question, possibly quoting one of them.

    Returns
    -------
    tuple[int, str, ShownFinding | None] | None
        The paragraph's index, the text to rewrite and the finding it
        answers: the whole sentence holding the finding's words when the
        check listed one at that line (a quoted one first, then a long
        sentence when the question asks to split, else the first),
        otherwise the paragraph holding the line. None when no paragraph
        holds it.

    """

    index = next(
        (i for i, paragraph in enumerate(document.paragraphs)
         if paragraph.start_line <= line <= paragraph.end_line),
        None,
    )
    if index is None:
        return None
    text = document.paragraphs[index].text
    at_line = [finding for finding in findings if finding.line == line]
    quoted = [f for f in at_line if f.original in question]
    split = [f for f in at_line if f.long_sentence] if re.search(
        r"\b(?:split|long)\b", question, re.I
    ) else []
    chosen = (quoted or split or at_line or [None])[0]
    if chosen is None:
        return index, text, None
    found = locate(chosen.original, text)
    if found is None:
        return index, text, chosen
    for start, sentence in sentences(text):
        if start <= found[1] < start + len(sentence) + 1:
            return index, sentence, chosen
    return index, text, chosen


def rewrite_prompt(
    text: str,
    paragraph: str,
    context: str,
    finding: ShownFinding | None,
) -> tuple[str, str]:
    """Return the prompts asking for rewrites of a sentence or paragraph.

    Parameters
    ----------
    text: str
        The text to rewrite.
    paragraph: str
        The paragraph holding it, as context.
    context: str
        What the model is told about the document (``document_context``).
    finding: ShownFinding | None
        The finding the rewrite answers, if any.

    Returns
    -------
    tuple[str, str]
        System and user prompt asking for two or three complete versions,
        each on an ``OPTION:`` line.

    """

    system = (
        "You are an experienced editor of scientific writing. You rewrite "
        "text so it reads better while saying exactly the same."
    )
    task = "Write two or three alternative versions of the text."
    if finding is not None and finding.long_sentence:
        task += (
            " It is too long: split it into two or three sentences, "
            "keeping all of its content."
        )
    elif finding is not None:
        task += f" Address this: {finding.reason}."
    parts = []
    if context:
        parts.append(f"The document:\n{context}")
    if paragraph != text:
        parts.append(
            "Paragraph, for reference only:\n" + paragraph
        )
    parts.append(f'Text to rewrite:\n"""\n{text}\n"""')
    parts.append(
        f"{task} Each version must say exactly what the text says: keep "
        "every fact, claim, number, name, term and acronym, the "
        "placeholders [cite], [author], [year] and (ref), citations such "
        "as (Smith et al., 2020), and every hedge "
        "or modal verb (\"virtually\", \"would\", \"may\"). Add nothing the "
        "paragraph does not say. Write each version in full, never "
        "shortened with \"...\".\n\nWrite each version on its own line:\n"
        "OPTION: <the full rewritten text>"
    )
    return system, "\n\n".join(parts)


def parse_options(reply: str) -> list[str]:
    """Return the versions a model reply offers.

    Parameters
    ----------
    reply: str
        The model's reply.

    Returns
    -------
    list[str]
        The text of each ``OPTION:`` (or numbered) line, without
        surrounding quotation marks, at most three, repeats removed.

    """

    found: list[str] = []
    for match in _OPTION.finditer(strip_thinking(reply)):
        text = match["text"].strip()
        if len(text) >= 2 and text[0] in "\"“" and text[-1] in "\"”":
            text = text[1:-1].strip()
        if text and text not in found:
            found.append(text)
    return found[:3]


def option_rejection(
    text: str, option: str, paragraph: str, glossary: Glossary
) -> str:
    """Return why a rewrite cannot be kept, or ``""``.

    Parameters
    ----------
    text: str
        The text rewritten.
    option: str
        A rewrite.
    paragraph: str
        The paragraph holding the text.
    glossary: Glossary
        The code checks' glossary.

    Returns
    -------
    str
        ``""`` when the rewrite can be kept; otherwise why not: it is cut
        short with an ellipsis, changes nothing, changes meaning
        (``meaning_change``), or adds more than two content words the
        paragraph does not have (a new claim, such as "offering
        additional functionality").

    """

    if _ELLIPSIS.search(option) and not _ELLIPSIS.search(text):
        return "it is cut short with an ellipsis"
    if " ".join(option.split()) == " ".join(text.split()):
        return "nothing changes"
    changed = meaning_change(text, option, glossary)
    if changed:
        return changed
    known = {word.casefold() for word in _CONTENT_WORD.findall(paragraph)}
    new = [
        word for word in dict.fromkeys(
            word.casefold() for word in _CONTENT_WORD.findall(option)
        )
        if word not in known and word not in _JOINING
        and word.rstrip("s") not in known and word + "s" not in known
    ]
    if len(new) > _NEW_WORDS_MAX:
        return "it adds words the paragraph does not have: " + ", ".join(
            new[:5]
        )
    return ""
