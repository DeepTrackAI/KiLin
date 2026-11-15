"""Reduce lines of LaTeX to the readable text they print.

Tools that read LaTeX sources need the prose of a line, not its commands:
``reference_check`` to find citing sentences, ``writing_check`` to give a
model a paragraph to check. Each line is reduced on its own, so line
numbers stay those of the file. Citation commands are replaced by what the
caller chooses (reference numbers, a placeholder).

Key Features
------------

- **Line by line**

    Comments, labels and environment commands are removed; formatting
    commands keep their words; accents are written as plain letters.

- **Caller-defined citations**

    Each ``\\cite``-family command is replaced by the caller's text.

Module Structure
----------------

- ``LATEX_CITE``, ``LATEX_COMMENT`` and ``NOT_CITING``

    The citation command, a comment, and citation-like commands that
    do not cite.

- ``readable_latex``

    One line of LaTeX as readable text.

Examples
--------
>>> from kilin.tools._shared.latex_text import readable_latex
>>> readable_latex(r"As \\emph{shown} \\cite{a}. % note", lambda m: "[1]")
'As shown [1]. '

"""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Final

__all__ = [
    "LATEX_CITE",
    "LATEX_COMMENT",
    "NOT_CITING",
    "readable_latex",
]

# A citation command with its optional notes and its keys.
LATEX_CITE: Final[re.Pattern[str]] = re.compile(
    r"\\(?P<name>[A-Za-z]*[Cc]ite[A-Za-z]*)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{(?P<keys>[^{}]*)\}"
)
# Citation-like commands that do not cite in the text (lower case).
NOT_CITING: Final[frozenset[str]] = frozenset({
    "nocite", "citeauthor", "citeyear", "citeyearpar",
})
# A comment: from an unescaped % to the end of the line.
LATEX_COMMENT: Final[re.Pattern[str]] = re.compile(r"(?<!\\)%.*$")
_INLINE_MATH: Final[re.Pattern[str]] = re.compile(
    r"\$([^$]*)\$|\\\((.*?)\\\)"
)
# A line that starts a section: its title becomes a sentence of its own.
_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*\\(?:part|chapter|section|subsection|subsubsection|paragraph)"
    r"(?![A-Za-z])"
)
_REFERENCE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\*?\s*\{[^{}]*\}"
)
_DROPPED: Final[re.Pattern[str]] = re.compile(
    r"\\(?:label|begin|end|includegraphics|input|include|bibliography"
    r"|bibliographystyle|addbibresource|vspace|hspace)\*?"
    r"(?:\s*\[[^\]]*\])?\s*\{[^{}]*\}"
)
_WRAPPED: Final[re.Pattern[str]] = re.compile(
    r"\\[A-Za-z]+\*?\s*(?:\[[^\]]*\])?\{([^{}]*)\}"
)
_COMMAND: Final[re.Pattern[str]] = re.compile(r"\\[A-Za-z]+\*?")
# A command that writes a character with an accent ("\'e", "\"{o}",
# "\v{c}"); a letter accent is not the start of a longer command.
_ACCENT: Final[re.Pattern[str]] = re.compile(
    r"\\(?:[`'^\"~=.]|[uvHcdbtk](?![A-Za-z]))\s*\{?([A-Za-z])\}?"
)


def readable_latex(
    line: str, cite: Callable[[re.Match[str]], str]
) -> str:
    """Return one line of LaTeX as readable text.

    Parameters
    ----------
    line: str
        One line.
    cite: Callable[[re.Match[str]], str]
        The text replacing each citation command (a ``LATEX_CITE`` match,
        with groups ``name`` and ``keys``).

    Returns
    -------
    str
        The line without its comment, citations as ``cite`` writes them,
        math kept without brackets (so "[0, 1]" is not a citation),
        section titles ending with a full stop so they do not join the
        next sentence, cross-references as "(ref)", formatting commands
        reduced to their words, other commands and braces removed.

    """

    text = LATEX_COMMENT.sub("", line)
    text = LATEX_CITE.sub(cite, text)
    text = _INLINE_MATH.sub(
        lambda match: (match.group(1) or match.group(2) or "")
        .replace("[", "(").replace("]", ")"),
        text,
    )
    heading = _HEADING.match(text) is not None
    text = _REFERENCE.sub("(ref)", text)
    text = _DROPPED.sub("", text)
    text = _ACCENT.sub(r"\1", text)
    previous = None
    while previous != text:
        previous, text = text, _WRAPPED.sub(r"\1", text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\\\", " ")
    text = _COMMAND.sub("", text)
    text = text.replace("{", "").replace("}", "").replace("~", " ")
    if heading and text.strip() and text.rstrip()[-1] not in ".?!:":
        text = text.rstrip() + "."
    return text
