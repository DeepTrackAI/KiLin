"""Mask non-code JavaScript text for conservative structural analysis.

The scanner keeps source offsets and line endings unchanged while replacing
comments, quoted strings, template literals and likely regular-expression
literals with spaces. Template interpolations are excluded deliberately:
recognizing their executable expressions would require a full parser. Slash
classification is heuristic, so the result is not a JavaScript AST.

Key Features
------------

- **Offset stability**

    Keep original line endings and character positions for source units.

- **Conservative lexical hints**

    Exclude likely non-code spans without executing source text.

Module Structure
----------------

- ``mask_non_code``

    Return an offset-preserving view of code outside literal text.

Examples
--------
>>> from kilin.sources.extensions._js_lex import mask_non_code
>>> mask_non_code("const x = 'text';").startswith("const x =")
True

"""

from __future__ import annotations

from typing import Final

__all__ = ["mask_non_code"]

_REGEX_PREFIX_WORDS: Final[frozenset[str]] = frozenset(
    {
        "await", "case", "delete", "do", "else", "in", "instanceof",
        "new", "of", "return", "throw", "typeof", "void", "yield",
    }
)


def _mask(chars: list[str], start: int, end: int) -> None:
    """Replace a non-code span with spaces, retaining line endings.

    Parameters
    ----------
    chars: list[str]
        Mutable copy of the original JavaScript text.
    start: int
        Inclusive beginning of the span.
    end: int
        Exclusive end of the span.

    """

    for position in range(start, end):
        if chars[position] not in "\r\n":
            chars[position] = " "


def _quoted_end(text: str, start: int) -> int:
    """Find the end of a quoted string or template literal.

    Parameters
    ----------
    text: str
        JavaScript source text.
    start: int
        Offset of the opening quote or backtick.

    Returns
    -------
    int
        Offset just past the closing delimiter or the available text.

    """

    quote = text[start]
    position = start + 1
    while position < len(text):
        char = text[position]
        if char == "\\":
            position += 2
            continue
        if char == quote:
            return position + 1
        if char in "\r\n" and quote != "`":
            return position
        position += 1
    return len(text)


def _regex_end(text: str, start: int) -> int | None:
    """Find a probable regular-expression literal, including its flags.

    Parameters
    ----------
    text: str
        JavaScript source text.
    start: int
        Offset of a possible opening slash.

    Returns
    -------
    int | None
        Exclusive literal end, or None if no closing slash was found.

    """

    position = start + 1
    in_class = False
    while position < len(text):
        char = text[position]
        if char in "\r\n":
            return None
        if char == "\\":
            position += 2
            continue
        if char == "[":
            in_class = True
        elif char == "]":
            in_class = False
        elif char == "/" and not in_class:
            position += 1
            while position < len(text) and text[position].isalpha():
                position += 1
            return position
        position += 1
    return None


def mask_non_code(text: str) -> str:
    """Return JavaScript source with comments and literals masked.

    Parameters
    ----------
    text: str
        Source text to inspect without executing it.

    Returns
    -------
    str
        Equal-length source with line endings unchanged. Masked spans
        cannot contribute declarations, braces, or call candidates.

    Notes
    -----
    Slash disambiguation uses preceding tokens. Unsupported JavaScript
    syntax is intentionally left as context rather than guessed as code.

    """

    chars = list(text)
    position = 0
    regex_allowed = True
    while position < len(text):
        char = text[position]
        next_char = text[position + 1] if position + 1 < len(text) else ""
        if char.isspace():
            position += 1
            continue
        if char == "/" and next_char == "/":
            end = text.find("\n", position + 2)
            end = len(text) if end < 0 else end
            _mask(chars, position, end)
            position = end
            continue
        if char == "/" and next_char == "*":
            end = text.find("*/", position + 2)
            end = len(text) if end < 0 else end + 2
            _mask(chars, position, end)
            position = end
            continue
        if char in ("'", '"', "`"):
            end = _quoted_end(text, position)
            _mask(chars, position, end)
            position = max(end, position + 1)
            regex_allowed = False
            continue
        if char == "/" and regex_allowed:
            end = _regex_end(text, position)
            if end is not None:
                _mask(chars, position, end)
                position = end
                regex_allowed = False
                continue
        if char.isalpha() or char in "_$":
            end = position + 1
            while end < len(text) and (
                text[end].isalnum() or text[end] in "_$"
            ):
                end += 1
            regex_allowed = text[position:end] in _REGEX_PREFIX_WORDS
            position = end
            continue
        if char.isdigit():
            end = position + 1
            while end < len(text) and (
                text[end].isalnum() or text[end] in "_."
            ):
                end += 1
            regex_allowed = False
            position = end
            continue
        regex_allowed = char not in ")}]."
        position += 1
    return "".join(chars)
