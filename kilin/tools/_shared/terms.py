"""Extract and validate exact search terms for registered tools.

Term-based tools share model prompts, deterministic validation, and
flexible matching. These routines live in private infrastructure so
no tool must import another executable tool for them.

Key Features
------------

- **Terms from user text**

    Accept only terms and document names the user actually supplied.

- **Deterministic matching**

    Match punctuation and spacing variants of exact terms.

Module Structure
----------------

- ``MAX_TERMS``, ``TERMS_REPLY_TOKENS`` and
  ``CATEGORIES_REPLY_TOKENS``

    Stable extraction limits and model reply budgets.

- ``quoted_terms`` and ``terms_prompt``

    Read user-quoted terms or request terms from the router.

- ``validated_terms``, ``validated_document``, ``document_phrase``

    Validate the reply and recognize named documents.

- ``term_pattern``

    Compile a robust exact-term search pattern.

Examples
--------
>>> from kilin.tools._shared.terms import quoted_terms
>>> quoted_terms('list "MRI"')
('MRI',)

"""

from __future__ import annotations

import json
import re
from typing import Final

__all__ = [
    "CATEGORIES_REPLY_TOKENS",
    "MAX_TERMS",
    "TERMS_REPLY_TOKENS",
    "document_phrase",
    "quoted_terms",
    "term_pattern",
    "terms_prompt",
    "validated_document",
    "validated_terms",
]

# More terms would make one answer hard to read; ask again for others.
MAX_TERMS: Final[int] = 5

# Settings shared by tools that collect mentions.
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
TERMS_REPLY_TOKENS: Final[int] = 128
CATEGORIES_REPLY_TOKENS: Final[int] = 256

# Extra field of the terms prompt when ``batch_classify`` needs categories.
_CATEGORY_FIELD: Final[str] = """
"categories": 2 to 6 short labels for the groups the user wants each mention
sorted into, taken from the message (for example "own code" and "citation
of other work")."""



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
        ``kilin.tools.batch_classify.validated_categories``.

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
