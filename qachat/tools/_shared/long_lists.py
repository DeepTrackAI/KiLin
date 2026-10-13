"""Offer to save a long list as a Markdown file instead of the chat.

A list of hundreds of mentions or sentences is hard to read in a chat.
Tools that list results call ``offer_export`` when a list is longer than
``CHAT_LIST_LIMIT``: the user chooses between saving the full list to
``outputs/`` as Markdown (the chat then shows the counts and where the
file is) and seeing the first items in the chat.

The saved file drops passage citations such as ``[3]``, which point at
evidence shown in the chat and mean nothing outside QAChat. Where nobody
can be asked (notebooks), nothing is saved and the tool shows the first
items as usual. A file never replaces an existing one; see
``qachat.tools._shared.outputs``.

Key Features
------------

- **Chat limit**

    One shared length above which tools offer a file.

- **Export offer**

    Ask the user, then save the full list without passage citations.

- **File names**

    Build a readable Markdown file name from the listed words.

Module Structure
----------------

- ``CHAT_LIST_LIMIT``

    Longest list shown in the chat; longer lists are offered as a file.

- ``offer_export``

    Ask whether to save a long list as a Markdown file, and save it.

- ``file_name``

    Return a Markdown file name built from a prefix and some words.

Examples
--------
>>> from qachat.tools._shared.long_lists import file_name, offer_export
>>> file_name("mentions", "Gradient descent")
'mentions_gradient_descent.md'
>>> saved = offer_export(
...     context,
...     count=30,
...     what="mentions of “MRI”",
...     name=file_name("mentions", "MRI"),
...     markdown=markdown,
... )

``context`` is the ``ToolContext`` a tool's ``run`` receives and
``markdown`` the full list; ``saved`` is None unless the user chose to save.

"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final

from qachat.tools._contract import ToolContext
from qachat.tools._shared.outputs import save_output_file
from qachat.tools._shared.planning import ask_choice

__all__ = ["CHAT_LIST_LIMIT", "file_name", "offer_export"]

# Longest list shown in the chat; longer lists are offered as a file.
CHAT_LIST_LIMIT: Final[int] = 20
# Passage citations, which mean nothing outside QAChat.
_CITATION: Final[re.Pattern[str]] = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
# Longest part of a file name taken from the listed words.
_SLUG_LENGTH: Final[int] = 40


def offer_export(
    context: ToolContext,
    *,
    count: int,
    what: str,
    name: str,
    markdown: str,
) -> Path | None:
    """Ask whether to save a long list as a Markdown file, and save it.

    Parameters
    ----------
    context: ToolContext
        QAChat's services (``choose``, ``project_root``).
    count: int
        Number of items in the list.
    what: str
        What the items are, such as "mentions of “MRI”".
    name: str
        Wanted file name without folder, such as ``mentions_mri.md``; a
        taken name gets ``-2``, ``-3``, ...
    markdown: str
        The full list as Markdown.

    Returns
    -------
    pathlib.Path | None
        The saved file (relative to the project) when the user chose to
        save it; None when the list is short, nobody can be asked
        (notebooks), or the user wants it in the chat.

    """

    if count <= CHAT_LIST_LIMIT or context.choose is None:
        return None

    index = ask_choice(
        f"Found {count} {what}. Save the full list as a Markdown file?",
        (
            f"Save all {count} to outputs/ as Markdown",
            f"Show the first {CHAT_LIST_LIMIT} here",
        ),
        context,
    )
    if index != 0:
        return None

    text = _CITATION.sub("", markdown).strip() + "\n"
    return save_output_file(context.project_root, name, text)


def file_name(prefix: str, words: str) -> str:
    """Return a Markdown file name built from a prefix and some words.

    Parameters
    ----------
    prefix: str
        Start of the name, such as ``"mentions"``.
    words: str
        Words describing the list, such as ``"gradient descent"``. Runs of
        characters other than lowercase ASCII letters and digits become one
        underscore, and the result is cut to 40 characters.

    Returns
    -------
    str
        A name such as ``mentions_gradient_descent.md``;
        ``<prefix>_list.md`` when no letter or digit remains.

    """

    slug = re.sub(r"[^a-z0-9]+", "_", words.casefold()).strip("_")
    slug = slug[:_SLUG_LENGTH]
    return f"{prefix}_{slug or 'list'}.md"
