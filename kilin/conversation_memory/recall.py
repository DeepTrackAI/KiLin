"""Recognize narrow requests for verifiable conversation history.

A request about whether the researcher previously discussed a topic should
be checked against saved message records, not inferred from embedding
similarity or the answering model's general knowledge. This detector is
intentionally conservative: ordinary scientific questions and quoted text
remain ordinary queries.

Key Features
------------

- **No additional model call**

    Detect explicit first-person conversation-recall questions locally.

- **Explicit historical topic**

    Extract the topic whose occurrence can be checked in saved messages.

Module Structure
----------------

- ``recall_topic``

    Return a historical lookup topic, or None for ordinary questions.

Examples
--------
>>> from kilin.conversation_memory.recall import recall_topic
>>> recall_topic("Did we discuss optical tweezers?")
'optical tweezers'

"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["recall_topic"]

_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:did|have) we (?:ever )?(?:discuss|discussed|mention|"
        r"mentioned|consider|considered|talk(?:ed)? about) (.+)",
        r"(?:do|can) you remember (?:if|whether) we (?:ever )?"
        r"(?:discussed|mentioned|considered|talked about) (.+)",
        r"(?:what|how) did we (?:say|decide|agree|conclude|discuss) "
        r"(?:about |on )?(.+)",
        r"(?:what|how) have we (?:said|decided|agreed|concluded|"
        r"discussed) (?:about |on )?(.+)",
    )
)


def recall_topic(question: str) -> str | None:
    """Extract a topic from an explicit historical-conversation request.

    Parameters
    ----------
    question: str
        Current researcher's question.

    Returns
    -------
    str | None
        Nonempty topic, or None when no narrowly supported pattern matches.

    Notes
    -----
    A matched pattern is a lookup request, not proof of a past discussion.
    Retrieval still has to inspect eligible saved conversation messages.

    """

    if not isinstance(question, str):
        raise ValueError("The conversation question must be text.")
    stripped = question.strip().rstrip("?.! ").strip()
    if not stripped or "\n" in stripped:
        return None
    for pattern in _PATTERNS:
        match = pattern.fullmatch(stripped)
        if match:
            topic = match.group(1).strip().strip('"\'')
            if len(topic) >= 3:
                return topic
    return None
