"""Validate short tool replies without confusing failure with no matches.

Sentence and item selectors request either numbered matches or an explicit
``NONE`` sentinel. Empty text and unparseable prose are not negative search
results, even if model generation ended normally.

Key Features
------------

- **Explicit negatives**

    Accept zero matches only when the model returns ``NONE``.

- **Bounded selections**

    Keep selected numbers within the supplied batch's identifiers.

Module Structure
----------------

- ``selected_numbers``

    Parse a completed selector reply or raise an incomplete-result error.

Examples
--------
>>> from kilin.tools._shared.model_results import selected_numbers
>>> selected_numbers("1, 3", {1, 2, 3})
[1, 3]
>>> selected_numbers("NONE", {1, 2, 3})
[]

"""

from __future__ import annotations

from collections.abc import Collection
import re

from kilin._model_completion import IncompleteModelResponse

__all__ = ["selected_numbers"]


def selected_numbers(reply: str, allowed: Collection[int]) -> list[int]:
    """Parse comma- or whitespace-separated item numbers, or ``NONE``.

    Parameters
    ----------
    reply: str
        Completed, thinking-free model reply.
    allowed: Collection[int]
        Valid item identifiers in the current batch.

    Returns
    -------
    list[int]
        Sorted unique valid identifiers. Out-of-batch numbers are ignored
        when at least one valid identifier remains, as in earlier tools.

    Raises
    ------
    IncompleteModelResponse
        If the reply is malformed or selects no valid number without the
        explicit ``NONE`` sentinel. No model retry is attempted.

    """

    value = reply.strip()
    if value.upper() == "NONE":
        return []
    if re.fullmatch(r"\d+(?:[\s,]+\d+)*", value):
        selected = sorted({int(n) for n in re.findall(r"\d+", value)}
                          & set(allowed))
        if selected:
            return selected
    raise IncompleteModelResponse(
        "The model did not return valid item numbers or NONE.",
        reason="invalid_output", raw_text=reply,
    )
