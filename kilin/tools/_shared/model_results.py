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

- ``incomplete_notice`` and ``partial_result``

    Mark independent failed items without turning them into negative findings.

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

from collections.abc import Collection, Sequence
from dataclasses import replace
import re

from kilin._model_completion import IncompleteModelResponse
from kilin.tools._contract import ToolResult

__all__ = ["incomplete_notice", "partial_result", "selected_numbers"]


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


def incomplete_notice(failures: Sequence[dict[str, object]]) -> str:
    """Describe uncompleted work without making claims about its evidence.

    Parameters
    ----------
    failures: Sequence[dict[str, object]]
        Item labels with safe ``IncompleteModelResponse.to_dict`` metadata.

    Returns
    -------
    str
        Markdown suffix; empty when all items completed. Raw model text is
        deliberately excluded.

    """

    if not failures:
        return ""
    return (
        "\n\n**Incomplete items** (not negative findings):\n"
        + "\n".join(
            f"- {item['item']}: {item['message']}" for item in failures
        )
    )


def partial_result(
    result: ToolResult,
    failures: Sequence[dict[str, object]],
    *,
    completed: bool = True,
) -> ToolResult:
    """Retain completed work and explicitly mark skipped independent items.

    Parameters
    ----------
    result: ToolResult
        Valid completed results only; never include truncated model output.
    failures: Sequence[dict[str, object]]
        Item labels and safe failure metadata collected by the tool.
    completed: bool, optional
        Whether any independent work completed. Defaults to ``True``.
        A valid zero-match batch counts as completed; a failed batch does not.

    Returns
    -------
    ToolResult
        Original result on success; otherwise a partial or incomplete result
        with an explicit notice and machine-readable item failures. The RAG
        adapter streams the suffix separately when the answer was streamed.

    """

    if not failures:
        return result
    return replace(
        result,
        answer=result.answer + incomplete_notice(failures),
        summary={
            **result.summary,
            "status": "partial" if completed else "incomplete",
            "reason": "item_completion",
            "message": "Some requested items could not be completed.",
            "incomplete_items": list(failures),
        },
    )
