"""Validate model completion before treating generated text as a result.

A model can emit nonempty text while exhausting its output budget inside a
thinking block. Completion is therefore separate from text availability.
This dependency-free module is shared by chat, answering and tools so that
an incomplete model reply cannot silently become an empty search result.

Key Features
------------

- **Explicit failures**

    Carry a machine-readable reason and the backend's finish reason.

- **Final-content checks**

    Reject budget exhaustion, unfinished thinking and missing final text.

- **No extra inference**

    Validate the existing reply without retries or additional model calls.

Module Structure
----------------

- ``IncompleteModelResponse``

    An operation has no complete, usable model result.

- ``strip_thinking`` and ``thinking_is_open``

    Separate final text from model-specific thinking delimiters.

- ``require_complete_reply``

    Check completion before a tool consumes or publishes generated text.

Examples
--------
>>> from kilin._model_completion import require_complete_reply
>>> require_complete_reply("<think>Check</think>Result", "stop")
>>> require_complete_reply("NONE", "stop")

"""

from __future__ import annotations

import re

__all__ = [
    "IncompleteModelResponse",
    "require_complete_reply",
    "strip_thinking",
    "thinking_is_open",
]


class IncompleteModelResponse(RuntimeError):
    """Report a model reply that must not be interpreted as a result.

    Attributes
    ----------
    reason: str
        Machine-readable failure category, such as ``token_limit``,
        ``unfinished_thinking``, ``empty_answer`` or ``invalid_output``.
    finish_reason: str | None
        Backend termination reason, when available.
    raw_text: str
        Original reply for diagnostics and streaming cleanup. Never use it
        as a successful answer or as evidence of a negative result.

    """

    reason: str
    finish_reason: str | None
    raw_text: str

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        finish_reason: str | None = None,
        raw_text: str = "",
    ) -> None:
        """Initialize a failure without discarding its generation status.

        Parameters
        ----------
        message: str
            Human-readable explanation of the failed check.
        reason: str
            Stable diagnostic category.
        finish_reason: str | None, optional
            Backend termination reason. Defaults to ``None``.
        raw_text: str, optional
            Unmodified model reply. Defaults to ``""``.

        """

        super().__init__(message)
        self.reason = reason
        self.finish_reason = (
            finish_reason if isinstance(finish_reason, str) else None
        )
        self.raw_text = raw_text

    def to_dict(self) -> dict[str, object]:
        """Return safe completion metadata, without the raw model text.

        Returns
        -------
        dict[str, object]
            JSON-compatible failure status for traces and saved turns.

        """

        return {
            "status": "incomplete",
            "reason": self.reason,
            "finish_reason": self.finish_reason,
            "message": str(self),
        }


def strip_thinking(text: str) -> str:
    """Remove thinking blocks, including one opened by the chat template.

    Parameters
    ----------
    text: str
        Generated model output.

    Returns
    -------
    str
        Final text with surrounding whitespace removed.

    """

    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.split("<think>", 1)[0].strip()


def thinking_is_open(text: str) -> bool:
    """Check whether the final thinking block lacks a closing delimiter.

    Parameters
    ----------
    text: str
        Generated text, which may start inside a template-opened block.

    Returns
    -------
    bool
        True if a visible opening delimiter has no subsequent close.
        A closing delimiter alone is valid for template-opened thinking.

    """

    depth = 0
    for tag in re.findall(r"</?think>", text):
        depth = depth + 1 if tag == "<think>" else max(0, depth - 1)
    return depth > 0


def require_complete_reply(
    text: str, finish_reason: str | None = None
) -> None:
    """Require a complete final response before parsing or publishing it.

    Parameters
    ----------
    text: str
        Full generated text, before stripping any thinking.
    finish_reason: str | None, optional
        Backend reason. ``length`` means the output budget was exhausted.
        ``None`` supports adapters without termination metadata, but still
        checks the text. Defaults to ``None``.

    Raises
    ------
    IncompleteModelResponse
        If generation exhausted its budget, left thinking unfinished or
        produced no final text. This also rejects a truncated reply which
        happens to contain parseable numbers or JSON.

    Notes
    -----
    This verifies completion, not factual correctness or a tool's output
    schema. Tool-specific validators must still check the returned text.
    An implicit, unclosed template thinking block cannot be identified from
    untagged text alone; the backend's finish reason remains authoritative.

    """

    if finish_reason == "length":
        raise IncompleteModelResponse(
            "The model reached its output-token limit before finishing.",
            reason="token_limit", finish_reason=finish_reason, raw_text=text,
        )
    if thinking_is_open(text):
        raise IncompleteModelResponse(
            "The model did not finish its thinking block.",
            reason="unfinished_thinking", finish_reason=finish_reason,
            raw_text=text,
        )
    if not strip_thinking(text):
        raise IncompleteModelResponse(
            "The model returned no final answer text.",
            reason="empty_answer", finish_reason=finish_reason, raw_text=text,
        )
