"""Define the supported embedding-query formats and Qwen3 query formatter.

This module provides the canonical query-format identifiers and a formatter
for instruction-prefixed Qwen3 embedding queries. The formatter accepts a
retrieval instruction and question supplied by its caller; project-specific
instructions are defined in the project-type configuration rather than here.

The formatter does not embed text, load models, or modify chat history.
Applying the format is a caller decision: ``EmbeddingSession`` embeds the
query as provided and does not apply this format automatically.

Key Features
------------

- **Shared format identifiers**

    Define the supported embedding-query formats and their ordered choices.

- **Validated Qwen3 formatting**

    Strip surrounding whitespace, require a single-line instruction, and
    preserve internal newlines in the question.

Module Structure
----------------

- ``QUERY_FORMAT_PLAIN`` and ``QUERY_FORMAT_QWEN3``

    Canonical identifiers for the supported embedding-query formats.

- ``QUERY_FORMAT_CHOICES``

    Ordered collection of supported query-format identifiers.

- ``QueryFormat``

    Type alias for the supported embedding-query formats.

- ``format_qwen3_query``

    Format a caller-supplied instruction and question as a Qwen3 query.

Examples
--------
>>> from kilin.query_formats import format_qwen3_query
>>> query = format_qwen3_query(
...     "How are sources split?",
...     "Retrieve relevant source code.",
... )
>>> print(query)
Instruct: Retrieve relevant source code.
Query: How are sources split?

"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias


__all__ = [
    "QUERY_FORMAT_PLAIN",
    "QUERY_FORMAT_QWEN3",
    "QUERY_FORMAT_CHOICES",
    "QueryFormat",
    "format_qwen3_query",
]

QueryFormat: TypeAlias = Literal["plain", "qwen3"]

QUERY_FORMAT_PLAIN: Final[QueryFormat] = "plain"
QUERY_FORMAT_QWEN3: Final[QueryFormat] = "qwen3"
QUERY_FORMAT_CHOICES: Final[tuple[QueryFormat, ...]] = (
    QUERY_FORMAT_PLAIN,
    QUERY_FORMAT_QWEN3,
)


def format_qwen3_query(
    question: str,
    instruction: str,
) -> str:
    """Format one retrieval query for a Qwen3 embedding model.

    Parameters
    ----------
    question: str
        User question. Leading and trailing whitespace is removed; internal
        line breaks are preserved.
    instruction: str
        Retrieval instruction. After surrounding whitespace is removed, it
        must contain exactly one nonempty line.

    Returns
    -------
    str
        Query text containing separate ``Instruct:`` and ``Query:`` lines.

    Raises
    ------
    ValueError
        If either argument is not nonempty text, or if the trimmed
        instruction contains a line break.

    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty text.")

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be non-empty text.")

    instruction_text = instruction.strip()
    if len(instruction_text.splitlines()) != 1:
        raise ValueError("Instruction must be a single line.")

    instruction_line = f"Instruct: {instruction_text}"
    question_line = f"Query: {question.strip()}"
    return f"{instruction_line}\n{question_line}"
