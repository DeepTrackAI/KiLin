"""Format instruction-prefixed queries for the Qwen3 retrieval path.

The module stores separate software-repository and literature-repository
instruction strings and joins a validated instruction and question into the
``Instruct: ...`` / ``Query: ...`` text expected by the configured retrieval
workflow. It does not embed text, load a model, or alter a chat history.
Applying this format is a caller decision; ``EmbeddingSession`` does not
apply it automatically.

Key Features
------------

- **Reusable instructions**

    Keep the software and literature query instructions in named string
    constants.

- **Explicit formatting**

    Validate and strip the supplied strings before constructing one embedding
    query.

Module Structure
----------------

- ``QUERY_FORMAT_PLAIN`` and ``QUERY_FORMAT_QWEN3``

    Canonical identifiers for the supported embedding-query formats.

- ``QUERY_FORMAT_CHOICES``

    Ordered collection of the supported query-format identifiers.

- ``QueryFormat``

    Type alias for the supported embedding-query formats.

- ``SOFTWARE_REPOSITORY_INSTRUCTION``

    Embedding-query instruction for software source retrieval.

- ``LITERATURE_REPOSITORY_INSTRUCTION``

    Embedding-query instruction for scientific literature retrieval.

- ``format_qwen3_query``

    Format one retrieval query for a Qwen3 embedding model.

Examples
--------
>>> import qachat
>>> query = qachat.format_qwen3_query(
...     "How are sources split?",
...     qachat.SOFTWARE_REPOSITORY_INSTRUCTION,
... )

"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias


__all__ = [
    "QUERY_FORMAT_PLAIN",
    "QUERY_FORMAT_QWEN3",
    "QUERY_FORMAT_CHOICES",
    "QueryFormat",
    "SOFTWARE_REPOSITORY_INSTRUCTION",
    "LITERATURE_REPOSITORY_INSTRUCTION",
    "format_qwen3_query",
]

QueryFormat: TypeAlias = Literal["plain", "qwen3"]

QUERY_FORMAT_PLAIN: Final[QueryFormat] = "plain"
QUERY_FORMAT_QWEN3: Final[QueryFormat] = "qwen3"
QUERY_FORMAT_CHOICES: Final[tuple[QueryFormat, ...]] = (
    QUERY_FORMAT_PLAIN,
    QUERY_FORMAT_QWEN3,
)

SOFTWARE_REPOSITORY_INSTRUCTION = (
    "Given a question about a software repository, retrieve relevant code "
    "and documentation that help answer the question."
)

LITERATURE_REPOSITORY_INSTRUCTION = (
    "Given a question about academic literature, retrieve relevant research "
    "papers, abstracts, and citations that help answer the question."
)


def format_qwen3_query(
    question: str,
    instruction: str,
) -> str:
    """Format one retrieval query for a Qwen3 embedding model.

    Parameters
    ----------
    question: str
        User question.
    instruction: str
        Description of what should be retrieved.

    Returns
    -------
    str
        Qwen3 query containing separate Instruct and Query lines.

    Raises
    ------
    ValueError
        If the question or instruction is not non-empty text.

    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty text.")

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be non-empty text.")

    instruction_line = f"Instruct: {instruction.strip()}"
    question_line = f"Query: {question.strip()}"
    return f"{instruction_line}\n{question_line}"
