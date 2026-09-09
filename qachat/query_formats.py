"""Prepare retrieval queries for model-specific embedding formats.

Attributes
----------
SOFTWARE_REPOSITORY_INSTRUCTION
    Default retrieval goal for a software repository.
LITERATURE_REPOSITORY_INSTRUCTION
    Default retrieval goal for an academic literature repository.

Functions
---------
format_qwen3_query
    Combine an instruction and question in Qwen3 Embedding query format.

"""

from __future__ import annotations


__all__ = [
    "SOFTWARE_REPOSITORY_INSTRUCTION",
    "LITERATURE_REPOSITORY_INSTRUCTION",
    "format_qwen3_query",
]

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
