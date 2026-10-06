"""Tool ``qachat_help``: answer questions about QAChat from its own docs.

Questions such as "how do I add a paper?", "will adding a file re-index
everything?", or "where are my conversations saved?" are about QAChat, not
the project. The answering model does not know QAChat, and the project's
index does not describe it, so this tool answers from QAChat's own
documentation: the main README and the desktop app's README, which describe
the behaviour the code implements.

The documentation is split into sections with the same Markdown reader
used for sources, ranked for the question with lexical (TF-IDF) search, and
the best sections are given to the answering model, whose answer streams to
the user with citations. The documentation never enters a project's index.

Key Features
------------

- **Docs first**: answers come from documentation, which states intended
  behaviour plainly; code is not read.
- **Separate knowledge**: QAChat's documentation is searched on its own,
  never mixed with project sources.

Module Structure
----------------

- ``HelpSettings``
- ``parse_settings``
- ``run``
- ``documentation``
- ``DOCUMENTATION_FILES``

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from qachat.chunking import Chunk, chunk_source_units
from qachat.retrievals import build_lexical_index, lexical_search
from qachat.retrievals.document_reading import estimate_tokens
from qachat.sources import extract_source_units, read_repository_files
from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)

__all__ = [
    "DOCUMENTATION_FILES",
    "HelpSettings",
    "documentation",
    "parse_settings",
    "run",
]

# QAChat's documentation, relative to the QAChat checkout.
DOCUMENTATION_FILES: Final[tuple[str, ...]] = (
    "README.md",
    "qachat/gui/README.md",
)
_QACHAT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# Evidence paths are shown under this name, apart from project sources.
_DOCS_LABEL: Final[str] = "QAChat documentation"


@dataclass(frozen=True, slots=True)
class HelpSettings:
    """Settings of ``qachat_help`` in a project type.

    Attributes
    ----------
    max_sections: int
        Documentation sections given to the model (1-20).
    answer_tokens: int
        Longest answer, in tokens (200-2000).

    """

    max_sections: int = 6
    answer_tokens: int = 700


def parse_settings(data: dict, where: str) -> HelpSettings:
    """Validate ``qachat_help`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    HelpSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(where, data, ("max_sections", "answer_tokens"))
    defaults = HelpSettings()
    return HelpSettings(
        max_sections=check_int(
            where, data, "max_sections", defaults.max_sections, 1, 20
        ),
        answer_tokens=check_int(
            where, data, "answer_tokens", defaults.answer_tokens, 200, 2000
        ),
    )


@lru_cache(maxsize=1)
def documentation(root: Path = _QACHAT_ROOT) -> tuple[Chunk, ...]:
    """Read QAChat's documentation and split it into sections.

    Parameters
    ----------
    root: pathlib.Path, optional
        QAChat checkout. Defaults to the one this module belongs to.

    Returns
    -------
    tuple[Chunk, ...]
        Sections of the files in ``DOCUMENTATION_FILES`` that exist, with
        paths under "QAChat documentation"; cached for the process. Empty
        when none exist (for example in an installed package).

    """

    present = [name for name in DOCUMENTATION_FILES if (root / name).is_file()]
    if not present:
        return ()
    files = read_repository_files(
        {"repository": str(root), "selected_files": present}
    )
    chunks = chunk_source_units(
        extract_source_units(files),
        count_tokens=estimate_tokens,
        overlap_tokens=0,
    )
    return tuple(
        Chunk(
            identifier=f"docs:{chunk.identifier}",
            path=Path(_DOCS_LABEL) / chunk.path,
            text=chunk.text,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            kind=chunk.kind,
            symbol=chunk.symbol,
            section=chunk.section,
            cell=chunk.cell,
        )
        for chunk in chunks
    )


@lru_cache(maxsize=1)
def _index(root: Path = _QACHAT_ROOT) -> object:
    return build_lexical_index(list(documentation(root)))


def run(
    settings: HelpSettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Answer a question about QAChat from its documentation.

    Parameters
    ----------
    settings: HelpSettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        QAChat's services (``write``, ``progress``, ``trace``).

    Returns
    -------
    ToolResult
        The streamed answer, the documentation sections as evidence, and
        which sections were used.

    """

    context.progress("Searching QAChat's documentation")
    sections = documentation()
    summary: dict[str, object] = {"name": "qachat_help"}
    if not sections:
        context.done("No QAChat documentation found", evidence_count=0)
        return ToolResult(
            "QAChat's documentation is not available in this installation. "
            "See the README in the QAChat repository.",
            (),
            summary,
        )
    hits = lexical_search(
        _index(), request.text, count=settings.max_sections
    )
    chosen = [hit.chunk for hit in hits if hit.score > 0]
    summary.update(
        {
            "sections": [
                f"{chunk.path.as_posix()}#{chunk.section or ''}"
                for chunk in chosen
            ]
        }
    )
    context.trace.emit(
        "tool.qachat_help.sections",
        component="tool",
        data=summary,
        turn_id=context.trace_turn_id,
    )
    context.done(
        f"Found {len(chosen)} documentation sections",
        evidence_count=len(chosen),
    )
    excerpts = "\n\n".join(
        f"[{number}] {chunk.section or chunk.path.name}\n{chunk.text.strip()}"
        for number, chunk in enumerate(chosen, start=1)
    )
    system = (
        "You answer questions about how the QAChat application works, "
        "using only the documentation excerpts you are given. You give "
        "practical steps."
    )
    prompt = f"""Question: {request.question}

QAChat documentation excerpts:

{excerpts or "none"}

Answer from these excerpts and cite them like [2]. If they do not answer
the question, say so plainly instead of guessing."""
    if request.guidance:
        prompt += f"\n\n{request.guidance}"
    answer = context.write(system, prompt, settings.answer_tokens)
    evidence = tuple(
        {
            "citation": number,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "text": chunk.text,
            "cell": chunk.cell,
        }
        for number, chunk in enumerate(chosen, start=1)
    )
    return ToolResult(answer, evidence, summary, streamed=True)
