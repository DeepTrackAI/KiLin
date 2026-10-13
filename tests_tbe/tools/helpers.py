"""Build chunks and a tool context from plain functions, for tool tests.

Tools receive QAChat's services through ``ToolContext``. These helpers
replace the models, ranking, and user prompts with plain functions, so a
tool test runs without loading a model.

"""

from __future__ import annotations

from pathlib import Path

from qachat.chunking import Chunk
from qachat.retrievals import CodeGraph
from qachat.tools import ToolContext
from qachat.trace_logging import NullTraceLogger


def chunk(path: str, lines: list[str], start: int = 1, **fields) -> Chunk:
    """Return a chunk of whole lines, numbered from ``start``.

    Parameters
    ----------
    path: str
        File path of the chunk.
    lines: list[str]
        Lines of text, without line endings.
    start: int, optional
        Line number of the first line. Defaults to ``1``.
    **fields
        Optional ``symbol``, ``section``, and ``cell`` values.

    Returns
    -------
    Chunk
        A text chunk whose identifier combines path, start, and section.

    """

    return Chunk(
        identifier=f"{path}:{start}:{fields.get('section')}",
        path=Path(path),
        text="".join(f"{line}\n" for line in lines),
        start_line=start,
        end_line=start + len(lines) - 1,
        kind="text",
        symbol=fields.get("symbol"),
        section=fields.get("section"),
        cell=fields.get("cell"),
    )


def context(
    chunks=(),
    *,
    sizes=None,
    router=None,
    model=None,
    write=None,
    rank=None,
    confirm=None,
    root=None,
    write_pdf=None,
    choose=None,
):
    """Return a ToolContext whose services are plain functions.

    Each keyword replaces one service; services left out answer with an
    empty or neutral value (an empty ranking, ``"{}"`` from the router,
    ``"[]"`` from the model). ``write`` falls back to ``model``. ``sizes``
    maps source files to their sizes; files in it but not indexed are
    reported as unsearchable.

    Returns
    -------
    tuple[ToolContext, list[tuple[str, str, dict]]]
        The context, and the events list that collects progress and done
        calls as ``(kind, message, details)``.

    """

    events: list[tuple[str, str, dict]] = []
    sizes = dict(sizes or {})
    built = ToolContext(
        project_root=Path(root) if root else Path("."),
        chunks=list(chunks),
        graph=CodeGraph(entities=(), relations=()),
        rank=rank or (lambda text: []),
        retrieval_count=20,
        unsearchable=lambda indexed: [
            path for path in sizes if path not in indexed
        ],
        source_files=lambda: sizes,
        ask_router=router or (lambda system, prompt, tokens: "{}"),
        ask_model=model or (lambda system, prompt, tokens: "[]"),
        write=write or model or (lambda system, prompt, tokens: ""),
        model_id="test/model",
        progress=lambda message, **details: events.append(
            ("progress", message, details)
        ),
        done=lambda message, **details: events.append(
            ("done", message, details)
        ),
        confirm=confirm,
        trace=NullTraceLogger(),
        trace_turn_id=None,
        write_pdf=write_pdf,
        choose=choose,
    )
    return built, events
