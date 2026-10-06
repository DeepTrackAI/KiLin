"""A tool context built from plain functions, for tool tests."""

from __future__ import annotations

from pathlib import Path

from qachat.chunking import Chunk
from qachat.retrievals import CodeGraph
from qachat.tools import ToolContext
from qachat.trace_logging import NullTraceLogger


def chunk(path: str, lines: list[str], start: int = 1, **fields) -> Chunk:
    """Return a chunk of whole lines, numbered from ``start``."""

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
):
    """Return a ToolContext whose services are plain functions.

    ``events`` collects progress and done calls as ``(kind, message,
    details)``.
    """

    events: list[tuple[str, str, dict]] = []
    sizes = dict(sizes or {})
    built = ToolContext(
        project_root=Path(root) if root else Path("."),
        chunks=list(chunks),
        graph=CodeGraph(entities=(), relations=()),
        rank=rank or (lambda text: []),
        retrieval_count=20,
        unsearchable=lambda indexed: [p for p in sizes if p not in indexed],
        source_files=lambda: sizes,
        ask_router=router or (lambda system, prompt, tokens: "{}"),
        ask_model=model or (lambda system, prompt, tokens: "[]"),
        write=write or model or (lambda system, prompt, tokens: ""),
        model_id="test/model",
        progress=lambda message, **d: events.append(("progress", message, d)),
        done=lambda message, **d: events.append(("done", message, d)),
        confirm=confirm,
        trace=NullTraceLogger(),
        trace_turn_id=None,
        write_pdf=write_pdf,
    )
    return built, events
