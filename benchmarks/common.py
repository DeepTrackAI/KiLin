"""Shared helpers for the CodeRAG-Bench evaluation notebooks.

Only the pieces that are genuinely benchmark-specific live here --
BEIR-shaped task data, query formatting for the benchmark's retrieval
instruction, and BEIR/pytrec_eval-style scoring. Chunking, index
building, and searching all go through ``qachat`` directly from each
notebook now (real ``scan_repository``/``chunk_source_units``,
``build_semantic_index`` batching, ``semantic_search``'s numpy path),
not a benchmark-only stand-in.

Classes
-------
BenchmarkTask
    One CodeRAG-Bench retrieval task in BEIR format.

Functions
---------
format_query
    Format one benchmark query for the Qwen3 embedding model.
evaluate_retrieval
    Score ranked results the same way CodeRAG-Bench and BEIR do.
corpus_already_written
    Check whether a project's corpus folder already holds files, so a
    notebook can skip rewriting a corpus already on disk.
local_run_timestamp
    Current time in Europe/Rome, for a result record and its filename.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytrec_eval

from qachat.query_formats import format_qwen3_query


__all__ = [
    "BenchmarkTask",
    "format_query",
    "evaluate_retrieval",
    "corpus_already_written",
    "local_run_timestamp",
]

LOCAL_TIMEZONE = ZoneInfo("Europe/Rome")


CODE_RETRIEVAL_INSTRUCTION = (
    "Given a coding problem description, retrieve the source code that "
    "solves it."
)


def format_query(question: str) -> str:
    """Format one benchmark query for the Qwen3 embedding model."""

    return format_qwen3_query(
        question=question,
        instruction=CODE_RETRIEVAL_INSTRUCTION,
    )


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One CodeRAG-Bench retrieval task in BEIR format.

    Attributes
    ----------
    name: str
        Short task name, used in result tables.
    corpus: dict[str, dict[str, str]]
        Document id -> {"title": ..., "text": ...}.
    queries: dict[str, str]
        Query id -> query text.
    qrels: dict[str, dict[str, int]]
        Query id -> {document id: relevance score}.

    """

    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def evaluate_retrieval(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Score ranked results the same way CodeRAG-Bench and BEIR do."""

    if k_values is None:
        k_values = [1, 3, 5, 10, 100]

    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    map_ = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}
    mrr = {f"MRR@{k}": 0.0 for k in k_values}

    cutoffs = ",".join(str(k) for k in k_values)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {
            f"map_cut.{cutoffs}",
            f"ndcg_cut.{cutoffs}",
            f"recall.{cutoffs}",
            f"P.{cutoffs}",
        },
    )
    scores = evaluator.evaluate(results)

    for query_scores in scores.values():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores[f"ndcg_cut_{k}"]
            map_[f"MAP@{k}"] += query_scores[f"map_cut_{k}"]
            recall[f"Recall@{k}"] += query_scores[f"recall_{k}"]
            precision[f"P@{k}"] += query_scores[f"P_{k}"]

    k_max = max(k_values)
    for query_id, doc_scores in results.items():
        top_hits = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[:k_max]
        relevant = {
            doc_id
            for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0
        }
        for k in k_values:
            for rank, (doc_id, _) in enumerate(top_hits[:k]):
                if doc_id in relevant:
                    mrr[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    query_count = len(scores) or 1
    qrel_count = len(qrels) or 1
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / query_count, 5)
        map_[f"MAP@{k}"] = round(map_[f"MAP@{k}"] / query_count, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / query_count, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / query_count, 5)
        mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"] / qrel_count, 5)

    return {
        "ndcg": ndcg,
        "map": map_,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def corpus_already_written(directory: Path) -> bool:
    """Return whether a project's corpus folder is already populated.

    Parameters
    ----------
    directory: pathlib.Path
        A project's ``repository/`` or ``documents/`` folder.

    Returns
    -------
    bool
        Whether the folder exists and already holds at least one file --
        lets a benchmark notebook skip rewriting a corpus that's already
        on disk from a previous run, instead of unconditionally
        overwriting every document again.

    """

    return directory.exists() and any(directory.iterdir())


def local_run_timestamp() -> tuple[str, str]:
    """Return the current time in Europe/Rome, for a saved result.

    Returns
    -------
    tuple[str, str]
        ``(timestamp, stamp)`` -- an ISO-8601 timestamp with the local
        UTC offset, for a result record's ``"timestamp"`` field, and a
        compact filename-safe stamp (``%Y%m%dT%H%M%S%z``) for naming
        the saved file. Uses ``Europe/Rome`` (CET/CEST), not UTC, so
        saved-result timestamps read naturally at a glance.

    """

    now = datetime.now(LOCAL_TIMEZONE)
    return now.isoformat(), now.strftime("%Y%m%dT%H%M%S%z")
