"""Provide task definitions and scoring helpers for KíLín benchmarks.

The benchmark notebooks exercise KíLín's real scanning, extraction,
chunking, indexing and retrieval paths. This module only owns evaluation
records, benchmark-specific query formatting and deterministic scoring.
The optional ``pytrec_eval`` dependency is loaded when retrieval scores
are calculated; the dogfooding helpers can run without that package.

Key Features
------------

- **Retrieval tasks**

    Represent CodeRAG-Bench and BEIR corpora as query, document and
    relevance dictionaries and compute standard ranking metrics.

- **Answer assessment**

    Compare KíLín-generated citations with supplied evidence and source
    locations, while keeping heuristic checks distinct from human review.

- **Reproducibility**

    Load hand-annotated gold records and produce local result timestamps.

Module Structure
----------------

- ``BenchmarkTask`` and ``GenerationTask``

    Frozen records for retrieval and answer-quality evaluations.

- ``format_query`` and ``evaluate_retrieval``

    Prepare CodeRAG-Bench queries and score retrieval results.

- ``corpus_already_written`` and ``local_run_timestamp``

    Utilities for creating and recording benchmark runs.

- ``load_generation_tasks``, ``extract_cited_indices``,
  ``citation_validity``, ``gold_evidence_indices`` and
  ``citation_coverage``

    Parse ground truth and calculate mechanistic citation checks.

- ``abstains`` and ``verbatim_overlap_chars``

    Heuristic answer checks that require human interpretation.

Examples
--------
>>> from benchmarks.common import citation_validity, format_query
>>> citation_validity("Supported [1, 2]", evidence_count=2)
(True, [])
>>> format_query("How is a file scanned?").startswith("Instruct:")
True

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from kilin.query_formats import format_qwen3_query


__all__ = [
    "BenchmarkTask",
    "GenerationTask",
    "format_query",
    "evaluate_retrieval",
    "corpus_already_written",
    "open_or_initialize",
    "local_run_timestamp",
    "load_generation_tasks",
    "final_answer",
    "extract_cited_indices",
    "citation_validity",
    "gold_evidence_indices",
    "citation_coverage",
    "abstains",
    "verbatim_overlap_chars",
]

LOCAL_TIMEZONE = ZoneInfo("Europe/Rome")


CODE_RETRIEVAL_INSTRUCTION = (
    "Given a coding problem description, retrieve the source code that "
    "solves it."
)


def format_query(question: str) -> str:
    """Format a coding question using the benchmark retrieval instruction.

    Parameters
    ----------
    question: str
        Coding task or natural-language retrieval query.

    Returns
    -------
    str
        Validated Qwen3 query with separate instruction and question lines.

    Raises
    ------
    ValueError
        If question is empty or not text.

    """

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
    """Compute BEIR-style information retrieval scores for ranked results.

    Parameters
    ----------
    qrels: dict[str, dict[str, int]]
        Relevance grades keyed by query and corpus document identifier.
    results: dict[str, dict[str, float]]
        Retrieved document scores keyed by query and document identifier.
    k_values: list[int] | None, optional
        Ranking cutoffs, or the standard 1, 3, 5, 10 and 100.

    Returns
    -------
    dict[str, dict[str, float]]
        Mean NDCG, MAP, recall, precision and MRR at each cutoff.

    Notes
    -----
    ``pytrec_eval`` is required only when scoring retrieval tasks;
    citation and dogfooding helpers can be imported without it.

    """

    import pytrec_eval

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


def open_or_initialize(project_root: Path, type_id: str):
    """Open a benchmark project, or create it in an empty folder.

    Parameters
    ----------
    project_root: pathlib.Path
        Benchmark project folder under ``sandbox/``.
    type_id: str
        Project type used when the project is created, such as ``"code"``
        or ``"literature"``.

    Returns
    -------
    kilin.project.QAProject
        The existing project when ``profile.ini`` is present; otherwise a
        new project.

    Notes
    -----
    ``initialize_project`` only creates a project in a nonexistent or
    empty folder, so a project from an earlier run is reopened. A folder
    without ``profile.ini`` is left over from an interrupted run and is
    deleted and created again.

    """

    from kilin.project import initialize_project, open_project

    if (project_root / "profile.ini").is_file():
        return open_project(project_root)
    if project_root.exists():
        shutil.rmtree(project_root)
    return initialize_project(project_root, type_id)


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


@dataclass(frozen=True, slots=True)
class GenerationTask:
    """One hand-annotated code-question and answer-quality task.

    Unlike ``BenchmarkTask``, there's no published table to check these
    against -- credibility comes from the citations and fact checklist
    below being verified against the actual source, not from a leaderboard.

    Attributes
    ----------
    task_id: str
        Short id used in result tables (e.g. ``"A1"``).
    question: str
        The question asked of a real KíLín ``RAGSession``.
    gold_citations: tuple[str, ...]
        ``"path:start-end"`` or ``"path:line"`` strings identifying the
        source location(s) a correct, well-cited answer should draw on.
        Empty for a deliberate abstention question. For a notebook cell,
        include the one-based cell number as ``"path#cell:start-end"`` --
        every cell's line numbers start at 1 (see ``Chunk`` in
        ``kilin/chunking.py``), so two different cells can share the same
        line range and a plain ``"path:start-end"`` citation would then
        match every chunk in that notebook rather than the one intended.
    fact_checklist: tuple[str, ...]
        Facts a correct answer must state, for human Tier-2 grading.
    requires_abstention: bool
        Whether the corpus deliberately does not answer this question --
        a correct response says the evidence is insufficient rather than
        guessing.
    reference_answer: str
        A reference answer written from the fact checklist, for a human
        grader to compare KíLín's actual answer against during Tier 2
        grading. Not used by any Tier 1 mechanical check.

    """

    task_id: str
    question: str
    gold_citations: tuple[str, ...]
    fact_checklist: tuple[str, ...]
    requires_abstention: bool = False
    reference_answer: str = ""


def load_generation_tasks(path: Path) -> list[GenerationTask]:
    """Load a hand-authored gold question set from its JSON file.

    Parameters
    ----------
    path: pathlib.Path
        A ``benchmarks/gold/*.json`` file: ``{"tasks": [...]}``, each item
        matching ``GenerationTask``'s fields.

    Returns
    -------
    list[GenerationTask]
        One task per entry, in file order.

    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GenerationTask(
            task_id=item["task_id"],
            question=item["question"],
            gold_citations=tuple(item.get("gold_citations", ())),
            fact_checklist=tuple(item.get("fact_checklist", ())),
            requires_abstention=bool(item.get("requires_abstention", False)),
            reference_answer=str(item.get("reference_answer", "")),
        )
        for item in data["tasks"]
    ]


# KíLín accepts both [1] and grouped [1, 2] citations.
_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def final_answer(answer: str) -> str:
    """Return the part of a model reply that is shown as the answer.

    Parameters
    ----------
    answer: str
        A chat-model reply, possibly with reasoning in ``<think>`` tags.

    Returns
    -------
    str
        The reply without its reasoning, stripped of surrounding space.

    Notes
    -----
    Reasoning is the model's working draft: citations, list literals and
    quoted evidence there are not claims made to the reader, so the
    citation, abstention and overlap checks read only the final answer.
    This mirrors ``kilin.answering.strip_thinking``.

    """

    text = re.sub(r"<think>.*?</think>", "", str(answer), flags=re.S)
    # A chat template that opens <think> itself leaves only the closing tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.split("<think>", 1)[0].strip()


def extract_cited_indices(answer: str) -> list[int]:
    """Extract numeric citations from standalone and grouped references.

    Parameters
    ----------
    answer: str
        A chat-model answer, as returned by ``RAGSession.ask``.

    Returns
    -------
    list[int]
        Cited indices from ``[1]`` or ``[1, 2]`` in the final answer
        (``final_answer``), in appearance order and including duplicates.
        Callers needing unique numbers must dedupe.

    """

    indices: list[int] = []
    for group in _CITATION_PATTERN.findall(final_answer(answer)):
        indices.extend(int(part.strip()) for part in group.split(","))
    return indices


def citation_validity(
    answer: str, evidence_count: int
) -> tuple[bool, list[int]]:
    """Check that every ``[n]`` citation in an answer refers to real evidence.

    Parameters
    ----------
    answer: str
        A chat-model answer.
    evidence_count: int
        Number of evidence items actually supplied for this turn -- valid
        citation numbers are ``1..evidence_count`` inclusive.

    Returns
    -------
    tuple[bool, list[int]]
        ``(all_valid, invalid_indices)`` -- ``invalid_indices`` lists any
        cited number outside that range, sorted and deduplicated.

    """

    cited = extract_cited_indices(answer)
    invalid = sorted({n for n in cited if n < 1 or n > evidence_count})
    return (len(invalid) == 0, invalid)


def _parse_gold_citation(spec: str) -> tuple[str, int | None, int, int]:
    """Parse source path, notebook cell and inclusive line bounds.

    Parameters
    ----------
    spec: str
        ``path:line`` or ``path:start-end`` with optional ``#cell``
        suffix before the colon. Notebook cell indices are one-based.

    Returns
    -------
    tuple[str, int | None, int, int]
        Source path, optional notebook cell, first line and last line.

    """

    path, _, span = spec.rpartition(":")
    if "-" in span:
        start_text, end_text = span.split("-", 1)
    else:
        start_text = end_text = span
    cell: int | None = None
    if "#" in path:
        path, _, cell_text = path.rpartition("#")
        cell = int(cell_text)
    return path, cell, int(start_text), int(end_text)


def gold_evidence_indices(
    evidence, gold_citations: tuple[str, ...]
) -> list[int]:
    """Find real evidence that overlaps the hand-annotated gold passages.

    Parameters
    ----------
    evidence: Sequence[kilin.rag.Evidence]
        The exact evidence KíLín retrieved and cited for one turn (e.g.
        ``rag_session.turns[-1].evidence``).
    gold_citations: tuple[str, ...]
        This question's gold ``"path:start-end"`` citation(s), optionally
        ``"path#cell:start-end"`` for a notebook cell.

    Returns
    -------
    list[int]
        Citation numbers (``Evidence.citation``) whose path, cell (when the
        gold citation names one), and line range overlap a gold citation. A
        gold path matches an evidence path by suffix, since ``Evidence.path``
        is relative to the project's configured source folder while a gold
        citation is written relative to the repo root. A gold citation with
        no ``#cell`` matches on path and line range alone, same as before --
        only a notebook citation needs the cell number, since every cell's
        line numbers start at 1 and would otherwise overlap every chunk in
        that file.

    """

    matches: list[int] = []
    for spec in gold_citations:
        gold_path, gold_cell, gold_start, gold_end = _parse_gold_citation(spec)
        for item in evidence:
            # Gold citations use forward slashes on every system.
            item_path = Path(item.path).as_posix()
            if not (
                gold_path == item_path
                or gold_path.endswith(item_path)
                or item_path.endswith(gold_path)
            ):
                continue
            if (
                gold_cell is not None
                and getattr(item, "cell", None) != gold_cell
            ):
                continue
            if item.start_line <= gold_end and item.end_line >= gold_start:
                matches.append(item.citation)
    return sorted(set(matches))


def citation_coverage(
    answer: str, evidence, gold_citations: tuple[str, ...]
) -> float | None:
    """Fraction of a question's gold-relevant evidence that got cited.

    Parameters
    ----------
    answer: str
        A chat-model answer.
    evidence: Sequence[kilin.rag.Evidence]
        The exact evidence supplied for this turn.
    gold_citations: tuple[str, ...]
        This question's gold citation(s).

    Returns
    -------
    float | None
        Covered fraction in ``[0, 1]``, or ``None`` when none of the gold
        citation(s) were even retrieved into evidence this turn -- that's a
        retrieval miss, not a generation/citation failure, and the two
        shouldn't be collapsed into one number.

    """

    gold_indices = gold_evidence_indices(evidence, gold_citations)
    if not gold_indices:
        return None
    cited = set(extract_cited_indices(answer))
    covered = sum(1 for index in gold_indices if index in cited)
    return covered / len(gold_indices)


_ABSTENTION_PATTERN = re.compile(
    r"insufficient evidence"
    r"|evidence is insufficient"
    r"|not enough evidence"
    r"|cannot answer"
    r"|(?:evidence|code|documentation|sources?) (?:provided |supplied )?"
    r"(?:does|do) not (?:directly )?"
    r"(?:confirm|address|mention|state|show|indicate|describe|contain|specify)"
    r"|\bno (?:explicit |direct )?(?:\w+[\s-]){0,4}(?:is|are) "
    r"(?:stated|mentioned|present|described|specified|found)"
    r"|\bnot (?:explicitly )?(?:stated|mentioned|specified|documented)\b"
    r"|\bno (?:mention|evidence|indication) of\b",
    re.IGNORECASE,
)

# A contrast after the abstention phrase ("Insufficient evidence is provided
# to confirm X, but ...") marks a hedge on an answer, not an abstention.
_HEDGE_PATTERN = re.compile(
    r"\b(?:but|however|although|though|yet)\b", re.IGNORECASE
)

# Quoted text, such as a quoted system prompt, is not the answer's own claim.
_QUOTATION_PATTERN = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d')

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _states_insufficient_evidence(sentence: str) -> bool:
    """Return whether one sentence concludes that evidence is missing.

    Parameters
    ----------
    sentence: str
        One sentence of the final answer.

    Returns
    -------
    bool
        Whether an abstention phrase occurs with no contrast after it.

    """

    match = _ABSTENTION_PATTERN.search(sentence)
    return match is not None and not _HEDGE_PATTERN.search(
        sentence[match.end():]
    )


def abstains(answer: str) -> bool:
    """Detect whether an answer concludes that the evidence is insufficient.

    Parameters
    ----------
    answer: str
        Model-generated answer to inspect.

    Returns
    -------
    bool
        Whether the first or last sentence of the final answer states that
        the evidence does not answer the question.

    Notes
    -----
    Reasoning (``final_answer``) and quoted text are ignored.
    The opening and closing sentences carry an answer's conclusion. Either
    counts as an abstention when it uses the requested phrase
    ``Insufficient evidence.`` or a common paraphrase ("the evidence does
    not confirm", "no ... is stated", "not explicitly mentioned"), unless a
    contrast follows ("..., but ..."), which marks a hedged answer. This
    recognizer cannot judge whether the question was in fact answerable,
    and unusual paraphrases may be missed. Interpret the result alongside
    the human Tier-2 worksheet.

    """

    text = _QUOTATION_PATTERN.sub('""', final_answer(answer))
    sentences = [
        sentence
        for sentence in _SENTENCE_BREAK.split(text)
        if sentence.strip()
    ]
    if not sentences:
        return False

    return any(
        _states_insufficient_evidence(sentence)
        for sentence in (sentences[0], sentences[-1])
    )


def verbatim_overlap_chars(answer: str, evidence) -> int:
    """Measure the longest substring shared with supplied source excerpts.

    Parameters
    ----------
    answer: str
        Generated answer to inspect.
    evidence: Sequence[kilin.rag.Evidence]
        Source excerpts supplied for the answer.

    Returns
    -------
    int
        Length in characters of the longest exact substring shared by the
        final answer (``final_answer``) and one excerpt.

    Notes
    -----
    Long overlaps signal passages for human review, not automatically
    improper copying. In the KíLín gold set question A3 explicitly asks
    to quote the system prompt, so an overlap there is expected.

    """

    answer = final_answer(answer)
    longest = 0
    matcher = SequenceMatcher(None, answer, "")
    for item in evidence:
        matcher.set_seq2(item.text)
        match = matcher.find_longest_match(0, len(answer), 0, len(item.text))
        longest = max(longest, match.size)
    return longest
