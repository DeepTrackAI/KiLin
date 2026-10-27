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
GenerationTask
    One hand-authored dogfooding question against QAChat's own repo.

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
load_generation_tasks
    Load a hand-authored gold question set from its JSON file.
extract_cited_indices
    Return every one-based [n] citation index referenced in an answer.
citation_validity
    Check that every citation in an answer refers to real, supplied evidence.
gold_evidence_indices
    Which of a turn's real evidence citation numbers cover a question's gold
    citation(s).
citation_coverage
    Fraction of a question's gold-relevant evidence that actually got cited.
abstains
    Heuristic check for whether an answer says the evidence is insufficient.
verbatim_overlap_chars
    Longest run of characters an answer shares verbatim with any one piece
    of supplied evidence.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pytrec_eval

from qachat.query_formats import format_qwen3_query


__all__ = [
    "BenchmarkTask",
    "GenerationTask",
    "format_query",
    "evaluate_retrieval",
    "corpus_already_written",
    "local_run_timestamp",
    "load_generation_tasks",
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


@dataclass(frozen=True, slots=True)
class GenerationTask:
    """One hand-authored dogfooding question against QAChat's own repo.

    Unlike ``BenchmarkTask``, there's no published table to check these
    against -- credibility comes from the citations and fact checklist
    below being verified against the actual source, not from a leaderboard.

    Attributes
    ----------
    task_id: str
        Short id used in result tables (e.g. ``"A1"``).
    question: str
        The question asked of QAChat's real ``RAGSession``.
    gold_citations: tuple[str, ...]
        ``"path:start-end"`` or ``"path:line"`` strings identifying the
        source location(s) a correct, well-cited answer should draw on.
        Empty for a deliberate abstention question. For a notebook cell,
        include the one-based cell number as ``"path#cell:start-end"`` --
        every cell's line numbers start at 1 (see ``Chunk`` in
        ``qachat/chunking.py``), so two different cells can share the same
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
        grader to compare QAChat's actual answer against during Tier 2
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
        A ``_si_article_/benchmarks/gold/*.json`` file containing
        ``{"tasks": [...]}`` entries matching ``GenerationTask``'s fields.

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


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_cited_indices(answer: str) -> list[int]:
    """Return every one-based ``[n]`` citation index referenced in an answer.

    Parameters
    ----------
    answer: str
        A chat-model answer, as returned by ``RAGSession.ask``.

    Returns
    -------
    list[int]
        Cited indices in order of appearance, duplicates included --
        callers that need a set should dedupe themselves.

    """

    return [int(match) for match in _CITATION_PATTERN.findall(answer)]


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
    """Parse a ``"path:line"`` or ``"path:start-end"`` gold citation string.

    A path suffixed with ``#<cell>`` (e.g. ``"tutorial.ipynb#34:1-8"``) names
    a specific one-based notebook cell -- required to disambiguate a
    notebook citation, since every cell's line numbers start at 1.
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
    """Find evidence citations overlapping the supplied gold citations.

    Parameters
    ----------
    evidence: Sequence[qachat.rag.Evidence]
        The exact evidence QAChat retrieved and cited for one turn (e.g.
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
            item_path = str(item.path)
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
    evidence: Sequence[qachat.rag.Evidence]
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
    r"|cannot answer (?:this|that) from the (?:supplied|available) evidence",
    re.IGNORECASE,
)


_ABSTENTION_DOMINANCE_RATIO = 0.5

_ABSTENTION_TRAILING_CHARS = re.compile(r"^[\s.:;,!\"'()\[\]]*$")


def abstains(answer: str) -> bool:
    """Heuristic check for whether an answer says the evidence is insufficient.

    Parameters
    ----------
    answer: str
        A chat-model answer.

    Returns
    -------
    bool
        Whether the answer, taken as a whole, *is* an abstention -- not
        merely whether it contains the abstention wording ("insufficient
        evidence" or a close paraphrase) somewhere inside it. QAChat's
        default system prompt (``qachat/rag.py``'s ``_SYSTEM_PROMPT``) only
        loosely asks the model to "say when evidence is insufficient";
        some benchmark runs instead use a stronger, run-scoped override
        (for example, in the ``60_qachat_dogfood_code.ipynb`` notebook,
        under ``_si_article_/benchmarks/notebooks/``) that asks for an
        exact "Insufficient evidence." reply when the evidence
        *clearly* does not address the question -- but even that
        instruction has a legitimate gap for a
        compound question the evidence only partially answers: a model
        that explains what it *does* know and then flags the unanswered
        part with that same closing phrase is abstaining correctly on that
        part, just not in the bare single-sentence form the instruction
        describes for a clean non-match. So an answer counts as an
        abstention if the matched phrase either (a) covers most of the
        answer's length (``_ABSTENTION_DOMINANCE_RATIO``), which is what a
        clean, terse abstention looks like, or (b) is the last substantive
        thing in the answer -- nothing but whitespace or trailing
        punctuation follows it -- which is what a hedged, partial-then-
        abstain answer looks like. Either way, a match stuck in the middle
        of a longer explanation (reciting the system prompt itself, or a
        reasoning trace that mentions "insufficient evidence" only in
        passing before answering anyway) is still not misclassified as an
        abstention. This is still a mechanical proxy, not a semantic one: a
        real abstention phrased some other way is missed here. Treat it as
        a fast first pass; the human Tier-2 read is what actually settles
        abstention correctness.

    """

    stripped = answer.strip()
    if not stripped:
        return False

    matches = list(_ABSTENTION_PATTERN.finditer(stripped))
    if not matches:
        return False

    last_match = matches[-1]
    matched_chars = last_match.end() - last_match.start()
    dominates = matched_chars / len(stripped) >= _ABSTENTION_DOMINANCE_RATIO

    trailing = stripped[last_match.end():]
    anchored_at_end = bool(_ABSTENTION_TRAILING_CHARS.match(trailing))

    return dominates or anchored_at_end


def verbatim_overlap_chars(answer: str, evidence) -> int:
    """Longest run of characters an answer shares verbatim with any evidence.

    Parameters
    ----------
    answer: str
        A chat-model answer.
    evidence: Sequence[qachat.rag.Evidence]
        The exact evidence supplied for this turn.

    Returns
    -------
    int
        The largest single matching block length (characters), across all
        evidence items, found via ``difflib.SequenceMatcher``. QAChat's
        system prompt explicitly forbids copying source text verbatim
        (``_SYSTEM_PROMPT`` in ``qachat/rag.py``) -- a long match here is a
        checkable compliance signal, not a judgment call. (One question in
        the gold set, A3, asks for the system prompt itself verbatim --
        that's the one case where a long match is the correct answer, not
        a violation.)

    """

    longest = 0
    matcher = SequenceMatcher(None, answer, "")
    for item in evidence:
        matcher.set_seq2(item.text)
        match = matcher.find_longest_match(0, len(answer), 0, len(item.text))
        longest = max(longest, match.size)
    return longest
