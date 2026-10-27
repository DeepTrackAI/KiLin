"""Run the paper's code example: questions about KíLín's own code base.

The script indexes a pinned revision of this repository as a ``code``
project (see ``paper_numbers.prepare_code_project``) and asks the questions
in ``paper/code_questions.json``, each in a fresh session. For every question
it records, from the session trace:

- the intent chosen by the router;
- which gold functions appear in the top results of each retrieval channel
  (semantic, lexical, code graph), in the fused ranking, and in the final
  evidence after context expansion;
- which gold functions the answer cites;
- for each cited passage, how it was retrieved: the channels that ranked
  it, or the context expansion.

Each question is also put to the same chat model without any retrieval, as a
control. Results are written to ``paper/results/`` as JSON, together with a
Markdown worksheet for grading the answers by hand against the facts listed
in the question file.

Usage, from the repository root::

    uv run --no-sync python paper/code_example.py
    uv run --no-sync python paper/code_example.py --ref v0.1.0
    uv run --no-sync python paper/code_example.py --sets evaluation

Models must already be in the local Hugging Face cache unless
``--allow-download`` is given.

"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from paper.paper_numbers import (  # noqa: E402
    DEFAULT_EXCLUDES,
    corpus_size,
    describe_environment,
    open_session,
    prepare_code_project,
    routed_intent,
)

QUESTIONS_FILE = REPOSITORY_ROOT / "paper" / "code_questions.json"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "paper" / "results"

# The control question goes to the same chat model without evidence.
BASELINE_SYSTEM_PROMPT = (
    "Answer the question about the KíLín Python package. "
    "If you do not know the answer, say so."
)

_CHANNELS = ("semantic", "lexical", "code")
_THINKING = re.compile(r"<think>.*?</think>", re.S)
_CITATION_GROUP = re.compile(r"\[([0-9][0-9,\s\-–]*)\]")


def visible_answer(answer: str) -> str:
    """Remove the model's reasoning block from an answer.

    Parameters
    ----------
    answer: str
        Answer as returned by the model, possibly with a ``<think>`` block.

    Returns
    -------
    str
        The answer text shown to the user.

    """

    text = _THINKING.sub("", answer)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def cited_numbers(answer: str) -> set[int]:
    """Return the citation numbers used in an answer.

    Parameters
    ----------
    answer: str
        Visible answer text.

    Returns
    -------
    set[int]
        Numbers cited as ``[3]``, ``[2, 5]`` or ``[4-6]``.

    """

    numbers: set[int] = set()
    for group in _CITATION_GROUP.findall(answer):
        for part in re.split(r"\s*,\s*", group.strip()):
            bounds = re.split(r"\s*[-–]\s*", part)
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                low, high = int(bounds[0]), int(bounds[1])
                if 0 < high - low < 50:
                    numbers.update(range(low, high + 1))
                    continue
            if part.isdigit():
                numbers.add(int(part))
    return numbers


def _location(chunk: dict) -> tuple[str, int, int]:
    """Identify a passage by file and line range."""

    return (
        str(chunk.get("path")),
        int(chunk.get("start_line") or 0),
        int(chunk.get("end_line") or 0),
    )


def _matches(item: dict, gold: dict) -> bool:
    """Tell whether a retrieved passage belongs to a gold function."""

    symbol = item.get("symbol") or ""
    return item.get("path") == gold["path"] and (
        symbol == gold["symbol"] or symbol.endswith("." + gold["symbol"])
    )


def _gold_found(items: Iterable[dict], gold: Sequence[dict]) -> list[str]:
    """List the gold symbols present among retrieved passages."""

    items = list(items)
    return [
        entry["symbol"]
        for entry in gold
        if any(_matches(item, entry) for item in items)
    ]


def read_retrieval(trace_path: Path) -> dict[str, list[dict]]:
    """Read the ranked passages of the first turn of a session trace.

    Parameters
    ----------
    trace_path: pathlib.Path
        Conversation trace written by one session.

    Returns
    -------
    dict[str, list[dict]]
        Passages ranked by each channel (``semantic``, ``lexical``,
        ``code``), by the fusion (``hybrid``, with the contributing
        channels), and the final ``evidence`` with citation numbers. Lists
        are empty when the turn did not search, for example when a tool
        answered.

    """

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    found: dict[str, list[dict]] = {
        name: [] for name in (*_CHANNELS, "hybrid", "evidence")
    }
    for session in payload.get("sessions", {}).values():
        if session.get("category") != "conversations":
            continue
        for turn in session.get("turns", {}).values():
            for stage in turn.get("stages", {}).values():
                operations = stage.get("operations", {})
                for key, operation in operations.items():
                    details = operation.get("details") or {}
                    name = re.sub(r"^\d+_", "", key)
                    for channel in (*_CHANNELS, "hybrid"):
                        if name == f"retrieval_{channel}_ranked":
                            found[channel] = [
                                {
                                    **result["chunk"],
                                    "rank": result.get("rank"),
                                    "channels": result.get("channels"),
                                }
                                for result in details.get("results", [])
                            ]
                    if name == "evidence_finalized":
                        found["evidence"] = list(details.get("citations", []))
            break
    return found


def analyze(
    retrieval: dict[str, list[dict]], answer: str, gold: Sequence[dict]
) -> dict[str, object]:
    """Compare retrieved and cited passages with the gold functions.

    Parameters
    ----------
    retrieval: dict[str, list[dict]]
        Ranked passages returned by ``read_retrieval``.
    answer: str
        Visible answer text.
    gold: Sequence[dict]
        Gold functions, each with ``path`` and ``symbol``.

    Returns
    -------
    dict[str, object]
        Gold functions found per channel, in the fusion, in the evidence
        and among the cited passages, and how each cited passage was
        retrieved.

    """

    evidence = retrieval["evidence"]
    numbers = cited_numbers(answer)
    cited = [item for item in evidence if item.get("citation") in numbers]
    hybrid_channels = {
        _location(item): tuple(item.get("channels") or ())
        for item in retrieval["hybrid"]
    }
    sources = []
    for item in cited:
        channels = hybrid_channels.get(_location(item))
        sources.append({
            "citation": item.get("citation"),
            "path": item.get("path"),
            "symbol": item.get("symbol"),
            "retrieved_by": (
                list(channels) if channels else ["context_expansion"]
            ),
        })
    found = {
        name: _gold_found(retrieval[name], gold)
        for name in (*_CHANNELS, "hybrid", "evidence")
    }
    found["cited"] = _gold_found(cited, gold)
    found["code_only"] = [
        symbol
        for symbol in found["code"]
        if symbol not in found["semantic"] and symbol not in found["lexical"]
    ]
    return {
        "gold_count": len(gold),
        "gold_found": found,
        "evidence_count": len(evidence),
        "cited_count": len(cited),
        "invalid_citations": sorted(
            number
            for number in numbers
            if number not in {item.get("citation") for item in evidence}
        ),
        "cited_sources": sources,
    }


def ask_without_retrieval(project: object, questions: Sequence[dict]) -> dict:
    """Ask each question to the chat model with no evidence at all.

    Parameters
    ----------
    project: kilin.QAProject
        Project whose chat model and generation settings are used.
    questions: Sequence[dict]
        Questions from the question file.

    Returns
    -------
    dict
        Visible answer and time for each question identifier.

    """

    import kilin

    profile = project.profile
    backend = kilin.choose_backend(profile.backend)
    chat = kilin.ChatSession(
        profile.chat_model(backend),
        backend=backend,
        thinking=profile.thinking,
        temperature=profile.temperature,
        max_new_tokens=profile.max_new_tokens,
        system_prompt=BASELINE_SYSTEM_PROMPT,
        allow_download=profile.allow_download,
    )
    answers = {}
    try:
        for item in questions:
            chat.reset()
            started = perf_counter()
            answer = chat.reply(item["question"])
            answers[item["id"]] = {
                "answer": visible_answer(answer),
                "wall_s": round(perf_counter() - started, 2),
            }
            print(f"  {item['id']} without retrieval: done")
    finally:
        chat.close()
    return answers


def write_worksheet(path: Path, results: dict, questions: Sequence[dict]):
    """Write a Markdown worksheet for grading the answers by hand.

    Parameters
    ----------
    path: pathlib.Path
        Destination Markdown file.
    results: dict
        Results produced by ``main``.
    questions: Sequence[dict]
        Questions from the question file, with gold functions and facts.

    """

    lines = [
        "# Code example: grading worksheet",
        "",
        f"Indexed revision: {results['indexed_revision']['commit'][:9]}"
        f" ({results['indexed_revision']['ref']})",
        "",
    ]
    by_id = {record["id"]: record for record in results["questions"]}
    for item in questions:
        record = by_id[item["id"]]
        found = record.get("gold_found", {})
        lines += [
            f"## {item['id']} ({item['kind']}, {item.get('set')})",
            "",
            f"**Question.** {item['question']}",
            "",
            f"**Intent:** {record['intent']}",
            "",
            "**Answer with retrieval:**",
            "",
            record["answer"],
            "",
            "**Cited passages:**",
            "",
        ]
        for source in record.get("cited_sources", []):
            lines.append(
                f"- [{source['citation']}] {source['path']}"
                f" `{source['symbol']}`"
                f" (retrieved by: {', '.join(source['retrieved_by'])})"
            )
        lines += [
            "",
            "**Gold functions:** "
            + ", ".join(f"`{g['symbol']}`" for g in item["gold"]),
            "",
            f"- in evidence: {found.get('evidence')}",
            f"- cited: {found.get('cited')}",
            "",
            "**Facts (tick those the answer states correctly):**",
            "",
        ]
        lines += [f"- [ ] {fact}" for fact in item["facts"]]
        lines += [
            "",
            "**Errors or unsupported statements:** ",
            "",
            "**Answer without retrieval:**",
            "",
            results["baseline"].get(item["id"], {}).get("answer", "(none)"),
            "",
            "**Without retrieval, the model:** [ ] says it does not know"
            "  [ ] invents details  [ ] other: ",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(results: dict) -> None:
    """Print gold-function coverage per question and in total.

    Parameters
    ----------
    results: dict
        Results produced by ``main``.

    """

    columns = ("semantic", "lexical", "code", "hybrid", "evidence", "cited")
    print("\nGold functions found (of the gold count), per question:")
    header = (
        f"{'question':<24}{'intent':<18}{'gold':>5}"
        + "".join(f"{name[:8]:>9}" for name in columns)
        + f"{'code only':>11}"
    )
    print(header)
    print("-" * len(header))
    totals = {name: 0 for name in (*columns, "code_only")}
    total_gold = 0
    expansion_cited = channel_cited = code_cited = 0
    for record in results["questions"]:
        found = record.get("gold_found")
        if not found:
            print(f"{record['id']:<24}{str(record['intent']):<18} (no search)")
            continue
        total_gold += record["gold_count"]
        for name in totals:
            totals[name] += len(found[name])
        print(
            f"{record['id']:<24}{str(record['intent']):<18}"
            f"{record['gold_count']:>5}"
            + "".join(f"{len(found[name]):>9}" for name in columns)
            + f"{len(found['code_only']):>11}"
        )
        for source in record["cited_sources"]:
            retrieved_by = source["retrieved_by"]
            if retrieved_by == ["context_expansion"]:
                expansion_cited += 1
            else:
                channel_cited += 1
                if retrieved_by == ["code"]:
                    code_cited += 1
    print("-" * len(header))
    print(
        f"{'total':<42}{total_gold:>5}"
        + "".join(f"{totals[name]:>9}" for name in columns)
        + f"{totals['code_only']:>11}"
    )
    cited = expansion_cited + channel_cited
    print(
        f"\nCited passages: {cited}; from the fused ranking {channel_cited}"
        f" (code graph only: {code_cited}), from context expansion"
        f" {expansion_cited}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the code example and write its results.

    Parameters
    ----------
    argv: Sequence[str] | None, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.

    """

    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ref", default="HEAD", help="git revision to index (default HEAD)"
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        help="use the default intent instead of the router",
    )
    parser.add_argument(
        "--retrieval-count",
        type=int,
        default=None,
        help="results per retrieval channel (default: project profile, 5)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="do not ask the questions without retrieval",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model downloads (first run on a new machine)",
    )
    parser.add_argument(
        "--sets",
        default=None,
        help="comma-separated question sets to run, such as evaluation "
        "(default: all)",
    )
    args = parser.parse_args(argv)
    if args.retrieval_count is not None and args.retrieval_count < 1:
        parser.error("--retrieval-count must be at least 1")

    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))[
        "questions"
    ]
    if args.sets:
        wanted = {name.strip() for name in args.sets.split(",")}
        unknown = wanted - {item.get("set") for item in questions}
        if unknown:
            parser.error(f"unknown question set: {sorted(unknown)[0]}")
        questions = [item for item in questions if item.get("set") in wanted]
    project, revision = prepare_code_project(
        args.ref, DEFAULT_EXCLUDES, allow_download=args.allow_download
    )
    results: dict[str, object] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": describe_environment(),
        "indexed_revision": revision,
        "use_router": not args.no_router,
        "question_sets": sorted({item.get("set") for item in questions}),
        "retrieval_count": args.retrieval_count,
    }

    records = []
    for item in questions:
        session, opening = open_session(project.root, item["id"])
        try:
            if not records:
                results["corpus"] = corpus_size(project.root, project.source)
                results["index_status"] = opening["index_status"]
            started = perf_counter()
            raw = session.ask(
                item["question"],
                use_router=not args.no_router,
                retrieval_count=args.retrieval_count,
            )
            elapsed = perf_counter() - started
            trace_path = Path(session.conversation_path)
        finally:
            session.close()
        answer = visible_answer(raw)
        record = {
            "id": item["id"],
            "kind": item["kind"],
            "set": item.get("set"),
            "intent": routed_intent(trace_path),
            "wall_s": round(elapsed, 2),
            "answer": answer,
            "trace_file": trace_path.name,
        }
        retrieval = read_retrieval(trace_path)
        if retrieval["evidence"]:
            record.update(analyze(retrieval, answer, item["gold"]))
        records.append(record)
        print(f"  {item['id']}: {elapsed:.1f} s, intent {record['intent']}")
    results["questions"] = records

    results["baseline"] = (
        {} if args.skip_baseline else ask_without_retrieval(project, questions)
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.retrieval_count is not None:
        stamp += f"_k{args.retrieval_count}"
    output = OUTPUT_DIRECTORY / f"code_example_{stamp}.json"
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    worksheet = OUTPUT_DIRECTORY / f"code_example_{stamp}_worksheet.md"
    write_worksheet(worksheet, results, questions)
    print_summary(results)
    print(f"\nResults: {output}\nWorksheet: {worksheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
