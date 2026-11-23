"""Measure the hardware and timing figures reported in the KíLín paper.

One run exports a pinned revision of this repository with ``git archive``,
mirrors it into a dedicated sandbox workspace as a ``code`` project, and
measures, through the public Python API:

1. the hardware and software environment;
2. the first (cold) opening, which builds every index;
3. a second opening with unchanged sources, which reuses the indexes;
4. a fixed set of questions, each asked in a fresh session so that no
   conversation history carries over, with per-stage timings read from the
   session trace by ``assets.trace_benchmark``;
5. peak memory of the process and of the computation backend.

Each opening writes its own trace file, so measurements of earlier runs are
never mixed in. Results are written as timing-only JSON to
``_si_article_/code_analysis/results/`` and printed as the values to copy
into the manuscript.
Prompts, answers and retrieved passages are not exported; they stay in the
trace of the sandbox workspace under ``sandbox/paper_numbers/``.

Usage, from the repository root::

    uv run --no-sync python _si_article_/code_analysis/paper_numbers.py
    uv run --no-sync python _si_article_/code_analysis/paper_numbers.py \\
        --ref v0.1.0 --repeats 3

Models must already be in the local Hugging Face cache unless
``--allow-download`` is given. The sandbox workspace is reused between runs;
its saved indexes are deleted before the cold opening. Evaluation
materials, including ``_si_article_``, are excluded by default.
``--include-examples`` disables those exclusions for exploratory runs;
such runs can retrieve the questions or previous answers and must not
be reported as uncontaminated evaluation.

"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import shutil
from tempfile import gettempdir
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from assets.trace_benchmark import (  # noqa: E402
    read_trace_operations,
    summarize_trace_turns,
)

# The sandbox workspace (project, indexes and traces) stays out of version
# control; the timing results are kept with the other paper results.
WORKSPACE_DIRECTORY = REPOSITORY_ROOT / "sandbox" / "paper_numbers"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

# Each question names code entities, so the code graph can take part, and
# has an answer that can be checked against the indexed source:
# - routing: answering.route_question, RouteDecision, default-intent fallback;
# - index_reuse: RAGSession._prepare_retrieval, source fingerprint and
#   embedding-model identity;
# - index_storage: build_semantic_index, save_indexes and load_indexes
#   (fingerprint and payload checksum).
QUESTIONS: tuple[tuple[str, str], ...] = (
    (
        "routing",
        "Explain how route_question assigns a question to an intent, and "
        "what happens when the router reply cannot be used.",
    ),
    (
        "index_reuse",
        "Explain how RAGSession decides whether to reuse the saved "
        "retrieval indexes or to rebuild them.",
    ),
    (
        "index_storage",
        "Explain how build_semantic_index embeds the chunks, and how "
        "save_indexes and load_indexes protect the saved index.",
    ),
)

# Initialization operations reported separately from question turns.
_STARTUP_OPERATIONS = (
    "session.ready",
    "model.embedding.load",
    "model.chat.load",
    "indexing",
    "index.code_graph.build",
    "index.semantic.build",
    "index.lexical.build",
    "index.load",
    "index.save",
)


def _run(command: Sequence[str]) -> str:
    """Return the stripped output of a command, or empty text on failure."""

    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, check=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def describe_environment() -> dict[str, object]:
    """Collect hardware, operating-system and package versions.

    Returns
    -------
    dict[str, object]
        Non-sensitive environment description.

    """

    system = platform.system()
    info: dict[str, object] = {
        "system": system,
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if system == "Darwin":
        info["os_version"] = _run(["sw_vers", "-productVersion"])
        info["model"] = _run(["sysctl", "-n", "hw.model"])
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = _run(["sysctl", "-n", "hw.memsize"])
        if memory.isdigit():
            info["memory_gb"] = round(int(memory) / 2**30)
    elif system == "Linux":
        info["os_version"] = platform.release()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            match = re.search(
                r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.M
            )
            if match:
                info["chip"] = match.group(1).strip()
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            match = re.search(r"^MemTotal:\s*(\d+)", meminfo.read_text(), re.M)
            if match:
                info["memory_gb"] = round(int(match.group(1)) / 2**20)
    else:
        info["os_version"] = platform.version()
        info["chip"] = platform.processor()
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import mlx.core as mx

        info["mlx"] = getattr(mx, "__version__", "unknown")
    except ImportError:
        pass
    return info


# Left out of the indexed corpus by default:
# - example_repos: bundled third-party example corpora (about 66 MB of text),
#   not part of the KíLín code base; indexing them would take hours;
# - assets and _si_article_: measurement scripts, evaluation questions,
#   previous results and source corpora must not become answer evidence;
# - benchmarks and paper: legacy locations in older pinned revisions.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    "example_repos",
    "assets",
    "_si_article_",
    "benchmarks",
    "paper",
)


def export_revision(
    ref: str,
    destination: Path,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> dict[str, object]:
    """Export one committed revision of this repository with git archive.

    Parameters
    ----------
    ref: str
        Git revision, tag or branch to export.
    destination: pathlib.Path
        Empty directory that receives the exported files.
    excludes: Sequence[str], optional
        Top-level folders left out of the export. Defaults to
        ``DEFAULT_EXCLUDES``, which omits evaluation materials and
        supplementary corpora. An explicit empty sequence includes
        them and is unsuitable for uncontaminated evaluation.

    Returns
    -------
    dict[str, object]
        The requested reference, the resolved commit hash and the excluded
        folders.

    Raises
    ------
    RuntimeError
        If the reference cannot be resolved or exported.

    Notes
    -----
    Only committed, tracked files are exported, so uncommitted changes,
    the sandbox and local environments are never indexed.

    """

    commit = _run(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", ref])
    if not commit:
        raise RuntimeError(f"Unknown git revision: {ref}")
    pathspecs = [f":(exclude){folder}" for folder in excludes]
    archive = subprocess.run(
        [
            "git", "-C", str(REPOSITORY_ROOT), "archive", "--format=tar",
            commit, "--", ".", *pathspecs,
        ],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        check=True,
    )
    return {"ref": ref, "commit": commit, "excluded": list(excludes)}


def _disable_downloads(profile_path: Path) -> None:
    """Set ``allow_download = false`` in a sandbox project profile."""

    text = profile_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^(\s*allow_download\s*=\s*).*$",
        r"\1false",
        text,
        flags=re.M,
    )
    if count != 1:
        raise RuntimeError("Could not find allow_download in profile.ini.")
    profile_path.write_text(updated, encoding="utf-8")


def prepare_code_project(
    ref: str, excludes: Sequence[str], allow_download: bool = False
) -> tuple[object, dict[str, object]]:
    """Export a revision of this repository and open it as a code project.

    Parameters
    ----------
    ref: str
        Git revision, tag or branch to index.
    excludes: Sequence[str]
        Top-level folders left out of the indexed corpus.
    allow_download: bool, optional
        Keep model downloads enabled in the project profile. Defaults to
        ``False``, which sets ``allow_download = false``.

    Returns
    -------
    tuple[kilin.QAProject, dict[str, object]]
        The sandbox project and the description of the indexed revision.

    Notes
    -----
    The revision is exported to a fixed folder in the system temporary
    directory. The first run creates the sandbox workspace
    ``project_kilin_paper_source``; later runs synchronize that same
    workspace, so only changed files are copied. A workspace that the
    installed KíLín cannot open (for example one written by an older
    version) is renamed with a time stamp and a new one is created, so
    traces of earlier runs are kept.

    """

    import kilin

    source = Path(gettempdir()) / "kilin_paper_source"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    revision = export_revision(ref, source, excludes)
    WORKSPACE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    workspace = WORKSPACE_DIRECTORY / f"project_{source.name}"
    project = None
    if workspace.exists():
        try:
            project = kilin.synchronize_project_workspace(source, workspace)
        except (OSError, ValueError) as error:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retired = workspace.with_name(f"{workspace.name}_old_{stamp}")
            print(
                f"Cannot reuse {workspace.name} ({error}); "
                f"renamed it to {retired.name}."
            )
            workspace.rename(retired)
    if project is None:
        project = kilin.prepare_project_workspace(
            source, "code", WORKSPACE_DIRECTORY
        )
        if project.root != workspace:
            raise RuntimeError(
                f"Expected the sandbox workspace {workspace}, "
                f"got {project.root}."
            )
    if not allow_download:
        _disable_downloads(project.root / "profile.ini")
    return project, revision


def corpus_size(project_root: Path, source: Path) -> dict[str, int]:
    """Count source files, chunks and code-graph records of a project.

    Parameters
    ----------
    project_root: pathlib.Path
        KíLín workspace containing ``indexes/retrieval.json``.
    source: pathlib.Path
        Mirrored source folder of the workspace.

    Returns
    -------
    dict[str, int]
        File, chunk, entity and relation counts.

    """

    payload = json.loads(
        (project_root / "indexes" / "retrieval.json").read_text(
            encoding="utf-8"
        )
    )
    graph = payload.get("code_graph") or {}
    return {
        "source_files": sum(1 for path in source.rglob("*") if path.is_file()),
        "chunks": len(payload.get("chunks", [])),
        "graph_entities": len(graph.get("entities", [])),
        "graph_relations": len(graph.get("relations", [])),
    }


def _reset_backend_peak() -> None:
    """Reset the backend's peak-memory counter when the backend offers one."""

    try:
        import mlx.core as mx

        reset = getattr(mx, "reset_peak_memory", None) or getattr(
            getattr(mx, "metal", None), "reset_peak_memory", None
        )
        if reset is not None:
            reset()
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_memory_gb() -> dict[str, float]:
    """Return peak resident memory and backend peak memory in gigabytes.

    Returns
    -------
    dict[str, float]
        ``process`` peak resident set size, plus ``mlx`` or ``cuda`` peak
        allocation when available. On Apple silicon the model memory of MLX
        may not be included in the process figure, so both are reported.

    """

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    scale = 1 if platform.system() == "Darwin" else 1024
    result = {"process": round(peak * scale / 2**30, 2)}
    try:
        import mlx.core as mx

        getter = getattr(mx, "get_peak_memory", None) or getattr(
            getattr(mx, "metal", None), "get_peak_memory", None
        )
        if getter is not None:
            result["mlx"] = round(getter() / 2**30, 2)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            peak_cuda = torch.cuda.max_memory_allocated()
            result["cuda"] = round(peak_cuda / 2**30, 2)
    except ImportError:
        pass
    return result


def _startup_operations(trace_path: Path, label: str) -> dict[str, float]:
    """Sum completed initialization operations of one traced opening."""

    totals: dict[str, float] = defaultdict(float)
    for row in read_trace_operations(trace_path, label):
        if (
            row.get("turn") is None
            and row.get("status") == "completed"
            and row.get("operation") in _STARTUP_OPERATIONS
        ):
            totals[str(row["operation"])] += float(row["duration_ms"])
    return {name: round(value / 1000.0, 2) for name, value in totals.items()}


def show_progress(event: object) -> None:
    """Print one updating progress line for indexing and model loading.

    Parameters
    ----------
    event: kilin.ProgressEvent
        Progress report emitted while a session opens.

    """

    parts = [f"{event.percent:5.1f}%", event.message]
    if event.current is not None and event.total is not None:
        parts.append(f"{event.current:,}/{event.total:,}")
    if event.eta_seconds is not None:
        minutes, seconds = divmod(round(event.eta_seconds), 60)
        parts.append(f"ETA {minutes}m {seconds:02d}s")
    print("\r  " + "  |  ".join(parts).ljust(78), end="", flush=True)


def routed_intent(trace_path: Path) -> str | None:
    """Return the intent recorded for the first turn of a session trace.

    Parameters
    ----------
    trace_path: pathlib.Path
        Conversation trace written by one session.

    Returns
    -------
    str | None
        Intent name, or None when the trace does not record one.

    """

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    for session in payload.get("sessions", {}).values():
        if session.get("category") != "conversations":
            continue
        for turn in session.get("turns", {}).values():
            match = re.search(r'"intent": "([a-z_]+)"', json.dumps(turn))
            if match:
                return match.group(1)
    return None


def open_session(project_root: Path, label: str) -> tuple[object, dict]:
    """Open a traced session and time its construction.

    Parameters
    ----------
    project_root: pathlib.Path
        KíLín workspace to open.
    label: str
        Name of this opening in the results.

    Returns
    -------
    tuple[kilin.RAGSession, dict]
        The open session and its measurements.

    """

    import kilin

    started = perf_counter()
    session = kilin.RAGSession(
        project_root,
        progress_callback=show_progress,
        trace=kilin.TraceConfig(enabled=True),
    )
    elapsed = perf_counter() - started
    print()
    return session, {
        "label": label,
        "wall_s": round(elapsed, 2),
        "index_status": session.index_status,
        "backend": session.backend,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full measurement and write the results.

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
        "--repeats", type=int, default=1, help="repetitions of the questions"
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        help="use the default intent instead of the router",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help=(
            "also index example_repos, assets, _si_article_ and legacy "
            "analysis folders (slow; includes evaluation material)"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model downloads (first run on a new machine)",
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import kilin

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": describe_environment(),
        "kilin_checkout": _run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"]
        ),
        "use_router": not args.no_router,
    }

    # 1. Export the pinned revision outside the repository and mirror it.
    excludes = () if args.include_examples else DEFAULT_EXCLUDES
    project, revision = prepare_code_project(
        args.ref, excludes, allow_download=args.allow_download
    )
    results["indexed_revision"] = revision
    project_root = project.root
    backend = kilin.choose_backend(project.profile.backend)
    results["models"] = {
        "chat": project.profile.chat_model(backend),
        "embedding": project.profile.embedding_model(backend),
    }

    # 2. Cold opening: remove saved indexes so every index is rebuilt.
    index_file = project_root / "indexes" / "retrieval.json"
    if index_file.exists():
        index_file.unlink()
    _reset_backend_peak()
    session, cold = open_session(project_root, "cold")
    trace_path = Path(session.conversation_path)
    session.close()
    cold["operations_s"] = _startup_operations(trace_path, "cold")
    results["cold_open"] = cold
    results["corpus"] = corpus_size(project_root, project.source)
    print(f"Cold opening: {cold['wall_s']} s ({cold['index_status']})")

    # 3. Warm opening with unchanged sources.
    session, warm = open_session(project_root, "warm")
    warm_trace = Path(session.conversation_path)
    session.close()
    warm["operations_s"] = _startup_operations(warm_trace, "warm")
    results["warm_open"] = warm
    print(f"Warm opening: {warm['wall_s']} s ({warm['index_status']})")

    # 4. Each question in a fresh session, so no conversation history or
    # earlier evidence carries over from one question to the next.
    questions: list[dict[str, object]] = []
    for repeat in range(1, args.repeats + 1):
        for question_id, question in QUESTIONS:
            label = f"{question_id}_r{repeat}"
            session, opening = open_session(project_root, label)
            try:
                started = perf_counter()
                answer = session.ask(question, use_router=not args.no_router)
                elapsed = perf_counter() - started
                trace_path = Path(session.conversation_path)
            finally:
                session.close()
            record: dict[str, object] = {
                "repeat": repeat,
                "question_id": question_id,
                "intent": routed_intent(trace_path),
                "open_s": opening["wall_s"],
                "wall_s": round(elapsed, 2),
                "answer_characters": len(answer),
                "answer_sha256": hashlib.sha256(
                    answer.encode("utf-8")
                ).hexdigest(),
                # To be filled in by hand after checking the answer
                # against the indexed source.
                "verdict": None,
                "trace_file": trace_path.name,
            }
            turns = summarize_trace_turns(
                read_trace_operations(trace_path, label)
            )
            for key, value in (turns[0] if turns else {}).items():
                if key in {"run", "turn"} or value is None:
                    continue
                if key.endswith("_ms"):
                    record[key[:-3] + "_s"] = round(value / 1000.0, 2)
                elif isinstance(value, float):
                    record[key] = round(value, 1)
                else:
                    record[key] = value
            questions.append(record)
            print(
                f"  {question_id} (repeat {repeat}): {elapsed:.1f} s,"
                f" intent {record['intent']}"
            )
    results["questions"] = questions
    results["peak_memory_gb"] = peak_memory_gb()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS_DIRECTORY / f"paper_numbers_{stamp}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nFull results: {output}")
    return 0


def _retrieval_s(item: dict) -> float | None:
    """Return the summed retrieval time of one question, if measured."""

    parts = [
        item[key]
        for key in ("semantic_s", "lexical_s", "code_s")
        if key in item
    ]
    return round(sum(parts), 2) if parts else None


def _spread(values: list[float]) -> str:
    """Format the median and range of a small sample."""

    from statistics import median

    if not values:
        return "n/a"
    middle = round(median(values), 2)
    if len(values) == 1:
        return f"{middle}"
    return f"{middle} (range {min(values)}-{max(values)})"


def print_summary(results: dict) -> None:
    """Print the values to copy into the manuscript.

    Parameters
    ----------
    results: dict
        Measurements collected by ``main``.

    Notes
    -----
    Questions answered by the language model and questions answered by a
    tool without generation have different costs, so their statistics are
    reported separately rather than pooled.

    """

    environment = results["environment"]
    corpus = results["corpus"]
    cold = results["cold_open"]
    warm = results["warm_open"]
    questions = results["questions"]

    print("\n--- Values for the manuscript ---")
    print(
        "Hardware:",
        environment.get("model", ""),
        environment.get("chip", ""),
        f"{environment.get('memory_gb', '?')} GB RAM,",
        environment.get("system"),
        environment.get("os_version", ""),
    )
    print("Backend:", cold["backend"], "| models:", results["models"])
    revision = results["indexed_revision"]
    print(
        "Indexed revision:", revision["ref"], revision["commit"][:9],
        "| excluded:", ", ".join(revision["excluded"]) or "none",
    )
    print(
        "Corpus:",
        corpus["source_files"], "files,",
        corpus["chunks"], "chunks,",
        corpus["graph_entities"], "graph entities,",
        corpus["graph_relations"], "relations",
    )
    print(
        "Cold opening:", cold["wall_s"], "s; of which",
        cold["operations_s"].get("index.semantic.build"), "s embedding",
    )
    print("Warm opening (indexes reused):", warm["wall_s"], "s")
    print("Peak memory (GB):", results["peak_memory_gb"])

    print("\nPer question (seconds):")
    header = (
        f"{'rep':>3}  {'question':<14} {'intent':<18} {'total':>6} "
        f"{'route':>6} {'retr.':>6} {'gen.':>6} {'tokens':>6} {'tok/s':>6}"
    )
    print(header)
    print("-" * len(header))
    for item in questions:
        by_model = item.get("llm_calls", 0) > 0
        retrieval_s = _retrieval_s(item)
        print(
            f"{item['repeat']:>3}  {item['question_id']:<14} "
            f"{str(item.get('intent')) + ('' if by_model else '*'):<18} "
            f"{item['wall_s']:>6} "
            f"{item.get('routing_s', '-'):>6} "
            f"{retrieval_s if retrieval_s is not None else '-':>6} "
            f"{item.get('generation_s', '-'):>6} "
            f"{item.get('output_tokens', '-') if by_model else '-':>6} "
            f"{item.get('output_tokens_per_second', '-'):>6}"
        )

    print("  * answered by a tool without language-model generation")
    model = [item for item in questions if item.get("llm_calls", 0) > 0]
    tool = [item for item in questions if item.get("llm_calls", 0) == 0]

    def values(items: list[dict], key: str) -> list[float]:
        return [item[key] for item in items if key in item]

    print(f"\nModel-answered questions (n={len(model)}), median and range:")
    print("  total:      ", _spread(values(model, "wall_s")), "s")
    print("  routing:    ", _spread(values(model, "routing_s")), "s")
    retrieval = [
        value for value in map(_retrieval_s, model) if value is not None
    ]
    print("  retrieval:  ", _spread(retrieval), "s")
    print("  generation: ", _spread(values(model, "generation_s")), "s")
    print("  tokens:     ", _spread(values(model, "output_tokens")))
    print(
        "  tokens/s:   ",
        _spread(values(model, "output_tokens_per_second")),
    )
    if tool:
        print(
            f"Tool-answered questions without generation (n={len(tool)}):",
            _spread(values(tool, "wall_s")), "s total",
        )

    repeats = {item["repeat"] for item in questions}
    if len(repeats) > 1:
        print("\nIdentical answers across repeats:")
        by_question: dict[str, set[str]] = defaultdict(set)
        for item in questions:
            by_question[item["question_id"]].add(item["answer_sha256"])
        for question_id, hashes in by_question.items():
            print(f"  {question_id:<14} {'yes' if len(hashes) == 1 else 'no'}")


if __name__ == "__main__":
    raise SystemExit(main())
