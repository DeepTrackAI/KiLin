"""Check that the router sends file questions to the right code intent.

Two groups of questions are routed with the code project type: questions
about the repository's own files, which belong to ``project_inventory``,
and questions about code that selects, reads, or skips files, which belong
to ``code_explanation``. Both groups must route correctly; a change to the
intent descriptions that fixes one group must not break the other.

Only the router runs: no project is opened and nothing is retrieved.
The router uses the packaged core router default unless ``--model``
overrides it. No project or GUI session is required.

Usage, from the repository root::

    uv run --no-sync python assets/router_check.py
    uv run --no-sync python assets/router_check.py --model Qwen/Qwen3-8B

The exit status is 1 when any question is routed to another intent.

"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
import sys
from time import perf_counter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Questions about the repository's own files.
INVENTORY_QUESTIONS = (
    "How many Python files are there?",
    "List the notebooks in the project.",
    "Which files were not indexed?",
    "Which file types are included in this project?",
    "How many MATLAB files does the repository have?",
    "Which files could not be read?",
)

# Questions about code that selects, reads, or skips files.
FILE_CODE_QUESTIONS = (
    "In which function are hidden folders skipped during scanning?",
    "How does scan_repository treat symbolic links?",
    "How does the loader skip hidden files?",
    "How are PDF files converted to text before they are indexed?",
    "Explain how build_repository_manifest computes the hashes of the "
    "source files.",
    "How does the importer handle a file that is too large?",
)


def cases() -> list[tuple[str, str]]:
    """Return the questions with their expected intents.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of expected intent and question.

    """

    return [
        *(("project_inventory", item) for item in INVENTORY_QUESTIONS),
        *(("code_explanation", item) for item in FILE_CODE_QUESTIONS),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Route every question and report the intents.

    Parameters
    ----------
    argv: Sequence[str] | None, optional
        Command-line arguments. Defaults to ``None``, which reads
        ``sys.argv``.

    Returns
    -------
    int
        0 when every question is routed as expected, otherwise 1.

    """

    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        help="Router model; defaults to the core router model.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "mlx", "torch"),
        help="Computation backend. Defaults to auto.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Download the model if it is not in the local cache.",
    )
    args = parser.parse_args(argv)

    import kilin
    from kilin.answering import route_question

    backend = kilin.choose_backend(args.backend)
    from kilin.model_settings import default_router_model
    model = args.model or default_router_model(backend)
    project = kilin.project_type(kilin.PROJECT_TYPE_ID_CODE)
    print(f"Router model: {model} ({backend})")
    router = kilin.ChatSession(
        model,
        backend=backend,
        thinking=False,
        temperature=0.0,
        allow_download=args.allow_download,
    )
    failures = 0
    try:
        for expected, question in cases():
            started = perf_counter()
            decision = route_question(router, project, [], question)
            elapsed = perf_counter() - started
            ok = decision.intent == expected
            failures += not ok
            print(
                f"{'OK' if ok else 'WRONG':5} {elapsed:4.1f}s "
                f"expected={expected:18} routed={decision.intent:18} "
                f"{question}"
            )
    finally:
        router.close()
    total = len(cases())
    print(f"\n{total - failures} of {total} routed as expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
