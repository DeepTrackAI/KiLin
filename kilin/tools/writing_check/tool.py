"""Tool ``writing_check``: check the grammar and style of a document.

"Check the grammar and style of draft.tex" is about one document of the
project, or a part of it. The tool:

1. Chooses the document (``choose_document``) and reads its selected
   source file into sections and paragraphs (``_document``): LaTeX,
   Markdown and text as written; Word and PDF through their source
   readers, as text.
2. Chooses the part the question names (``_scope``): a section, a
   paragraph, lines, or quoted text; the whole document otherwise. A part
   the document does not have is offered as a choice of its headings.
3. Runs the checks that need no model on the whole document
   (``_checks``): acronyms, terms, spelling, promotional words, repeated
   words, long sentences; their glossary goes into every model call.
4. Shows the plan (one model call per paragraph and check) when it is
   large, then checks each paragraph of the part for grammar and, at
   level ``all`` or ``style``, style (``_review``), with the document's
   card or a one-call brief. Every suggested change is checked before it
   is kept.
5. Answers with the first findings, by section, and saves the full report
   to ``outputs/``.

The tool never changes the document: it suggests changes for the author.

Key Features
------------

- **Context from the whole document**

    Glossary and card cover the whole document even when one section is
    checked; only findings in the chosen part are reported.

- **Checked suggestions**

    A model suggestion that changes a number, negation, citation or the
    authors' terms, or quotes words not in the text, is dropped.

Module Structure
----------------

- ``WritingCheckSettings`` and ``parse_settings``

    Settings of ``writing_check`` in a project type.

- ``run``

    Check the document or part a question names.

- ``report_markdown`` and ``chat_answer``

    The saved report and the chat answer.

Examples
--------
>>> from kilin.tools.writing_check import parse_settings
>>> parse_settings({"level": "grammar"}, "example").level
'grammar'

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Final

from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.choose_document import choose_document
from kilin.tools._shared.model_results import (
    incomplete_notice,
    partial_result,
)
from kilin.tools._shared.outputs import save_output_file
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)
from kilin.tools._shared.source_access import (
    selected_paths,
    selected_source_file,
)
from kilin.tools.writing_check import _configuration as _config
from kilin.tools.writing_check._checks import (
    LEVEL_CONSISTENCY,
    LEVEL_GRAMMAR,
    LEVEL_STYLE,
    Finding,
    check_document,
)
from kilin.tools.writing_check._document import (
    LATEX_SUFFIXES,
    MARKDOWN_SUFFIXES,
    Document,
    read_document,
)
from kilin.tools.writing_check._review import (
    document_brief,
    document_context,
    review_paragraph,
)
from kilin.tools.writing_check._scope import (
    Scope,
    ScopeProblem,
    choose_scope,
    section_scope,
)
from kilin.workspace_resources import PROJECT_SOURCES_DIRECTORY

__all__ = [
    "WritingCheckSettings",
    "chat_answer",
    "parse_settings",
    "report_markdown",
    "run",
]

# Files read through their source reader, and read as Markdown after it.
_READ_BY_EXTENSION: Final[frozenset[str]] = frozenset(
    {".docx", ".doc", ".pdf"}
)
_MARKDOWN_AFTER_READING: Final[frozenset[str]] = frozenset({".docx"})
# Order of the levels in counts.
_LEVEL_ORDER: Final[tuple[str, ...]] = (
    LEVEL_GRAMMAR, LEVEL_CONSISTENCY, LEVEL_STYLE,
)
_WHOLE_DOCUMENT: Final[str] = "The whole document"


@dataclass(frozen=True, slots=True)
class WritingCheckSettings:
    """Settings of ``writing_check`` in a project type.

    Attributes
    ----------
    level: str
        Model checks to run: ``grammar``, ``style`` or ``all`` (both).
        The checks done in code always run.
    confirmation: ConfirmationPolicy
        When to show the plan first: always, when_large, or never.
    large_batches: int
        With ``when_large``, plans with more model calls than this ask
        first (0-1000).
    quick_paragraphs: int
        Paragraphs checked for grammar by the quick version (1-1000).
    chat_findings: int
        Findings shown in the chat; all are in the saved report (0-200).

    """

    level: str = _config.DEFAULT_LEVEL
    confirmation: ConfirmationPolicy = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES
    quick_paragraphs: int = _config.DEFAULT_QUICK_PARAGRAPHS
    chat_findings: int = _config.DEFAULT_CHAT_FINDINGS


def parse_settings(data: dict, where: str) -> WritingCheckSettings:
    """Validate ``writing_check`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    WritingCheckSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range.

    """

    reject_unknown(
        where, data,
        ("level", "confirmation", "large_batches", "quick_paragraphs",
         "chat_findings"),
    )
    defaults = WritingCheckSettings()
    return WritingCheckSettings(
        level=check_choice(where, data, "level", defaults.level,
                           _config.LEVELS),
        confirmation=check_choice(
            where, data, "confirmation", defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 1000
        ),
        quick_paragraphs=check_int(
            where, data, "quick_paragraphs", defaults.quick_paragraphs, 1,
            1000,
        ),
        chat_findings=check_int(
            where, data, "chat_findings", defaults.chat_findings, 0, 200
        ),
    )


def run(
    settings: WritingCheckSettings,
    request: ToolRequest,
    context: ToolContext,
) -> ToolResult:
    """Check the grammar and style of the document a question names.

    Parameters
    ----------
    settings: WritingCheckSettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        Tool services: the project's files, the model, and the user.

    Returns
    -------
    ToolResult
        The answer with the first findings, no evidence passages (the
        findings quote the document), and a summary of the counts; a
        partial result when some model calls failed.

    """

    summary: dict[str, object] = {"name": "writing_check"}
    context.progress("Finding the document")
    chosen = choose_document(
        request, context,
        question="Which document should I check?",
        example="check the grammar and style of draft.tex",
        prefer=(".tex", ".md", ".docx", ".txt"),
    )
    if isinstance(chosen, str):
        context.done("No document chosen", evidence_count=0)
        return ToolResult(chosen, (), summary)
    path, reason, guessed = chosen
    summary.update(
        {"document": path.as_posix(), "reason": reason, "guessed": guessed}
    )

    text, read_as = _source_text(path, context)
    if text is None:
        context.done("Document not readable", evidence_count=0)
        return ToolResult(
            f"I could not read {path.name} from the project's selected "
            "sources. Select it and refresh the indexes, then ask again.",
            (), summary,
        )
    document = read_document(read_as, text)
    document = Document(path, document.sections, document.paragraphs)

    scope = _scope(request.question, document, context)
    if isinstance(scope, str):
        context.done("No part chosen", evidence_count=0)
        return ToolResult(scope, (), summary)
    if scope.text:
        document = read_document(Path("question.txt"), scope.text)
        document = Document(path, (), document.paragraphs)
        scope = Scope(tuple(range(len(document.paragraphs))), scope.label)
    summary["scope"] = scope.label
    if not scope.paragraphs:
        context.done("No prose in the part", evidence_count=0)
        return ToolResult(
            f"There is no prose to check in {scope.label} of {path.name}.",
            (), summary,
        )

    context.progress("Checking acronyms, terms and spelling")
    code_findings, glossary = check_document(document)
    in_scope = set(scope.paragraphs)
    findings = [f for f in code_findings if f.paragraph in in_scope]

    card_reader = getattr(context, "card", None)
    card = card_reader(path) if card_reader is not None else None
    levels = (
        (LEVEL_GRAMMAR, LEVEL_STYLE) if settings.level == "all"
        else (settings.level,)
    )
    paragraphs = scope.paragraphs
    calls = len(paragraphs) * len(levels) + (0 if card else 1)
    quick = min(len(paragraphs), settings.quick_paragraphs)
    plan = ToolPlan(
        tool="writing_check",
        summary=(
            f"Check {len(paragraphs)} paragraphs of {scope.label} of "
            f"{path.name} for {' and '.join(levels)}, one model call per "
            "paragraph and check"
            + ("" if card else ", plus one to describe the document")
            + "."
        ),
        scope=f"{scope.label}; acronyms, terms and spelling checked "
        "without the model.",
        items=len(paragraphs),
        batches=calls,
        batches_to_run=calls,
        estimated_seconds=calls * _call_seconds(),
        quick_items=quick,
        quick_seconds=(quick + (0 if card else 1)) * _call_seconds(),
        quick_label=f"Grammar of the first {quick} paragraphs",
    )
    choice, _asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    notes = []
    failures: list[dict[str, object]] = []
    dropped = 0
    checked = 0
    if choice == PLAN_CHOICE_CANCEL:
        notes.append("The model checks were cancelled; only the checks "
                     "done without the model are shown.")
    else:
        if choice == PLAN_CHOICE_QUICK:
            paragraphs, levels = paragraphs[:quick], (LEVEL_GRAMMAR,)
            first = "paragraph" if quick == 1 else f"{quick} paragraphs"
            notes.append(
                f"Quick version: grammar of the first {first} of "
                f"{len(scope.paragraphs)}."
            )
        brief = ""
        if not card:
            context.progress("Describing the document")
            brief = document_brief(document, context.ask_model)
        about = document_context(card, brief, glossary)
        for position, index in enumerate(paragraphs, start=1):
            for level in levels:
                checked += 1
                context.progress(
                    f"Checking {level} of paragraph {position} of "
                    f"{len(paragraphs)}"
                )
                kept, rejected, failure = review_paragraph(
                    level, document, index, about, glossary,
                    context.ask_model,
                )
                findings += kept
                dropped += rejected
                if failure:
                    failures.append({
                        "item": (
                            f"Paragraph at line "
                            f"{document.paragraphs[index].start_line} "
                            f"({level})"
                        ),
                        "message": failure,
                        "reason": "model_result",
                    })
    findings = _unique(findings)
    if dropped:
        changes = (
            "1 suggested change was" if dropped == 1
            else f"{dropped} suggested changes were"
        )
        notes.append(
            f"{changes} dropped: quoting words not in the text, or "
            "changing numbers, negations, citations or the document's "
            "terms."
        )

    counts = Counter(finding.level for finding in findings)
    saved = save_output_file(
        context.project_root,
        f"{path.stem}_writing_check.md",
        report_markdown(document, scope, findings, notes)
        + incomplete_notice(failures),
    ) if findings or failures else None
    summary.update({
        "paragraphs": len(scope.paragraphs),
        "levels": list(levels),
        "card": bool(card),
        "findings": {level: counts[level] for level in _LEVEL_ORDER},
        "dropped": dropped,
        "model_calls": checked,
        "report": saved.as_posix() if saved else None,
    })
    context.done(f"Checked {len(scope.paragraphs)} paragraphs",
                 evidence_count=0)
    answer = chat_answer(
        document, scope, findings, settings.chat_findings, notes, saved
    )
    return partial_result(
        ToolResult(answer, (), summary), failures,
        completed=checked == 0 or len(failures) < checked,
    )


def _source_text(
    path: Path, context: ToolContext
) -> tuple[str | None, Path]:
    """Return a selected document's text, and the path to read it as.

    Parameters
    ----------
    path: pathlib.Path
        The document's indexed path.
    context: ToolContext
        Tool services, for the project and its selected sources.

    Returns
    -------
    tuple[str | None, pathlib.Path]
        The text, or None when the file is not a selected, readable
        source; and the path whose suffix says how to read the text (a
        Word file is read as Markdown after its source reader, a PDF as
        plain text).

    """

    root = Path(context.project_root)
    source = getattr(context, "source_root", None)
    if source is None:
        source = root / PROJECT_SOURCES_DIRECTORY
    file = selected_source_file(
        root, Path(source), path, selected_paths(context)
    )
    if file is None:
        return None, path
    suffix = path.suffix.lower()
    if suffix not in _READ_BY_EXTENSION:
        try:
            return file.read_text(encoding="utf-8", errors="replace"), path
        except OSError:
            return None, path
    try:
        reader = importlib.import_module(
            f"kilin.sources.extensions.sources_{suffix[1:]}"
        )
        text = reader.read_source_file(file, path).text
    except (OSError, ValueError, ImportError):
        return None, path
    kind = ".md" if suffix in _MARKDOWN_AFTER_READING else ".txt"
    return text, path.with_suffix(kind)


def _scope(
    question: str, document: Document, context: ToolContext
) -> Scope | str:
    """Return the part to check, asking the user when it is unclear.

    Parameters
    ----------
    question: str
        The question.
    document: Document
        The document.
    context: ToolContext
        Tool services, for asking the user.

    Returns
    -------
    Scope | str
        The part to check; or, when the question names a part the
        document does not have and nobody can be asked or the user
        cancels, a message listing the headings.

    """

    found = choose_scope(question, document)
    if isinstance(found, Scope):
        return found
    choose = getattr(context, "choose", None)
    options = (*found.headings, _WHOLE_DOCUMENT)
    if choose is not None and found.headings:
        picked = choose(f"{found.message} Which part should I check?",
                        options)
        if picked is not None and 0 <= picked < len(options):
            if options[picked] == _WHOLE_DOCUMENT:
                return Scope(tuple(range(len(document.paragraphs))),
                             "the whole document")
            names = [f"{s.number} {s.title}".strip()
                     for s in document.sections]
            return section_scope(names.index(options[picked]), document)
    listed = "; ".join(found.headings) or "none"
    return (
        f"{found.message} The headings of {document.path.name} are: "
        f"{listed}. Name one, for example: check the grammar of section "
        f"{_example(found.headings)}."
    )


def _example(headings: Sequence[str]) -> str:
    """Return a heading to show in an example question.

    Parameters
    ----------
    headings: Sequence[str]
        The document's headings.

    Returns
    -------
    str
        The first heading's number, else its title, else "2".

    """

    if not headings:
        return "2"
    first = headings[0]
    number = first.split(" ", 1)[0]
    return number if number[:1].isdigit() else f"“{first}”"


def _call_seconds() -> float:
    """Return the estimated time of one paragraph call, in seconds.

    Returns
    -------
    float
        A planning estimate, not a limit.

    """

    return (
        _config.CALL_SECONDS
        + _config.PROMPT_OVERHEAD_TOKENS / _config.PROMPT_TOKENS_PER_SECOND
        + _config.REPLY_TOKENS / _config.REPLY_TOKENS_PER_SECOND
    )


def _unique(findings: Sequence[Finding]) -> list[Finding]:
    """Return findings in reading order, without repeats.

    Parameters
    ----------
    findings: Sequence[Finding]
        Findings from the code checks and the model.

    Returns
    -------
    list[Finding]
        One finding per paragraph, words and suggestion (case ignored),
        the first kept, sorted by paragraph, line and position.

    """

    seen = set()
    kept = []
    for finding in findings:
        key = (finding.paragraph, finding.original.casefold(),
               finding.suggestion.casefold())
        if key in seen:
            continue
        seen.add(key)
        kept.append(finding)
    kept.sort(key=lambda f: (f.paragraph, f.line, f.offset))
    return kept


def _section_name(document: Document, finding: Finding) -> str:
    """Return the heading a finding falls under.

    Parameters
    ----------
    document: Document
        The document.
    finding: Finding
        A finding.

    Returns
    -------
    str
        "2.1 Data" or the title alone; "Before the first heading" when
        the paragraph has no section.

    """

    index = document.paragraphs[finding.paragraph].section
    if index is None:
        return "Before the first heading" if document.sections else ""
    section = document.sections[index]
    return f"{section.number} {section.title}".strip()


def _counts(findings: Sequence[Finding]) -> str:
    """Return the number of findings per level, as words.

    Parameters
    ----------
    findings: Sequence[Finding]
        The findings.

    Returns
    -------
    str
        "3 grammar, 2 consistency, 5 style findings", or "No findings".

    """

    counts = Counter(finding.level for finding in findings)
    parts = [f"{counts[level]} {level}" for level in _LEVEL_ORDER
             if counts[level]]
    if not parts:
        return "No findings"
    return ", ".join(parts) + (" finding" if len(findings) == 1
                               else " findings")


def _cell(text: str) -> str:
    """Return text safe in a Markdown table cell.

    Parameters
    ----------
    text: str
        Any text.

    Returns
    -------
    str
        The text on one line, with ``|`` escaped.

    """

    return " ".join(text.split()).replace("|", "\\|")


def report_markdown(
    document: Document,
    scope: Scope,
    findings: Sequence[Finding],
    notes: Sequence[str],
) -> str:
    """Return the full report, to save as Markdown.

    Parameters
    ----------
    document: Document
        The document.
    scope: Scope
        The part checked.
    findings: Sequence[Finding]
        Every finding, in reading order.
    notes: Sequence[str]
        Notes on how the check ran.

    Returns
    -------
    str
        A title, the part and the counts, the notes, and one table per
        section: line, level, original, suggestion and reason.

    """

    lines = [
        f"# Writing check: {document.path.name}", "",
        f"Checked: {scope.label}. {_counts(findings)}.", "",
    ]
    lines += [f"*{note}*" for note in notes]
    current = None
    for finding in findings:
        name = _section_name(document, finding)
        if name != current:
            current = name
            lines += [
                "", f"## {name or 'Text'}", "",
                "| Line | Level | Original | Suggestion | Reason |",
                "| ---: | --- | --- | --- | --- |",
            ]
        lines.append(
            f"| {finding.line} | {finding.level} | {_cell(finding.original)}"
            f" | {_cell(finding.suggestion) or '—'} | "
            f"{_cell(finding.reason)} |"
        )
    return "\n".join(lines) + "\n"


def chat_answer(
    document: Document,
    scope: Scope,
    findings: Sequence[Finding],
    limit: int,
    notes: Sequence[str],
    saved: Path | None,
) -> str:
    """Return the chat answer.

    Parameters
    ----------
    document: Document
        The document.
    scope: Scope
        The part checked.
    findings: Sequence[Finding]
        Every finding, in reading order.
    limit: int
        Findings to show; the rest are counted.
    notes: Sequence[str]
        Notes on how the check ran.
    saved: pathlib.Path | None
        The saved report, if any.

    Returns
    -------
    str
        Markdown: the part and counts, then up to ``limit`` findings under
        their section headings ("line 12 · grammar: “a” → “b” (reason)"),
        the notes, and where the full report is.

    """

    lines = [
        f"**Writing check of {document.path.name}** ({scope.label})", "",
        f"{_counts(findings)}.",
    ]
    current = None
    for finding in findings[:limit]:
        name = _section_name(document, finding)
        if name != current:
            current = name
            lines += ["", f"**{name}**" if name else ""]
        change = f"“{finding.original}”"
        if finding.suggestion:
            change += f" → “{finding.suggestion}”"
        lines.append(
            f"- line {finding.line} · {finding.level}: {change} "
            f"({finding.reason.rstrip('.')})"
        )
    hidden = len(findings) - min(len(findings), limit)
    if hidden:
        lines += ["", f"*… and {hidden} more in the report.*"]
    for note in notes:
        lines += ["", f"*{note}*"]
    if saved is not None:
        lines += ["", f"The full report is saved as `{saved.name}` in "
                  "outputs/."]
    return "\n".join(lines)
