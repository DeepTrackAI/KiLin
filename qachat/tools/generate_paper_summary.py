"""Tool ``generate_paper_summary``: a structured summary of one paper.

Turns one paper into a summary that follows a report template, such as
``research_briefing`` (the one-page Research Briefing format of
Nature-family journals) or ``research_summary`` (a compact structured
summary), and saves it as Markdown in the project's ``target/`` folder.
The question picks the template ("research briefing"); the intent's
default applies otherwise. It is QAChat's reference example of a custom
tool (see tutorial 150): it composes existing capabilities instead of
re-implementing them.

1. Choose the paper: a ``target/`` file or an indexed source the question
   names, the only ``target/`` file when the question says "my paper", or
   the indexed paper that best matches the question.
2. Load it: ``load_target`` for ``target/`` files; indexed passages for
   sources.
3. Read it in parts: ``batch_review.take_notes`` writes notes with passage
   numbers, because a whole paper is too long for one small-model prompt.
4. Write the headline and each section in separate calls; Python
   assembles the headings, so sections cannot go missing or out of order.
5. Check each piece against its limit (``section_problem``) and ask the
   model once to fix exactly that problem; then check the whole report
   (``check_report``).
6. Save it to ``target/<paper>_summary.md`` (never overwriting a file),
   where it is never indexed.

Key Features
------------

- **Template-driven**: sections and their guidance come from JSON.
- **Checked, then fixed once per piece**: no open-ended loop.
- **Grounded**: the chat answer cites passages; the saved file states its
  source.

Module Structure
----------------

- ``SummarySettings``
- ``parse_settings``
- ``run``
- ``choose_paper``
- ``choose_template``
- ``headline_prompt``
- ``section_prompt``
- ``fix_prompt``
- ``summary_file_text``

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Final

from qachat.chunking import Chunk
from qachat.report_templates import (
    SECTION_SOURCE_BEGINNING,
    SECTION_SOURCE_CITATION,
    count_words,
    SECTION_SOURCE_REFERENCES,
    ReportTemplate,
    TemplateSection,
    check_report,
    load_template,
    section_problem,
)
from qachat.retrievals.document_reading import (
    dominant_path,
    estimate_tokens,
    named_paths,
)
from qachat.tools.batch_review import plan_parts, take_notes
from qachat.tools.contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from qachat.tools.load_target import (
    TARGET_FOLDER,
    find_target,
    free_target_path,
    load_target,
    save_target_file,
    target_files,
)
from qachat.tools.planning import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_CHOICES,
    CONFIRMATION_WHEN_LARGE,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)

__all__ = [
    "SummarySettings",
    "choose_paper",
    "choose_template",
    "fix_prompt",
    "headline_prompt",
    "paper_beginning",
    "paper_citation",
    "parse_settings",
    "reference_list",
    "run",
    "section_prompt",
    "summary_file_text",
    "ungrounded",
]

# One or more passage citations, with separators: " [1], [3]", "[2][24]".
_CITATION = re.compile(r"(?:\s*,?\s*\[\d+(?:\s*[,–-]\s*\d+)*\])+")
# Words that point at the user's own paper in target/.
_OWN_PAPER = re.compile(
    r"\b(target|my (paper|manuscript|pdf|draft|article)|this (pdf|paper))\b",
    flags=re.I,
)
_SUFFIXES: Final[tuple[str, ...]] = (".pdf", ".md", ".txt")
_PROMPT_TOKENS_PER_SECOND: Final[float] = 700.0
_REPLY_TOKENS_PER_SECOND: Final[float] = 25.0


@dataclass(frozen=True, slots=True)
class SummarySettings:
    """Settings of ``generate_paper_summary`` in a project type.

    Attributes
    ----------
    template: str
        Default report template in ``qachat/report_templates``.
    templates: tuple[str, ...]
        Templates the question may ask for by name or title ("research
        briefing", "research summary"); the default is always allowed.
    part_tokens: int
        Estimated tokens of paper per reading part (1000-16000).
    note_tokens: int
        Longest notes for one part, in tokens (100-1500).
    section_tokens: int
        Longest section without a word limit, in tokens (100-2000).
    repair_attempts: int
        Targeted fixes allowed per piece that misses its limit (0-4).
    max_model_calls: int
        Most reading parts (1-200).
    confirmation: ConfirmationPolicy
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more parts than this ask first.

    """

    template: str = "research_summary"
    templates: tuple[str, ...] = ()
    part_tokens: int = 6000
    note_tokens: int = 500
    section_tokens: int = 600
    repair_attempts: int = 1
    max_model_calls: int = 20
    confirmation: ConfirmationPolicy = CONFIRMATION_WHEN_LARGE
    large_batches: int = 3


def parse_settings(data: dict, where: str) -> SummarySettings:
    """Validate ``generate_paper_summary`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    SummarySettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range, or the template does not
        exist or is malformed.

    """

    names = (
        "template",
        "templates",
        "part_tokens",
        "note_tokens",
        "section_tokens",
        "repair_attempts",
        "max_model_calls",
        "confirmation",
        "large_batches",
    )
    reject_unknown(where, data, names)
    defaults = SummarySettings()
    template = data.get("template", defaults.template)
    if not isinstance(template, str):
        raise ValueError(f"{where!r} template must be a template name.")
    load_template(template)  # Fails early, naming the known templates.
    templates = data.get("templates", [])
    if not isinstance(templates, list) or not all(
        isinstance(name, str) for name in templates
    ):
        raise ValueError(f"{where!r} templates must be a list of names.")
    for name in templates:
        load_template(name)
    return SummarySettings(
        template=template,
        templates=tuple(dict.fromkeys([template, *templates])),
        part_tokens=check_int(
            where, data, "part_tokens", defaults.part_tokens, 1000, 16000
        ),
        note_tokens=check_int(
            where, data, "note_tokens", defaults.note_tokens, 100, 1500
        ),
        section_tokens=check_int(
            where, data, "section_tokens", defaults.section_tokens, 100, 2000
        ),
        repair_attempts=check_int(
            where, data, "repair_attempts", defaults.repair_attempts, 0, 4
        ),
        max_model_calls=check_int(
            where, data, "max_model_calls", defaults.max_model_calls, 1, 200
        ),
        confirmation=check_choice(
            where,
            data,
            "confirmation",
            defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 100
        ),
    )


def choose_paper(
    request: ToolRequest, context: ToolContext
) -> tuple[Path, tuple[Chunk, ...], str, bool] | None:
    """Choose the paper to summarize and return its passages.

    Parameters
    ----------
    request: ToolRequest
        The question; ``request.text`` adds the earlier questions it points
        back to ("that paper").
    context: ToolContext
        QAChat's services.

    Returns
    -------
    tuple[pathlib.Path, tuple[Chunk, ...], str, bool] | None
        The paper's path (``target/...`` or the indexed source path), its
        passages in order, why it was chosen, and whether it is a guess the
        user should confirm; None when no paper fits.

    Notes
    -----
    In order: a file named in the question, then in the earlier questions
    it points back to, by path, name, stem, or first author ("Smith's
    paper" for ``smith2020.pdf``), ``target/`` files first; the only
    ``target/`` file when the question says "target" or "my paper"; and
    otherwise the indexed PDF that best matches the question. Several
    files with the same name are resolved by search and confirmed.

    """

    root = context.project_root
    targets = [
        path
        for path in target_files(root)
        if path.suffix.lower() in _SUFFIXES
    ]
    sources = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    ranked = context.rank(request.text) if context.chunks else []
    for words in dict.fromkeys([request.question, request.text]):
        named_target = named_paths(words, targets, authors=True)
        if len(named_target) == 1:
            document = load_target(root, named_target[0])
            return document.path, document.passages, "named target file", (
                False
            )
        named = named_paths(words, sources, authors=True)
        if named:
            path = dominant_path(ranked, named) or named[0]
            passages = tuple(c for c in context.chunks if c.path == path)
            if len(named) == 1:
                return path, passages, "named source file", False
            return path, passages, (
                f"best match among {len(named)} files with that name"
            ), True
    if _OWN_PAPER.search(request.question):
        only = find_target(request.question, root, suffixes=_SUFFIXES)
        if only is not None:
            document = load_target(root, only)
            return document.path, document.passages, (
                "the only target file"
            ), False
    papers = [path for path in sources if path.suffix.lower() == ".pdf"]
    top = ranked[: context.retrieval_count]
    path = dominant_path(top, papers or None) or dominant_path(top)
    if path is None:
        return None
    passages = tuple(c for c in context.chunks if c.path == path)
    return path, passages, "best match for the question", True


def choose_template(question: str, settings: SummarySettings) -> str:
    """Pick the template a question asks for.

    Parameters
    ----------
    question: str
        The question.
    settings: SummarySettings
        The allowed templates and the default.

    Returns
    -------
    str
        The first allowed template whose title ("Research Briefing") or name
        with spaces ("research briefing") occurs in the question; otherwise
        the default.

    """

    text = " ".join(question.casefold().split())
    for name in settings.templates or (settings.template,):
        template = load_template(name)
        if template.title.casefold() in text or (
            name.replace("_", " ") in text
        ):
            return name
    return settings.template


_REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited)\s*:?\s*$",
    flags=re.I | re.M,
)
_ITEM_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)")
_BEGINNING_CHARACTERS: Final[int] = 1500
_REFERENCE_CHARACTERS: Final[int] = 6000
# Share of an item's distinctive words that must appear in its source.
_GROUNDED_SHARE: Final[float] = 0.6


def paper_citation(
    file: Path | None, passages: tuple[Chunk, ...]
) -> str | None:
    """Write a paper's citation from its metadata and first page.

    Parameters
    ----------
    file: pathlib.Path | None
        The paper on disk, for its PDF metadata; None when unknown.
    passages: tuple[Chunk, ...]
        The paper's passages, for the DOI, year, and author count.

    Returns
    -------
    str | None
        For example ``Patil, A. C. et al. Deep-Learning Investigation of
        ... https://doi.org/10.1038/... (2025).``: the first author and the
        title come from the PDF metadata; the DOI and year are added only
        when printed on the first page. None when there is no metadata
        title, so the model writes it from the first page instead.

    """

    if file is None or file.suffix.lower() != ".pdf" or not file.is_file():
        return None
    try:
        from pypdf import PdfReader

        metadata = PdfReader(file).metadata or {}
    except Exception:  # Unreadable metadata: let the model write it.
        return None
    title = " ".join(str(metadata.get("/Title") or "").split()).rstrip(".")
    if len(title.split()) < 3:
        return None
    first = paper_beginning(passages)
    authors = [
        name.strip()
        for name in re.split(r";|,| and ", str(metadata.get("/Author") or ""))
        if name.strip() and not name.strip().lower().startswith("for ")
    ]
    parts = []
    if authors:
        surname, initials = _author(authors[0])
        many = len(authors) > 1 or bool(
            re.search(
                rf"{re.escape(surname)}[\s\S]{{0,60}}?,\s*[A-Z]", first
            )
        )
        parts.append(f"{surname}, {initials}" + (" et al." if many else ""))
    parts.append(f"{title}.")
    doi = re.search(r"https?://doi\.org/\S+[0-9A-Za-z]", first)
    year = re.search(
        r"(?:Published|Accepted)[^\n]{0,30}?\b((?:19|20)\d\d)\b"
        r"|\(((?:19|20)\d\d)\)",
        first,
    )
    tail = []
    if doi:
        tail.append(doi.group(0))
    if year:
        tail.append(f"({year.group(1) or year.group(2)})")
    if tail:
        parts.append(" ".join(tail) + ".")
    return " ".join(parts)


def _author(name: str) -> tuple[str, str]:
    """Split "Anoop C. Patil" into ("Patil", "A. C.")."""

    words = name.split()
    if len(words) == 1:
        return words[0], ""
    initials = " ".join(
        "-".join(f"{part[0].upper()}." for part in word.split("-") if part)
        for word in words[:-1]
    )
    return words[-1], initials


def paper_beginning(passages: tuple[Chunk, ...]) -> str:
    """Return the start of a paper, where title and authors are printed.

    Parameters
    ----------
    passages: tuple[Chunk, ...]
        The paper's passages, in order.

    Returns
    -------
    str
        About the first 1,500 characters.

    """

    return "\n".join(chunk.text for chunk in passages[:3])[
        :_BEGINNING_CHARACTERS
    ]


def reference_list(passages: tuple[Chunk, ...]) -> str:
    """Return the paper's reference list, found by its heading.

    Parameters
    ----------
    passages: tuple[Chunk, ...]
        The paper's passages, in order.

    Returns
    -------
    str
        Text from the last "References" (or "Bibliography") heading,
        about 6,000 characters at most; empty when there is none.

    """

    text = "\n".join(chunk.text for chunk in passages)
    headings = list(_REFERENCES_HEADING.finditer(text))
    if not headings:
        return ""
    start = headings[-1].end()
    return text[start : start + _REFERENCE_CHARACTERS].strip()


def ungrounded(content: str, source: str, *, items: bool) -> list[int]:
    """Find content the given source text does not support.

    Parameters
    ----------
    content: str
        Section content.
    source: str
        The text it must come from (the first page or reference list).
    items: bool
        Check each list item separately (for reference lists); otherwise
        check the whole content.

    Returns
    -------
    list[int]
        One-based numbers of unsupported items, or ``[1]`` for unsupported
        whole content. An item is supported when at least 60% of its
        distinctive words (four letters or more) occur in the source.

    """

    vocabulary = set(_distinctive(source))
    pieces = (
        [match.group(1) for line in content.splitlines()
         if (match := _ITEM_LINE.match(line))]
        if items
        else [content]
    )
    missing: list[int] = []
    for number, piece in enumerate(pieces, start=1):
        words = _distinctive(_CITATION.sub("", piece))
        if words and sum(w in vocabulary for w in words) < (
            _GROUNDED_SHARE * len(words)
        ):
            missing.append(number)
    return missing


def _distinctive(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z]{4,}", text.casefold())
        if word not in ("this", "that", "with", "from", "paper", "their")
    ]


_NO_PLACEHOLDERS: Final[str] = (
    "Never write placeholders such as 'Author, A.', 'Title', 'Journal', "
    "'volume', 'pages', or '(year)'; leave out any detail the notes do not "
    "give."
)


def _notes_block(notes: list[tuple[int, list[str]]]) -> str:
    return "\n\n".join(
        f"Part {part}:\n" + ("\n".join(lines) if lines else "NOTHING")
        for part, lines in notes
    )


def headline_prompt(
    template: ReportTemplate,
    name: str,
    notes: list[tuple[int, list[str]]],
) -> tuple[str, str]:
    """Return the prompts that write a template's headline.

    Parameters
    ----------
    template: ReportTemplate
        A template with ``headline_characters``.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is one line.

    """

    system = (
        "You write headlines for QAChat. You reply with the headline only, "
        "on one line, without quotes or a '#'."
    )
    user = f"""Write the headline of a {template.title} of {name}.

{template.headline_guidance} At most {template.headline_characters}
characters.

Reading notes:

{_notes_block(notes)}"""
    return system, user


def section_prompt(
    template: ReportTemplate,
    section: TemplateSection,
    name: str,
    notes: list[tuple[int, list[str]]],
    written: list[tuple[str, str]],
    source: str = "",
) -> tuple[str, str]:
    """Return the prompts that write one section from reading notes.

    Parameters
    ----------
    template: ReportTemplate
        The report template.
    section: TemplateSection
        The section to write.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.
    written: list[tuple[str, str]]
        Sections written so far, so they are not repeated.
    source: str, optional
        The section's source text (the first page or the reference list),
        when its template source is not the notes. Defaults to ``""``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is the section's content.

    """

    system = (
        "You write one section of a structured summary of a scientific "
        "paper for QAChat. You use only the reading notes you are given "
        "and never invent results."
    )
    earlier = "\n".join(
        f"- {heading}: " + " ".join(content.split()[:30]) + " ..."
        for heading, content in written
    )
    user = f"""Write the "{section.name}" section of a {template.title} of
{name}.

What it contains: {section.guidance}
Length: {_length(section)}

Rules:
- Use only information in the notes. If they do not cover this section,
  write "Not stated in the paper."
- {_NO_PLACEHOLDERS}
- Keep names, numbers, and versions exactly as in the notes.
- Mark your own inference with "Inferred:".
- Cite passage numbers from the notes, like [12].
- Reply with the section's content only, without its heading.

Sections already written (do not repeat them):
{earlier or "none"}

Reading notes:

{_notes_block(notes)}"""
    if section.source == SECTION_SOURCE_BEGINNING:
        user += f"""

The beginning of the paper (take the citation from here):

{source or "Not available."}"""
    elif section.source == SECTION_SOURCE_REFERENCES:
        user += f"""

The paper's reference list (choose only from here and copy entries as
they appear; if it is not available, write "Not stated in the paper."):

{source or "Not available."}"""
    return system, user


def fix_prompt(
    section: TemplateSection,
    content: str,
    problem: str,
    source: str = "",
    notes: list[tuple[int, list[str]]] | None = None,
) -> tuple[str, str]:
    """Return the prompts that fix one section's length or form.

    Parameters
    ----------
    section: TemplateSection
        The section.
    content: str
        Its current content.
    problem: str
        What is wrong, from ``section_problem`` or the source check.
    source: str, optional
        The section's source text, when it has one. Defaults to ``""``.
    notes: list[tuple[int, list[str]]] | None, optional
        Reading notes, so a section that is too short can be extended with
        real content rather than padding. Defaults to ``None``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is the fixed content.

    """

    system = (
        "You revise one section of a summary for QAChat. You keep its facts "
        "and citations and change only what the problem requires."
    )
    user = f"""Problem: {problem}

Rewrite the "{section.name}" section so it has {_length(section)}. Keep
its facts and passage citations. To make it longer, add relevant details
from the reading notes below, never filler; to make it shorter, keep the
most important points. Add nothing the notes do not support.
{_NO_PLACEHOLDERS} Reply with the section's content only, without its
heading.

Current content:

{content}"""
    if notes:
        user += f"""

Reading notes:

{_notes_block(notes)}"""
    if source:
        user += f"""

Use only this source text for the section:

{source}"""
    return system, user


def _length(section: TemplateSection) -> str:
    """Describe a section's length, such as "about 135 words (120-150)".

    The midpoint is the target, so the sections together fit the page.

    """

    if section.inline:
        return "one line only"
    if section.words:
        low, high = section.words
        return f"about {(low + high) // 2} words ({low}-{high})"
    if section.items:
        return (
            f"{section.items[0]}-{section.items[1]} items, one per line "
            "starting with '- '"
        )
    return "a few concise sentences"


def _content(reply: str, section: TemplateSection) -> str:
    """The section text from a reply, without a repeated heading.

    In a reference-list section, the paper's own reference numbers ("[2]
    Smith ...") are removed, because in QAChat ``[n]`` cites passage n.

    """

    lines = reply.strip().splitlines()
    while lines and (
        lines[0].lstrip("#* ").strip().rstrip(":").casefold()
        == section.name.casefold()
        or not lines[0].strip()
    ):
        lines.pop(0)
    if section.inline:
        # One line, without a repeated label ("This is a summary of ...").
        line = next((x.strip() for x in lines if x.strip()), "")
        line = re.sub(
            rf"^\**\s*{re.escape(section.name)}\s*:?\**\s*:?\s*",
            "",
            line,
            flags=re.I,
        )
        return line.strip()
    if section.source == SECTION_SOURCE_REFERENCES:
        lines = [
            re.sub(r"^(\s*(?:[-*•]|\d+[.)])\s+)\[\d+\]\s*", r"\1", line)
            for line in lines
        ]
    return "\n".join(lines).strip()


def summary_file_text(summary: str, source: Path) -> str:
    """Return the summary as saved: citations removed, source stated.

    Parameters
    ----------
    summary: str
        The checked summary, with ``[n]`` citations.
    source: pathlib.Path
        The paper.

    Returns
    -------
    str
        Markdown for the file. Passage numbers only make sense inside
        QAChat, so they are removed; a line under the title names the
        source.

    """

    text = _CITATION.sub("", summary).strip()
    note = (
        f"*Source: {source.name}. Written by QAChat from the paper's text; "
        "check important details against the paper.*"
    )
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join([lines[0], "", note, *lines[1:]]) + "\n"
    return f"{note}\n\n{text}\n"


def run(
    settings: SummarySettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Summarize one paper with a template and save it to ``target/``.

    Parameters
    ----------
    settings: SummarySettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        QAChat's services.

    Returns
    -------
    ToolResult
        One line saying where the Markdown and PDF were saved, then the
        saved Markdown itself; every passage of the paper as evidence; and
        a summary of the checks, fixes, and files.

    """

    trace = context.trace
    template = load_template(choose_template(request.question, settings))
    context.progress("Choosing the paper to summarize")
    summary: dict[str, object] = {
        "name": "generate_paper_summary",
        "template": template.name,
    }
    try:
        chosen = choose_paper(request, context)
    except (FileNotFoundError, ValueError) as error:
        context.done("Could not load the paper", evidence_count=0)
        return ToolResult(str(error), (), summary)
    if chosen is None:
        files = ", ".join(str(p) for p in target_files(context.project_root))
        context.done("No paper found", evidence_count=0)
        return ToolResult(
            "I couldn't tell which paper to summarize. Name the file, for "
            "example: turn smith2020.pdf into a research briefing. Papers "
            f"in {TARGET_FOLDER}/: {files or 'none'}.",
            (),
            summary,
        )
    path, passages, reason, guessed = chosen
    texts = [chunk.text for chunk in passages]
    trace.emit(
        "tool.generate_paper_summary.loaded",
        component="tool",
        data={
            "paper": path.as_posix(),
            "reason": reason,
            "passages": len(passages),
            "template": template.name,
        },
        turn_id=context.trace_turn_id,
    )
    parts = plan_parts(passages, settings.part_tokens)
    to_read = parts[: settings.max_model_calls]
    tokens = sum(estimate_tokens(text) for text in texts)
    plan = ToolPlan(
        tool="generate_paper_summary",
        summary=(
            f"Summarize {path.name} as a {template.title}: read "
            f"{len(passages)} passages in {len(parts)} parts, then write "
            f"{len(template.sections)} sections."
        ),
        scope=f"{path.as_posix()} ({reason})",
        items=len(passages),
        batches=len(parts),
        batches_to_run=len(to_read),
        estimated_seconds=round(
            _estimate(to_read, texts, settings, template), 1
        ),
        quick_items=len(parts[0]),
        quick_seconds=round(
            _estimate(parts[:1], texts, settings, template), 1
        ),
        details=(
            *(
                (
                    "QAChat chose this paper from your question; name the "
                    "file if it is not the right one.",
                )
                if guessed
                else ()
            ),
            f"About {tokens:,} tokens",
            f"Saved to {TARGET_FOLDER}/{path.stem}_summary.md",
            f"Model: {context.model_id}",
        ),
    )
    # A guessed paper is always shown before any work, whatever its size.
    choice, asked = confirm_plan(
        plan,
        CONFIRMATION_ALWAYS if guessed else settings.confirmation,
        settings.large_batches,
        context,
    )
    summary.update(
        {
            "paper": path.as_posix(),
            "reason": reason,
            "passages": len(passages),
            "plan": plan.to_dict(),
            "choice": choice,
            "asked": asked,
        }
    )
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled before reading", evidence_count=0)
        return ToolResult(
            f"Cancelled before summarizing {path.name}; no model calls "
            "were made.",
            (),
            summary,
        )

    started = perf_counter()
    runs = parts[:1] if choice == PLAN_CHOICE_QUICK else to_read
    task = (
        f"Notes for a {template.title} of this paper. Copy the paper's "
        "title, authors, journal, and year when they appear, every figure "
        "caption with its figure number, and the references the paper "
        "relies on most. Note what each section needs:\n"
        + "\n".join(
            f"- {section.name}: {section.guidance}"
            for section in template.sections
        )
    )
    notes = take_notes(
        task, path.name, texts, runs, settings.note_tokens, context
    )
    # Python owns the structure: the model writes the headline and each
    # section separately, and each piece is checked against its limit.
    headline = template.title
    repairs = 0
    fixes: list[str] = []
    if template.headline_characters:
        context.progress("Writing the headline")
        system, prompt = headline_prompt(template, path.name, notes)
        headline = _headline(context.ask_model(system, prompt, 60))
        if len(headline) > template.headline_characters:
            fixes.append(f"headline: {len(headline)} characters")
            repairs += 1
            headline = _headline(
                context.ask_model(
                    system,
                    prompt
                    + f"\n\nYour headline \"{headline}\" is too long. "
                    f"Write one of at most {template.headline_characters} "
                    "characters.",
                    60,
                )
            )
    sources = {
        SECTION_SOURCE_BEGINNING: paper_beginning(passages),
        SECTION_SOURCE_REFERENCES: reference_list(passages),
    }
    written: list[tuple[str, str]] = []
    unsupported: list[str] = []
    for section in template.sections:
        if section.source == SECTION_SOURCE_CITATION:
            # Copied, not written: the model never touches the citation.
            citation = paper_citation(_paper_file(path, context), passages)
            if citation:
                written.append((section.name, citation))
                continue
        context.progress(f"Writing {section.name}")
        source = sources.get(
            SECTION_SOURCE_BEGINNING
            if section.source == SECTION_SOURCE_CITATION
            else section.source,
            "",
        )
        system, prompt = section_prompt(
            template, section, path.name, notes, written, source
        )
        tokens = _section_tokens(section, settings)
        content = _content(context.ask_model(system, prompt, tokens), section)
        problem = _problem(section, content, source)
        for _ in range(settings.repair_attempts):
            if problem is None:
                break
            context.progress(f"Fixing: {problem}")
            fixes.append(problem)
            repairs += 1
            short = bool(
                section.words and count_words(content) < section.words[0]
            )
            system, prompt = fix_prompt(
                section, content, problem, source, notes if short else None
            )
            content = _content(
                context.ask_model(system, prompt, tokens), section
            )
            problem = _problem(section, content, source)
        if problem and section.words and (
            count_words(content) > section.words[1]
        ):
            # Too long after its fixes: cut whole trailing sentences.
            content = _trim_words(content, section.words[1])
            problem = _problem(section, content, source)
        if problem and section.source == SECTION_SOURCE_REFERENCES:
            # References the paper does not list are dropped, not shown.
            content = _grounded_items(content, source)
            problem = _problem(section, content, source)
        if problem and problem not in (section_problem(section, content),):
            unsupported.append(problem)
        written.append((section.name, content))
    if template.max_total_words:
        written = _fit_total(written, template)
    inline = {s.name for s in template.sections if s.inline}
    draft = "\n\n".join(
        [f"# {headline}"]
        + [
            f"**{name}:** " + " ".join(content.split())
            if name in inline
            else f"## {name}\n{content}"
            for name, content in written
        ]
    )
    context.progress("Checking the template")
    check = check_report(draft, template)
    checks = [fixes, check.problems(), unsupported]
    trace.emit(
        "tool.generate_paper_summary.checked",
        component="tool",
        data={"checks": checks, "repairs": repairs, "valid": check.valid},
        turn_id=context.trace_turn_id,
    )
    file_text = summary_file_text(draft, path)
    saved = save_target_file(
        context.project_root, f"{path.stem}_summary.md", file_text
    )
    pdf, pdf_note = _write_pdf(file_text, saved, template, context)
    trace.emit(
        "tool.generate_paper_summary.saved",
        component="tool",
        data={"file": saved.as_posix()},
        turn_id=context.trace_turn_id,
    )
    read = sum(len(part) for part in runs)
    # The answer says where the files are and then shows the saved file
    # itself, so the chat and the file never differ. Checks and fixes are
    # recorded in the turn and the trace.
    files = f"`{saved.as_posix()}`" + (
        f" and `{pdf.as_posix()}`" if pdf else ""
    )
    lead = [
        f"Done: the {template.title.lower()} of {path.name} is saved as "
        f"{files}."
    ]
    if pdf_note:
        lead.append(pdf_note)
    if read < len(passages):
        lead.append(
            f"It is based on {read} of {len(passages)} passages"
            + (
                " (quick version: the first part)."
                if choice == PLAN_CHOICE_QUICK
                else " (the model-call limit was reached)."
            )
        )
    summary.update(
        {
            "passages_read": read,
            "notes": sum(len(lines) for _, lines in notes),
            "valid": check.valid and not unsupported,
            "problems": check.problems() + unsupported,
            "repairs": repairs,
            "saved": saved.as_posix(),
            "saved_pdf": pdf.as_posix() if pdf else None,
            "seconds": round(perf_counter() - started, 1),
            "model": context.model_id,
        }
    )
    trace.emit(
        "tool.generate_paper_summary.result",
        component="tool",
        data={
            key: value
            for key, value in summary.items()
            if key not in ("name", "plan")
        },
        turn_id=context.trace_turn_id,
    )
    context.done(f"Saved {saved.as_posix()}", evidence_count=len(passages))
    evidence = tuple(
        {
            "citation": index + 1,
            "path": chunk.path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "text": chunk.text,
            "cell": chunk.cell,
        }
        for index, chunk in enumerate(passages)
    )
    answer = " ".join(lead) + "\n\n---\n\n" + file_text.strip()
    return ToolResult(answer, evidence, summary)


def _problem(
    section: TemplateSection, content: str, source: str
) -> str | None:
    """A section's length problem, or content its source does not support."""

    problem = section_problem(section, content)
    if problem or not source or "not stated" in content.casefold():
        return problem
    if section.source == SECTION_SOURCE_REFERENCES:
        missing = ungrounded(content, source, items=True)
        if missing:
            numbers = ", ".join(str(number) for number in missing)
            return (
                f"{section.name} item {numbers} is not in the paper's "
                "reference list."
            )
    elif section.source in (SECTION_SOURCE_BEGINNING, SECTION_SOURCE_CITATION):
        if ungrounded(content, source, items=False):
            return (
                f"{section.name} does not match the beginning of the paper."
            )
        invented = _unknown_names(content, source)
        if invented:
            return (
                f"{section.name} names {', '.join(invented)}, which the "
                "beginning of the paper does not; copy names exactly."
            )
    return None


# Capitalized words that are not names.
_NOT_NAMES: Final[frozenset[str]] = frozenset(
    "this the that these and for with from into in on at by of an et al "
    "is are was a its their our nature journal".split()
)


def _unknown_names(content: str, source: str) -> list[str]:
    """Capitalized words (names, title words) the source does not contain."""

    # Join words hyphenated across lines ("Spec-\ntra") before comparing.
    joined = re.sub(r"-\s*\n\s*", "", source)
    known = set(re.findall(r"[a-z]+", joined.casefold()))
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", _CITATION.sub("", content))
    return sorted(
        {
            name
            for name in names
            if name.casefold() not in known
            and name.casefold() not in _NOT_NAMES
        }
    )


def _trim_words(content: str, limit: int) -> str:
    """Drop whole trailing sentences until the content has ``limit`` words.

    Lists keep whole items; the first sentence is always kept.

    """

    if count_words(content) <= limit:
        return content
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", content.strip())
    while len(pieces) > 1 and count_words(" ".join(pieces)) > limit:
        pieces.pop()
    return " ".join(pieces)


def _write_pdf(
    text: str, saved: Path, template: ReportTemplate, context: ToolContext
) -> tuple[Path | None, str | None]:
    """Write the PDF next to the Markdown when a PDF writer is available.

    Returns the PDF's project path (or None) and a note for the answer
    when the PDF is longer than one page or could not be written.

    """

    if context.write_pdf is None:
        return None, None
    target = free_target_path(context.project_root, f"{saved.stem}.pdf")
    try:
        pages = context.write_pdf(text, target, label=template.title)
    except Exception as error:  # The Markdown is saved either way.
        return None, f"The PDF could not be written: {error}"
    pdf = Path(TARGET_FOLDER) / target.name
    context.trace.emit(
        "tool.generate_paper_summary.pdf",
        component="tool",
        data={"file": pdf.as_posix(), "pages": pages},
        turn_id=context.trace_turn_id,
    )
    if pages > 1:
        return pdf, f"The PDF needed {pages} pages, even at a small font."
    return pdf, None


def _paper_file(path: Path, context: ToolContext) -> Path | None:
    """The paper on disk: ``target/...`` or below the project's sources."""

    root = context.project_root
    for candidate in (root / path, root / "sources" / path):
        if candidate.is_file():
            return candidate
    return None


def _fit_total(
    written: list[tuple[str, str]], template: ReportTemplate
) -> list[tuple[str, str]]:
    """Trim the longest prose sections toward their midpoints to fit.

    Sections are never cut below the middle of their range; reference
    lists and one-line sections are left alone.

    """

    limits = {s.name: s.words for s in template.sections if s.words}
    sections = dict(written)
    total = sum(count_words(text) for text in sections.values())
    while total > template.max_total_words:
        over = [
            (count_words(text) - sum(limits[name]) // 2, name)
            for name, text in sections.items()
            if name in limits
            and count_words(text) > sum(limits[name]) // 2
        ]
        if not over:
            break
        _, name = max(over)
        trimmed = _trim_words(
            sections[name], count_words(sections[name]) - 1
        )
        if trimmed == sections[name]:
            break
        sections[name] = trimmed
        total = sum(count_words(text) for text in sections.values())
    return [(name, sections[name]) for name, _ in written]


def _grounded_items(content: str, references: str) -> str:
    """Keep only list items found in the reference list."""

    missing = set(ungrounded(content, references, items=True))
    kept: list[str] = []
    number = 0
    for line in content.splitlines():
        if _ITEM_LINE.match(line):
            number += 1
            if number in missing:
                continue
        kept.append(line)
    return "\n".join(kept).strip()


def _headline(reply: str) -> str:
    """The headline from a reply: first line, without '#' or quotes."""

    line = next((x for x in reply.strip().splitlines() if x.strip()), "")
    return line.strip().lstrip("#").strip().strip('"“”*').strip()


def _section_tokens(
    section: TemplateSection, settings: SummarySettings
) -> int:
    """Reply room for one section: twice its word limit, or the setting."""

    if section.words:
        return 2 * section.words[1] + 80
    if section.items:
        return 90 * section.items[1] + 40
    return settings.section_tokens


def _estimate(
    parts: list[list[int]],
    texts: list[str],
    settings: SummarySettings,
    template: ReportTemplate,
) -> float:
    """Approximate seconds: reading parts, then one call per section."""

    total = 0.0
    for part in parts:
        prompt = 300 + sum(estimate_tokens(texts[index]) for index in part)
        total += 1.0 + prompt / _PROMPT_TOKENS_PER_SECOND
        total += settings.note_tokens / 2 / _REPLY_TOKENS_PER_SECOND
    notes = len(parts) * settings.note_tokens * 0.6
    calls = len(template.sections) + (1 if template.headline_characters else 0)
    for section in template.sections:
        reply = _section_tokens(section, settings) * 0.6
        total += 1.0 + reply / _REPLY_TOKENS_PER_SECOND
    total += calls * (notes + 400) / _PROMPT_TOKENS_PER_SECOND
    return total
