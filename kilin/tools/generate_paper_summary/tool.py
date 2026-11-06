"""Tool ``generate_paper_summary``: turn one paper into a templated report.

The tool turns one paper into a report that follows a Markdown template
from ``report_template``, such as ``research_briefing`` (the one-page
Research Briefing format of Nature-family journals), and saves it in the
project's ``outputs/`` folder. Templates live in the project's
``templates/`` folder, which receives only the project type's required
templates on creation; the question picks one by title or name, and the
intent's default applies otherwise. It is KíLín's reference example of a
custom tool (see tutorial 160): it composes existing capabilities instead
of re-implementing them.

1. Choose the paper: a file in ``outputs/`` the question names by its
   file name, the only ``outputs/`` file when the question says "my paper"
   and names no indexed source, or else the indexed source
   ``choose_document`` picks (asking the user when unclear). Reports KíLín
   saved in ``outputs/`` are never taken for a paper the question names
   only by its source's name.
2. Show the plan; a guessed paper is always confirmed before any work.
3. Read it in parts: the shared note-taking service writes notes with passage
   numbers, because a whole paper is too long for one small-model prompt.
4. Write the headline and each section in separate calls; Python
   assembles the headings, so sections cannot go missing or out of order.
   The citation and the reference items are copied by code: the model
   only chooses which reference entries fit, by number.
5. Check each piece against its limit (``section_problem``) and ask the
   model to fix exactly that problem; then check the whole report
   (``check_report``).
6. Save it to ``outputs/<paper>_summary.md`` (never overwriting a file),
   where it is never indexed, and to a PDF next to it when the caller
   provides a PDF writer.

A template or paper that cannot be read gives an answer saying so, without
model calls. Problems that remain after the allowed fixes are recorded in
the result summary and the trace; an invalid report is not published.
The checks verify structure and length, not truth: the model can still misstate
what the paper says. The citation uses only the PDF metadata and the first
page; without a metadata title it names the file. Reference entries are
found with layout rules, so an unusual layout can split or join entries
wrongly, but every entry is the paper's own text.

Key Features
------------

- **Template-driven**

    Sections, limits, and guidance come from Markdown templates.

- **Nothing bibliographic is invented**

    The citation comes from the file's metadata, and references are copied
    from the paper's own list.

- **Checked, then fixed per piece**

    Each piece gets a fixed number of targeted fixes; there is no
    open-ended loop.

- **Planned before running**

    The user sees the scope and the estimated time, and can continue, run
    a quick version on the first part, or cancel.

Module Structure
----------------

- ``SummarySettings``

    Settings of ``generate_paper_summary`` in a project type.

- ``parse_settings``

    Validate the settings from a project type.

- ``choose_paper``

    Choose the paper to summarize and return its passages.

- ``choose_template``

    Pick the template a question asks for.

- ``paper_citation``

    Write a paper's citation from its metadata and first page.

- ``paper_beginning``

    Return the start of a paper, where title and authors are printed.

- ``REFERENCE_LAYOUT``

    This tool's thresholds for the shared reference-entry extractor
    (``kilin.tools._shared.references``).

- ``headline_prompt``

    Return the prompts that write a template's headline.

- ``section_prompt``

    Return the prompts that write one section from reading notes.

- ``references_prompt``

    Return the prompts that choose reference entries, by number only.

- ``fix_prompt``

    Return the prompts that fix one section's length or form.

- ``summary_file_text``

    Return the summary as saved: citations removed, source stated.

- ``run``

    Summarize one paper with a template and save it to ``outputs/``.

Private helpers write, fix, trim, and assemble the pieces of the report;
they are not part of the public API.

Examples
--------
>>> from kilin.tools.generate_paper_summary import (
...     choose_template,
...     parse_settings,
... )
>>> from kilin.tools.generate_paper_summary import SHIPPED_TEMPLATES_DIR
>>> settings = parse_settings({"repair_attempts": 2}, "paper_summary")
>>> template = choose_template(
...     "make a research briefing of it",
...     SHIPPED_TEMPLATES_DIR,
...     settings.template,
... )
>>> template.title
'Research Briefing'

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter
from typing import Final

from kilin.tools._shared.model_results import (
    incomplete_notice,
    partial_result,
)
from kilin._model_completion import IncompleteModelResponse
from kilin.chunking import Chunk
from kilin.retrievals.document_reading import (
    estimate_tokens,
    named_paths,
)
from kilin.tools._shared.document_notes import (
    NOTHING,
    plan_parts,
    take_notes,
)
from kilin.tools._shared.choose_document import choose_document
from kilin.tools._shared.references import (
    ReferenceLayout,
    reference_entries,
)
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.outputs import (
    OUTPUTS_FOLDER,
    find_output,
    load_output,
    output_files,
    publish_output_file,
    save_output_file,
    staged_output_file,
)
from kilin.tools._shared.planning import (
    CONFIRMATION_ALWAYS,
    CONFIRMATION_CHOICES,
    CONFIRMATION_WHEN_LARGE,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ConfirmationPolicy,
    ToolPlan,
    confirm_plan,
)
from kilin.workspace_resources import (
    PROJECT_SOURCES_DIRECTORY,
    WorkspaceResource,
    templates_folder,
)
from kilin.tools.generate_paper_summary._configuration import (
    BEGINNING_CHARACTERS,
    BEGINNING_PASSAGES,
    CALL_SECONDS,
    COMMON_WORDS,
    DEFAULT_CONFIRMATION,
    DEFAULT_LARGE_BATCHES,
    DEFAULT_MAX_MODEL_CALLS,
    DEFAULT_PART_TOKENS,
    DEFAULT_REFERENCE_ITEMS,
    DEFAULT_REPAIR_ATTEMPTS,
    DEFAULT_TEMPLATE,
    EARLIER_SECTION_WORDS,
    ENTRY_MIN_WORDS,
    HEADLINE_TOKENS,
    NUMBERED_ENTRIES_MIN,
    NOTE_REPLY_TOKENS,
    SECTION_REPLY_TOKENS,
    ORGANISATION_WORDS,
    PART_PROMPT_TOKENS,
    PROMPT_TOKENS_PER_SECOND,
    REFERENCES_REPLY_TOKENS,
    REFERENCES_TOKENS,
    REFERENCE_CHARACTERS,
    REPLY_SHARE,
    REPLY_TOKENS_PER_SECOND,
    RUNNING_LINE_REPEATS,
    RUNNING_LINE_WORDS,
    SECTION_PROMPT_TOKENS,
)
from kilin.tools.generate_paper_summary._report_template import (
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_REFERENCES,
    SHIPPED_TEMPLATES_DIR,
    ReportTemplate,
    TemplateSection,
    check_report,
    count_words,
    load_template,
    section_problem,
    template_names,
)

__all__ = [
    "REFERENCE_LAYOUT",
    "SummarySettings",
    "choose_paper",
    "choose_template",
    "fix_prompt",
    "headline_prompt",
    "paper_beginning",
    "paper_citation",
    "parse_settings",
    "workspace_resources",
    "references_prompt",
    "run",
    "section_prompt",
    "summary_file_text",
]

# One or more passage citations, with separators: " [1], [3]", "[2][24]".
# They are removed from the saved file, where passage numbers mean nothing.
_CITATION = re.compile(r"(?:\s*,?\s*\[\d+(?:\s*[,–-]\s*\d+)*\])+")
# Words that point at the user's own paper in outputs/. "This paper" is
# not among them: it usually means the paper the conversation is about.
_OWN_PAPER = re.compile(
    r"\b(outputs?|my (paper|manuscript|pdf|draft|article))\b",
    flags=re.I,
)
# File types in outputs/ that can be a paper.
_SUFFIXES: Final[tuple[str, ...]] = (".pdf", ".md", ".txt")
# Template setting: a plain file name, so it cannot point outside templates/.
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][\w-]*")

# Numeric and vocabulary heuristics come from this tool's defaults.json.
_BEGINNING_PASSAGES: Final[int] = BEGINNING_PASSAGES
_BEGINNING_CHARACTERS: Final[int] = BEGINNING_CHARACTERS
_ORGANISATION_WORDS: Final[frozenset[str]] = ORGANISATION_WORDS

# Thresholds of the shared reference-entry extractor, from defaults.json.
REFERENCE_LAYOUT: Final[ReferenceLayout] = ReferenceLayout(
    max_characters=REFERENCE_CHARACTERS,
    numbered_entries_min=NUMBERED_ENTRIES_MIN,
    running_line_repeats=RUNNING_LINE_REPEATS,
    running_line_words=RUNNING_LINE_WORDS,
    entry_min_words=ENTRY_MIN_WORDS,
)
# Item range of a references section that has no item limit of its own.
_DEFAULT_REFERENCE_ITEMS: Final[tuple[int, int]] = DEFAULT_REFERENCE_ITEMS
# Vocabulary used to identify distinctive reference terms.
_COMMON_WORDS: Final[tuple[str, ...]] = COMMON_WORDS

# Prompt rule shared by the writing and the fixing prompts.
_NO_PLACEHOLDERS: Final[str] = (
    "Never write placeholders such as 'Author, A.', 'Title', 'Journal', "
    "'volume', 'pages', or '(year)'; leave out any detail the notes do not "
    "give."
)
# Words of each earlier section shown to the model, so it does not repeat
# them.
_EARLIER_SECTION_WORDS: Final[int] = EARLIER_SECTION_WORDS
# Longest replies: a one-line headline, and a few reference numbers.
_HEADLINE_TOKENS: Final[int] = HEADLINE_TOKENS
_REFERENCES_TOKENS: Final[int] = REFERENCES_TOKENS

# Rough speeds of an 8B 4-bit model on Apple silicon, for estimates only.
_PROMPT_TOKENS_PER_SECOND: Final[float] = PROMPT_TOKENS_PER_SECOND
_REPLY_TOKENS_PER_SECOND: Final[float] = REPLY_TOKENS_PER_SECOND
# Fixed time of one model call, in seconds, for estimates only.
_CALL_SECONDS: Final[float] = CALL_SECONDS
# Instruction tokens around one reading part, and around one section.
_PART_PROMPT_TOKENS: Final[int] = PART_PROMPT_TOKENS
_SECTION_PROMPT_TOKENS: Final[int] = SECTION_PROMPT_TOKENS
# Share of its reply room a reply uses on average, for estimates only.
_REPLY_SHARE: Final[float] = REPLY_SHARE
# Estimated reply of a reference choice ("4, 17").
_REFERENCES_REPLY_TOKENS: Final[int] = REFERENCES_REPLY_TOKENS


@dataclass(frozen=True, slots=True)
class SummarySettings:
    """Settings of ``generate_paper_summary`` in a project type.

    Attributes
    ----------
    template: str
        Template used when the question names none: a file name without
        ``.md`` in the project's ``templates/`` folder.
    part_tokens: int
        Estimated tokens of paper per reading part (1000-16000).
    repair_attempts: int
        Targeted fixes allowed per section that misses its limit (0-4).
    max_model_calls: int
        Most reading parts (1-200).
    confirmation: "always" | "when_large" | "never"
        When to ask before running.
    large_batches: int
        With ``when_large``, plans with more parts than this ask first
        (0-100).

    """

    template: str = DEFAULT_TEMPLATE
    part_tokens: int = DEFAULT_PART_TOKENS
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    confirmation: ConfirmationPolicy = DEFAULT_CONFIRMATION
    large_batches: int = DEFAULT_LARGE_BATCHES


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
        Validated settings; missing ones take their defaults.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range, or the template name is
        not a plain file name. Whether the template exists is checked when
        the tool runs, because each project has its own templates.

    """

    names = (
        "template",
        "part_tokens",
        "repair_attempts",
        "max_model_calls",
        "confirmation",
        "large_batches",
    )
    reject_unknown(where, data, names)

    defaults = SummarySettings()
    template = data.get("template", defaults.template)
    if not isinstance(template, str) or not _TEMPLATE_NAME.fullmatch(
        template
    ):
        raise ValueError(
            f"{where!r} template must be a template file name without .md."
        )

    return SummarySettings(
        template=template,
        part_tokens=check_int(
            where, data, "part_tokens", defaults.part_tokens, 1000, 16000
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


def workspace_resources(
    settings: SummarySettings,
) -> tuple[WorkspaceResource, ...]:
    """Declare the report template needed by one configured intent.

    Parameters
    ----------
    settings: SummarySettings
        Validated tool configuration with a default template name.

    Returns
    -------
    tuple[WorkspaceResource, ...]
        One shipped template to install when available, otherwise
        an empty tuple for a user-provided custom template.

    Notes
    -----
    The core deduplicates resources required by multiple intents. Template
    existence and destination safety are checked during project planning.

    """

    name = f"{settings.template}.md"
    source = SHIPPED_TEMPLATES_DIR / name
    if not source.exists():
        # A custom project-type template may be supplied by the user later.
        return ()
    return (WorkspaceResource(source, f"templates/{name}"),)


def choose_paper(
    request: ToolRequest, context: ToolContext, what: str = "summary"
) -> tuple[Path, tuple[Chunk, ...], str, bool] | str:
    """Choose the paper to summarize and return its passages.

    Parameters
    ----------
    request: ToolRequest
        The question, with the earlier turns it points back to.
    context: ToolContext
        KíLín's services.
    what: str, optional
        What will be made, for the question to the user, such as
        "research briefing". Defaults to ``"summary"``.

    Returns
    -------
    tuple[pathlib.Path, tuple[Chunk, ...], str, bool] | str
        The paper's path (``outputs/...`` or the indexed source path), its
        passages in order, why it was chosen, and whether it is a guess the
        user should confirm; or a message to answer with when no paper fits
        or the user cancelled.

    Raises
    ------
    FileNotFoundError
        If a chosen ``outputs/`` file is missing (from ``load_output``).
    ValueError
        If a chosen ``outputs/`` file cannot be read (from ``load_output``).

    Notes
    -----
    A file in ``outputs/`` the question names by its file name or path, or
    the only one when it says "outputs" or "my paper" and names no indexed
    source, comes first. A looser match (by file stem or author, as in
    ``named_paths``) never selects an ``outputs/`` file, so a report such
    as ``smith20_reference_check.md`` is not taken for ``smith20.pdf``.
    Otherwise ``choose_document`` decides from the question and the
    earlier turns it points back to, and asks the user when the paper is
    unclear.

    """

    root = context.project_root
    outputs = [
        path
        for path in output_files(root)
        if path.suffix.lower() in _SUFFIXES
    ]

    def from_outputs(
        name: Path, reason: str
    ) -> tuple[Path, tuple[Chunk, ...], str, bool]:
        """Load an ``outputs/`` file; it is never a guess.

        Parameters
        ----------
        name: Path
            Filename or project name to use.
        reason: str
            Description of how the output file was selected.

        Returns
        -------
        tuple[Path, tuple[Chunk, ...], str, bool]
            Named output file, selected chunks, selection reason, and flag.

        """

        document = load_output(root, name)
        return document.path, document.passages, reason, False

    # Only an exact file name or path selects an outputs/ file: by stem or
    # author, "tiunn26braph2.pdf" would also match the saved report
    # "tiunn26braph2_reference_check.md".
    text = request.question.replace("\\", "/").casefold()
    hits = [
        path for path in outputs
        if path.as_posix().casefold() in text or path.name.casefold() in text
    ]
    if len(hits) == 1:
        return from_outputs(hits[0], "named in the question")

    sources = list(dict.fromkeys(chunk.path for chunk in context.chunks))
    if _OWN_PAPER.search(request.question) and not named_paths(
        request.question, sources, authors=True
    ):
        only = find_output(request.question, root, suffixes=_SUFFIXES)
        if only is not None:
            return from_outputs(only, "the only outputs file")

    chosen = choose_document(
        request,
        context,
        question=f"Which paper should I turn into a {what}?",
        example=f"turn smith2020.pdf into a {what}",
        prefer=(".pdf",),
    )
    if isinstance(chosen, str):
        return chosen

    path, reason, guessed = chosen
    passages = tuple(c for c in context.chunks if c.path == path)
    return path, passages, reason, guessed


def choose_template(
    question: str, folder: Path, default: str
) -> ReportTemplate:
    """Pick the template a question asks for.

    Parameters
    ----------
    question: str
        The question.
    folder: pathlib.Path
        The project's templates folder.
    default: str
        Template used when the question names none.

    Returns
    -------
    ReportTemplate
        The first template in the folder whose title ("Research
        Briefing"), file name ("research_briefing"), or file name with
        spaces occurs in the question; otherwise the default, read from
        the folder. Missing project-owned templates are not restored.

    Raises
    ------
    ValueError
        If the chosen template does not exist or breaks a template rule;
        the message names the line and lists the project's templates.

    """

    text = " ".join(question.casefold().split())
    for name in template_names(folder):
        words = (name.casefold(), name.casefold().replace("_", " "))
        try:
            template = load_template(name, folder)
        except ValueError:
            # Broken templates are reported only when they are chosen.
            if any(word in text for word in words):
                raise
            continue

        if template.title.casefold() in text or any(
            word in text for word in words
        ):
            return template

    return load_template(default, folder)  # Raises, listing the templates.


def paper_citation(file: Path | None, passages: tuple[Chunk, ...]) -> str:
    """Write a paper's citation from its metadata and first page.

    The language model never writes or edits the citation.

    Parameters
    ----------
    file: pathlib.Path | None
        The paper on disk, for its PDF metadata; None when unknown.
    passages: tuple[Chunk, ...]
        The paper's passages, for the DOI, year, and author count.

    Returns
    -------
    str
        For example ``Smith, J. A. et al. A Study of
        ... https://doi.org/10.1038/... (2025).``: the first author and the
        title come from the PDF metadata; the DOI and year are added only
        when printed on the first page. Without a metadata title of at
        least three words, the file name, saying so (for example
        ``smith2020.pdf (the file has no title metadata).``).

    """

    fallback = (
        f"{file.name if file else 'The paper'} (the file has no title "
        "metadata)."
    )
    if file is None or file.suffix.lower() != ".pdf" or not file.is_file():
        return fallback

    try:
        from pypdf import PdfReader

        metadata = PdfReader(file).metadata or {}
    except Exception:  # Unreadable metadata: name the file instead.
        return fallback

    title = " ".join(str(metadata.get("/Title") or "").split()).rstrip(".")
    if len(title.split()) < 3:
        return fallback

    first = paper_beginning(passages)
    authors = [
        name.strip()
        for name in re.split(r";|,| and ", str(metadata.get("/Author") or ""))
        if name.strip() and not name.strip().lower().startswith("for ")
    ]

    parts: list[str] = []
    if authors:
        surname, initials = _author(authors[0])
        # One metadata author can still be the first of many: the first
        # page then prints another name after the surname.
        many = len(authors) > 1 or bool(
            re.search(
                rf"{re.escape(surname)}[\s\S]{{0,60}}?,\s*[A-Z]", first
            )
        )
        if initials:
            etal = " et al." if many else ""
            parts.append(f"{surname}, {initials}{etal}")
        else:
            parts.append(f"{surname.rstrip('.')}.")
    parts.append(f"{title}.")

    doi = re.search(r"https?://doi\.org/\S+[0-9A-Za-z]", first)
    year = re.search(
        r"(?:Published|Accepted)[^\n]{0,30}?\b((?:19|20)\d\d)\b"
        r"|\(((?:19|20)\d\d)\)",
        first,
    )
    tail: list[str] = []
    if doi:
        tail.append(doi.group(0))
    if year:
        tail.append(f"({year.group(1) or year.group(2)})")
    if tail:
        parts.append(" ".join(tail) + ".")

    return " ".join(parts)


def _author(name: str) -> tuple[str, str]:
    """Split an author's name into surname and initials.

    Parameters
    ----------
    name: str
        One author from the PDF metadata, such as "Jane A. Smith".

    Returns
    -------
    tuple[str, str]
        For example ``("Smith", "J. A.")``; hyphenated first names give
        ``"J.-P."``. A one-word name, a name of more than four words, or an
        organisation ("The XYZ Team") keeps its whole name, without
        initials.

    """

    words = name.split()
    if len(words) == 1 or len(words) > 4 or any(
        word.isdigit() or word.casefold() in _ORGANISATION_WORDS
        for word in words
    ):
        return " ".join(words), ""

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
        The first 1,500 characters of the first three passages.

    """

    return "\n".join(chunk.text for chunk in passages[:_BEGINNING_PASSAGES])[
        :_BEGINNING_CHARACTERS
    ]


def _distinctive(text: str) -> list[str]:
    """Return the words of four or more letters that tell texts apart.

    Parameters
    ----------
    text: str
        Any text, such as reading notes or a reference entry.

    Returns
    -------
    list[str]
        Lowercase words of at least four letters, in order, without the
        common words in ``_COMMON_WORDS``.

    """

    return [
        word
        for word in re.findall(r"[a-z]{4,}", text.casefold())
        if word not in _COMMON_WORDS
    ]


def _notes_block(notes: list[tuple[int, list[str]]]) -> str:
    """Return reading notes as prompt text, one block per part.

    Parameters
    ----------
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.

    Returns
    -------
    str
        ``Part <n>:`` followed by its note lines, or ``NOTHING`` when the
        part has none; blocks separated by a blank line.

    """

    return "\n\n".join(
        f"Part {part}:\n" + ("\n".join(lines) if lines else NOTHING)
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
        "You write headlines for KíLín. You reply with the headline only, "
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
        Sections written so far, so they are not repeated; the prompt shows
        the first 30 words of each.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is the section's content.

    """

    system = (
        "You write one section of a structured summary of a scientific "
        "paper for KíLín. You use only the reading notes you are given "
        "and never invent results."
    )
    earlier = "\n".join(
        f"- {heading}: "
        + " ".join(content.split()[:_EARLIER_SECTION_WORDS])
        + " ..."
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
    return system, user


def references_prompt(
    template: ReportTemplate,
    section: TemplateSection,
    name: str,
    notes: list[tuple[int, list[str]]],
    entries: list[str],
) -> tuple[str, str]:
    """Return the prompts that choose reference entries, by number only.

    Parameters
    ----------
    template: ReportTemplate
        The report template.
    section: TemplateSection
        A section filled from ``{references}``, with an item limit.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines, for what the paper is about.
    entries: list[str]
        The paper's reference entries, from the shared
        ``reference_entries``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt. The reply is entry numbers only;
        the entries themselves are copied by code, so the model cannot
        change or invent a reference.

    """

    low, high = section.items or _DEFAULT_REFERENCE_ITEMS
    system = (
        "You choose references for KíLín. You reply with entry numbers "
        "only, separated by commas, and nothing else."
    )
    guidance = section.guidance.replace("{references}", "the list below")
    listed = "\n".join(
        f"{number}. {entry}" for number, entry in enumerate(entries, start=1)
    )
    user = f"""For the "{section.name}" section of a {template.title} of
{name}, choose {low} to {high} entries from the paper's reference list.

What the section needs: {guidance}

Reply with the entry numbers only, for example: 4, 17

What the paper is about (reading notes):

{_notes_block(notes)}

The paper's reference list:

{listed}"""
    return system, user


def fix_prompt(
    section: TemplateSection,
    content: str,
    problem: str,
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
        What is wrong, from ``section_problem``.
    notes: list[tuple[int, list[str]]] | None, optional
        Reading notes, so a section that is too short can be extended with
        real content rather than padding. Defaults to ``None``.

    Returns
    -------
    tuple[str, str]
        System prompt and user prompt; the reply is the fixed content.

    """

    system = (
        "You revise one section of a summary for KíLín. You keep its facts "
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
    return system, user


def _length(section: TemplateSection) -> str:
    """Describe a section's length for a prompt.

    Parameters
    ----------
    section: TemplateSection
        The section.

    Returns
    -------
    str
        For example "about 135 words (120-150)": the midpoint is the
        target, so the sections together fit the page. Labelled lines are
        "one line only", and sections without limits "a few concise
        sentences".

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
    """Return the section text from a reply, without a repeated heading.

    Parameters
    ----------
    reply: str
        The model's reply.
    section: TemplateSection
        The section the reply is for.

    Returns
    -------
    str
        The reply without leading blank lines or lines repeating the
        section's name (as a heading, in bold, or with a colon). For a
        labelled line, only the first remaining line, without a repeated
        label.

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
        KíLín, so they are removed; a closing line names the source, so
        the headline and citation stay together at the top.

    """

    text = _CITATION.sub("", summary).strip()
    note = (
        f"*Source: {source.name}. Written by KíLín from the paper's text; "
        "check important details against the paper.*"
    )
    return f"{text}\n\n{note}\n"


def run(
    settings: SummarySettings, request: ToolRequest, context: ToolContext
) -> ToolResult:
    """Summarize one paper with a template and save it to ``outputs/``.

    Parameters
    ----------
    settings: SummarySettings
        The intent's settings.
    request: ToolRequest
        The question.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    ToolResult
        One line saying where the Markdown and PDF were saved, then the
        saved Markdown itself; every passage of the paper as evidence; and
        a summary of the checks, fixes, and files. When the template or
        the paper cannot be used, or the user cancels, a message saying so
        and no evidence.

    Raises
    ------
    IncompleteModelResponse
        If no usable cited notes can be produced for a report. The
        normal template and paper-selection errors return ToolResult.

    """

    trace = context.trace
    summary: dict[str, object] = {"name": "generate_paper_summary"}
    try:
        template = choose_template(
            request.question,
            templates_folder(context.project_root),
            settings.template,
        )
    except ValueError as error:
        context.done("Could not read the template", evidence_count=0)
        return ToolResult(str(error), (), summary)
    summary["template"] = template.name

    context.progress("Choosing the paper to summarize")
    try:
        chosen = choose_paper(request, context, template.title.lower())
    except (FileNotFoundError, ValueError) as error:
        context.done("Could not load the paper", evidence_count=0)
        return ToolResult(str(error), (), summary)

    if isinstance(chosen, str):
        context.done("No paper chosen", evidence_count=0)
        return ToolResult(chosen, (), summary)

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
    paper_tokens = sum(estimate_tokens(text) for text in texts)
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
            _estimate(to_read, texts, template), 1
        ),
        quick_items=len(parts[0]),
        quick_seconds=round(
            _estimate(parts[:1], texts, template), 1
        ),
        details=(
            *(
                (
                    "KíLín chose this paper from your question; name the "
                    "file if it is not the right one.",
                )
                if guessed
                else ()
            ),
            f"About {paper_tokens:,} tokens",
            f"Saved to {OUTPUTS_FOLDER}/{path.stem}_summary.md",
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
    failures: list[dict[str, object]] = []
    notes = take_notes(
        _notes_task(template),
        path.name,
        texts,
        runs,
        context,
        what="paper",
        whole=choice != PLAN_CHOICE_QUICK,
        failures=failures,
    )
    reading_incomplete = bool(failures)
    read_indices = {index for number, _ in notes for index in runs[number - 1]}
    read = len(read_indices)

    if not any(lines for _, lines in notes):
        if failures:
            return partial_result(
                ToolResult(
                    "No usable notes are available. No report file was saved.",
                    (), {**summary, "valid": False, "saved": None,
                         "saved_pdf": None, "passages_read": read},
                ), failures, completed=False,
            )
        raise IncompleteModelResponse(
            "No usable cited notes were produced for the report.",
            reason="no_notes",
        )

    # Python owns the structure: the model writes the headline and each
    # section separately, and each piece is checked against its limit.
    try:
        headline, fixes = _write_headline(template, path.name, notes, context)
    except IncompleteModelResponse as error:
        headline, fixes = template.title, []
        failures.append({"item": "Headline", **error.to_dict()})
    # The PDF on disk lets the shared extractor read two columns in order.
    entries = reference_entries(
        passages, REFERENCE_LAYOUT, _paper_file(path, context)
    )
    written: list[tuple[str, str]] = []
    accepted: list[tuple[str, str]] = []
    unsupported: list[str] = []
    for section in template.sections:
        try:
            if section.source == SECTION_SOURCE_CITATION:
                # Bibliographic data is copied, not generated.
                content = paper_citation(_paper_file(path, context), passages)
            elif section.source == SECTION_SOURCE_REFERENCES:
                context.progress(f"Choosing {section.name}")
                content, problem = _fill_references(
                    template, section, path.name, notes, entries, context
                )
                if problem:
                    unsupported.append(problem)
            else:
                content, section_fixes = _write_section(
                    template, section, path.name, notes, accepted, settings,
                    context, incomplete_reading=reading_incomplete,
                )
                fixes.extend(section_fixes)
            accepted.append((section.name, content))
        except IncompleteModelResponse as error:
            failures.append({"item": section.name, **error.to_dict()})
            content = f"*Not completed: {error}*"
        # Failed placeholders remain visible, but never feed later sections.
        written.append((section.name, content))

    if template.max_total_words:
        written = _fit_total(written, template)
    if failures:
        # A partial artifact may contain validated sections and explicit
        # omissions, never an unmarked invalid section rescued by another's
        # failure. Validate after any document-wide shortening as well.
        accepted_names = {name for name, _ in accepted}
        for index, (section, (name, content)) in enumerate(
            zip(template.sections, written)
        ):
            if name not in accepted_names:
                continue
            problem = section_problem(section, content)
            if problem:
                failures.append({
                    "item": name, "message": problem,
                    "reason": "report_validation",
                })
                written[index] = (name, f"*Not completed: {problem}*")
                accepted_names.remove(name)
        accepted = [(name, text) for name, text in accepted
                    if name in accepted_names]
    draft = _draft(headline, written, template)

    context.progress("Checking the template")
    check = check_report(draft, template)
    repairs = len(fixes)
    checks = [fixes, check.problems(), unsupported]
    trace.emit(
        "tool.generate_paper_summary.checked",
        component="tool",
        data={"checks": checks, "repairs": repairs, "valid": check.valid},
        turn_id=context.trace_turn_id,
    )

    if (not check.valid or unsupported or failures) and not (
        failures and accepted
    ):
        problems = check.problems() + unsupported
        summary.update({
            "status": "incomplete",
            "reason": "report_validation",
            "message": "The generated report failed its template checks.",
            "valid": False,
            "problems": problems,
            "repairs": repairs,
            "passages_read": read,
            "notes": sum(len(lines) for _, lines in notes),
            "saved": None,
            "saved_pdf": None,
            "seconds": round(perf_counter() - started, 1),
            "model": context.model_id,
        })
        trace.emit(
            "tool.generate_paper_summary.incomplete", component="tool",
            data=summary, turn_id=context.trace_turn_id,
        )
        context.done(
            "Report validation failed; no file saved", evidence_count=0
        )
        return partial_result(
            ToolResult(
                "The report could not be completed because its validation "
                "failed. No report file was saved.\n\n"
                + "\n".join(f"- {problem}" for problem in problems),
                (), summary,
            ), failures, completed=False,
        )

    file_text = summary_file_text(draft, path)
    suffix = "_summary.md"
    if failures:
        suffix = "_summary_partial.md"
        file_text = (
            "> **Partial report:** incomplete items are marked below. "
            "Do not treat omissions as negative evidence.\n\n"
            + file_text + incomplete_notice(failures) + "\n"
        )
    saved = save_output_file(
        context.project_root, path.stem + suffix, file_text
    )
    # A partial draft must not replace the completed report or its PDF.
    pdf, pdf_note = (
        (None, "No PDF was produced for this partial report.") if failures
        else _write_pdf(file_text, saved, template, context)
    )
    trace.emit(
        "tool.generate_paper_summary.saved",
        component="tool",
        data={"file": saved.as_posix()},
        turn_id=context.trace_turn_id,
    )

    # The answer says where the files are and then shows the saved file
    # itself, so the chat and the file never differ. Checks and fixes are
    # recorded in the turn and the trace.
    files = f"`{saved.as_posix()}`" + (
        f" and `{pdf.as_posix()}`" if pdf else ""
    )
    lead = [
        ("Partial report: the " if failures else "Done: the ")
        + f"{template.title.lower()} of {path.name} is saved as "
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
                else " (reading coverage is incomplete)."
            )
        )

    summary.update(
        {
            "passages_read": read,
            "notes": sum(len(lines) for _, lines in notes),
            "valid": check.valid and not unsupported and not failures,
            "problems": check.problems() + unsupported,
            "repairs": repairs,
            "saved": saved.as_posix(),
            "saved_pdf": pdf.as_posix() if pdf else None,
            "seconds": round(perf_counter() - started, 1),
            "model": context.model_id,
        }
    )
    if failures:
        summary.update({
            "status": "partial", "reason": "item_completion",
            "message": "Some report items could not be completed.",
            "incomplete_items": failures,
        })
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
    context.done(
        f"Saved {saved.as_posix()}",
        evidence_count=read if failures else len(passages),
    )

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
        if not failures or index in read_indices
    )
    answer = " ".join(lead) + "\n\n---\n\n" + file_text.strip()
    return ToolResult(answer, evidence, summary)


def _notes_task(template: ReportTemplate) -> str:
    """Return the note-taking task for a template.

    Parameters
    ----------
    template: ReportTemplate
        The report template.

    Returns
    -------
    str
        What the notes must capture: bibliographic details, figure
        captions, key references, and the guidance of every section the
        model writes from notes.

    """

    return (
        f"Notes for a {template.title} of this paper. Copy the paper's "
        "title, authors, journal, and year when they appear, every figure "
        "caption with its figure number, and the references the paper "
        "relies on most. Note what each section needs:\n"
        + "\n".join(
            f"- {section.name}: {section.guidance}"
            for section in template.sections
            if section.source == SECTION_SOURCE_NOTES
        )
    )


def _write_headline(
    template: ReportTemplate,
    name: str,
    notes: list[tuple[int, list[str]]],
    context: ToolContext,
) -> tuple[str, list[str]]:
    """Write the headline, asking once more when it is too long.

    Parameters
    ----------
    template: ReportTemplate
        The report template.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    tuple[str, list[str]]
        The headline, and the fix made (``"headline: <n> characters"``) if
        any. A template without a headline limit gives its title and no
        model call. The second headline is not checked again here;
        ``check_report`` reports it if it is still too long.

    """

    if not template.headline_characters:
        return template.title, []

    context.progress("Writing the headline")
    system, prompt = headline_prompt(template, name, notes)
    headline = _headline(context.ask_model(system, prompt, _HEADLINE_TOKENS))
    if len(headline) <= template.headline_characters:
        return headline, []

    fixes = [f"headline: {len(headline)} characters"]
    headline = _headline(
        context.ask_model(
            system,
            prompt
            + f"\n\nYour headline \"{headline}\" is too long. "
            f"Write one of at most {template.headline_characters} "
            "characters.",
            _HEADLINE_TOKENS,
        )
    )
    return headline, fixes


def _write_section(
    template: ReportTemplate,
    section: TemplateSection,
    name: str,
    notes: list[tuple[int, list[str]]],
    written: list[tuple[str, str]],
    settings: SummarySettings,
    context: ToolContext,
    *,
    incomplete_reading: bool = False,
) -> tuple[str, list[str]]:
    """Write one section from the notes, then fix it toward its limits.

    Parameters
    ----------
    template: ReportTemplate
        The report template.
    section: TemplateSection
        A section written from notes.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.
    written: list[tuple[str, str]]
        Sections written so far, so they are not repeated.
    settings: SummarySettings
        The intent's repair-attempt setting.
    context: ToolContext
        KíLín's services.
    incomplete_reading: bool, optional
        Warn that failed reading parts are not evidence of absence.

    Returns
    -------
    tuple[str, list[str]]
        The content and the problems that were sent for fixing. Each fix
        names the exact problem; a section that is too short also gets the
        notes, so it can grow with real content. A section still too long
        after its fixes loses whole trailing sentences.

    """

    context.progress(f"Writing {section.name}")
    system, prompt = section_prompt(template, section, name, notes, written)
    coverage = (
        "\n\nSome reading parts failed. Use only the supplied notes. "
        "When these do not cover a topic, say it could not be established "
        "from the available notes, not that it is absent from the paper."
        if incomplete_reading else ""
    )
    content = _content(
        context.ask_model(system, prompt + coverage, None), section
    )
    problem = section_problem(section, content)

    fixes: list[str] = []
    for _ in range(settings.repair_attempts):
        if problem is None:
            break

        context.progress(f"Fixing: {problem}")
        fixes.append(problem)
        short = bool(
            section.words and count_words(content) < section.words[0]
        )
        system, prompt = fix_prompt(
            section, content, problem, notes if short else None
        )
        content = _content(
            context.ask_model(system, prompt + coverage, None), section
        )
        problem = section_problem(section, content)

    if problem and section.words and (
        count_words(content) > section.words[1]
    ):
        # Too long after its fixes: cut whole trailing sentences.
        content = _trim_words(content, section.words[1])

    return content, fixes


def _fill_references(
    template: ReportTemplate,
    section: TemplateSection,
    name: str,
    notes: list[tuple[int, list[str]]],
    entries: list[str],
    context: ToolContext,
) -> tuple[str, str | None]:
    """Fill a ``{references}`` section with entries copied verbatim.

    Parameters
    ----------
    template: ReportTemplate
        The report template.
    section: TemplateSection
        A section filled from ``{references}``.
    name: str
        The paper's file name.
    notes: list[tuple[int, list[str]]]
        Part numbers and note lines.
    entries: list[str]
        The paper's reference entries, from the shared
        ``reference_entries``.
    context: ToolContext
        KíLín's services.

    Returns
    -------
    tuple[str, str | None]
        The content (one ``- `` line per entry) and a problem when the
        paper has no readable reference list (then without a model call).

    Notes
    -----
    The model replies with entry numbers; numbers outside the list and
    repeated numbers are ignored, and at most the item maximum is kept.
    When it gives too few, the entries sharing the most distinctive words
    with the notes complete the list.

    """

    low, high = section.items or _DEFAULT_REFERENCE_ITEMS
    if not entries:
        return (
            "- Not available: the paper's reference list could not be read.",
            f"{section.name}: the paper's reference list could not be read.",
        )

    system, prompt = references_prompt(
        template, section, name, notes, entries
    )
    reply = context.ask_model(system, prompt, _REFERENCES_TOKENS)
    chosen: list[int] = []
    for number in re.findall(r"\d+", reply):
        index = int(number) - 1
        if 0 <= index < len(entries) and index not in chosen:
            chosen.append(index)
    chosen = chosen[:high]

    if len(chosen) < low:
        vocabulary = set(_distinctive(_notes_block(notes)))
        ranked = sorted(
            (i for i in range(len(entries)) if i not in chosen),
            key=lambda i: -sum(
                w in vocabulary for w in set(_distinctive(entries[i]))
            ),
        )
        chosen += ranked[: low - len(chosen)]

    context.trace.emit(
        "tool.generate_paper_summary.references",
        component="tool",
        data={"entries": len(entries), "reply": reply, "chosen": chosen},
        turn_id=context.trace_turn_id,
    )
    return "\n".join(f"- {entries[i]}" for i in chosen), None


def _draft(
    headline: str, written: list[tuple[str, str]], template: ReportTemplate
) -> str:
    """Assemble the report's Markdown from its pieces.

    Parameters
    ----------
    headline: str
        The headline, or the template's title.
    written: list[tuple[str, str]]
        Section names and contents, in the template's order.
    template: ReportTemplate
        The report template, for which sections are labelled lines.

    Returns
    -------
    str
        ``# headline``, then each labelled line as ``**Name:** content`` on
        one line and each section as ``## Name`` with its content, separated
        by blank lines.

    """

    inline = {s.name for s in template.sections if s.inline}
    return "\n\n".join(
        [f"# {headline}"]
        + [
            f"**{name}:** " + " ".join(content.split())
            if name in inline
            else f"## {name}\n{content}"
            for name, content in written
        ]
    )


def _trim_words(content: str, limit: int) -> str:
    """Drop whole trailing sentences until the content has ``limit`` words.

    Parameters
    ----------
    content: str
        Section content.
    limit: int
        Most words.

    Returns
    -------
    str
        The content when it fits; otherwise its leading sentences, joined
        by single spaces. The first sentence is always kept, so the result
        can still be over the limit. A sentence ends at ``.``, ``!``, or
        ``?`` followed by a capital letter, ``[``, or ``(``.

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

    Parameters
    ----------
    text: str
        The saved Markdown.
    saved: pathlib.Path
        The Markdown's project path (``outputs/...``).
    template: ReportTemplate
        The report template, for the PDF's label.
    context: ToolContext
        KíLín's services (``write_pdf``).

    Returns
    -------
    tuple[pathlib.Path | None, str | None]
        The PDF's project path, or None when there is no writer or it
        failed; and a note for the answer when the PDF is longer than one
        page or could not be written.

    """

    if context.write_pdf is None:
        return None, None

    try:
        with staged_output_file(
            context.project_root, suffix=".pdf"
        ) as staged:
            pages = context.write_pdf(
                text, staged, label=template.title
            )
            if staged.stat().st_size == 0:
                raise ValueError("The PDF renderer produced an empty file.")
            pdf = publish_output_file(
                context.project_root, f"{saved.stem}.pdf", staged
            )
    except Exception as error:  # The Markdown is saved either way.
        return None, f"The PDF could not be written: {error}"

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
    """Return the paper on disk, for its PDF metadata and page layout.

    Parameters
    ----------
    path: pathlib.Path
        The paper's path: ``outputs/...`` (relative to the project) or an
        indexed source path (relative to the project's ``sources/``).
    context: ToolContext
        KíLín's services (``project_root``).

    Returns
    -------
    pathlib.Path | None
        The first of these that is a file, or None.

    """

    root = context.project_root
    for candidate in (root / path, root / PROJECT_SOURCES_DIRECTORY / path):
        if candidate.is_file():
            return candidate

    return None


def _fit_total(
    written: list[tuple[str, str]], template: ReportTemplate
) -> list[tuple[str, str]]:
    """Trim the longest prose sections toward their midpoints to fit.

    Parameters
    ----------
    written: list[tuple[str, str]]
        Section names and contents, in the template's order.
    template: ReportTemplate
        A template with ``max_total_words``.

    Returns
    -------
    list[tuple[str, str]]
        The sections, in the same order. While the total is over
        ``max_total_words``, the section furthest above the middle of its
        word range loses its last sentence. Sections are never cut below
        that middle; reference lists and one-line sections are left alone,
        so the total can stay over the limit.

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


def _headline(reply: str) -> str:
    """Return the headline from a reply: first line, without marks.

    Parameters
    ----------
    reply: str
        The model's reply.

    Returns
    -------
    str
        The first nonblank line, without leading ``#``, surrounding quotes,
        or ``*``.

    """

    line = next((x for x in reply.strip().splitlines() if x.strip()), "")
    return line.strip().lstrip("#").strip().strip('"“”*').strip()


def _section_reply_estimate(section: TemplateSection) -> int:
    """Estimate section output for the plan, not generation control.

    Parameters
    ----------
    section: TemplateSection
        The section.

    Returns
    -------
    int
        Two tokens per word of the word maximum plus 80; else 90 tokens
        per item of the item maximum plus 40; otherwise the packaged section
        estimate.

    """

    if section.words:
        return 2 * section.words[1] + 80

    if section.items:
        return 90 * section.items[1] + 40

    return SECTION_REPLY_TOKENS


def _estimate(
    parts: list[list[int]],
    texts: list[str],
    template: ReportTemplate,
) -> float:
    """Return the approximate seconds of a run, for the plan.

    Parameters
    ----------
    parts: list[list[int]]
        Passage indices of the parts to read.
    texts: list[str]
        Every passage's text.
    template: ReportTemplate
        The report template.

    Returns
    -------
    float
        Seconds to read the parts (as ``batch_review`` estimates them),
        then one call for the headline and one per written section, each
        with all the notes in its prompt.

    """

    total = 0.0
    for part in parts:
        prompt = _PART_PROMPT_TOKENS + sum(
            estimate_tokens(texts[index]) for index in part
        )
        total += _CALL_SECONDS + prompt / _PROMPT_TOKENS_PER_SECOND
        # Planning estimates are independent of generation limits.
        total += NOTE_REPLY_TOKENS / _REPLY_TOKENS_PER_SECOND

    notes = len(parts) * NOTE_REPLY_TOKENS * _REPLY_SHARE
    written = [
        section
        for section in template.sections
        if section.source != SECTION_SOURCE_CITATION
    ]
    calls = len(written) + (1 if template.headline_characters else 0)
    for section in written:
        reply = (
            _REFERENCES_REPLY_TOKENS
            if section.source == SECTION_SOURCE_REFERENCES
            else _section_reply_estimate(section) * _REPLY_SHARE
        )
        total += _CALL_SECONDS + reply / _REPLY_TOKENS_PER_SECOND
    total += (
        calls * (notes + _SECTION_PROMPT_TOKENS) / _PROMPT_TOKENS_PER_SECOND
    )

    return total
