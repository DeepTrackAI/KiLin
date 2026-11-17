"""Tool ``project_summary``: one report on all the documents of a project.

A project template (``scope: project``) says what the report holds:
sections written from all documents, and per-document sections listing the
documents that meet a condition. The tool:

1. Chooses the template the question names, the project's only project
   template, or the one the user chooses.
2. Plans the work (one call per document, one per section) and asks
   before a large run; the quick version reads a shorter excerpt of each
   document (its card and opening).
3. Reads every document once (``_documents``, ``_mapping``): notes for each
   section, and checked per-document answers.
4. Writes the report (``_reduce``), checks it against the template, fixes
   sections that miss their limits, and saves it to ``outputs/``, closed by
   the numbered list of documents it cites.

Key Features
------------

- **The whole project**

    Every document is read, not only what a search finds.

- **Any project template**

    Shipped or written by the user, chosen by name or from a list.

- **Checked**

    Per-document entries need evidence; sections are checked against
    their limits.

Module Structure
----------------

- ``ProjectSummarySettings`` and ``parse_settings``

    The settings of ``project_summary`` in a project type.

- ``SHIPPED_TEMPLATES_DIR``, ``TEMPLATE_CHOOSE`` and
  ``workspace_resources``

    The project templates shipped with KíLín, and their installation.

- ``project_templates``, ``choose_template`` and ``routing``

    The project's project templates, the one a question asks for, and the
    routing they add.

- ``run``

    Write the report and save it.

Examples
--------
>>> from kilin.tools.project_summary import parse_settings
>>> parse_settings({"template": "project_overview"}, "x").template
'project_overview'

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final

from kilin._model_completion import strip_thinking
from kilin.retrievals.document_reading import estimate_tokens
from kilin.tools._contract import (
    ToolContext,
    ToolRequest,
    ToolResult,
    ToolRouting,
    check_choice,
    check_int,
    reject_unknown,
)
from kilin.tools._shared.outputs import save_output_file
from kilin.tools._shared.planning import (
    CONFIRMATION_CHOICES,
    PLAN_CHOICE_CANCEL,
    PLAN_CHOICE_QUICK,
    ToolPlan,
    confirm_plan,
)
from kilin.tools._shared.report_templates import (
    SCOPE_PROJECT,
    SECTION_SOURCE_DOCUMENTS,
    SECTION_SOURCE_KEY_DOCUMENTS,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_PER_DOCUMENT,
    ReportTemplate,
    TemplateSection,
    check_report,
    load_template,
    section_problem,
    template_names,
)
from kilin.tools.project_summary import _configuration as _config
from kilin.tools.project_summary._documents import (
    ProjectDocument,
    document_excerpt,
    project_documents,
    query_words,
)
from kilin.tools.project_summary._mapping import (
    DocumentNotes,
    map_prompt,
    parse_notes,
    template_texts,
    topics,
)
from kilin.tools.project_summary._reduce import (
    chosen_numbers,
    condense_prompt,
    documents_line,
    documents_list,
    fix_prompt,
    headline_prompt,
    key_documents_prompt,
    note_groups,
    per_document_markdown,
    section_prompt,
)
from kilin.workspace_resources import WorkspaceResource, templates_folder

__all__ = [
    "SHIPPED_TEMPLATES_DIR",
    "TEMPLATE_CHOOSE",
    "ProjectSummarySettings",
    "choose_template",
    "parse_settings",
    "project_templates",
    "routing",
    "run",
    "workspace_resources",
]

# Project templates shipped with KíLín.
SHIPPED_TEMPLATES_DIR: Final[Path] = Path(__file__).parent / "templates"
# The ``template`` value using any project template of the project.
TEMPLATE_CHOOSE: Final[str] = "choose"
_TEMPLATE_NAME = re.compile(r"[A-Za-z0-9][\w-]*")
_WRITE_WORDS: Final[str] = (
    r"\b(?:write|prepare|make|generate|create|produce|draft|fill(?:\s+in)?|"
    r"do|give\s+me|build)\b[^\n]{0,60}?"
)
_FOR_WORDS: Final[str] = r"\s+(?:of|for|on|from|about|across)\b"
_ROUTED_DESCRIPTION: Final[int] = 120


@dataclass(frozen=True, slots=True)
class ProjectSummarySettings:
    """Settings of ``project_summary`` in a project type.

    Attributes
    ----------
    template: str
        The project template's file name without ``.md``, or
        ``TEMPLATE_CHOOSE`` to use the one the question names or the user
        chooses.
    document_tokens: int
        Tokens read from each document (300-16000).
    max_documents: int
        Most documents read (1-2000); the others are reported as not read.
    repair_attempts: int
        Rewrites of a section that misses its limits (0-4).
    confirmation: str
        ``always``, ``when_large`` or ``never``.
    large_batches: int
        With ``when_large``, plans with more model calls than this ask
        first (0-1000).

    """

    template: str = _config.DEFAULT_TEMPLATE
    document_tokens: int = _config.DEFAULT_DOCUMENT_TOKENS
    max_documents: int = _config.DEFAULT_MAX_DOCUMENTS
    repair_attempts: int = _config.DEFAULT_REPAIR_ATTEMPTS
    confirmation: str = _config.DEFAULT_CONFIRMATION
    large_batches: int = _config.DEFAULT_LARGE_BATCHES


def parse_settings(data: dict, where: str) -> ProjectSummarySettings:
    """Validate ``project_summary`` settings from a project type.

    Parameters
    ----------
    data: dict
        Settings from the intent's ``tool`` object, without ``name``.
    where: str
        Location for errors.

    Returns
    -------
    ProjectSummarySettings
        Validated settings.

    Raises
    ------
    ValueError
        If a setting is unknown or out of range, or the template is not a
        plain file name.

    """

    reject_unknown(
        where, data,
        ("template", "document_tokens", "max_documents", "repair_attempts",
         "confirmation", "large_batches"),
    )
    defaults = ProjectSummarySettings()
    template = data.get("template", defaults.template)
    if not isinstance(template, str) or not _TEMPLATE_NAME.fullmatch(
        template
    ):
        raise ValueError(
            f"{where!r} template must be a template file name without .md."
        )
    return ProjectSummarySettings(
        template=template,
        document_tokens=check_int(
            where, data, "document_tokens", defaults.document_tokens,
            300, 16000,
        ),
        max_documents=check_int(
            where, data, "max_documents", defaults.max_documents, 1, 2000
        ),
        repair_attempts=check_int(
            where, data, "repair_attempts", defaults.repair_attempts, 0, 4
        ),
        confirmation=check_choice(
            where, data, "confirmation", defaults.confirmation,
            CONFIRMATION_CHOICES,
        ),
        large_batches=check_int(
            where, data, "large_batches", defaults.large_batches, 0, 1000
        ),
    )


def workspace_resources(
    settings: ProjectSummarySettings,
) -> tuple[WorkspaceResource, ...]:
    """Declare the project templates one configured intent needs.

    Parameters
    ----------
    settings: ProjectSummarySettings
        Validated tool configuration.

    Returns
    -------
    tuple[WorkspaceResource, ...]
        Every shipped project template for ``TEMPLATE_CHOOSE``; otherwise
        the named one when it is shipped, else none (a template the user
        provides).

    """

    if settings.template == TEMPLATE_CHOOSE:
        sources = sorted(SHIPPED_TEMPLATES_DIR.glob("*.md"))
    else:
        source = SHIPPED_TEMPLATES_DIR / f"{settings.template}.md"
        sources = [source] if source.exists() else []
    return tuple(
        WorkspaceResource(source, f"templates/{source.name}")
        for source in sources
    )


def project_templates(
    folder: Path,
) -> list[tuple[str, ReportTemplate | None, str]]:
    """Return the project templates in a templates folder.

    Parameters
    ----------
    folder: pathlib.Path
        A project's templates folder.

    Returns
    -------
    list[tuple[str, ReportTemplate | None, str]]
        Each project template's file name, the template (None when it
        breaks a rule) and the error (``""`` when valid), by name; a file
        that breaks a rule is listed when it says ``scope: project``.
        Templates for one document are left out.

    """

    if not folder.is_dir():
        return []
    found: list[tuple[str, ReportTemplate | None, str]] = []
    for name in template_names(folder):
        try:
            template = load_template(name, folder, _config.TEMPLATE_RULES)
        except ValueError as error:
            text = (folder / f"{name}.md").read_text(encoding="utf-8")
            if re.search(r"^scope:\s*project\s*$", text, re.M | re.I):
                found.append((name, None, str(error)))
            continue
        if template.scope == SCOPE_PROJECT:
            found.append((name, template, ""))
    return found


def _named(question: str, name: str, template: ReportTemplate | None) -> bool:
    """Tell whether a question names a template.

    Parameters
    ----------
    question: str
        The question, in lower case with single spaces.
    name: str
        The template's file name.
    template: ReportTemplate | None
        The template, for its title.

    Returns
    -------
    bool
        True when the file name, the file name with spaces, or the title
        occurs in the question.

    """

    words = [name.casefold(), name.casefold().replace("_", " ")]
    if template is not None:
        words.append(template.title.casefold())
    return any(word in question for word in words)


def choose_template(
    question: str,
    folder: Path,
    default: str,
    choose: Callable[[str, tuple[str, ...]], int | None] | None = None,
) -> ReportTemplate | None:
    """Pick the project template a question asks for, or ask which.

    Parameters
    ----------
    question: str
        The question.
    folder: pathlib.Path
        The project's templates folder.
    default: str
        The template used when the question names none, or
        ``TEMPLATE_CHOOSE``.
    choose: Callable[[str, tuple[str, ...]], int | None] | None, optional
        ``ToolContext.choose``. Defaults to ``None``.

    Returns
    -------
    ReportTemplate | None
        The project template the question names; otherwise the default,
        or with ``TEMPLATE_CHOOSE`` the only one or the user's choice;
        None when the user cancels.

    Raises
    ------
    ValueError
        If the named template breaks a rule, there is no project template,
        or nobody can be asked which; the message says how to go on.

    """

    text = " ".join(question.casefold().split())
    found = project_templates(folder)
    for name, template, error in found:
        if _named(text, name, template):
            if template is None:
                raise ValueError(f"`{name}.md` cannot be used: {error}")
            return template
    if default != TEMPLATE_CHOOSE:
        return load_template(default, folder, _config.TEMPLATE_RULES)
    usable = [template for _, template, _ in found if template is not None]
    if not usable:
        raise ValueError(
            "This project has no project report template. A project "
            "template is a Markdown file in `templates/` with `scope: "
            "project` in its front matter; `templates/README.md` explains "
            "the format."
        )
    if len(usable) == 1:
        return usable[0]
    if choose is None:
        raise ValueError(
            "Which project report? Name one in the question, for example "
            f"“write a {usable[0].title.lower()} of the project”: "
            + ", ".join(f"{t.title} (`{t.name}.md`)" for t in usable) + "."
        )
    index = choose(
        "Which report on the whole project?",
        tuple(f"{t.title} ({t.name})" for t in usable),
    )
    return None if index is None else usable[index]


def routing(
    settings: ProjectSummarySettings,
    root: Path,
    others: tuple[ProjectSummarySettings, ...] = (),
) -> ToolRouting | None:
    """Return the routing the project's project templates add.

    Parameters
    ----------
    settings: ProjectSummarySettings
        The intent's settings.
    root: pathlib.Path
        The project's workspace.
    others: tuple[ProjectSummarySettings, ...], optional
        Other intents using this tool; templates they name are left to
        them. Defaults to none.

    Returns
    -------
    ToolRouting | None
        For ``TEMPLATE_CHOOSE``: a cue per valid project template whose
        title or file name has at least two words, after a word asking to
        write ("write a lab overview of the project") or before "of",
        "for", "on", "from", "about" or "across"; and a description of the
        templates for the router. None otherwise, or without templates.

    """

    if settings.template != TEMPLATE_CHOOSE:
        return None
    claimed = {other.template for other in others}
    usable = [
        template for name, template, _ in
        project_templates(templates_folder(root))
        if template is not None and name not in claimed
    ]
    if not usable:
        return None
    cues: list[str] = []
    for template in usable:
        phrases = {
            " ".join(phrase.casefold().split())
            for phrase in (template.title, template.name.replace("_", " "))
            if len(phrase.replace("_", " ").split()) >= 2
        }
        if not phrases:
            continue
        alternatives = "|".join(
            r"[\s_-]+".join(re.escape(word) for word in phrase.split())
            for phrase in sorted(phrases)
        )
        cues.append(rf"{_WRITE_WORDS}\b(?:{alternatives})\b")
        cues.append(rf"\b(?:{alternatives})\b{_FOR_WORDS}")

    def short(text: str) -> str:
        """Return a description cut for the router.

        Parameters
        ----------
        text: str
            A template's description.

        Returns
        -------
        str
            Without its final full stop, cut at a word with "…" when long.

        """

        text = " ".join(text.split()).rstrip(". ")
        if len(text) <= _ROUTED_DESCRIPTION:
            return text
        return text[:_ROUTED_DESCRIPTION].rsplit(" ", 1)[0] + "…"

    described = "; ".join(
        f"{t.title} ({t.name}): {short(t.description)}" for t in usable
    )
    return ToolRouting(
        tuple(cues),
        f"Project report templates in this project: {described}.",
    )


def _seconds(prompt_tokens: int, reply_tokens: int) -> float:
    """Return the estimated duration of one model call.

    Parameters
    ----------
    prompt_tokens: int
        Tokens sent.
    reply_tokens: int
        Tokens written.

    Returns
    -------
    float
        Seconds, from the tool's estimates.

    """

    return (
        _config.CALL_SECONDS
        + prompt_tokens / _config.PROMPT_TOKENS_PER_SECOND
        + reply_tokens / _config.REPLY_TOKENS_PER_SECOND
    )


def _plan(
    template: ReportTemplate,
    documents: Sequence[ProjectDocument],
    settings: ProjectSummarySettings,
    left_out: int,
) -> ToolPlan:
    """Return the plan of one project report.

    Parameters
    ----------
    template: ReportTemplate
        The project template.
    documents: Sequence[ProjectDocument]
        The documents to read.
    settings: ProjectSummarySettings
        The settings, for the excerpt size.
    left_out: int
        Documents beyond ``max_documents``.

    Returns
    -------
    ToolPlan
        One call per document and one per written section, with estimated
        times for full and shorter (quick) reading.

    """

    count = len(documents)
    writing = len(topics(template)) + sum(
        s.source == SECTION_SOURCE_KEY_DOCUMENTS for s in template.sections
    ) + (1 if template.headline_characters else 0)
    write_seconds = writing * _seconds(
        _config.SECTION_PROMPT_TOKENS + _config.GROUP_TOKENS // 2,
        _config.SECTION_REPLY_TOKENS,
    )
    read = count * _seconds(
        _config.MAP_PROMPT_TOKENS + settings.document_tokens,
        _config.MAP_REPLY_TOKENS,
    )
    quick = count * _seconds(
        _config.MAP_PROMPT_TOKENS + _config.QUICK_DOCUMENT_TOKENS,
        _config.MAP_REPLY_TOKENS,
    )
    cards = sum(document.card is not None for document in documents)
    scope = f"{count} documents of the project, {cards} with a card"
    if left_out:
        scope += f"; {left_out} more are left out (the limit is {count})"
    return ToolPlan(
        tool="project_summary",
        summary=f"Write a {template.title} from every document.",
        scope=scope + ".",
        items=count,
        batches=count + writing,
        batches_to_run=count + writing,
        estimated_seconds=read + write_seconds,
        quick_items=count,
        quick_seconds=quick + write_seconds,
        details=(
            f"Reads about {settings.document_tokens} tokens of each "
            "document, then writes each section from all of them.",
        ),
        quick_label="Shorter reading",
        quick_note="Each document's card and opening only",
    )


def _clean(content: str, section: TemplateSection) -> str:
    """Return a model-written section without a repeated heading.

    Parameters
    ----------
    content: str
        The model's reply.
    section: TemplateSection
        The section it writes.

    Returns
    -------
    str
        Without thinking, a leading heading or label naming the section,
        and surrounding blank lines; a labelled line is its first line.

    """

    lines = strip_thinking(content).strip().splitlines()
    name = re.escape(section.name)
    while lines and (
        re.match(r"^\s*#+\s", lines[0])
        or re.match(rf"^\**\s*{name}\s*:?\**\s*:?\s*$", lines[0], re.I)
    ):
        lines.pop(0)
    text = "\n".join(lines).strip()
    if section.inline:
        first = next((line.strip() for line in lines if line.strip()), "")
        text = re.sub(rf"^\**\s*{name}\s*:?\**\s*:?\s*", "", first,
                      flags=re.I)
    return text


def _write_section(
    template: ReportTemplate,
    section: TemplateSection,
    notes: dict[int, DocumentNotes],
    documents: Sequence[ProjectDocument],
    settings: ProjectSummarySettings,
    context: ToolContext,
) -> str:
    """Write one section from all documents' notes, fixing its length.

    Parameters
    ----------
    template: ReportTemplate
        The project template.
    section: TemplateSection
        A ``notes`` section.
    notes: dict[int, DocumentNotes]
        Each read document's notes, by number.
    documents: Sequence[ProjectDocument]
        The documents.
    settings: ProjectSummarySettings
        The settings, for repair attempts.
    context: ToolContext
        Tool services, for model calls.

    Returns
    -------
    str
        The section's content; when the notes do not fit one call, written
        from each group's condensed facts.

    """

    lines = [
        (number, found.notes[section.name])
        for number, found in sorted(notes.items())
        if section.name in found.notes
    ]
    groups = note_groups(lines, _config.GROUP_TOKENS)
    if len(groups) > 1:
        condensed = [
            strip_thinking(context.ask_model(
                *condense_prompt(section, group), None
            )).strip()
            for group in groups
        ]
        text = "\n".join(condensed)
    else:
        text = "\n".join(f"[{n}] {note}" for n, note in lines)
    content = _clean(context.ask_model(
        *section_prompt(template, section, text, len(documents)), None
    ), section)
    for _ in range(settings.repair_attempts):
        problem = section_problem(section, content, _config.TEMPLATE_RULES)
        if not problem:
            break
        content = _clean(context.ask_model(
            *fix_prompt(section, content, problem), None
        ), section) or content
    return content


def run(
    settings: ProjectSummarySettings,
    request: ToolRequest,
    context: ToolContext,
) -> ToolResult:
    """Write one report on all the documents of a project and save it.

    Parameters
    ----------
    settings: ProjectSummarySettings
        Validated settings.
    request: ToolRequest
        The question.
    context: ToolContext
        Tool services: the project's passages, cards, model, confirmation
        and progress.

    Returns
    -------
    ToolResult
        The report, citing documents as ``[n]`` with each document's first
        passage as evidence ``n``, and where it was saved; or why it could
        not be written.

    """

    summary: dict[str, object] = {"name": "project_summary"}
    folder = templates_folder(context.project_root)
    try:
        template = choose_template(
            request.question, folder, settings.template,
            getattr(context, "choose", None),
        )
    except ValueError as error:
        context.done("No project template", evidence_count=0)
        return ToolResult(str(error), (), summary)
    if template is None:
        context.done("No template chosen", evidence_count=0)
        return ToolResult("No template was chosen.", (), summary)
    summary["template"] = template.name

    documents = project_documents(
        context.chunks, getattr(context, "card", None)
    )
    if not documents:
        context.done("No documents", evidence_count=0)
        return ToolResult(
            "The project has no indexed documents to report on.", (),
            summary,
        )
    left_out = max(0, len(documents) - settings.max_documents)
    documents = documents[:settings.max_documents]
    plan = _plan(template, documents, settings, left_out)
    choice, asked = confirm_plan(
        plan, settings.confirmation, settings.large_batches, context
    )
    summary.update({"documents": len(documents), "left_out": left_out,
                    "asked": asked, "choice": choice})
    if choice == PLAN_CHOICE_CANCEL:
        context.done("Cancelled", evidence_count=0)
        return ToolResult(
            f"The {template.title} was not written (cancelled).", (),
            summary,
        )
    tokens = (
        _config.QUICK_DOCUMENT_TOKENS if choice == PLAN_CHOICE_QUICK
        else settings.document_tokens
    )

    words = query_words(template_texts(template))
    notes: dict[int, DocumentNotes] = {}
    for document in documents:
        context.progress(
            f"Reading document {document.number} of {len(documents)}"
        )
        excerpt = document_excerpt(document, words, tokens)
        try:
            reply = context.ask_model(
                *map_prompt(template, document, excerpt), None
            )
            notes[document.number] = parse_notes(
                reply, template, document.number, excerpt
            )
        except Exception as error:  # One document must not stop the rest.
            notes[document.number] = DocumentNotes(
                document.number, failure=str(error) or type(error).__name__
            )
    failed = [n for n, found in notes.items() if found.failure]
    summary["failed"] = failed

    written: dict[str, str] = {}
    failures: list[str] = []
    for section in template.sections:
        context.progress(f"Writing {section.name}")
        try:
            if section.source == SECTION_SOURCE_NOTES:
                written[section.name] = _write_section(
                    template, section, notes, documents, settings, context
                )
            elif section.source == SECTION_SOURCE_PER_DOCUMENT:
                written[section.name] = per_document_markdown(
                    section, documents, notes
                )
            elif section.source == SECTION_SOURCE_DOCUMENTS:
                written[section.name] = documents_line(documents)
            elif section.source == SECTION_SOURCE_KEY_DOCUMENTS:
                lines = {
                    n: " ".join(found.notes.values())
                    for n, found in notes.items()
                }
                reply = context.ask_model(
                    *key_documents_prompt(section, documents, lines), None
                )
                most = section.items[1] if section.items else 3
                titles = {d.number: d.title for d in documents}
                written[section.name] = "\n".join(
                    f"- {titles[n]} [{n}]"
                    for n in chosen_numbers(reply, len(documents), most)
                )
        except Exception as error:  # Reported; the other sections go on.
            failures.append(f"{section.name}: {error}")
            written[section.name] = "(This section could not be written.)"

    headline = template.title
    if template.headline_characters:
        try:
            reply = context.ask_model(*headline_prompt(
                template,
                [(name, text) for name, text in written.items()],
            ), None)
            line = next((x.strip() for x in strip_thinking(reply)
                         .splitlines() if x.strip()), "")
            line = line.lstrip("# ").strip().rstrip(".")
            if line:
                headline = line[:template.headline_characters]
        except Exception as error:  # The title stands in.
            failures.append(f"headline: {error}")

    parts = [f"# {headline}"]
    for section in template.sections:
        text = written.get(section.name, "")
        if section.inline:
            parts.append(f"**{section.name}:** {text}")
        else:
            parts.append(f"## {section.name}\n\n{text}")
    report = "\n\n".join(parts)
    check = check_report(report, template, _config.TEMPLATE_RULES)
    final = f"{report}\n\n---\n\n{documents_list(documents)}\n"
    saved = save_output_file(
        context.project_root, f"{template.name}_project_report.md", final
    )
    problems = [*check.lengths, *failures]
    if failed:
        problems.append(
            "Not read: " + ", ".join(f"[{n}]" for n in failed) + "."
        )
    summary.update({
        "report": saved.as_posix(), "valid": check.valid and not failures,
        "estimated_tokens": estimate_tokens(final),
    })
    note = f"\n\n*Saved to `{saved.as_posix()}`.*"
    if problems:
        note += "\n\n*Check:* " + " ".join(problems)
    evidence = tuple(
        {
            "citation": document.number,
            "path": document.path,
            "start_line": document.passages[0].start_line,
            "end_line": document.passages[0].end_line,
            "symbol": None,
            "text": document.passages[0].text,
            "cell": getattr(document.passages[0], "cell", None),
        }
        for document in documents
    )
    context.done(
        f"Wrote the {template.title}", evidence_count=len(evidence)
    )
    return ToolResult(final.rstrip() + note, evidence, summary)
