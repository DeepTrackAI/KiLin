"""Read Markdown report templates and check reports against them.

A report template is a Markdown file that anyone can read and edit. It
holds the structure of a report only: the headline, the sections in order,
their word or item limits, and the instruction for writing each one. Tools
such as ``generate_paper_summary`` write the content; this module reads the
template and checks the finished report in Python, so a report is not
trusted merely because the model was asked to follow the format.

Shared by the tools that write reports from templates
(``generate_paper_summary`` for one document, ``project_summary`` for a
whole project). Each tool ships its own templates in its package and
passes its own limits as ``TemplateRules``. Each project has its own
``templates/`` folder, next to ``sources/`` and ``outputs/`` and never
indexed; adding a ``.md`` file there adds a format, deleting it removes the
format only from that project.

Templates are read fresh on every load, so edits apply at once. A template
that breaks a rule raises ``ValueError`` naming the template and, where
possible, the line. The checks count words and list items with simple
patterns; they do not judge whether the content is correct or relevant.

Key Features
------------

- **Front matter (optional)**

    ``key: value`` lines between two ``---`` lines at the top. Keys are
    ``title``, ``description``, ``max_words`` (all sections together),
    ``reference`` (where the format comes from), and ``scope``:
    ``document`` (the default: a report on one document) or ``project`` (a
    report on all the documents of a project).

- **Headline**

    A ``#`` line. With a limit, ``# Headline (max 90 characters)``, the
    model writes the headline; without one, the line is the report's fixed
    title.

- **Sections and limits**

    ``## Name (limits)`` lines, in the order the report uses; the text under
    a heading is the instruction for writing it. Limits are written in
    brackets at the end of a heading: ``(60–150 words)``, ``(max 100
    words)``, ``(2–3 items)``, or ``(3 items)``. A heading without brackets
    has no limit.

- **Labelled lines**

    ``**Label:** {citation}`` puts a one-line field in the report, such as
    the citation of a briefing.

- **Placeholders**

    In a document template, ``{citation}`` is the paper's citation, copied
    by code from its metadata and never written by the model, and
    ``{references}`` in a section's text means its items are chosen from
    the paper's own reference list and copied word for word; such a
    section needs an item limit. In a project template, ``{documents}``
    is the list of documents the report covers, written by code, and
    ``{key_documents}`` means a section's items are chosen from the
    project's documents (with an item limit).

- **Per-document sections (project templates)**

    ``## Name (per document)`` with a ``When:`` line (a condition) and a
    ``Do:`` line (what to write for each document that meets it, with an
    optional ``(max 40 words)``): the report lists one entry per matching
    document, assembled by code. ``(per document, table)`` with ``Do: |
    Dataset | Size |`` gives one table row per document instead. At
    most 26 such sections are allowed per template, because each maps to
    one of the letters A–Z in the document-reading protocol.

- **Sections covering every document (project templates)**

    ``## Scope (80-120 words, all documents)``: the section must cite
    every document the report reads.

- **Checks in Python**

    Count words and list items, and report missing, repeated, empty,
    unexpected, out-of-order, and too long or too short sections.

Module Structure
----------------

- ``TemplateRules``

    A tool's limits: how far word limits may be missed, and the smallest
    ``max_words``.

- ``SectionSource``

    Type alias for ``"notes"``, ``"citation"``, or ``"references"``.

- ``SECTION_SOURCE_NOTES``, ``SECTION_SOURCE_CITATION``, and
  ``SECTION_SOURCE_REFERENCES``

    Canonical string values for where a section's content comes from.

- ``SECTION_SOURCE_CHOICES``

    Ordered collection of the section sources.

- ``TemplateSection``

    One section or labelled line of a template.

- ``ReportTemplate``

    A validated report template.

- ``ReportCheck``

    The result of checking a report against its template.

- ``template_names``

    List the templates in a folder.

- ``load_template``

    Read and validate one template file.

- ``parse_template``

    Read and validate a template's Markdown text.

- ``count_words``

    Count the words in section content, ignoring citations and markup.

- ``count_items``

    Count the list items in section content.

- ``section_problem``

    Describe what is wrong with one section's content, if anything.

- ``check_report``

    Check a Markdown report against its template.

Private helpers read the front matter, the limits, and the report's
sections; they are not part of the public API.

Examples
--------
>>> from kilin.tools._shared.report_templates import (
...     TemplateRules, check_report, parse_template,
... )
>>> rules = TemplateRules(limit_tolerance=0.1, min_total_words=50)
>>> template = parse_template(
...     "# Headline (max 90 characters)\\n\\n## The Bio (10-20 words)\\n"
...     "Who they are.\\n", "bio", rules)
>>> template.headline_characters
90
>>> check_report("# Headline\\n\\n## The Bio\\nText.", template, rules).valid
False

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias


__all__ = [
    "SECTION_SOURCE_CHOICES",
    "SCOPE_DOCUMENT",
    "SCOPE_PROJECT",
    "SECTION_SOURCE_CITATION",
    "SECTION_SOURCE_DOCUMENTS",
    "SECTION_SOURCE_KEY_DOCUMENTS",
    "SECTION_SOURCE_NOTES",
    "SECTION_SOURCE_PER_DOCUMENT",
    "SECTION_SOURCE_REFERENCES",
    "ReportCheck",
    "ReportTemplate",
    "SectionSource",
    "TemplateRules",
    "TemplateScope",
    "TemplateSection",
    "check_report",
    "count_items",
    "count_words",
    "load_template",
    "parse_template",
    "section_problem",
    "template_names",
]


# What a section is written from: the reading notes; the paper's citation,
# copied by code; the paper's reference list, copied by code; or, in a
# project report, the documents covered (code), items chosen from them, or
# one entry per matching document (assembled by code).
SectionSource: TypeAlias = Literal[
    "notes", "citation", "references", "documents", "key_documents",
    "per_document",
]

SECTION_SOURCE_NOTES: Final[SectionSource] = "notes"
SECTION_SOURCE_CITATION: Final[SectionSource] = "citation"
SECTION_SOURCE_REFERENCES: Final[SectionSource] = "references"
SECTION_SOURCE_DOCUMENTS: Final[SectionSource] = "documents"
SECTION_SOURCE_KEY_DOCUMENTS: Final[SectionSource] = "key_documents"
SECTION_SOURCE_PER_DOCUMENT: Final[SectionSource] = "per_document"
SECTION_SOURCE_CHOICES: Final[tuple[SectionSource, ...]] = (
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_REFERENCES,
    SECTION_SOURCE_DOCUMENTS,
    SECTION_SOURCE_KEY_DOCUMENTS,
    SECTION_SOURCE_PER_DOCUMENT,
)
# What a template reports on: one document, or a whole project.
TemplateScope: TypeAlias = Literal["document", "project"]
SCOPE_DOCUMENT: Final[TemplateScope] = "document"
SCOPE_PROJECT: Final[TemplateScope] = "project"

_FRONT_MATTER_KEYS: Final[tuple[str, ...]] = (
    "title",
    "description",
    "max_words",
    "reference",
    "scope",
)
# Placeholders a template may use in a section's text, by scope.
_PLACEHOLDER_NAMES: Final[dict[str, frozenset[str]]] = {
    "document": frozenset(("citation", "references")),
    "project": frozenset(("documents", "key_documents")),
}
# Words in a heading's brackets marking a per-document section.
_PER_DOCUMENT: Final[str] = "per document"
# Map responses identify per-document sections with letters A through Z.
_MAX_PER_DOCUMENT_SECTIONS: Final[int] = 26
_TABLE: Final[str] = "table"
_ALL_DOCUMENTS: Final[str] = "all documents"
# "When:" and "Do:" lines of a per-document section; a "Do:" line's
# optional word limit for each entry: "(max 40 words)", "(20-40 words)".
_WHEN = re.compile(r"^when\s*:\s*(.*)$", re.I)
_DO = re.compile(r"^do\s*:\s*(.*)$", re.I)
_ENTRY_LIMIT = re.compile(
    r"\s*\((?:max\s+(\d+)|(\d+)\s*[–-]\s*(\d+))\s+words\)\.?\s*$",
    re.I,
)

# A heading's name and its bracketed limits: "The Bio (60–150 words)".
_LIMITS = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
# One limit: "60–150 words", "max 100 words", "3 items", "max 90 characters".
_RANGE = re.compile(
    r"^(?:(\d+)\s*[–-]\s*(\d+)|max\s+(\d+)|(\d+))\s+"
    r"(words|items|characters)$",
    flags=re.I,
)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
# A "## Section" heading and a "# Headline" line; closing hashes are dropped.
_HEADING = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_TITLE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
# A labelled line: "**This is a summary of:** content".
_INLINE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
# A list item: "- ", "* ", "• ", "1. ", or "1) " followed by text.
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
# Passage citations such as "[3]" or "[2, 5]", which are not words.
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


@dataclass(frozen=True, slots=True)
class TemplateRules:
    """A tool's limits for reading templates and checking reports.

    Each tool owns these values in its ``defaults.json``.

    Attributes
    ----------
    limit_tolerance: float
        Fraction by which word limits (of a section and of the whole
        report) may be missed either way, as editors allow; item limits
        are exact.
    min_total_words: int
        Smallest ``max_words`` a template may set: a lower total cannot
        hold a useful report.

    """

    limit_tolerance: float
    min_total_words: int


@dataclass(frozen=True, slots=True)
class TemplateSection:
    """One section or labelled line of a template.

    Attributes
    ----------
    name: str
        The heading text, without its limits.
    guidance: str
        The instruction under the heading (for a labelled line, the text
        after the label).
    words: tuple[int, int] | None
        Allowed word count of the content, or None for no limit.
    items: tuple[int, int] | None
        Allowed number of list items, or None for no limit.
    source: SectionSource
        ``notes`` (written by the model from its reading notes),
        ``citation`` (the paper's citation, copied by code),
        ``references`` (items copied from the paper's reference list),
        ``documents`` (the documents a project report covers, by code),
        ``key_documents`` (items chosen from the project's documents), or
        ``per_document`` (one entry per document meeting ``when``).
    inline: bool
        Whether it is written as one labelled line (``**Name:** content``)
        instead of a ``##`` section.
    when: str
        A per-document section's condition (``When:``), or ``""``.
    do: str
        What a per-document section writes for each matching document
        (``Do:``, without its word limit), or ``""``; for a table, the
        header row.
    columns: tuple[str, ...]
        A per-document table's column names, or empty. For a
        per-document section, ``words`` limits each entry.
    all_documents: bool
        Whether a project report's section must cite every document with
        a note on it (``all documents`` in its heading's brackets).

    """

    name: str
    guidance: str = ""
    words: tuple[int, int] | None = None
    items: tuple[int, int] | None = None
    source: SectionSource = SECTION_SOURCE_NOTES
    inline: bool = False
    when: str = ""
    do: str = ""
    columns: tuple[str, ...] = ()
    all_documents: bool = False


@dataclass(frozen=True, slots=True)
class ReportTemplate:
    """A validated report template.

    Attributes
    ----------
    name: str
        Template ID: the file name without ``.md``.
    title: str
        What the report is called, such as "Research Briefing".
    description: str
        What the template is for.
    sections: tuple[TemplateSection, ...]
        Sections and labelled lines, in the report's order.
    headline_characters: int | None
        When set, the report starts with a model-written ``# headline`` of
        at most this many characters; otherwise with ``# title``.
    headline_guidance: str
        What the headline should be.
    max_total_words: int | None
        Most words in all sections together (for example to fit one page),
        or None for no limit.
    reference: str
        Where the format comes from.
    scope: TemplateScope
        ``document`` for a report on one document, ``project`` for a
        report on a whole project.

    """

    name: str
    title: str
    description: str
    sections: tuple[TemplateSection, ...]
    headline_characters: int | None = None
    headline_guidance: str = ""
    max_total_words: int | None = None
    reference: str = ""
    scope: TemplateScope = SCOPE_DOCUMENT


@dataclass(frozen=True, slots=True)
class ReportCheck:
    """The result of checking a report against its template.

    Attributes
    ----------
    missing: tuple[str, ...]
        Sections without a heading.
    duplicated: tuple[str, ...]
        Sections whose heading appears more than once.
    empty: tuple[str, ...]
        Sections with a heading but no content.
    out_of_order: bool
        Whether the template's sections appear in a different order.
    unexpected: tuple[str, ...]
        ``##`` headings that are not template sections.
    lengths: tuple[str, ...]
        Sections outside their word or item limits, and a report over its
        total word limit, each described in one sentence.
    headline: str | None
        What is wrong with the headline, if anything.

    """

    missing: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()
    empty: tuple[str, ...] = ()
    out_of_order: bool = False
    unexpected: tuple[str, ...] = ()
    lengths: tuple[str, ...] = ()
    headline: str | None = None

    @property
    def valid(self) -> bool:
        """Whether the report follows the template (no problems found).

        Returns
        -------
        bool
            True when the report follows the template (no problems found).

        """

        return not self.problems()

    def problems(self) -> list[str]:
        """Describe each problem in one sentence, for repairs and users.

        Returns
        -------
        list[str]
            Headline, missing, repeated, empty, order, unexpected, and
            length problems, in this order; empty when the report is valid.

        """

        found: list[str] = []
        if self.headline:
            found.append(self.headline)
        if self.missing:
            found.append("Missing sections: " + ", ".join(self.missing) + ".")
        if self.duplicated:
            found.append(
                "Sections that appear more than once: "
                + ", ".join(self.duplicated)
                + "."
            )
        if self.empty:
            found.append("Empty sections: " + ", ".join(self.empty) + ".")
        if self.out_of_order:
            found.append("The sections are not in the template's order.")
        if self.unexpected:
            found.append(
                "Headings that are not template sections: "
                + ", ".join(self.unexpected)
                + "."
            )
        found.extend(self.lengths)

        return found


def template_names(directory: Path) -> list[str]:
    """List the templates in a folder.

    Parameters
    ----------
    directory: pathlib.Path
        Explicit template folder, usually the project-owned directory.

    Returns
    -------
    list[str]
        Template IDs (file names without ``.md``), sorted; not
        ``README.md``, which explains the format.

    Raises
    ------
    ValueError
        If the template directory itself is a symbolic link.

    """

    folder = Path(directory)
    if folder.is_symlink():
        raise ValueError(f"Template folder cannot be a link: {folder}")
    return sorted(
        path.stem
        for path in folder.glob("*.md")
        if path.is_file() and not path.is_symlink() and path.stem
        and path.stem.casefold() != "readme"  # Explains the format.
    )


def load_template(
    name: str, directory: Path, rules: TemplateRules
) -> ReportTemplate:
    """Read and validate one template file.

    Parameters
    ----------
    name: str
        Template ID, such as ``"research_briefing"``.
    directory: pathlib.Path
        Explicit project-owned template folder; never defaults to the
        packaged templates.
    rules: TemplateRules
        The calling tool's limits.

    Returns
    -------
    ReportTemplate
        The template, read fresh so edits apply at once.

    Raises
    ------
    ValueError
        If the template does not exist, its name contains a path separator,
        or it breaks a rule; the message names the line.

    """

    folder = Path(directory)
    if folder.is_symlink():
        raise ValueError(f"Template folder cannot be a link: {folder}")
    valid_name = (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and not any(char in name for char in ("/", "\\", "\x00"))
    )
    path = folder / f"{name}.md" if valid_name else None
    if path is None or path.is_symlink() or not path.is_file():
        raise ValueError(
            f"Unknown report template {name!r}. Templates in {directory}: "
            f"{', '.join(template_names(directory)) or 'none'}."
        )

    return parse_template(path.read_text(encoding="utf-8"), name, rules)


def parse_template(
    text: str, name: str, rules: TemplateRules
) -> ReportTemplate:
    """Read and validate a template's Markdown text.

    Parameters
    ----------
    text: str
        The template, following the rules in this module's docstring.
    name: str
        Template ID, used in the result and in error messages.
    rules: TemplateRules
        The calling tool's limits (the smallest ``max_words``).

    Returns
    -------
    ReportTemplate
        The validated template. Without front matter ``title`` or a ``#``
        line, the title is the ID in title case.

    Raises
    ------
    ValueError
        If a rule is broken, as ``"<name>, line <n>: <problem>"`` (or
        ``"<name>: <problem>"`` for rules about the whole template).

    """

    lines = text.splitlines()
    fields, body_start = _front_matter(lines, name)
    scope = fields.get("scope", SCOPE_DOCUMENT).casefold()
    if scope not in _PLACEHOLDER_NAMES:
        raise ValueError(
            f"{name}: scope must be document (a report on one document) or "
            "project (a report on a whole project)."
        )
    title = fields.get("title", name.replace("_", " ").title())
    headline_characters: int | None = None
    headline_guidance = ""
    sections: list[TemplateSection] = []
    # The heading being read, as (kind, name, limits, line number, marks);
    # its guidance lines are collected in ``body`` until the next heading.
    current: tuple[
        str, str, dict[str, tuple[int, int]], int, frozenset[str]
    ] | None = None
    body: list[str] = []

    def finish() -> None:
        """Store the heading being read, with its collected guidance."""

        nonlocal headline_guidance

        if current is None:
            return

        kind, label, limits, line, marks = current
        if kind == "headline":
            headline_guidance = " ".join(" ".join(body).split())
        elif _PER_DOCUMENT in marks:
            sections.append(
                _per_document(name, scope, line, label, limits, marks, body)
            )
        else:
            guidance = " ".join(" ".join(body).split())
            sections.append(
                _section(name, scope, line, label, limits, guidance,
                         marks=marks)
            )

    for number, raw in enumerate(lines[body_start:], start=body_start + 1):
        line = raw.strip()
        heading = _HEADING.match(line)
        top = _TITLE.match(line)
        labelled = _INLINE.match(line)

        if heading or labelled:
            finish()
            label, limits, marks = _limits(
                name, number, (heading or labelled).group(1)
            )
            if heading:
                current, body = (
                    ("section", label, limits, number, marks), []
                )
            else:
                if marks:
                    raise ValueError(
                        f"{name}, line {number}: only ## sections can be "
                        "per document."
                    )
                # A labelled line is complete on its own line.
                content = labelled.group(2).strip()
                sections.append(
                    _section(
                        name, scope, number, label, limits, content,
                        inline=True,
                    )
                )
                current, body = None, []

        elif top:
            if current is not None or sections or headline_characters:
                raise ValueError(
                    f"{name}, line {number}: the '#' headline must come "
                    "before the sections, and only once."
                )

            label, limits, marks = _limits(name, number, top.group(1))
            if set(limits) - {"characters"} or marks:
                raise ValueError(
                    f"{name}, line {number}: a headline limit is written "
                    "like (max 90 characters)."
                )

            if "characters" in limits:
                headline_characters = limits["characters"][1]
                current, body = (
                    ("headline", label, limits, number, marks), []
                )
            else:
                title = label

        elif current is not None:
            body.append(line)

    finish()

    if not any(not section.inline for section in sections):
        raise ValueError(f"{name}: a template needs at least one ## section.")

    if sum(
        section.source == SECTION_SOURCE_PER_DOCUMENT for section in sections
    ) > _MAX_PER_DOCUMENT_SECTIONS:
        raise ValueError(
            f"{name}: a project template can contain at most "
            f"{_MAX_PER_DOCUMENT_SECTIONS} per-document sections."
        )

    names = [section.name.casefold() for section in sections]
    for index, key in enumerate(names):
        if key in names[:index]:
            raise ValueError(
                f"{name}: the section {sections[index].name!r} appears "
                "twice."
            )

    total = fields.get("max_words")
    if total is not None and (
        not total.isdigit() or int(total) < rules.min_total_words
    ):
        raise ValueError(
            f"{name}: max_words must be a whole number of at least "
            f"{rules.min_total_words}."
        )

    return ReportTemplate(
        name=name,
        title=title,
        description=fields.get("description", ""),
        sections=tuple(sections),
        headline_characters=headline_characters,
        headline_guidance=headline_guidance,
        max_total_words=int(total) if total else None,
        reference=fields.get("reference", ""),
        scope=scope,
    )


def _front_matter(lines: list[str], name: str) -> tuple[dict[str, str], int]:
    """Read ``key: value`` lines between two ``---`` lines at the top.

    Parameters
    ----------
    lines: list[str]
        The template's lines.
    name: str
        Template ID, for error messages.

    Returns
    -------
    tuple[dict[str, str], int]
        The fields (keys in lower case, values with single spaces) and the
        index of the first line after the front matter; ``({}, 0)`` when
        the first nonblank line is not ``---``.

    Raises
    ------
    ValueError
        If a line is not ``key: value`` with a known key, or the closing
        ``---`` is missing.

    """

    start = next((i for i, line in enumerate(lines) if line.strip()), None)
    if start is None or lines[start].strip() != "---":
        return {}, 0

    fields: dict[str, str] = {}
    for index in range(start + 1, len(lines)):
        line = lines[index].strip()
        if line == "---":
            return fields, index + 1

        if not line:
            continue

        key, colon, value = line.partition(":")
        key = key.strip().lower()
        if not colon or key not in _FRONT_MATTER_KEYS:
            raise ValueError(
                f"{name}, line {index + 1}: front matter lines are "
                f"'key: value' with a key among "
                f"{', '.join(_FRONT_MATTER_KEYS)}."
            )
        fields[key] = " ".join(value.split())

    raise ValueError(f"{name}: the front matter has no closing '---'.")


def _limits(
    name: str, number: int, heading: str
) -> tuple[str, dict[str, tuple[int, int]], frozenset[str]]:
    """Split a heading such as "The Bio (60–150 words)" into name and limits.

    Parameters
    ----------
    name: str
        Template ID, for error messages.
    number: int
        Line number of the heading, for error messages.
    heading: str
        The heading text, without its ``#`` marks.

    Returns
    -------
    tuple[str, dict[str, tuple[int, int]], frozenset[str]]
        The name with single spaces, the limits by unit (``words``,
        ``items``, or ``characters``) as ``(low, high)`` (``max N`` gives
        ``(0, N)`` and ``N`` gives ``(N, N)``), and the marks written with
        them (``per document``, ``table``, ``all documents``). A heading
        without brackets has none.

    Raises
    ------
    ValueError
        If the brackets hold something other than limits, or a minimum is
        larger than its maximum.

    """

    match = _LIMITS.match(heading.strip())
    if not match:
        return " ".join(heading.split()), {}, frozenset()

    limits: dict[str, tuple[int, int]] = {}
    marks: set[str] = set()
    for part in match.group(2).split(","):
        word = " ".join(part.split()).casefold()
        if word in (_PER_DOCUMENT, _TABLE, _ALL_DOCUMENTS):
            marks.add(word)
            continue
        found = _RANGE.match(" ".join(part.split()))
        if not found:
            raise ValueError(
                f"{name}, line {number}: cannot read the limit "
                f"({match.group(2)}); write for example (60–150 words), "
                "(max 100 words), (2–3 items), (max 90 characters), or "
                "(per document)."
            )

        low, high, most, exact, unit = found.groups()
        if most:
            low, high = "0", most
        elif exact:
            low = high = exact

        if int(low) > int(high):
            raise ValueError(
                f"{name}, line {number}: the limit's minimum is larger than "
                "its maximum."
            )
        limits[unit.lower()] = (int(low), int(high))

    if _TABLE in marks and _PER_DOCUMENT not in marks:
        raise ValueError(
            f"{name}, line {number}: a table section is written (per "
            "document, table)."
        )
    if _ALL_DOCUMENTS in marks and _PER_DOCUMENT in marks:
        raise ValueError(
            f"{name}, line {number}: a per-document section already covers "
            "every document; leave out (all documents)."
        )
    return " ".join(match.group(1).split()), limits, frozenset(marks)


def _section(
    name: str,
    scope: str,
    number: int,
    label: str,
    limits: dict[str, tuple[int, int]],
    guidance: str,
    *,
    inline: bool = False,
    marks: frozenset[str] = frozenset(),
) -> TemplateSection:
    """Build one section, checking its limits and placeholders.

    Parameters
    ----------
    name: str
        Template ID, for error messages.
    scope: str
        The template's scope, which decides the placeholders allowed.
    number: int
        Line number of the heading, for error messages.
    label: str
        The section's name, without limits.
    limits: dict[str, tuple[int, int]]
        Limits from ``_limits``.
    guidance: str
        The instruction for the section, or a labelled line's content.
    inline: bool, optional
        Whether this is a labelled line. Defaults to ``False``.
    marks: frozenset[str], optional
        Marks from the heading's brackets; ``all documents`` is allowed on
        a project template's section written from notes. Defaults to none.

    Returns
    -------
    TemplateSection
        The section; ``{citation}`` or ``{documents}`` makes its source
        ``citation`` or ``documents``, and otherwise ``{references}`` or
        ``{key_documents}`` makes it ``references`` or ``key_documents``.

    Raises
    ------
    ValueError
        If the name is empty or holds ``#``, the section has a character
        limit, a placeholder is unknown or of the other scope, a section
        choosing items (``{references}``, ``{key_documents}``) has no item
        limit, or a labelled line is empty.

    """

    if not label or "#" in label:
        raise ValueError(f"{name}, line {number}: give the section a name.")

    if "characters" in limits:
        raise ValueError(
            f"{name}, line {number}: only the headline has a character "
            "limit; sections use words or items."
        )

    placeholders = set(_PLACEHOLDER.findall(guidance))
    allowed = _PLACEHOLDER_NAMES[scope]
    unknown = placeholders - allowed
    if unknown:
        other = next(
            (other for other, names in _PLACEHOLDER_NAMES.items()
             if sorted(unknown)[0] in names),
            None,
        )
        use = " or ".join(f"{{{item}}}" for item in sorted(allowed))
        raise ValueError(
            f"{name}, line {number}: "
            + (
                f"{{{sorted(unknown)[0]}}} is for {other} templates"
                if other else
                f"unknown placeholder {{{sorted(unknown)[0]}}}"
            )
            + f"; use {use}."
        )

    source = SECTION_SOURCE_NOTES
    if placeholders & {"citation", "documents"}:
        source = (
            SECTION_SOURCE_CITATION if "citation" in placeholders
            else SECTION_SOURCE_DOCUMENTS
        )
    elif placeholders & {"references", "key_documents"}:
        chosen = sorted(placeholders & {"references", "key_documents"})[0]
        source = (
            SECTION_SOURCE_REFERENCES if chosen == "references"
            else SECTION_SOURCE_KEY_DOCUMENTS
        )
        if "items" not in limits:
            raise ValueError(
                f"{name}, line {number}: a section filled from "
                f"{{{chosen}}} needs an item limit, such as (2–3 items)."
            )

    if inline and source not in (
        SECTION_SOURCE_CITATION, SECTION_SOURCE_DOCUMENTS
    ) and not guidance:
        raise ValueError(
            f"{name}, line {number}: write what the labelled line holds, or "
            + ("{citation}." if scope == SCOPE_DOCUMENT else "{documents}.")
        )

    if _ALL_DOCUMENTS in marks and (
        scope != SCOPE_PROJECT or source != SECTION_SOURCE_NOTES
    ):
        raise ValueError(
            f"{name}, line {number}: (all documents) is for sections of "
            "project templates written from the documents' notes."
        )
    return TemplateSection(
        name=label,
        guidance=guidance,
        words=limits.get("words"),
        items=limits.get("items"),
        source=source,
        inline=inline,
        all_documents=_ALL_DOCUMENTS in marks,
    )


def _per_document(
    name: str,
    scope: str,
    number: int,
    label: str,
    limits: dict[str, tuple[int, int]],
    marks: frozenset[str],
    body: list[str],
) -> TemplateSection:
    """Build one per-document section from its ``When:`` and ``Do:`` lines.

    Parameters
    ----------
    name: str
        Template ID, for error messages.
    scope: str
        The template's scope; must be ``project``.
    number: int
        Line number of the heading, for error messages.
    label: str
        The section's name.
    limits: dict[str, tuple[int, int]]
        Limits from the heading; none are allowed.
    marks: frozenset[str]
        ``per document``, and ``table`` for a table.
    body: list[str]
        The lines under the heading.

    Returns
    -------
    TemplateSection
        Source ``per_document``, with ``when``, ``do`` (without its word
        limit), the entry word limit as ``words``, the table's ``columns``,
        and the other lines as ``guidance``.

    Raises
    ------
    ValueError
        If the template is not a project template, the heading has limits,
        ``When:`` or ``Do:`` is missing or empty, a table's ``Do:`` is not
        a header row with at least one column, or the section uses a
        placeholder.

    """

    where = f"{name}, line {number}"
    if scope != SCOPE_PROJECT:
        raise ValueError(
            f"{where}: per-document sections are for project templates "
            "(scope: project)."
        )
    if limits:
        raise ValueError(
            f"{where}: a per-document section has no limits in its heading; "
            "limit each entry on its Do: line, such as (max 40 words)."
        )
    when = do = ""
    other: list[str] = []
    for line in body:
        if match := _WHEN.match(line):
            when = " ".join(match.group(1).split())
        elif match := _DO.match(line):
            do = " ".join(match.group(1).split())
        else:
            other.append(line)
    if not when or not do:
        raise ValueError(
            f"{where}: a per-document section needs a When: line (which "
            "documents) and a Do: line (what to write for each)."
        )
    guidance = " ".join(" ".join(other).split())
    if _PLACEHOLDER.search(" ".join((when, do, guidance))):
        raise ValueError(
            f"{where}: a per-document section has no placeholders."
        )
    words = None
    if found := _ENTRY_LIMIT.search(do):
        most, low, high = found.groups()
        words = (0, int(most)) if most else (int(low), int(high))
        do = do[:found.start()].strip()
    columns: tuple[str, ...] = ()
    if _TABLE in marks:
        columns = tuple(
            cell.strip() for cell in do.strip().strip("|").split("|")
            if cell.strip()
        )
        if not do.startswith("|") or not columns:
            raise ValueError(
                f"{where}: a table's Do: line names its columns, such as "
                "Do: | Dataset | Size | Licence |"
            )
    return TemplateSection(
        name=label,
        guidance=guidance,
        words=words,
        source=SECTION_SOURCE_PER_DOCUMENT,
        when=when,
        do=do,
        columns=columns,
    )


def count_words(text: str) -> int:
    """Count the words in section content, ignoring citations and markup.

    Parameters
    ----------
    text: str
        Section content.

    Returns
    -------
    int
        Number of words: runs of letters, digits, ``_``, apostrophes, and
        hyphens, after removing ``[n]`` passage citations.

    """

    return len(re.findall(r"[\w'’-]+", _CITATION.sub("", text)))


def count_items(text: str) -> int:
    """Count list items (``-``, ``*``, ``•``, or ``1.``) in content.

    Parameters
    ----------
    text: str
        Section content.

    Returns
    -------
    int
        Number of list items.

    """

    return sum(1 for line in text.splitlines() if _ITEM.match(line))


def section_problem(
    section: TemplateSection, content: str, rules: TemplateRules
) -> str | None:
    """Describe what is wrong with one section's content, if anything.

    Parameters
    ----------
    section: TemplateSection
        The template section.
    content: str
        The section's content, without its heading.
    rules: TemplateRules
        The calling tool's limits.

    Returns
    -------
    str | None
        For example "The Reason has 92 words; it should have 120-150.";
        None when the content is within its limits. Word limits allow
        ``rules.limit_tolerance`` and apply to ``notes`` sections only; item
        limits are exact and apply to ``notes``, ``references`` and
        ``key_documents`` sections. ``citation``, ``documents`` and
        ``per_document`` sections, written by code, are only checked for
        being empty.

    """

    if not content.strip():
        return f"{section.name} is empty."

    if section.source in (
        SECTION_SOURCE_CITATION, SECTION_SOURCE_DOCUMENTS,
        SECTION_SOURCE_PER_DOCUMENT,
    ):
        return None

    if section.words and section.source == SECTION_SOURCE_NOTES:
        words = count_words(content)
        low, high = section.words
        if not _within(words, low, high, rules.limit_tolerance):
            return (
                f"{section.name} has {words} words; it should have "
                f"{low}-{high}."
            )

    if section.items:
        items = count_items(content)
        low, high = section.items
        if not low <= items <= high:
            return (
                f"{section.name} lists {items} items; it should list "
                f"{low}-{high}."
            )

    return None


def check_report(
    markdown: str, template: ReportTemplate, rules: TemplateRules
) -> ReportCheck:
    """Check a Markdown report against its template.

    Parameters
    ----------
    markdown: str
        The report, with a ``#`` first line, labelled lines, and
        ``## Section`` headings.
    template: ReportTemplate
        The template.
    rules: TemplateRules
        The calling tool's limits.

    Returns
    -------
    ReportCheck
        Missing, repeated, empty, out-of-order, and unexpected sections;
        sections outside their limits (as ``section_problem`` reports
        them); a total over ``max_total_words`` (with
        ``rules.limit_tolerance``; per-document sections do not count);
        and headline problems. Heading text is compared ignoring case and
        surrounding spaces.

    """

    expected = {
        section.name.casefold(): section for section in template.sections
    }
    inline = {
        section.name.casefold() for section in template.sections
        if section.inline
    }
    found = _report_sections(markdown, inline)

    seen = [heading.casefold() for heading, _ in found]
    known = [heading for heading in seen if heading in expected]
    missing = tuple(
        section.name
        for section in template.sections
        if section.name.casefold() not in seen
    )
    duplicated = tuple(
        expected[key].name
        for key in dict.fromkeys(known)
        if known.count(key) > 1
    )
    empty = tuple(
        expected[heading.casefold()].name
        for heading, content in found
        if heading.casefold() in expected and not content
    )
    order = [section.name.casefold() for section in template.sections]
    positions = [order.index(key) for key in dict.fromkeys(known)]
    unexpected = tuple(
        heading for heading, _ in found if heading.casefold() not in expected
    )

    lengths: list[str] = []
    for heading, content in found:
        section = expected.get(heading.casefold())
        if section is not None and content:
            problem = section_problem(section, content, rules)
            if problem:
                lengths.append(problem)

    # Per-document sections grow with the project; they do not count.
    total = sum(
        count_words(content)
        for heading, content in found
        if heading.casefold() in expected
        and expected[heading.casefold()].source
        != SECTION_SOURCE_PER_DOCUMENT
    )
    limit = template.max_total_words
    if limit and total > limit * (1 + rules.limit_tolerance):
        lengths.append(
            f"The report has {total} words in its sections; it may have at "
            f"most {limit}."
        )

    headline = None
    if template.headline_characters:
        first_line = next(
            (line.strip() for line in markdown.splitlines() if line.strip()),
            "",
        )
        match = _TITLE.match(first_line)
        if not match:
            headline = "The report must start with a '# ' headline."
        elif len(match.group(1)) > template.headline_characters:
            headline = (
                f"The headline has {len(match.group(1))} characters; it "
                f"may have at most {template.headline_characters}."
            )

    return ReportCheck(
        missing=missing,
        duplicated=duplicated,
        empty=tuple(dict.fromkeys(empty)),
        out_of_order=positions != sorted(positions),
        unexpected=unexpected,
        lengths=tuple(dict.fromkeys(lengths)),
        headline=headline,
    )


def _report_sections(
    markdown: str, inline: set[str]
) -> list[tuple[str, str]]:
    """Split a report into its sections and labelled lines.

    Parameters
    ----------
    markdown: str
        The report.
    inline: set[str]
        Case-folded names of the template's labelled lines. Other
        ``**Label:**`` lines stay part of the section they are in.

    Returns
    -------
    list[tuple[str, str]]
        ``(heading, content)`` pairs in report order, with stripped
        content. Text before the first heading (the headline) is left out.

    """

    found: list[tuple[str, str]] = []
    current: str | None = None
    body: list[str] = []
    for line in markdown.splitlines():
        match = _HEADING.match(line.strip())
        labelled = _INLINE.match(line.strip())
        if labelled and labelled.group(1).strip().casefold() in inline:
            # "**This is a summary of:** citation" is a one-line section.
            if current is not None:
                found.append((current, "\n".join(body).strip()))
            found.append(
                (labelled.group(1).strip(), labelled.group(2).strip())
            )
            current, body = None, []
        elif match:
            if current is not None:
                found.append((current, "\n".join(body).strip()))
            current, body = match.group(1).strip(), []
        elif current is not None:
            body.append(line)

    if current is not None:
        found.append((current, "\n".join(body).strip()))

    return found


def _within(value: int, low: int, high: int, tolerance: float) -> bool:
    """Return whether a count is within limits, allowing the tolerance.

    Parameters
    ----------
    value: int
        The count.
    low: int
        The limit's minimum.
    high: int
        The limit's maximum.
    tolerance: float
        Fraction by which the limits may be missed either way.

    Returns
    -------
    bool
        Whether ``value`` is within ``low`` and ``high`` widened by
        ``tolerance`` either way.

    """

    return low * (1 - tolerance) <= value <= high * (1 + tolerance)
