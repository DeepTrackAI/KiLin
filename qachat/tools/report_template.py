"""Read Markdown report templates and check reports against them.

A report template is a Markdown file that anyone can read and edit. It
holds the structure of a report only: the headline, the sections in order,
their word or item limits, and the instruction for writing each one. Tools
such as ``generate_paper_summary`` write the content; this module reads the
template and checks the finished report in Python, so a report is not
trusted merely because the model was asked to follow the format.

Templates shipped with QAChat live in ``qachat/tools/templates/``, for
example ``research_briefing.md`` (the one-page Research Briefing format of
Nature-family journals). Each project has its own ``templates/`` folder,
next to ``sources/`` and ``outputs/`` and never indexed, which starts with a
copy of the shipped templates. Adding a ``.md`` file there adds a format;
deleting one removes it for that project only.

Templates are read fresh on every load, so edits apply at once. A template
that breaks a rule raises ``ValueError`` naming the template and, where
possible, the line. The checks count words and list items with simple
patterns; they do not judge whether the content is correct or relevant.

Key Features
------------

- **Front matter (optional)**

    ``key: value`` lines between two ``---`` lines at the top. Keys are
    ``title``, ``description``, ``max_words`` (all sections together), and
    ``reference`` (where the format comes from).

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

    ``{citation}`` is the paper's citation, copied by code from its
    metadata and never written by the model. ``{references}`` in a section's
    text means its items are chosen from the paper's own reference list and
    copied word for word; such a section needs an item limit.

- **Checks in Python**

    Count words and list items, and report missing, repeated, empty,
    unexpected, out-of-order, and too long or too short sections.

Module Structure
----------------

- ``SHIPPED_TEMPLATES_DIR``

    Folder of the templates shipped with QAChat.

- ``TEMPLATES_FOLDER``

    Name of the templates folder inside a project.

- ``LIMIT_TOLERANCE``

    Fraction by which word limits may be missed either way.

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

- ``templates_folder``

    Return a project's templates folder, creating it when missing.

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
>>> from qachat.tools.report_template import check_report, load_template
>>> template = load_template("research_briefing")
>>> template.headline_characters
90
>>> check = check_report("# Headline\\n\\n## The Bio\\nText.", template)
>>> check.valid
False

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Final, Literal, TypeAlias

__all__ = [
    "LIMIT_TOLERANCE",
    "SECTION_SOURCE_CHOICES",
    "SECTION_SOURCE_CITATION",
    "SECTION_SOURCE_NOTES",
    "SECTION_SOURCE_REFERENCES",
    "SHIPPED_TEMPLATES_DIR",
    "TEMPLATES_FOLDER",
    "ReportCheck",
    "ReportTemplate",
    "SectionSource",
    "TemplateSection",
    "check_report",
    "count_items",
    "count_words",
    "load_template",
    "parse_template",
    "section_problem",
    "template_names",
    "templates_folder",
]

# Templates shipped with QAChat; new projects get a copy.
SHIPPED_TEMPLATES_DIR: Final[Path] = Path(__file__).parent / "templates"
# A project's own templates, next to sources/ and outputs/; never indexed.
TEMPLATES_FOLDER: Final[str] = "templates"
# Word limits (of each section and of the whole report) allow this fraction
# either way, as editors do. Item limits are exact.
LIMIT_TOLERANCE: Final[float] = 0.1

# What a section is written from: the reading notes; the paper's citation,
# copied by code; or the paper's reference list, copied by code.
SectionSource: TypeAlias = Literal["notes", "citation", "references"]

SECTION_SOURCE_NOTES: Final[SectionSource] = "notes"
SECTION_SOURCE_CITATION: Final[SectionSource] = "citation"
SECTION_SOURCE_REFERENCES: Final[SectionSource] = "references"
SECTION_SOURCE_CHOICES: Final[tuple[SectionSource, ...]] = (
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_REFERENCES,
)

_FRONT_MATTER_KEYS: Final[tuple[str, ...]] = (
    "title",
    "description",
    "max_words",
    "reference",
)
# Placeholders a template may use in a section's text.
_PLACEHOLDER_NAMES: Final[frozenset[str]] = frozenset(
    ("citation", "references")
)
# Smallest ``max_words``: a lower total cannot hold a useful report.
_MIN_TOTAL_WORDS: Final[int] = 50

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
    source: "notes" | "citation" | "references"
        ``notes`` (written by the model from its reading notes),
        ``citation`` (the paper's citation, copied by code), or
        ``references`` (items copied from the paper's reference list).
    inline: bool
        Whether it is written as one labelled line (``**Name:** content``)
        instead of a ``##`` section.

    """

    name: str
    guidance: str = ""
    words: tuple[int, int] | None = None
    items: tuple[int, int] | None = None
    source: SectionSource = SECTION_SOURCE_NOTES
    inline: bool = False


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

    """

    name: str
    title: str
    description: str
    sections: tuple[TemplateSection, ...]
    headline_characters: int | None = None
    headline_guidance: str = ""
    max_total_words: int | None = None
    reference: str = ""


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
        """Whether the report follows the template (no problems found)."""

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


def templates_folder(project_root: Path) -> Path:
    """Return ``<project>/templates``, creating it with the shipped ones.

    Parameters
    ----------
    project_root: pathlib.Path
        The project folder.

    Returns
    -------
    pathlib.Path
        The project's templates folder. When it does not exist yet, it is
        created with a copy of every shipped template; an existing folder
        is left as the user keeps it (deleted templates stay deleted).

    """

    folder = Path(project_root) / TEMPLATES_FOLDER
    if not folder.exists():
        folder.mkdir(parents=True)
        for path in sorted(SHIPPED_TEMPLATES_DIR.glob("*.md")):
            shutil.copyfile(path, folder / path.name)

    return folder


def template_names(directory: Path = SHIPPED_TEMPLATES_DIR) -> list[str]:
    """List the templates in a folder.

    Parameters
    ----------
    directory: pathlib.Path, optional
        Template folder. Defaults to the shipped templates.

    Returns
    -------
    list[str]
        Template IDs (file names without ``.md``), sorted.

    """

    return sorted(path.stem for path in Path(directory).glob("*.md"))


def load_template(
    name: str, directory: Path = SHIPPED_TEMPLATES_DIR
) -> ReportTemplate:
    """Read and validate one template file.

    Parameters
    ----------
    name: str
        Template ID, such as ``"research_briefing"``.
    directory: pathlib.Path, optional
        Template folder, such as a project's ``templates/``. Defaults to
        the shipped templates.

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

    path = Path(directory) / f"{name}.md"
    if "/" in name or "\\" in name or not path.is_file():
        raise ValueError(
            f"Unknown report template {name!r}. Templates in {directory}: "
            f"{', '.join(template_names(directory)) or 'none'}."
        )

    return parse_template(path.read_text(encoding="utf-8"), name)


def parse_template(text: str, name: str) -> ReportTemplate:
    """Read and validate a template's Markdown text.

    Parameters
    ----------
    text: str
        The template, following the rules in this module's docstring.
    name: str
        Template ID, used in the result and in error messages.

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
    title = fields.get("title", name.replace("_", " ").title())
    headline_characters: int | None = None
    headline_guidance = ""
    sections: list[TemplateSection] = []
    # The heading being read, as (kind, name, limits, line number); its
    # guidance lines are collected in ``body`` until the next heading.
    current: tuple[str, str, dict[str, tuple[int, int]], int] | None = None
    body: list[str] = []

    def finish() -> None:
        """Store the heading being read, with its collected guidance."""

        nonlocal headline_guidance

        if current is None:
            return

        kind, label, limits, line = current
        guidance = " ".join(" ".join(body).split())
        if kind == "headline":
            headline_guidance = guidance
        else:
            sections.append(_section(name, line, label, limits, guidance))

    for number, raw in enumerate(lines[body_start:], start=body_start + 1):
        line = raw.strip()
        heading = _HEADING.match(line)
        top = _TITLE.match(line)
        labelled = _INLINE.match(line)

        if heading or labelled:
            finish()
            label, limits = _limits(
                name, number, (heading or labelled).group(1)
            )
            if heading:
                current, body = ("section", label, limits, number), []
            else:
                # A labelled line is complete on its own line.
                content = labelled.group(2).strip()
                sections.append(
                    _section(
                        name, number, label, limits, content, inline=True
                    )
                )
                current, body = None, []

        elif top:
            if current is not None or sections or headline_characters:
                raise ValueError(
                    f"{name}, line {number}: the '#' headline must come "
                    "before the sections, and only once."
                )

            label, limits = _limits(name, number, top.group(1))
            if set(limits) - {"characters"}:
                raise ValueError(
                    f"{name}, line {number}: a headline limit is written "
                    "like (max 90 characters)."
                )

            if "characters" in limits:
                headline_characters = limits["characters"][1]
                current, body = ("headline", label, limits, number), []
            else:
                title = label

        elif current is not None:
            body.append(line)

    finish()

    if not any(not section.inline for section in sections):
        raise ValueError(f"{name}: a template needs at least one ## section.")

    names = [section.name.casefold() for section in sections]
    for index, key in enumerate(names):
        if key in names[:index]:
            raise ValueError(
                f"{name}: the section {sections[index].name!r} appears "
                "twice."
            )

    total = fields.get("max_words")
    if total is not None and (
        not total.isdigit() or int(total) < _MIN_TOTAL_WORDS
    ):
        raise ValueError(
            f"{name}: max_words must be a whole number of at least "
            f"{_MIN_TOTAL_WORDS}."
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
) -> tuple[str, dict[str, tuple[int, int]]]:
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
    tuple[str, dict[str, tuple[int, int]]]
        The name with single spaces, and the limits by unit (``words``,
        ``items``, or ``characters``) as ``(low, high)``; ``max N`` gives
        ``(0, N)`` and ``N`` gives ``(N, N)``. A heading without brackets
        has no limits.

    Raises
    ------
    ValueError
        If the brackets hold something other than limits, or a minimum is
        larger than its maximum.

    """

    match = _LIMITS.match(heading.strip())
    if not match:
        return " ".join(heading.split()), {}

    limits: dict[str, tuple[int, int]] = {}
    for part in match.group(2).split(","):
        found = _RANGE.match(" ".join(part.split()))
        if not found:
            raise ValueError(
                f"{name}, line {number}: cannot read the limit "
                f"({match.group(2)}); write for example (60–150 words), "
                "(max 100 words), (2–3 items), or (max 90 characters)."
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

    return " ".join(match.group(1).split()), limits


def _section(
    name: str,
    number: int,
    label: str,
    limits: dict[str, tuple[int, int]],
    guidance: str,
    *,
    inline: bool = False,
) -> TemplateSection:
    """Build one section, checking its limits and placeholders.

    Parameters
    ----------
    name: str
        Template ID, for error messages.
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

    Returns
    -------
    TemplateSection
        The section; ``{citation}`` makes its source ``citation`` and
        otherwise ``{references}`` makes it ``references``.

    Raises
    ------
    ValueError
        If the name is empty or holds ``#``, the section has a character
        limit, a placeholder is unknown, a ``{references}`` section has no
        item limit, or a labelled line is empty.

    """

    if not label or "#" in label:
        raise ValueError(f"{name}, line {number}: give the section a name.")

    if "characters" in limits:
        raise ValueError(
            f"{name}, line {number}: only the headline has a character "
            "limit; sections use words or items."
        )

    placeholders = set(_PLACEHOLDER.findall(guidance))
    unknown = placeholders - _PLACEHOLDER_NAMES
    if unknown:
        raise ValueError(
            f"{name}, line {number}: unknown placeholder "
            f"{{{sorted(unknown)[0]}}}; use {{citation}} or {{references}}."
        )

    source = SECTION_SOURCE_NOTES
    if "citation" in placeholders:
        source = SECTION_SOURCE_CITATION
    elif "references" in placeholders:
        source = SECTION_SOURCE_REFERENCES
        if "items" not in limits:
            raise ValueError(
                f"{name}, line {number}: a section filled from "
                "{references} needs an item limit, such as (2–3 items)."
            )

    if inline and source != SECTION_SOURCE_CITATION and not guidance:
        raise ValueError(
            f"{name}, line {number}: write what the labelled line holds, or "
            "{citation}."
        )

    return TemplateSection(
        name=label,
        guidance=guidance,
        words=limits.get("words"),
        items=limits.get("items"),
        source=source,
        inline=inline,
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
    """Count the list items (``-``, ``*``, ``•``, or ``1.`` lines) in content.

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


def section_problem(section: TemplateSection, content: str) -> str | None:
    """Describe what is wrong with one section's content, if anything.

    Parameters
    ----------
    section: TemplateSection
        The template section.
    content: str
        The section's content, without its heading.

    Returns
    -------
    str | None
        For example "The Reason has 92 words; it should have 120-150.";
        None when the content is within its limits. Word limits allow
        ``LIMIT_TOLERANCE`` and apply to ``notes`` sections only; item
        limits are exact and apply to ``notes`` and ``references``
        sections. A ``citation`` section is only checked for being empty.

    """

    if not content.strip():
        return f"{section.name} is empty."

    if section.source == SECTION_SOURCE_CITATION:
        return None

    if section.words and section.source == SECTION_SOURCE_NOTES:
        words = count_words(content)
        low, high = section.words
        if not _within(words, low, high):
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


def check_report(markdown: str, template: ReportTemplate) -> ReportCheck:
    """Check a Markdown report against its template.

    Parameters
    ----------
    markdown: str
        The report, with a ``#`` first line, labelled lines, and
        ``## Section`` headings.
    template: ReportTemplate
        The template.

    Returns
    -------
    ReportCheck
        Missing, repeated, empty, out-of-order, and unexpected sections;
        sections outside their limits (as ``section_problem`` reports
        them); a total over ``max_total_words`` (with ``LIMIT_TOLERANCE``);
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
            problem = section_problem(section, content)
            if problem:
                lengths.append(problem)

    total = sum(
        count_words(content)
        for heading, content in found
        if heading.casefold() in expected
    )
    limit = template.max_total_words
    if limit and total > limit * (1 + LIMIT_TOLERANCE):
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


def _within(value: int, low: int, high: int) -> bool:
    """Return whether a count is within limits, allowing the tolerance.

    Parameters
    ----------
    value: int
        The count.
    low: int
        The limit's minimum.
    high: int
        The limit's maximum.

    Returns
    -------
    bool
        Whether ``value`` is within ``low`` and ``high`` widened by
        ``LIMIT_TOLERANCE`` either way.

    """

    return low * (1 - LIMIT_TOLERANCE) <= value <= high * (
        1 + LIMIT_TOLERANCE
    )
