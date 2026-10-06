"""Report templates: the structure a generated report must follow.

A template is a JSON file in this folder (``<name>.json``) that lists a
report's sections in order, with guidance and optional limits: a word range
for prose, an item range for lists, and a model-written headline with a
character limit. Templates hold structure only; tools such as
``generate_paper_summary`` write the content, and ``check_report`` verifies
the structure and limits in Python, so a report is not trusted merely
because the model was asked to follow the format.

Key Features
------------

- **Templates as data**

    ``load_template`` reads and validates one template; ``template_names``
    lists them. Add a JSON file to add a template.

- **Deterministic checks**

    ``check_report`` finds missing, repeated, empty, out-of-order, too long
    or too short sections, and a missing or overlong headline.

Module Structure
----------------

- ``TemplateSection``
- ``ReportTemplate``
- ``ReportCheck``
- ``template_names``
- ``load_template``
- ``check_report``
- ``report_skeleton``
- ``count_words``
- ``count_items``
- ``section_problem``
- ``template_guide``

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Final, Literal, TypeAlias

__all__ = [
    "LIMIT_TOLERANCE",
    "SECTION_SOURCE_BEGINNING",
    "SECTION_SOURCE_CHOICES",
    "SECTION_SOURCE_NOTES",
    "SECTION_SOURCE_REFERENCES",
    "SectionSource",
    "REPORT_TEMPLATES_DIR",
    "ReportCheck",
    "ReportTemplate",
    "TemplateSection",
    "GUIDE_MARKER",
    "check_report",
    "count_items",
    "count_words",
    "load_template",
    "report_skeleton",
    "section_problem",
    "template_guide",
    "template_names",
]

REPORT_TEMPLATES_DIR: Final[Path] = Path(__file__).parent
# Word and item limits allow this fraction either way, as editors do.
LIMIT_TOLERANCE: Final[float] = 0.1
# What a section is written from: the reading notes, the beginning of the
# document (title, authors, venue), or its reference list.
SectionSource: TypeAlias = Literal["notes", "beginning", "references"]
SECTION_SOURCE_NOTES: Final[SectionSource] = "notes"
SECTION_SOURCE_BEGINNING: Final[SectionSource] = "beginning"
SECTION_SOURCE_REFERENCES: Final[SectionSource] = "references"
SECTION_SOURCE_CHOICES: Final[tuple[SectionSource, ...]] = (
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_BEGINNING,
    SECTION_SOURCE_REFERENCES,
)
_HEADING = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_TITLE = re.compile(r"^#\s+(.+?)\s*#*\s*$")
_INLINE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


@dataclass(frozen=True, slots=True)
class TemplateSection:
    """One section of a template.

    Attributes
    ----------
    name: str
        The exact heading text.
    required: bool
        Whether the report must contain it, with content.
    guidance: str
        What belongs here, for the writer.
    words: tuple[int, int] | None
        Allowed word count of the content, or None for no limit.
    items: tuple[int, int] | None
        Allowed number of list items, or None for no limit.
    source: SectionSource
        What the writer also sees besides the notes: ``beginning`` (the
        document's first page, for citations) or ``references`` (its
        reference list; items must come from it).
    inline: bool
        Written as one labelled line (``**Name:** content``) instead of a
        ``##`` section, as the "This is a summary of:" line of a
        briefing.

    """

    name: str
    required: bool = True
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
        Template ID (the file name without ``.json``).
    title: str
        What the report is called, such as "Research Briefing".
    description: str
        What the template is for.
    sections: tuple[TemplateSection, ...]
        Sections in their required order.
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
        Required sections without a heading.
    duplicated: tuple[str, ...]
        Template sections whose heading appears more than once.
    empty: tuple[str, ...]
        Required sections with a heading but no content.
    out_of_order: bool
        Whether the template's sections appear in a different order.
    unexpected: tuple[str, ...]
        ``##`` headings that are not template sections.
    lengths: tuple[str, ...]
        Sections outside their word or item limits, described.
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
        """True when the report follows the template."""

        return not self.problems()

    def problems(self) -> list[str]:
        """Describe each problem in one sentence, for repairs and users."""

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


def template_names(directory: Path = REPORT_TEMPLATES_DIR) -> list[str]:
    """List the available templates.

    Parameters
    ----------
    directory: pathlib.Path, optional
        Template folder. Defaults to this package's folder.

    Returns
    -------
    list[str]
        Template IDs, sorted.

    """

    return sorted(path.stem for path in Path(directory).glob("*.json"))


def _range(name: str, item: dict, key: str) -> tuple[int, int] | None:
    value = item.get(key)
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(type(number) is int and number >= 0 for number in value)
        or value[0] > value[1]
    ):
        raise ValueError(
            f"Template {name!r}: {key} must be [minimum, maximum] whole "
            "numbers."
        )
    return value[0], value[1]


@lru_cache(maxsize=None)
def load_template(
    name: str, directory: Path = REPORT_TEMPLATES_DIR
) -> ReportTemplate:
    """Read and validate one template.

    Parameters
    ----------
    name: str
        Template ID, such as ``"research_briefing"``.
    directory: pathlib.Path, optional
        Template folder. Defaults to this package's folder.

    Returns
    -------
    ReportTemplate
        The template; cached for the process.

    Raises
    ------
    ValueError
        If the template does not exist or is malformed; the message names
        the problem.

    """

    path = Path(directory) / f"{name}.json"
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not path.is_file():
        raise ValueError(
            f"Unknown report template {name!r}. Choose one of: "
            f"{', '.join(template_names(directory)) or 'none'}."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if data.get("output_format", "markdown") != "markdown":
        raise ValueError(f"Template {name!r}: only markdown is supported.")
    title = data.get("title")
    sections = data.get("sections")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Template {name!r} needs a title.")
    if not isinstance(sections, list) or not sections:
        raise ValueError(f"Template {name!r} needs a list of sections.")
    parsed: list[TemplateSection] = []
    for item in sections:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Template {name!r}: each section needs a name.")
        section = TemplateSection(
            name=" ".join(item["name"].split()),
            required=item.get("required", True) is True,
            guidance=str(item.get("guidance", "")).strip(),
            words=_range(name, item, "words"),
            items=_range(name, item, "items"),
            source=item.get("source", SECTION_SOURCE_NOTES),
            inline=item.get("inline", False) is True,
        )
        if section.source not in SECTION_SOURCE_CHOICES:
            raise ValueError(
                f"Template {name!r}: section source must be one of: "
                f"{', '.join(SECTION_SOURCE_CHOICES)}."
            )
        if not section.name or "#" in section.name:
            raise ValueError(
                f"Template {name!r}: section names must be plain text."
            )
        parsed.append(section)
    names = [section.name.casefold() for section in parsed]
    if len(set(names)) < len(names):
        raise ValueError(f"Template {name!r} repeats a section name.")
    headline = data.get("headline")
    characters = None
    guidance = ""
    if headline is not None:
        characters = (
            headline.get("max_characters")
            if isinstance(headline, dict)
            else None
        )
        if type(characters) is not int or characters < 10:
            raise ValueError(
                f"Template {name!r}: headline.max_characters must be a "
                "whole number of at least 10."
            )
        guidance = str(headline.get("guidance", "")).strip()
    total = data.get("max_total_words")
    if total is not None and (type(total) is not int or total < 50):
        raise ValueError(
            f"Template {name!r}: max_total_words must be a whole number of "
            "at least 50."
        )
    return ReportTemplate(
        name=name,
        title=" ".join(title.split()),
        description=str(data.get("description", "")).strip(),
        sections=tuple(parsed),
        headline_characters=characters,
        headline_guidance=guidance,
        max_total_words=total,
        reference=str(data.get("reference", "")).strip(),
    )


def report_skeleton(template: ReportTemplate) -> str:
    """Return the headings a report must use, in order.

    Parameters
    ----------
    template: ReportTemplate
        The template.

    Returns
    -------
    str
        The ``#`` line (the title, or a headline placeholder) and one
        ``## Section`` line per section.

    """

    first = (
        f"# <headline of at most {template.headline_characters} characters>"
        if template.headline_characters
        else f"# {template.title}"
    )
    return "\n".join(
        [first]
        + [
            f"**{section.name}:** ..."
            if section.inline
            else f"## {section.name}"
            for section in template.sections
        ]
    )


# First line of a generated guide; a file starting with it is QAChat's to
# rewrite.
GUIDE_MARKER: Final[str] = "<!-- Generated by QAChat from a report template"


def template_guide(template: ReportTemplate) -> str:
    """Write a human-readable guide to a template, for users to open.

    Parameters
    ----------
    template: ReportTemplate
        The template.

    Returns
    -------
    str
        Markdown: what the format is for and where it comes from, the
        headline rule, each section with its limit and guidance, and the
        total length. It is generated from the template, so it always
        matches what QAChat checks.

    """

    lines = [
        f"{GUIDE_MARKER} ({template.name}.json); regenerated when the "
        "template changes. Copy it to keep your own notes. -->",
        "",
        f"# {template.title} template",
        "",
        template.description,
    ]
    if template.reference:
        lines += ["", f"*{template.reference}*"]
    lines += ["", "## How QAChat fills it", ""]
    lines += [
        "- Ask, for example: \"Turn the … paper into a "
        f"{template.title.lower()}\". The paper can be in `target/` or "
        "among the project's sources.",
        "- QAChat reads the whole paper in parts, writes each section "
        "from its notes, and checks every limit below in Python; a "
        "section that misses its limit is sent back for a targeted fix, "
        "and any limit still missed is reported with the result.",
        "- The result is saved in `target/` as Markdown and, in the "
        "desktop app, as a one-page PDF.",
    ]
    if template.headline_characters:
        lines += [
            "",
            "## Headline",
            "",
            f"At most {template.headline_characters} characters. "
            f"{template.headline_guidance}",
        ]
    for number, section in enumerate(template.sections, start=1):
        if section.words:
            limit = f"{section.words[0]}-{section.words[1]} words"
        elif section.items:
            limit = f"{section.items[0]}-{section.items[1]} items"
        else:
            limit = "no fixed length"
        source = {
            SECTION_SOURCE_BEGINNING: " Taken from the paper's first page.",
            SECTION_SOURCE_REFERENCES: (
                " Chosen from the paper's own reference list."
            ),
        }.get(section.source, "")
        shown = (
            f" Written as one line: **{section.name}:** ..."
            if section.inline
            else ""
        )
        lines += [
            "",
            f"## {number}. {section.name}",
            "",
            f"*{limit}.*{source}{shown}",
            "",
            section.guidance,
        ]
    if template.max_total_words:
        lines += [
            "",
            "## Length",
            "",
            f"At most {template.max_total_words} words in all sections "
            "together, so it fits on one page.",
        ]
    return "\n".join(lines) + "\n"


def count_words(text: str) -> int:
    """Count the words of report content, ignoring citations and markup.

    Parameters
    ----------
    text: str
        Section content.

    Returns
    -------
    int
        Number of words.

    """

    return len(re.findall(r"[\w'’-]+", _CITATION.sub("", text)))


def count_items(text: str) -> int:
    """Count list items (``-``, ``*``, ``•``, or ``1.`` lines) in content.

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
        None when the content is within its limits (with
        ``LIMIT_TOLERANCE``).

    """

    if not content.strip():
        return f"{section.name} is empty." if section.required else None
    if section.words:
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
        The report, with a ``#`` first line and ``## Section`` headings.
    template: ReportTemplate
        The template.

    Returns
    -------
    ReportCheck
        Missing, repeated, empty, out-of-order, and unexpected sections;
        sections outside their word or item limits (with
        ``LIMIT_TOLERANCE``); and headline problems. Heading text is
        compared ignoring case and surrounding spaces.

    """

    expected = {
        section.name.casefold(): section for section in template.sections
    }
    found: list[tuple[str, str]] = []  # (heading, content)
    current: str | None = None
    body: list[str] = []
    first_line = next(
        (line.strip() for line in markdown.splitlines() if line.strip()), ""
    )
    inline = {
        section.name.casefold() for section in template.sections
        if section.inline
    }
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
    seen = [heading.casefold() for heading, _ in found]
    known = [heading for heading in seen if heading in expected]
    missing = tuple(
        section.name
        for section in template.sections
        if section.required and section.name.casefold() not in seen
    )
    duplicated = tuple(
        expected[key].name
        for key in dict.fromkeys(known)
        if known.count(key) > 1
    )
    empty = tuple(
        expected[heading.casefold()].name
        for heading, content in found
        if heading.casefold() in expected
        and expected[heading.casefold()].required
        and not content
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


def _within(value: int, low: int, high: int) -> bool:
    """Whether a count is within limits, allowing ``LIMIT_TOLERANCE``."""

    return low * (1 - LIMIT_TOLERANCE) <= value <= high * (
        1 + LIMIT_TOLERANCE
    )
