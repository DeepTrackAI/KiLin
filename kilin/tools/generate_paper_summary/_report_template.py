"""Read report templates and check reports with this tool's limits.

The template format, its parser and the report checks are shared with the
other report tools (``kilin.tools._shared.report_templates``). This module
binds them to ``generate_paper_summary``'s own limits, from its
``defaults.json``, and holds the templates the tool ships, so the tool's
callers keep the same names.

Key Features
------------

- **This tool's limits**

    ``LIMIT_TOLERANCE`` and the smallest ``max_words``, as
    ``TEMPLATE_RULES``.

- **Shipped templates**

    ``SHIPPED_TEMPLATES_DIR``: the research briefing, popular science,
    technical summary and experimental protocol templates, and the guide
    to the format.

Module Structure
----------------

- ``SHIPPED_TEMPLATES_DIR``, ``LIMIT_TOLERANCE`` and ``TEMPLATE_RULES``

    The tool's templates folder and limits.

- ``load_template``, ``parse_template``, ``section_problem`` and
  ``check_report``

    The shared functions with this tool's limits.

- ``ReportCheck``, ``ReportTemplate``, ``TemplateSection``,
  ``SectionSource``, the ``SECTION_SOURCE_*`` values, ``count_words``,
  ``count_items`` and ``template_names``

    Re-exported from the shared module.

Examples
--------
>>> from kilin.tools.generate_paper_summary._report_template import (
...     SHIPPED_TEMPLATES_DIR, check_report, load_template,
... )
>>> template = load_template("research_briefing", SHIPPED_TEMPLATES_DIR)
>>> template.headline_characters
90
>>> check = check_report("# Headline\\n\\n## The Bio\\nText.", template)
>>> check.valid
False

"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from kilin.tools._shared import report_templates as _shared
from kilin.tools._shared.report_templates import (
    SECTION_SOURCE_CHOICES,
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_REFERENCES,
    ReportCheck,
    ReportTemplate,
    SectionSource,
    TemplateRules,
    TemplateSection,
    count_items,
    count_words,
    template_names,
)
from kilin.tools.generate_paper_summary._configuration import (
    LIMIT_TOLERANCE as _CONFIG_LIMIT_TOLERANCE,
    MIN_TOTAL_WORDS,
)

__all__ = [
    "LIMIT_TOLERANCE",
    "SECTION_SOURCE_CHOICES",
    "SECTION_SOURCE_CITATION",
    "SECTION_SOURCE_NOTES",
    "SECTION_SOURCE_REFERENCES",
    "SHIPPED_TEMPLATES_DIR",
    "TEMPLATE_RULES",
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
]

# Templates shipped with KíLín; the core selects relevant files.
SHIPPED_TEMPLATES_DIR: Final[Path] = Path(__file__).parent / "templates"
# Word limits (of each section and of the whole report) allow this fraction
# either way, as editors do. Item limits are exact.
LIMIT_TOLERANCE: Final[float] = _CONFIG_LIMIT_TOLERANCE
TEMPLATE_RULES: Final[TemplateRules] = TemplateRules(
    limit_tolerance=LIMIT_TOLERANCE, min_total_words=MIN_TOTAL_WORDS
)


def load_template(name: str, directory: Path) -> ReportTemplate:
    """Read and validate one template file with this tool's limits.

    Parameters
    ----------
    name: str
        Template ID, such as ``"research_briefing"``.
    directory: pathlib.Path
        Explicit project-owned template folder.

    Returns
    -------
    ReportTemplate
        The template (``kilin.tools._shared.report_templates``).

    """

    return _shared.load_template(name, directory, TEMPLATE_RULES)


def parse_template(text: str, name: str) -> ReportTemplate:
    """Read and validate a template's text with this tool's limits.

    Parameters
    ----------
    text: str
        The template's Markdown.
    name: str
        Template ID, used in the result and in error messages.

    Returns
    -------
    ReportTemplate
        The validated template.

    """

    return _shared.parse_template(text, name, TEMPLATE_RULES)


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
        The problem, or None when the content is within its limits.

    """

    return _shared.section_problem(section, content, TEMPLATE_RULES)


def check_report(markdown: str, template: ReportTemplate) -> ReportCheck:
    """Check a Markdown report against its template with this tool's limits.

    Parameters
    ----------
    markdown: str
        The report.
    template: ReportTemplate
        The template.

    Returns
    -------
    ReportCheck
        The result (``kilin.tools._shared.report_templates``).

    """

    return _shared.check_report(markdown, template, TEMPLATE_RULES)
