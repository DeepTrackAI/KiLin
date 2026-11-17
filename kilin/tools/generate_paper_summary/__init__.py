"""Produce a validated paper summary using project-owned templates.

This tool package owns its code, ``defaults.json`` and shipped Markdown
templates. A project type may override the tool's ordinary settings; only
its selected packaged template is installed in a new project's workspace.
Shared model execution, output handling, and planning remain private
infrastructure in ``kilin.tools`` rather than other executable tools.

Key Features
------------

- **Tool-local configuration**

    Validate per-tool defaults and merge project-type settings explicitly.

- **Own packaged resources**

    Declare template files for project creation without GUI involvement.

- **Report validation**

    Parse templates, check report structure, and preserve user edits.

Module Structure
----------------

- ``SummarySettings``, ``parse_settings``, ``run`` and
  ``workspace_resources``

    Validated tool settings, execution, and packaged resource selection.

- ``ReportTemplate``, ``TemplateSection``, ``load_template`` and
  ``check_report``

    Tool-owned template parsing and deterministic report checks.

- ``SHIPPED_TEMPLATES_DIR``

    Resource folder owned by this tool package.

Examples
--------
>>> from kilin.tools.generate_paper_summary import parse_settings
>>> parse_settings({}, "example").template
'research_briefing'

"""

from kilin.tools.generate_paper_summary._report_template import (
    LIMIT_TOLERANCE,
    SECTION_SOURCE_CHOICES,
    SECTION_SOURCE_CITATION,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_REFERENCES,
    SHIPPED_TEMPLATES_DIR,
    ReportCheck,
    ReportTemplate,
    SectionSource,
    TemplateSection,
    check_report,
    count_items,
    count_words,
    load_template,
    parse_template,
    section_problem,
    template_names,
)
from kilin.tools.generate_paper_summary.tool import (
    REFERENCE_LAYOUT,
    TEMPLATE_CHOOSE,
    SummarySettings,
    choose_paper,
    choose_template,
    fix_prompt,
    headline_prompt,
    paper_beginning,
    paper_citation,
    parse_settings,
    references_prompt,
    routing,
    run,
    section_prompt,
    summary_file_text,
    templates_answer,
    workspace_resources,
)

__all__ = [
    "LIMIT_TOLERANCE",
    "REFERENCE_LAYOUT",
    "SECTION_SOURCE_CHOICES",
    "SECTION_SOURCE_CITATION",
    "SECTION_SOURCE_NOTES",
    "SECTION_SOURCE_REFERENCES",
    "SHIPPED_TEMPLATES_DIR",
    "ReportCheck",
    "ReportTemplate",
    "SectionSource",
    "SummarySettings",
    "TEMPLATE_CHOOSE",
    "TemplateSection",
    "check_report",
    "choose_paper",
    "choose_template",
    "count_items",
    "count_words",
    "fix_prompt",
    "headline_prompt",
    "load_template",
    "paper_beginning",
    "paper_citation",
    "parse_settings",
    "parse_template",
    "references_prompt",
    "routing",
    "run",
    "section_problem",
    "section_prompt",
    "summary_file_text",
    "template_names",
    "templates_answer",
    "workspace_resources",
]
