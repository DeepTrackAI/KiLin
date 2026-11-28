"""Expose the registered ``project_summary`` tool.

It writes one report on all the documents of a project from a project
template (``scope: project``): it reads a short excerpt of every document,
takes notes and per-document checks, and writes the report from them. Its
implementation and configuration are local to this package.

Key Features
------------

- **Tool execution**

    Parse validated settings and execute the named tool.

- **Independent defaults**

    Read this package's ``defaults.json`` configuration.

Module Structure
----------------

- ``PLUGIN``

    Versioned descriptor discovered by the tool registry.

- ``parse_settings`` and ``run``

    The registry entry points for this tool.

- ``ProjectSummarySettings``, ``routing`` and ``workspace_resources``

    The settings, the routing from the project's templates, and the
    templates installed in new projects.

- ``_documents``, ``_mapping`` and ``_reduce``

    The project's documents and their excerpts, each document's notes and
    checks, and the report written from them.

Examples
--------
>>> from kilin.tools.project_summary import parse_settings
>>> callable(parse_settings)
True

"""

from kilin.tools import Tool

from .tool import (
    SHIPPED_TEMPLATES_DIR,
    TEMPLATE_CHOOSE,
    ProjectSummarySettings,
    choose_template,
    parse_settings,
    project_templates,
    routing,
    run,
    workspace_resources,
)

__all__ = [
    "PLUGIN",
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


PLUGIN = Tool(
    name="project_summary",
    summary="Write one report on all the documents of a project with a "
    "project template, reading every document, and save it to "
    "outputs/.",
    parse_settings=parse_settings,
    run=run,
    routing=routing,
    workspace_resources=workspace_resources,
    version="1.0.0",
    api_version=1,
)
