"""Write one report on all the documents of a project, from a template.

The ``project_summary`` tool reads a short excerpt of every document of a
project, takes notes and per-document checks for a project template
(``scope: project``), and writes the report from them. Its implementation
is local to this package.

Key Features
------------

- **Every document**

    A map step reads each document once, so the report is not limited to
    the passages a search finds.

Module Structure
----------------

- ``_documents`` and ``_mapping``

    The project's documents and their excerpts, and each document's notes
    and checks.

Examples
--------
>>> import kilin.tools.project_summary
>>> kilin.tools.project_summary.__all__
[]

"""

__all__: list[str] = []
