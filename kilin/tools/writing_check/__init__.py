"""The ``writing_check`` tool: grammar and style of a long document.

The tool is being built in stages; this package holds its parts so far.
It reads a document into sections and paragraphs (``_document``) so that
each paragraph can be checked with its section as context and each
finding reported with its line, chooses the part a question asks to
check (``_scope``), and runs the checks that need no model (``_checks``).

Key Features
------------

- **Paragraphs with their place**

    Section, number and source lines for every paragraph of prose.

Module Structure
----------------

- ``_document``

    Sections and paragraphs of a ``.tex``, ``.md`` or ``.txt`` source.

- ``_scope``

    The part of the document a question asks to check.

- ``_checks`` and ``_configuration``

    The checks done in code, the glossary for the model, and their
    validated word lists and limits.

Examples
--------
>>> import kilin.tools.writing_check

"""

__all__: list[str] = []
