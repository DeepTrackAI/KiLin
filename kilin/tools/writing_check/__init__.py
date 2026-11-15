"""The ``writing_check`` tool: grammar and style of a long document.

The tool is being built in stages; this package holds its parts so far.
It reads a document into sections and paragraphs (``_document``) so that
each paragraph can be checked with its section as context and each
finding reported with its line, and chooses the part a question asks to
check (``_scope``).

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

Examples
--------
>>> import kilin.tools.writing_check

"""

__all__: list[str] = []
