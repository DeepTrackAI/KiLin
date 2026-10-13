"""Private services shared by registered QAChat tools.

These modules implement document selection, saved-output handling,
planning, list presentation, mention scanning, source-path
disambiguation, sentence splitting, and term filtering. They are internal
services, not executable tools or externally configurable plugins.

Key Features
------------

- **Shared operations**

    Reuse common algorithms without importing another executable tool.

- **Private implementation**

    Keep shared infrastructure separate from tool-owned packages.

Module Structure
----------------

- ``choose_document``, ``document_notes``, ``long_lists``,
  ``outputs``, ``planning``, ``term_filter``, and ``terms``

    Private support modules used by multiple registered tools.

- ``mentions``, ``paths``, ``scan_settings``,
  ``sentences``, and ``earlier_turns``

    Shared source and follow-up algorithms.

Examples
--------
>>> from qachat.tools._shared.planning import PLAN_CHOICE_CONTINUE
>>> PLAN_CHOICE_CONTINUE
'continue'

"""

__all__: list[str] = []
