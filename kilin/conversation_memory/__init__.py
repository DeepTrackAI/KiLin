"""Keep project conversation memory distinct from research sources.

Saved conversations and the independent feedback store are authoritative.
A rebuildable memory index gives previous questions and selected answers a
separate, bounded retrieval channel. Neither stored memories nor their
provenance identifiers are valid primary-source evidence or citations.

Key Features
------------

- **Stable feedback identity**

    Completed UUID4 turns retain per-message reactions across branches.

- **Project-local indexing**

    Embed eligible messages incrementally under intent and feedback policy.

- **Separated retrieval**

    Supply memory hits without changing documentary evidence indexes.

Module Structure
----------------

- ``models`` and ``feedback``

    Stable message identity and durable Save/Like/Dislike state.

- ``index`` and ``persistence``

    Policy-aware synchronization and checksum-verified derived vectors.

- ``retrieval``

    Independent semantic/lexical ranking, budgets, and provenance rendering.

Examples
--------
>>> from kilin.conversation_memory.index import memory_fingerprint
>>> len(memory_fingerprint("user", "Earlier question"))
64

"""

__all__: list[str] = []
