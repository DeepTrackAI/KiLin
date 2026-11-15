"""Manage project-local feedback for persisted conversation messages.

This package stores deliberate user selections independently of conversation
checkpoints and retrieved source evidence. It does not index or retrieve
conversation content; those capabilities are introduced separately.

Key Features
------------

- **Durable identity**

    Assign new completed turns stable UUIDs, preserved in branches.

- **Independent feedback**

    Record Save, Like, Dislike, and reset in a project-owned file.

Module Structure
----------------

- ``models``

    Turn identity and validated feedback records.

- ``feedback``

    Atomic, revision-checked feedback persistence.

Examples
--------
>>> from kilin.conversation_memory.models import new_turn_uuid
>>> len(new_turn_uuid())
32

"""

__all__: list[str] = []
