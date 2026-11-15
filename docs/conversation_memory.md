# Conversation memory: committed identities and feedback

This document describes the first implemented conversation-memory runtime
component: **durable turn identities and authoritative user feedback**.
Project/intent policies are defined in [Configuration](configuration.md).
Semantic memory indexing, memory retrieval, filtering disliked answers from
model context, and GUI Save/Like/Dislike controls are **not yet implemented**.
They belong to later PRs. Source evidence and remembered discussions must
remain distinct in prompts when retrieval is added.

## Persistent turn identity

Every newly constructed completed `kilin.rag.ConversationTurn` receives a
canonical UUID4 hexadecimal `conversation_turn_uuid`. In the ordinary answer
path, this object is constructed with its UTC `timestamp` immediately before
the checkpoint, following successful generation. Both are included in the
same saved turn. Only a successful checkpoint makes that turn durable.

This UUID is **not** the one-based router turn number or the trace
correlation ID. It does not derive from timestamps, question text, or answer
text; independent answers have separate UUIDs even if the text matches.
`ConversationStore.create(branch_from=...)` copies saved turn UUIDs without
reassigning them, including truncated branches. Duplicate UUIDs *within a
single conversation* are rejected. Saved turns without UUID4 identities are
unsupported and are not silently repaired or migrated. Reading or branching
such a conversation fails without modifying its JSON.

The original question and answer remain inside the same committed turn.
Feedback keys add a message role (`user` for the question and `assistant`
for its answer), allowing independent selections for each message.

## Project-owned feedback

```text
project/
    memory/
        feedback.json
        .feedback.json.lock
    logs/traces/
        conversation_*.json
    indexes/
```

The `memory/` folder is created for every new project but feedback JSON is
created **only on an explicit update**. Feedback is not saved inside
conversation checkpoints, trace events, ordinary source files, or derived
indexes. `open_project()` does not create or migrate memory files.
When the configured source directory is the project root, the managed
`memory/` folder is excluded from source scans. A registered tool cannot
install a resource into this reserved folder.

The `FeedbackStore` API lives in
`kilin.conversation_memory.feedback`. It accepts a workspace root and
provides `load()`, `update(...)` and `reset(...)`. It is independent of Qt
and model loading. A write requires an active conversation ID, persisted
turn UUID, and message role. Missing or archived origins cannot be newly
rated. If a branch retains the UUID, feedback can be changed using that
branch's active conversation ID.

The feedback JSON has one unversioned format:

```json
{
  "format": "conversation-memory-feedback",
  "entries": {
    "4fb3849d0cb4455a98e14073cbaec047": {
      "assistant": {"saved": true, "reaction": "liked"}
    }
  }
}
```

The UUID above illustrates the record shape, not a literal test fixture.
Each message can be explicitly saved, and separately rated `neutral`,
`liked`, or `disliked`. A Like implicitly selects the message for future
memory indexing; a Dislike makes it ineligible even if it remains saved.
A neutral, unsaved record is removed rather than stored. Feedback does
not certify generated assertions as primary evidence.

**Feedback belongs to the turn identity, not its text or conversation path.**
When a branch retains a turn UUID, it observes the same feedback. When an
unrelated answer happens to have the same text, its UUID and feedback are
independent. Archiving a conversation does not delete feedback: another
active branch can continue to reference its inherited turn. Future memory
retrieval must restrict origins to active conversations and reconcile stale
records, without deleting authoritative feedback prematurely.

## Atomicity and revision checks

`FeedbackStore.load()` returns an immutable `FeedbackSnapshot` with an
optimistic SHA-256 revision token and a map of role-specific ratings. A
missing feedback file gives an empty snapshot without filesystem writes.
`update()` verifies input types and active project membership, acquires an
exclusive companion-file lock using the existing cross-process protocol,
re-reads the latest state, optionally checks `expected_revision`, and
atomically replaces just `feedback.json`. Stale revisions raise
`FeedbackConflictError`; competing live locks raise
`ConversationLockedError`. An unsuccessful write leaves the previous JSON
intact. Updates without a revision still merge different entries under
exclusive ownership rather than replacing them from an old snapshot.

The feedback lock is independent of conversation-file ownership; no
transaction spans both files. An answer checkpoint can commit even if
later feedback persistence fails. A rated conversation can be archived
between membership verification and later memory indexing; the indexing
stage must re-check active origins before including content.

The on-disk schema is strict and has no automatic migration path. Invalid,
oversized, or symlinked feedback files cause a clear error and are not
silently reset. Feedback is retained across rebuilds of disposable indexes.

## Tests and follow-on integration

Relevant deterministic tests include:

```sh
python -m pytest -q \
    tests/kilin/conversation_memory/ \
    tests/kilin/test_rag__conversation.py \
    tests/kilin/test_conversations.py \
    tests/kilin/test_project_layout.py
```

The follow-on indexing PR will use UUID plus role for occurrence identity
and a separate content fingerprint for embedding reuse. A repeated answer
may reuse an embedding but must not inherit another answer's reaction.
Retrieved conversation memory must have its own provenance, prompt section
and token budget, never the numbered citations of primary-source evidence.
