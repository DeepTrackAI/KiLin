# Conversation memory: identity, feedback, indexing and retrieval

This document describes the implemented conversation-memory core:
**durable turn identities, authoritative feedback, disliked-answer context
exclusion, and project-local semantic/lexical retrieval**. Project/intent
policies are defined in [Configuration](configuration.md). GUI Save/Like/
Dislike controls and idle-phase indexing integration are deferred to PR5;
the headless core already provides feedback and explicit synchronization.
Source evidence and retrieved memory remain separate in answering prompts.

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
        conversation_memory.json
        .conversation_memory.json.lock
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
active branch can continue to reference its inherited turn. Memory retrieval restricts origins to active conversations and reconciles
stale records, without deleting authoritative feedback prematurely.

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

## Excluding disliked answers from conversational context

When `history.exclude_disliked_answers` is true (the default for all five
project types), `RAGSession` reads the project-owned `feedback.json` before
each question and excludes saved turns whose **assistant** feedback is
`disliked`. This policy works even when long-term conversation memory is
disabled. Feedback on the `user` message does not exclude its paired answer
from chat history; the question and answer can be rated independently.

The exclusion applies consistently to router summaries, related-turn
references, recent/relevant model history, search-query follow-up wording,
reused evidence, tool `EarlierTurn` inputs, tool language context, and
transitive related-turn traversal. If a follow-up does not name an eligible
turn, deterministic fallback uses the last eligible turn, not the last saved
turn. Recent history counts eligible turns but retains the original one-based
numbers; removing an entry never renumbers a saved conversation.

For model prompts, the complete disliked question/answer pair is omitted.
The original question and answer, evidence, and diagnostic checkpoints remain
stored and visible in the conversation. They are not erased or edited by a
reaction. Reopening a conversation and changing answering models also restore
filtered chat context. Resetting Dislike, or changing the saved project setting
to `exclude_disliked_answers: false`, restores normal history behavior.

Malformed feedback fails explicitly instead of silently including rejected
answers. A missing feedback file means no exclusions and does not create a
file or call an LLM. The history filter does not affect source indexes or
control memory retrieval: the separate index has its own content policy.
Explicit quotations typed by the user remain the user's current question,
regardless of the earlier answer's feedback.

## Independent indexing and provenance

`kilin.conversation_memory.index.synchronize_memory()` reads current saved
conversation files in `logs/traces/` and the authoritative feedback snapshot.
Its persistent index under `indexes/conversation_memory.json` is **derived,
rebuildable data**. A separate checksum validates the index payload; the same
atomic file-write and advisory-lock mechanisms used elsewhere protect index
replacement. Corrupt indexes are rebuilt rather than migrated. Invalid
conversation or feedback content fails explicitly rather than being trusted.
Archiving or deleting a conversation removes that origin on the next sync;
active branches continue to point to their inherited turn UUID. Metadata
contains the active conversation ID, one-based turn number, timestamp, role,
and turn UUID. Cross-project memory is never searched.

Intent policies are applied to the intent originally saved with each turn.
`never` forbids all indexing; `explicit` requires Save or Like; `automatic`
uses the project's question and answer policy. A disliked message is never
indexed, even if saved. A disliked **assistant** response does not forbid
independently indexing its paired user question. Answers marked as incomplete
or containing only hidden thinking are not indexed. `answers = none` forbids
assistant entries even when explicitly selected. Source/document and tool
indexes are unaffected by these decisions.

Each indexed occurrence is keyed by `(conversation_turn_uuid, role)`. Its
embedding key is instead the SHA-256 fingerprint of role and normalized
searchable text. Identical messages from distinct turns retain separate
feedback but reuse one embedding. Inherited branches share a single indexed
occurrence with multiple active origins. Search results avoid repeating the
same fingerprint within one answer. Hidden model thinking is removed from
indexed assistant text; the saved conversation remains unchanged.

`RAGSession.sync_conversation_memory()` is the GUI-independent entry point.
It checks active file size/mtime signatures, feedback revision, effective
indexing policy and model identity. If unchanged, it neither writes nor
re-embeds. New or changed content is embedded using the **already loaded**
embedding model and previously saved compatible vectors are reused. A model
identity change invalidates incompatible vectors. The next eligible memory
search catches up after a completed answer, archive, feedback edit or policy
change; a future GUI may invoke this method during idle time. Disabled memory
does not read or embed prior conversation content.

## Strictly separate answer-prompt channels

`RAGSession.ask()` retrieves memory only when the project has enabled it,
the current intent permits `memory.read`, and the normal answering model is
producing the answer. Tool-owned generation retains its existing tool-context
contract; no memory is inserted into its source evidence. Recent/relevant
same-conversation history is a third, separately selected channel, and turns
already shown through it are excluded from long-term memory hits.

The answering prompt uses **independent memory and source sections**:

- **Project source evidence:** Numbered source-file excerpts, unchanged
  citation identifiers and evidence-token budget.
- **Conversation memory:** Earlier quoted messages labeled `Memory M1`, etc.,
  with conversation/turn identity, role and UTC timestamp. These are
  navigational references, **not source citations**.
- **Current chat history:** Recent/relevant saved turns, filtered according
  to the project's disliked-answer context policy.

An additional system-level provenance instruction forbids treating memory as
independently validated research evidence, reusing historical citations as
current ones, or allowing retrieved text to override project or intent rules.
Reviewing's researcher-only judgment boundary remains in force. The memory
budget (`max_hits` and estimated `max_tokens`) never reduces the source
passage budget or renumbers citations. When no memory is returned, the
source-only prompt remains byte-identical to the existing path.

Only semantic/lexical relevance determines whether a message qualifies for
retrieval; a bounded Save/Like boost changes the order of *relevant memory*
hits, not source rankings. Memory query formatting has its own instruction
instead of reusing the source-evidence retrieval instruction. There is no
new summarizing model call. Memory is not used to validate citation claims.

The index contains prior research discussions and embeddings, potentially
including sensitive or speculative content. It is not encrypted or anonymized.
See [Security](security.md) for the local trust boundary.

## Tests and verification

Focused deterministic tests live in
`tests/kilin/test__conversation_memory_index.py`,
`tests/kilin/test__conversation_memory_retrieval.py`, and
`tests/kilin/test_rag__conversation_memory.py`, alongside the PR2 and PR3
feedback, checkpoint and disliked-history tests. They exercise branch
identity, duplicate content, explicit/hard-never policies, edits to feedback,
index corruption, archiving, model changes, search budgets and the four
source/memory prompt configurations. Native model/backend and Qt tests
remain necessary for end-to-end production verification.
