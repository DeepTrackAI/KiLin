# Project and conversation diagnostic traces

The project and conversation traces share a single event-number namespace
within each workspace. The existing grouped JSON format remains the diagnostic
format; there is no second journal of event payloads.

## Files and ownership

- `logs/traces/project.json` holds project-level diagnostic sessions.
- Conversation trace files retain the existing conversation checkpoint and
  grouped diagnostic sessions. Their existing exclusive writer ownership
  is unchanged.
- `logs/traces/.sequence.json` contains only the trace format version,
  allocation high-water mark, and a persistent incomplete-recording flag.
  It must accompany a copied test workspace. Its lock companion, like other
  ownership companions, must not be removed while writers are active.

The project trace and allocation counter take short write leases. Each
project-trace writer replaces only its own hierarchical session projection;
other writers' sessions are preserved. No workspace-wide session lock is
introduced. A project trace does not own a conversation checkpoint.

## Event identity

Each newly recorded event in a managed project receives a positive project-wide
`sequence` number before publication. This order is allocation order, not a
claim that operations in different processes completed sequentially. A process
can reserve a number and fail before its event reaches disk. Sequence gaps or
an incomplete-recording flag should therefore be treated as reasons to reject
or investigate a recording rather than silently reordering or renumbering it.

The hierarchical projection includes explicit event type, component, event ID,
trace ID, sequence, timing and correlation metadata. An operation span retains
its start metadata in `event` and its terminal metadata in `completion`.
The original status of the start is retained as `started_status`; the existing
summary status and completion time remain available for inspection. Sampled
progress checkpoints retain event identity as well. Top-level event counts
continue to describe the publishing logger run, not the entire history.

Conversation archival preserves the event numbers in the archived file. The
new active conversation does not re-emit those historical events. Standalone
loggers outside a managed project retain local numbering (`sequence_scope`
`logger`).

The `trace_format_version` and `sequence_scope` distinguish the new project
recordings from earlier diagnostics. Existing historical recordings are not
renumbered and must not be assumed replayable. A missing or malformed shared
counter for an already project-numbered workspace is not repaired by guessing
new numbers. Optional trace-write failures retain the existing warning/raise
policy; the incomplete flag is persisted on a best-effort basis. The replay
validator must independently check for gaps and missing data.

Missing-counter discovery includes managed JSON traces with custom names and
case variants such as `conversation.json` or `custom.JSON`, including archived
files. Initialization stops if a history directory or JSON candidate is
unreadable or linked, because the recorder cannot establish that older
numbered events are absent. Ordinary counter allocation does not rescan the
trace history.

## Recording coverage

The GUI maintains a project trace alongside the active conversation trace.
Conversation switching and archival retain their existing ownership rules.
Before archival, pending diagnostics are published; a trace that already
contains all recorded events is left byte-for-byte unchanged.

### Project operations

Project creation starts its trace only after the destination has been safely
reserved. The creation trace is owned by workspace preparation and is closed
before the profile is published. Its completion and closing writes remain
inside the creation rollback boundary: a strict diagnostic failure removes
the new trace and other files owned by that attempt. Warning policy preserves
successful creation and marks recording incomplete where possible. A creation
conflict does not write to the occupied destination. Caller-owned sinks retain
their caller's lifetime and receive completion after profile publication.

A successful creation or subsequent import emits `workspace.prepare.started`,
`workspace.source.copied` for each processed source, and
`workspace.prepare.completed`, correlated by a `batch_id`. The start records
requested source paths, selected files, project type, import policy, and
profile settings. Each copy event records the original source, its kind, the
managed workspace-relative destination, and the actual copied and preserved
original-relative filenames. For a direct file, the managed destination is the
file itself, which may have been renamed to avoid a collision. For a folder,
append each copied relative filename to the managed destination. Do not infer
historical imports from the current external directory contents.

Synchronization records `workspace.sync.started`, the applied decisions in
`workspace.sync.file` or `workspace.sync.detached`, and
`workspace.sync.completed`, with a common batch ID and actual results. Source
removal and explicit resource edits also leave project diagnostics. These
operations are not automatically considered safe to replay: additive source
imports can be reconstructed from the retained workspace copies, whereas
replacement, deletion, or manual resource edits may destroy historical inputs.

Index preparation, including model loading, index reuse/rebuild decisions,
source-card reconciliation, and preparation progress, is recorded in the
project trace. The content manifest already computed by indexing is included
in `index.check.completed` or `index.empty`; it records relative filenames,
sizes, and SHA-256 digests without rereading files for tracing. Comparing these
manifests can detect changed or removed indexed source content. Source-card
regeneration is recorded separately. Question-related model switches and
inference, evidence, and mandatory checkpoints remain in conversation traces.

### Configuration and lifecycle

The GUI emits `gui.project.creation.settings` with the original request,
`explicit_settings`, and `provenance`. A direct API request is treated as
explicit. A full GUI settings form is compared recursively to the defaults
last sent for that project type, and the comparison baseline is retained in
`displayed_defaults`. Unchanged fields are inherited; lists are treated as
whole values. Reaffirming a displayed value is indistinguishable from leaving
it unchanged. If the comparison baseline is unavailable, explicit settings
are recorded as null rather than guessed. The original request is preserved
regardless of provenance; this classification does not change the settings
applied during ordinary creation.

`gui.project.open` spans identify creation, resumption, or an internal reopen
with the selected conversation ID. `gui.project.configuration.effective`
records the effective profile, effort levels, router models, search depths,
source-card settings, behavior and tools after saved defaults are established.
A failed diagnostic snapshot is recorded explicitly. Subsequent saved profile,
model, tool, behavior and card settings changes are traced in the project.
`gui.project.closed` records the selected conversation when the GUI releases
the project. A standalone RAG session similarly records its initialization and
closure in its project sink; switching its conversation does not change that
sink's lifetime. The replacement conversation sink must have the same tracing
activation as the project sink. A mismatch is rejected before any session or
ownership change. Changing activation requires closing the session and
opening another with the desired configuration.

If RAG initialization fails, constructed model sessions and owned trace sinks
are closed, including when final readiness notification fails. Cleanup errors
are attached to the original exception rather than replacing it.

### Conversation operations

The shared project trace records the conversation catalog and active identity
at each opening. Explicit create, select, rename, delete, and reset commands
have `gui.conversation.*` operation spans with request data, results, and
terminal status. Creation records whether the command created or reused an
empty conversation, the selected conversation, and its branch origin and
inherited turn count, if any. Inherited turns retain their persistent UUIDs;
branching does not ask their questions again. Deletion records the archive
and replacement identity, including whether a replacement was created.
Selecting the already active conversation is recorded as a no-op. Generated
titles are separate diagnostics, not explicit rename requests.

Each `gui.question.request` span in the conversation trace retains the
question passed to the backend, the title question, requested retrieval
count, generation settings, effort selectors, and active conversation ID.
The GUI's full-form generation settings are compared with the saved profile
before applying the request. Reaffirming an unchanged value is not separately
observable. Partial/direct requests are explicit. The resolved selection is
recorded in `gui.question.settings.resolved`. The core turn is correlated to
the request by its parent span. `state.checkpoint.committed` and
`gui.question.saved` identify the persisted turn UUID and checkpoint, so the
original request is not confused with a branch's inherited turn. Settings
changes performed while answering are part of that question operation, not
additional user commands. Direct RAG calls retain their own core turn input
and checkpoint diagnostics, but are not GUI command recordings.

The GUI request span also covers delivery of the saved answer to the frontend.
A delivery failure can therefore fail that outer span after a successful core
turn and checkpoint. Validation must inspect the persisted checkpoint and
core outcome independently of the GUI request's terminal status.

Routing and tool choices record a proposal and decision (or failure) with a
common interaction ID and turn correlation. Decisions retain whether a
callback was actually invoked, its response, and the chosen option label as
well as the original index. Plans record the semantic action, confirmation
policy, and whether confirmation was requested. Automatic defaults are not
user choices. Invalid or ended GUI input and callback failures are recorded
separately; recording them does not change ordinary cancellation behavior.
A replay must resolve the original label/action against the newly offered
choices, not blindly reuse the index.

Feedback and memory exclusions record their requested mutations and revision,
conversation ID, persistent turn UUID, role, and committed rating/revision.
Manual memory updates have project-level spans; automatic memory updates
have turn-level spans. Result metadata describes the eligible records,
origins, fingerprints, priorities, and index counts without duplicating the
message texts or vectors. Memory remains derived from the authoritative
conversation and feedback state.

### Recording completeness

The creation event carries `operation_recording_version` in addition to the
structural trace-format version. Workspaces created before full operation
recording do not acquire this marker on reopening. A fresh workspace with
tracing enabled throughout is required for a reference experiment. The
marker is necessary but not sufficient: independent validation must also
check event coverage, sequence continuity, command completion, referenced
files and identities. An unrecorded change cannot always be detected from
traces alone. Interrupted or unsuccessful commands are not silently treated
as completed replay inputs.

These records are the inputs to future timeline reconstruction and replay,
not a replay implementation. See [Trace recording](trace_recording.md) for
activation and failure policy.
