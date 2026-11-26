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

## Recording coverage

The GUI maintains a project trace alongside the active conversation trace.
Conversation switching and archival retain their existing ownership rules.
Before archival, pending diagnostics are published; a trace that already
contains all recorded events is left byte-for-byte unchanged.

### Project operations

Project creation starts its trace only after the destination has been safely
reserved. The creation trace is owned by workspace preparation and is closed
before the GUI activates the project. Rollback removes trace files created by
an unsuccessful preparation attempt together with other new workspace files.
A creation conflict does not write to the occupied destination.

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
sink's lifetime.

These records provide project-operation history, not a workspace replay
implementation. Complete replay also requires explicit recording of
conversation operations and independent validation of recording completeness.
See [Trace recording](trace_recording.md) for activation and failure policy.
