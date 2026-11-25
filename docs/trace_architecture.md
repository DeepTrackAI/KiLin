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

## Scope of PR0A

The GUI maintains a project trace alongside the active conversation trace.
Conversation switching and archival retain their existing ownership rules.
Project-operation routing and complete recording of workspace operations are
introduced separately in PR0B; conversation operations are covered by PR0C.
This PR alone does not make a workspace replayable.
