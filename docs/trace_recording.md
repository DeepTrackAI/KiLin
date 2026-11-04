# Diagnostic trace recording

KíLín uses **one diagnostic tracing behavior**. Enabled traces record all
available structured diagnostic events, including complete prompts, model
responses, retrieved content and token IDs. There is no standard/forensic
mode switch and no automatic redaction or sanitization.

## Enabling or disabling tracing

A `RAGSession` creates an enabled trace when `trace=None` (the default):

```python
session = kilin.RAGSession(project_directory)
```

A low-level `TraceConfig` is disabled by default; activate it explicitly when
constructing a `TraceLogger` or supplying configuration to another caller:

```python
from pathlib import Path
import kilin

path = Path("/path/to/workspace/logs/traces/conversation.json")
config = kilin.TraceConfig(enabled=True, output_path=path)
logger = kilin.TraceLogger(config, default_directory=path.parent)
try:
    logger.emit("custom.ready", component="example", data={"ok": True})
finally:
    logger.close()
```

Pass `kilin.TraceConfig(enabled=False)` to a RAG session to disable diagnostic
events. This **does not disable conversation persistence**. The session still
holds exclusive ownership of its conversation path and saves checkpoints
there; see [conversation-file ownership](conversation_ownership.md).

`TraceConfig.include_embedding_vectors=True` is an independent opt-in for
full numerical embedding vectors, which can be large. `failure_policy` stays
independent and controls optional event-write errors (`"warn"` or `"raise"`);
checkpoint write failures always propagate. Output location and tracing
activation also remain configurable.

### Project-creation ordering

The engine creates a project before opening its trace/conversation files;
precreating a trace inside a fresh destination would make that destination
occupied. Core preparation can use a caller-owned external sink and emits
preparation events only after reservation. For a later import, create/open
the project first, start the trace, then call
`add_project_sources(..., trace=trace)`.

GUI creation conflicts and preparation failures before session switching
do not append errors to the previous project's trace. The
[GUI activation lifecycle](gui_architecture.md#creation-conflicts-and-activation)
owns session-switch ordering and the
[workspace contract](project_workspaces.md#failure-behavior-and-limitations)
owns filesystem recovery.

## Recorded data

An enabled logger always makes its `include_content` and `include_token_ids`
properties true. A disabled `NullTraceLogger` makes them false, allowing
callers to avoid constructing expensive diagnostic artifacts. These are sink capability
properties, not configurable tracing modes. Both built-in and custom events
are recorded without mode-based filtering.

A grouped `conversation_*.json` file holds diagnostic events and resumable
conversation state. The latter stores the question, answer, evidence, history,
and checkpoint identity needed to reopen the conversation. Disabling tracing
does not remove this state. A legacy `"mode": "forensic"` string is still
emitted in the `trace.started` event for readers of older files; this is a
fixed historical metadata value, not a configuration option. Existing JSON
files with a `"standard"` or `"forensic"` marker remain readable. Previously
stored events are preserved when reopening, without retroactive filtering.

Tracing only captures information that the execution components expose:
external backends and custom tools may not offer every diagnostic detail.
Token IDs are included when available; embeddings are included only when
explicitly enabled. Diagnostic coverage follows the producing component;
the logger does not synthesize missing backend details.

For a direct model answer, `llm.call.input` records the resolved request
in both `thinking` and `chat_template.enable_thinking`, matching prompt
rendering and tokenization. Turn parameters instead retain `thinking` as
the configured session value used on reopening and record
`thinking_effective` separately. The latter is absent on tool-written
answers and older turns. These values describe the requested template
policy, not proof that a particular model complied. No extra event or
checkpoint is introduced for an intent override.

Every tool run is recorded as a `tool.run` span in the turn: its `name`, the
`settings` it ran with (including any the project changed in Settings →
Tools), its `internal` values (the other sections of its `defaults.json`,
such as `MODEL_CALLS`, which only the tool sets), how long it took, and
whether it completed or failed. Saving tool
settings records `project.tool_settings.saved` with the tool and its changed
values.

### Project-type metadata

The [identifier schema](configuration.md#canonical-identifiers-and-schemas)
distinguishes profile `type_id` from qualified `project_type_id` in turn and
intent metadata; former project-kind keys are not emitted. An empty source collection reports
`index.empty` and index status `empty` without fabricating passages or
saving a retrieval bundle. Source-entity `kind` metadata is unchanged.

## Persistence and write failures

Trace updates use the [atomic-write and ownership contract](conversation_ownership.md#atomic-writes-and-cleanup),
not an independent persistence mechanism. The distinction here is between
optional diagnostic events and mandatory resumable-state checkpoints.

Optional diagnostic event writes use the configured `failure_policy`. Required
conversation checkpoint writes propagate failures and restore the logger's
previous in-memory resume state. When the GUI updates active conversation
metadata, it delegates the write to the active trace so later diagnostic
events cannot undo a rename or timestamp change from an obsolete snapshot.
These are single-file guarantees, not transactions across metadata, model
state, or multiple files; no power-loss durability is claimed.

## Stopped and failed answers

An acknowledged cooperative cancellation saves no turn: the conversation's
`resume_state.turns` gains a turn only at its checkpoint. The GUI can retain
partial text and show "Stopped at your request" without that text becoming
resumable history. The [Stop protocol](gui_protocol.md#errors-stop-and-restart)
owns the request, token-boundary checks, completion race and forced fallback.

A Stop is easy to find in the trace file. Under
`sessions["05_conversations"]["turns"]`, the turn's operation has `"status":
"failed"`, and its `result` has `"error_type": "GenerationCancelled"` and
`"message": "Stopped by the user."`. `sessions["06_runtime"]` also gets a
`gui_answer_cancelled` operation with `"reason": "Stopped by the user."`,
timed when the answer ended. Any other failure has a different
`error_type`.

If the window must terminate the engine, the trace retains events already
written before termination. A mid-turn exit can leave an operation without
a completion/failure event; reopening numbers later turns above it. Events
other than throttled progress updates rewrite the file as they are emitted,
but unwritten events cannot be recovered. Do not infer a successful answer
from the mere presence of a traced turn operation.

An answer that **fails** with an error records the failure in the trace and
adds no turn; the GUI shows the error and puts the question back. An answer
the model could not finish (for example, it reached its token limit) is
different: the trace records `generation.incomplete`, and the turn **is**
saved, with an answer explaining that no reliable result was produced and
the details under its `completion` parameter. The
[completion contract](answering_pipeline.md#model-output-completion-and-bounded-recovery)
owns incomplete versus partial outcomes and their use in later history.

## Migration from earlier trace configurations

Remove any `mode=...` argument from `TraceConfig` construction and any imports
of `TraceMode`, `TRACE_MODE_STANDARD`, `TRACE_MODE_FORENSIC`,
`TRACE_MODE_CHOICES`, or `DEFAULT_TRACE_MODE`. These options are no longer
part of KíLín's Python API. The `TRACE.DEFAULT_MODE` entry is also removed
from `kilin_defaults.json`; no replacement setting is needed. The change
does not migrate, rewrite, or delete older conversation files.

## Privacy, storage, and deletion

Traces are private *only to the extent that their files and environment are
private*. Their content can include secrets, confidential code or documents,
model prompts and outputs, traceback contents, local paths, and tool stdout.
There is no privacy-safe export or automatic scrubbing. Keep these files out
of public repositories, control filesystem permissions, and treat backups
and archived conversations as sensitive.

Deleting a conversation JSON also deletes its persisted resume state, not
just diagnostic events. Archived copies, backups, and external copies may
remain after deletion. Ordinary file deletion is not guaranteed secure
physical erasure. For the GUI's archival and lock rules, see
[conversation-file ownership](conversation_ownership.md).

## Performance analysis

For read-only timing analysis of existing conversation traces, and an optional
standardized-question notebook, see [Performance analysis](performance.md).
The analysis does not require a new trace mode or change the persistence
behavior described above. Raw traces remain sensitive project data.

## Tests

The tracing contract is tested without network or model dependencies:

```sh
python -m pytest -q \
    tests/kilin/test_trace_logging.py \
    tests/kilin/test_rag.py \
    tests/kilin/test__conversation_lifecycle.py
```

The project targets Python 3.12; run these tests under that interpreter and
verify the OS-specific file-lock behavior on supported platforms.


### Change impact and verification

Diagnostic writes can fail without the same policy as a required checkpoint.
Preserve that distinction when changing trace event names, projection or
write frequency. Review `trace_logging.py`, `conversation_format.py`, the
producer and any consumer such as `assets/trace_benchmark.py` together;
trace layout changes can break analysis even when the visible answer works.

Run the suites above and [benchmark parsing](../tests/test__trace_benchmark.py).
Use fault injection to check checkpoint failures, optional write failures,
reopen and active metadata updates. When the shared conversation structure
or lock ownership changes, also follow
[conversation verification](conversation_ownership.md#regression-testing).
Timing optimization must not silently weaken checkpoint or diagnostic
visibility guarantees.

## Related

[Conversation ownership](conversation_ownership.md), [Answering](answering_pipeline.md),
[Performance](performance.md), [Security](security.md).
