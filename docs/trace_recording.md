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
explicitly enabled. The JSON format and existing event identifiers are
otherwise unchanged by this simplification.

For a direct model answer, `llm.call.input` records the resolved request
in both `thinking` and `chat_template.enable_thinking`, matching prompt
rendering and tokenization. Turn parameters instead retain `thinking` as
the configured session value used on reopening and record
`thinking_effective` separately. The latter is absent on tool-written
answers and older turns. These values describe the requested template
policy, not proof that a particular model complied. No extra event or
checkpoint is introduced for an intent override.

## Persistence and write failures

Each trace update writes through a uniquely created temporary file in the
same directory as its destination. Successful updates atomically replace the
existing JSON. Failed writes leave the earlier JSON in place and remove the
temporary file; this avoids collisions with predictable `.tmp` names and
pre-existing temporary-file symlinks.

Optional diagnostic event writes use the configured `failure_policy`. Required
conversation checkpoint writes propagate failures and restore the logger's
previous in-memory resume state. When the GUI updates active conversation
metadata, it delegates the write to the active trace so later diagnostic
events cannot undo a rename or timestamp change from an obsolete snapshot.
These are single-file guarantees, not transactions across metadata, model
state, or multiple files; no power-loss durability is claimed.

## Stopped and failed answers

**Stop** sends the `cancel_answer` command. Every model call of the answer
checks it after each generated token, so the answer ends at the next token
while the engine keeps running, and the project and models stay loaded. No
turn is saved: the conversation's `resume_state.turns`, which the chat shows
and the model remembers, only gains a turn at its checkpoint. The page keeps
the partial answer on screen, with its steps ending in "Stopped at your
request", and puts the question back in the message box.

A Stop is easy to find in the trace file. Under
`sessions["05_conversations"]["turns"]`, the turn's operation has `"status":
"failed"`, and its `result` has `"error_type": "GenerationCancelled"` and
`"message": "Stopped by the user."`. `sessions["06_runtime"]` also gets a
`gui_answer_cancelled` operation with `"reason": "Stopped by the user."`,
timed when the answer ended. Any other failure has a different
`error_type`.

Only if the engine does not end the answer within 30 seconds does the
window restart it as a last resort. The trace then keeps every event written
before the restart (each event except throttled progress updates rewrites
the file at once) as a turn that ends without a completion or failure
event, and the reopened file numbers later turns above it.

An answer that **fails** with an error records the failure in the trace and
adds no turn; the GUI shows the error and puts the question back. An answer
the model could not finish (for example, it reached its token limit) is
different: the trace records `generation.incomplete`, and the turn **is**
saved, with an answer explaining that no reliable result was produced and
the details under its `completion` parameter.

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

## Tests

The tracing contract is tested without network or model dependencies:

```sh
python -m pytest -q \
    tests/kilin/test_trace_logging.py \
    tests/kilin/test_rag.py \
    tests/kilin/test_conversation__lifecycle.py
```

The project targets Python 3.12; run these tests under that interpreter and
verify the OS-specific file-lock behavior on supported platforms.

## Performance analysis

For read-only timing analysis of existing conversation traces, and an optional
standardized-question notebook, see [Performance analysis](performance.md).
The analysis does not require a new trace mode or change the persistence
behavior described above. Raw traces remain sensitive project data.

## Project-type metadata

Profile snapshots use `type_id`; qualified type identifiers in turn
parameters and intent metadata use `project_type_id`. The prerelease
project-kind keys are not emitted. An empty source collection reports
`index.empty` and index status `empty` without fabricating passages or
saving a retrieval bundle. Source-entity `kind` metadata is unchanged.

## Project-creation ordering

The desktop engine creates a project before opening its conversation and
trace files. This prevents diagnostic setup from making an empty destination
nonempty or writing into an existing project before a collision is rejected.
A creation conflict produces the GUI `workspace_conflict` event without
changing the existing trace. Preparation failures before session switching
are displayed without appending an error to the previous project's trace.

Core creation can still receive a caller-owned external trace sink; it emits
preparation events only after obtaining the destination reservation. Do not
precreate a trace inside a destination intended for a new project. To trace
later source imports inside the project, create/open it first, start the
trace, and call `add_project_sources(..., trace=trace)`.
