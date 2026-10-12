# Diagnostic trace recording

QAChat uses **one diagnostic tracing behavior**. Enabled traces record all
available structured diagnostic events, including complete prompts, model
responses, retrieved content and token IDs. There is no standard/forensic
mode switch and no automatic redaction or sanitization.

## Enabling or disabling tracing

A `RAGSession` creates an enabled trace when `trace=None` (the default):

```python
session = qachat.RAGSession(project_directory)
```

A low-level `TraceConfig` is disabled by default; activate it explicitly when
constructing a `TraceLogger` or supplying configuration to another caller:

```python
from pathlib import Path
import qachat

path = Path("/path/to/workspace/logs/traces/conversation.json")
config = qachat.TraceConfig(enabled=True, output_path=path)
logger = qachat.TraceLogger(config, default_directory=path.parent)
try:
    logger.emit("custom.ready", component="example", data={"ok": True})
finally:
    logger.close()
```

Pass `qachat.TraceConfig(enabled=False)` to a RAG session to disable diagnostic
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
properties true. A disabled `NullTraceLogger` makes them false, so callers
to avoid constructing expensive diagnostic artifacts. These are sink capability
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

## Migration from earlier trace configurations

Remove any `mode=...` argument from `TraceConfig` construction and any imports
of `TraceMode`, `TRACE_MODE_STANDARD`, `TRACE_MODE_FORENSIC`,
`TRACE_MODE_CHOICES`, or `DEFAULT_TRACE_MODE`. These options are no longer
part of QAChat's Python API. The `TRACE.DEFAULT_MODE` entry is also removed
from `qachat_defaults.json`; no replacement setting is needed. The change
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
    tests/qachat/test_trace_logging.py \
    tests/qachat/test_rag.py \
    tests/qachat/test_conversation__lifecycle.py
```

The project targets Python 3.12; run these tests under that interpreter and
verify the OS-specific file-lock behavior on supported platforms.
