# Answering, routing, tools and conversation lifecycle

## Top-level API and state ownership

`kilin.rag.RAGSession` is the coordinator for one opened project,
retrieval state, live chat/embedding sessions, optional separate router and
one active conversation. The Python entry point for a question is:

```python
import kilin

session = kilin.RAGSession("/path/to/existing/project")
try:
    answer = session.ask("Which module builds the lexical index?")
finally:
    session.close()
```

The path is illustrative; opening a session can load models and may access
the Hugging Face cache/network according to the project's settings. For
reproducible local-only execution set `allow_download = false` in the saved
profile before opening and ensure models are cached.

`RAGSession` owns the authoritative `ConversationTurn` list, selected
conversation path and project/index state. `ChatSession` separately owns the
loaded answering model and in-memory formatted messages; the router may
share that session or be a separate loaded chat model. The optional GUI
adapts callbacks and persists presentation-only metadata. A `ToolContext`
provides scoped operations for registered built-in tools, not arbitrary
filesystem or Python execution.

## One turn, in execution order

![Sequence of routing, evidence acquisition, a single model or tool answer producer, citation checks and an atomic completed-turn checkpoint.](figures/answer-sequence.svg)

*Figure: An ordinary successful turn passes through an optional evidence stage, one answer producer, citation-ID checks and persistence before completion.*

```text
ask(question, options)
   |
   +-- validate settings, mark token usage, start trace/turn
   +-- route_question(): shortcut or router model -> RouteDecision
   +-- resolve intent and related turn IDs / history policy
   +-- select retrieval mode and a search query
   |     +-- search -> ranked chunks and named-file context
   |     +-- document -> selected full document with budget
   |     +-- reuse -> previous evidence
   |     `-- none -> no new evidence
   +-- choose exactly one answer producer
   |     +-- configured built-in tool -> ToolResult
   |     `-- ChatSession.reply() with history + evidence prompt
   +-- deterministic citation-ID check; collect token usage
   +-- form ConversationTurn + compact representation
   +-- persist checkpoint; append committed turn
   `-- notify completed phase (observer)
```

The state machine uses `ANSWER_PHASE_*` identifiers (`routing`, `routed`,
`retrieving`, `evidence_ready`, `generating`, `saving`, `completed`, and
`processing`). Not every intent performs every phase; tools may control
additional progress and confirmation. See [GUI protocol](gui_protocol.md)
for event wire fields and the distinction between backend and frontend
operation states. A callback's failure **before** commitment can abort the
turn; a failure in the final completed notification is traced but should not
recast a successfully committed answer as a failed generation.

## Intent routing and fallback

`answering.route_question()` takes the question, project type and compact
representations of prior turns. First, a configured shortcut pattern can
match without an additional LLM call. Otherwise a router model produces a
structured decision (`RouteDecision`) with intent, standalone/follow-up
judgment and related prior-turn identifiers. Parsed IDs are checked against
available history and additional reference or distinctive-term heuristics;
not every related ID proposed by the model is automatically accepted.
Invalid or unusable router output falls back to the project's default
intent rather than interpreting an arbitrary tool name.

Project types currently package several intents for `code` and `literature`:
a mixture of normal answering, named-document reading, source inventory,
mention finding/classification, report generation, related-turn reuse and
small talk. Intent JSON selects retrieval and history behavior, a system
instruction, optional answer guidance, a request-local thinking policy,
and possibly a registered tool.
`project_type._parse_intent()` defaults to `search` for ordinary intents,
`none` for tools if no mode is specified, `hybrid` strategy and `relevant`
history. Always inspect the validated `Intent` record rather than relying
on optional fields being present verbatim in a JSON file.

`use_router=False` bypasses classification and uses the project's default
intent. A GUI effort selection or router-model choice is not itself an
intent. Router calls have their own token limits and can increase answer
latency; see [Package defaults](package_defaults.md).

## History and follow-up semantics

Saved turns include original question, answer, evidence, timestamp and
parameters. `turn_representation()` constructs a compact summary used when
routing later questions. `refers_back()` identifies follow-up phrasing;
`search_text()` incorporates earlier user wording according to the verified
decision. `select_history()` applies the intent's policy:

- `none`: do not send previous turns to the answering prompt.
- `recent`: send a recent bounded window.
- `relevant`: choose validated related turns where available.

The prompt history contains selected prior **questions and answers**, not
full prior evidence payloads. Related-turn evidence may be carried forward
separately when an intent permits it. This distinction affects privacy,
context size and the interpretation of follow-up answers. The router's
summary is derived from saved turns; it is not an automatically generated
long-term memory system.

## Retrieval options and evidence

`ask()` supports per-turn `retrieval_count`, `retrieval_query`,
`reuse_evidence`, answer guidance and generation settings. The original
question remains the question to answer even when search text differs.
`RAGSession` then chooses among `search`, `document`, `reuse` and `none`.
The retrieved evidence supplies exact text and source coordinates, with
citation numbers referenced by the answer. A named-document operation may
use extra document-scoped ranking or a full-document token budget; its
semantics cannot be emulated reliably by global top-k alone.

The evidence budget is not a guaranteed universal hard cap for every
prompt. Retrieved chunks, related evidence, full-document selection, chat
history and tool notes have different limits. See
[Retrieval pipeline](retrieval_pipeline.md) for how each channel and
`DocumentSelection` works.

## Answer production: model versus tool

For a normal answer, KíLín rebuilds `ChatSession` history from the selected
turns and project/intent system prompts, constructs an evidence prompt and
calls `ChatSession.reply()`. Generation settings permit validated
`thinking`, `temperature` and `max_new_tokens`. The implementation restores
chat history if a pre-commit failure would otherwise leave uncommitted
messages in memory. A model's `finish_reason = length` may be stored to
indicate output truncated by its generation cap.

An intent can override thinking for this direct model answer only. A
missing or null policy inherits the session setting; an explicit boolean
is passed to `ChatSession.reply(thinking=...)` without reconfiguring the
session. The configured `thinking` remains resumable state, while
`thinking_effective` records the requested value for this answer. Tool
intents cannot declare explicit overrides. See
[Per-intent thinking policy](configuration.md#per-intent-thinking-policy).

For a tool intent, `tools/_registry.run_tool()` invokes the registered
package with a `ToolRequest` and a `ToolContext`. It returns `ToolResult`
containing `answer`, numbered evidence, summary metadata and the `streamed`
flag. `streamed=True` means the tool already sent output through the
provided writer and it must not be emitted again. The context can supply
controlled model calls, ranking, notes, output-file access and confirmation.
It is not a generic arbitrary plugin or unrestricted system command API.

Some tools plan potentially large operations and ask the GUI for a
`continue`, `quick` or `cancel` choice via `ToolPlan`. The engine may accept
a `plan_choice` command **during** an active answer. A headless call with no
confirmation callback follows the tool's specified default execution path.
See [Extension interfaces](extension_interfaces.md) and
[Tool packages](tool_packages.md).

`generate_paper_summary` may request PDF output through an **optional**
`pdf_writer` supplied by the GUI. The core tool cannot presume PySide6 or
Qt is installed. Any PDF page-count and metadata validation should be
performed by the writer/tool according to their own documented contracts.

## Citation checks and answer truthfulness

After model/tool output, `_invalid_citations()` checks whether cited numeric
identifiers correspond to supplied evidence. Invalid references are recorded
in turn parameters. This is a **diagnostic**, not semantic verification of
scientific claims, and it does not trigger a second generation attempt or
auto-correction of the text. An answer may contain unsupported or mistaken
claims even if all citation indices resolve.

The GUI's citation previews are derived from saved exact evidence and source
coordinates. They must not silently substitute a different file or invent
supporting text if the source has changed since indexing.

## Persistence and exactly one committed turn

`_save_conversation()` writes resumable state using the active trace sink or
a dedicated atomic file writer when tracing is disabled. A successfully
committed turn is appended once to the session; a failure before commitment
must not be misreported as an answer. Enabled traces and saved conversation
state share the `conversation_*.json` representation; tracing-disabled
sessions still write their checkpoints and own the conversation lock.

`ConversationStore` manages multiple named conversations (create, select,
rename, touch, archive). `RAGSession.switch_conversation()` loads a different
conversation without intentionally reloading models/indexes; it acquires and
validates the next file before releasing the previous owner. The GUI's exact
conversation resume takes an optional `conversation_id`; without one it
opens the most recently updated record. Unknown explicit IDs must fail,
not silently choose an unrelated conversation.

`reset()` archives and clears the active conversation. A GUI Stop terminates
the child process and therefore may discard an incomplete, uncheckpointed
answer; it does **not** resume partially generated tokens. Locking, trace
ownership and failure recovery are specified in
[Conversation ownership](conversation_ownership.md) and
[Trace recording](trace_recording.md).

## Instrumentation and failure boundaries

Trace spans cover routing, retrieval, model loading and generation; content
traces can include prompts and source excerpts. Model backend failures,
missing dependencies, invalid tool results, source errors and checkpoint
I/O errors propagate through the core API. Do not add automatic retries or
additional router/model calls just to hide these failures: that would
change both results and latency.

Relevant regression areas: `test_rag.py`, `test_answering.py`,
`test_conversations.py`, `test_trace_logging.py`, tool-package tests and
`tests/kilin/gui/` protocol/recovery tests. Exact test filenames should be
confirmed against the current tree before running an isolated subset.

Related: [Architecture](architecture.md),
[Retrieval pipeline](retrieval_pipeline.md),
[Model management](model_management.md),
[Tool packages](tool_packages.md), [GUI architecture](gui_architecture.md).

### Tool model-call inheritance

Registered tools use `ToolContext.ask_model` for internal judgments and
`ToolContext.write` for streamed final answers. Both preserve the current
answering model's thinking and temperature, while the per-call output
limit does not exceed either the tool budget or the session's
`max_new_tokens` setting. The router keeps its separate deterministic
nonthinking behavior. Isolated tool replies never enter persistent chat
history.
