# Modular tools in KíLín

## Ownership and boundaries

KíLín's registered tools are Python subpackages under `kilin/tools/`.
The central registry (`kilin/tools/_registry.py`) connects a stable tool
identifier from a project type to `parse_settings` and `run`. It can also
register an optional `workspace_resources` function. Project-type JSON may
select and configure registered tools, but cannot load arbitrary code.

Each tool owns its implementation, its validated packaged `defaults.json`,
and its local `README.md`. When a tool needs packaged templates, those
resources also live in its package. The workspace planner rejects files
outside their declaring tool package. Only the configured resources are
installed into new project workspaces. The core always creates the
`templates/` folder, including when the selected tools have no templates.

The following files and directories form private infrastructure rather
than tools callable by project types:

- `_contract.py`: request, context, result, and validation contracts.
- `_registry.py`: controlled registration and tool selection.
- `_defaults.py`: cached, defensive loading of tool-owned JSON files.
- `_shared/`: reusable source-processing, conversation, planning, and
  output services. Shared algorithms never import executable tool packages.

Do not import one executable tool to reuse an implementation detail from
another. Extract the reusable operation into a private `_shared/` module
with its own tests. Tool packages may import private shared services,
but shared infrastructure should not depend on a tool's configuration.

## Standard tool layout

```text
kilin/tools/example_tool/
    __init__.py        # Tool's documented exports
    tool.py            # Settings dataclass, parser, runner
    _configuration.py  # Strictly validate defaults on import
    defaults.json      # Tool-owned settings and internal parameters
    README.md          # Explanation of every default
    templates/         # Only when this tool ships resources
```

The reference `generate_paper_summary` tool additionally has a private
report-template parser and a pure report HTML formatter. Qt-based PDF writing
remains an optional private GUI adapter (`kilin/gui/_report_pdf.py`), passed
to the tool as a callback. The tool does not import the GUI, and it
saves validated Markdown even without a PDF backend. Other tools can add private
modules as needed.
All registered built-in tools use packages, not flat `.py` modules. Use
their documented package exports; obsolete flat-file import paths have no
compatibility aliases.

## Generated output safety

Tools that create files use the private shared output service in
`kilin/tools/_shared/outputs.py`. Names must be single, portable basenames;
absolute paths, parent traversal, path separators, and Windows device names
are rejected. Existing filenames and symbolic links are not overwritten.

Markdown and optional PDF reports are written to exclusively created,
hidden staging files inside the project's `outputs/` directory. Complete
files are preferably published through an atomic, no-replace hard link.
Where the filesystem cannot create hard links, an exclusive-create copy
preserves the no-overwrite guarantee, though the copy is not atomic. If a
name is already occupied, the next numbered name is used. Staging files
are cleaned up on both success and failure. Tool writers must not choose
a free name and open that path for ordinary truncating output.

The PDF callback receives a staging filename, **not** the eventual public
filename. A renderer can replace the temporary file's contents; only the
tool publishes the finished report. If PDF rendering fails, the saved
Markdown report remains available.

## Configuration precedence

A tool's effective settings are resolved before `run()` through the
[configuration layers](configuration.md#tool-configuration-and-resources).
That page owns packaged/intent/project precedence, `[tool.<name>]`
persistence, reset behavior and the restriction on differing per-intent
values. The tool remains responsible for validating all accepted settings.

Internal sections such as `MODEL_CALLS` and `ESTIMATES` are tool-owned,
process-cached resources, not settings that a project can override. Their
exact values and constraints belong in the tool's README and
`_configuration.py`. Invalid JSON or missing, unknown, mistyped or
out-of-range values must fail validation rather than change policy silently.

## Retrieval evidence budgets and tool-owned limits

Source-first retrieval budgets (`max_sources`, `passages_per_source`,
`max_passages`, `evidence_tokens`, `coverage`) belong to an ordinary
model-answering intent's `retrieval` object in its project-type JSON.
They are not generic tool inference limits. An intent with a `tool`
executes the registered tool in place of the ordinary answer pipeline;
its own validated `part_tokens`, batch sizes, number of model calls,
confirmation policy and other exposed settings apply instead.

Tools can reuse the pure `kilin.retrievals.evidence_budget.EvidenceLimits`
and `allocate_evidence` functions for **their own** source/passage limits,
but must explicitly expose, validate and document any such settings in
their own tool package. Existing tools do not silently inherit source-first
budgets.

See [Source-first configuration and coverage](retrieval_pipeline.md#per-intent-source-first-configuration)
for exact accepted values, limits, estimate semantics, and coverage notices.

## Tool retrieval and settings

`ToolContext.rank(query)` remains a complete hybrid ranking of original
chunks. For tools needing source-card discovery and answer-ready bounded
evidence, `ToolContext.retrieve(RetrievalRequest(...))` is an optional
service. `RetrievalRequest` from `kilin.retrievals.requests` contains the
strategy, explicit `EvidenceLimits`, source scope, optional comparison, and
optional registered card-kind filter; `RetrievalResult` contains original
chunks, selected source paths, and coverage metadata. No answering model is
called by retrieval. Tools do not automatically inherit the ordinary
question-answering intent's evidence budgets.

For `suggest_reference`, the tool-local `SETTINGS` values in
`defaults.json` define `DISCOVERY_STRATEGY` (default `hybrid`) and
`DISCOVERY_TOKENS` (default `8000`). The project-type intent's `tool` object
may override these settings, followed by saved project `[tool.name]`
settings. When `source_first` is explicitly selected, candidate papers
are source-ranked using scientific-paper cards where present. The tool
continues to verify claims against separately ranked original passages;
that second passage ranking can incur another query embedding. Its existing
`FILES`, `PASSAGES`, and `PASSAGE_TOKENS` remain authoritative for how many
papers are judged and how much content is read for each judgment.

Other tools, especially exhaustive scanners and full-document readers,
retain their existing retrieval behavior. A selected source does not imply
that every passage was read or that the source supports the claim.

## Creating a tool

Follow the [registration and extension checklist](extension_interfaces.md#implementing-a-built-in-tool)
for the parser, runner, registry entry, project intent, packaging and tests.
The layout and shared-service boundary above remain part of that contract.
A tool needing workspace files declares resources; it does not copy them
from a GUI handler. Tool-specific parameters, outputs and confirmation cost
belong in its own README rather than in a second global tool manual.

## Generation settings for tools

The implementation owners are `tools/_model_calls.py`,
`answering.side_reply()` and the adapters in `RAGSession._run_tool()`.
The policy separates internal judgments from final visible writing while
preserving the active answering model, temperature and user allowance.

| Adapter | Thinking policy | Returned and displayed text |
| --- | --- | --- |
| `ToolContext.ask_model` | Tool-owned `MODEL_CALLS.THINKING` | Returns thinking-free text for parsers; completed thinking is captured separately for display and persistence. |
| `ToolContext.write` | Inherits the active session setting | Streams and returns the full user-visible reply, including thinking. |
| `ToolContext.ask_router` | Separate deterministic, nonthinking router policy | Isolated router reply; not controlled by the tool's answering-model policy. |

Side calls restore isolated conversation history on success and failure;
internal temporary thinking overrides are restored as well. Their replies
are not inserted as extra persistent conversation turns. Models still need
to support the existing template control; a policy request does not guarantee
that an unsupported model obeys it.

### Internal thinking policy

Each tool declares one `MODEL_CALLS` section in its packaged `defaults.json`:

```json
{
  "MODEL_CALLS": {
    "THINKING": "inherit"
  }
}
```

This is a valid excerpt, not a complete replacement defaults file.
`THINKING` accepts `inherit`, `on` or `off` for internal `ask_model` calls.
There is no short-reply threshold or token-based thinking switch, and `write`
is unaffected. See each tool's README/defaults for its chosen policy: the
selection tools and document-processing tools can deliberately opt out of
thinking for machine-readable requests without changing the user's main
session. Changing the packaged policy requires a process/kernel restart.

### Output-limit resolution

The allowance is **per call**, includes thinking and final output together,
and incorporates the active per-question `generation_settings`. It is not
a target prose length or an aggregate allowance for a complete tool run.
Input passage/batch/reading budgets and report-template word limits have
different purposes.

1. A thinking call uses the active session maximum, ignoring a local cap.
2. A nonthinking call without a local cap also inherits that maximum.
3. A nonthinking call with a cap uses the smaller of that cap and the
   session maximum. It cannot silently raise the user's selected maximum.

With a session maximum of `None`, inherited calls keep the backend/context-
derived budget; an explicit nonthinking local cap still applies. Caps must
be positive integers, excluding booleans. User limits are not rounded or
restricted to powers of two, and no minimum generation length is imposed.

Use `None` for ordinary analysis, source explanations, document notes,
report sections, reference judgments and variable-length selection lists.
A small cap is reserved for genuinely short metadata. The report tool's
`WRITING.HEADLINE_TOKENS` and `WRITING.REFERENCES_TOKENS` are such exceptions:
they apply only when thinking is off and are documented in its
[README](../kilin/tools/generate_paper_summary/README.md).

Router intent decisions, document selection, term/topic extraction,
category/statement extraction and GUI conversation titles have separate
nonthinking limits; they are not capped by the answering session's maximum.
Their owners are the [answering defaults](package_defaults.md#answering--routing-and-conversation-heuristics),
shared tool services and the GUI title helper. Do not copy these limits into
a new universal tool budget. A tool's `ESTIMATES.*_REPLY_TOKENS` values are
planning estimates only, particularly approximate when thinking is enabled.

### Completion, display and accounting

Output-limit resolution itself never retries a call. The core's separate
[completion and recovery contract](answering_pipeline.md#model-output-completion-and-bounded-recovery)
permits one narrowly eligible internal thinking recovery; it excludes final
streaming `write`, direct answers and routers. Both attempts count toward
actual usage, even though planning and existing `model_calls` counters count
logical requests.

Successful internal replies are stripped of thinking before tool parsers
consume them. The coordinator emits captured thinking after each completed
call, not the intermediate structured reply, and saves it before the tool's
composed answer. Failed/truncated text remains diagnostic, never parser
input or successful thinking displayed to the user. `write` retains the
full streamed reply. The tool summary records `internal_generation.policy`
and `internal_generation.calls` with requested thinking, caps, attempts and
completion status; model traces record the effective allowance. See
[trace records](trace_recording.md#recorded-data) for diagnostic ownership.

### Compatibility notes for customized tool configuration

Customized intent tool objects must not retain `answer_tokens`,
`note_tokens` or `section_tokens`. The corresponding packaged `SETTINGS`
keys `ANSWER_TOKENS`, `NOTE_TOKENS` and `SECTION_TOKENS` are not accepted.
Unknown settings fail validation rather than overriding the user allowance
or becoming ineffective settings.

`MODEL_CALLS` accepts only `THINKING`; remove `SHORT_REPLY_THINKING` and
`SHORT_REPLY_MAX_TOKENS` from older customized files. The old selection caps
`find_passages.JUDGE_REPLY_TOKENS`, `filter_results.REPLY_TOKENS` and
`batch_classify.REPLY_TOKEN_LIMIT_PER_ITEM` must also be removed from
customized estimate groups; their validators no longer accept those keys.
Use the current validated defaults and restart the process/kernel after
edits. These settings changes require neither source reindexing nor
rewriting historical answers.

## Partial tool completion

A tool may catch `IncompleteModelResponse` at an independent item boundary
and retain checked successes using the shared `partial_result` helper. The
[completion contract](answering_pipeline.md#independent-work-and-partial-results)
owns status meanings, dependency failure, history reuse and partial-report
publication. Tools must not add another generation retry, parse truncated
text or turn an unassessed item into a negative finding.

Keep the normal evidence/template checks in that path. Output publication
still follows [generated output safety](#generated-output-safety); a partial
result does not authorize overwriting a completed report.

## Change impact and verification

The adapter boundary keeps model-policy changes out of individual tools.
When changing it, review `ToolContext`, `_model_calls.py`, `side_reply` and
`RAGSession._run_tool` together, then inspect each affected tool parser.
New caps, retries or model calls change user cost and completion behavior;
they are not merely implementation cleanups.

Run [model-call policy tests](../tests/kilin/tools/test__model_calls.py),
[budget inheritance](../tests/kilin/test_rag__tool_budgets.py),
[thinking capture](../tests/kilin/test_rag__tool_thinking.py),
[bounded recovery](../tests/kilin/test_rag__tool_recovery.py),
[partial results](../tests/kilin/tools/test__partial_recovery.py) and
[output publication](../tests/kilin/tools/_shared/test_outputs.py), plus the
affected tool's tests. Shared-service tests live under
`tests/kilin/tools/_shared/`; native PDF acceptance remains a separate GUI
check. Preserve a useful headless path without Qt.

## Related

[Extension interfaces](extension_interfaces.md), [Configuration](configuration.md),
[Answering pipeline](answering_pipeline.md), [Project workspaces](project_workspaces.md).
