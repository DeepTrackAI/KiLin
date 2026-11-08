# Extension interfaces and tool architecture

## Extension philosophy

KíLín is extensible through **explicit, validated components**, not by
loading arbitrary Python code mentioned in a project JSON. A new reader,
project type or registered tool must preserve provenance, settings and
workspace ownership and must not require Qt in the core. This document
explains the implemented extension points and where modification is needed;
there is no general-purpose runtime plugin discovery service in this
snapshot.

| Extension | Registration / owner | Main invariant |
| --- | --- | --- |
| Project type | `kilin/project_types/<type_id>.json` read by `project_type.py` | Validate all intents, tools, sources and prompts before use. |
| New source format | `sources/extensions/sources_<suffix>.py`, dispatched by `sources.py` | Return well-formed `SourceFile` / `SourceUnit` records with exact provenance. |
| Code graph language | `retrievals/code_graph/extensions/` | Extract structural hints without executing source code. |
| Built-in tool | `tools/<name>/` plus `tools/_registry.py` | Validate settings; honor `ToolRequest`, `ToolContext`, `ToolResult`. |
| Tool workspace asset | Tool's `workspace_resources(settings)` and `project_layout.py` | Declare safe relative destinations, install in core, preserve user edits. |
| Model backend | `chat.py`, `embeddings.py`, `runtime.py` | Maintain validated session, tokenizer and vector/generation contracts; not a JSON-only extension. |
| Retrieval strategy | `retrievals/strategies/` and `rag.py` integration | Preserve chunk alignment, ranking and evidence coordinates; not dynamically registered today. |

## Creating a project type

The shipped examples are `code.json` and `literature.json`. A new type is a
packaged `<type_id>.json` resource with validated `display_name`, `description`,
`extensions`, `max_file_bytes`, `retrieval_instruction`, `system_prompt`,
optional `answer_guidance`, `capabilities`, history settings, a valid
`default_intent`, an `intents` object and optional empty-chat prompts. The
loader derives the type ID from the filename, rejects duplicate JSON keys,
and validates **all** type files on load.

`capabilities` accepts `code_graph` (`true`, `false` or `"auto"`) and
`paper_index` (`true` or `false`, default `false`). With `paper_index`,
document passages are indexed with their paper's title, author, section and
page, each document gets a paper card, and intents may use the `papers`
retrieval strategy with `max_papers` (1-50, default 3) and
`passages_per_paper` (0-10, default 0); see
[Retrieval pipeline](retrieval_pipeline.md). A `papers` intent in a type
without `paper_index` is rejected.

The registry requires at least one valid definition, not either standard
type. `ProjectTypeId` is an unrestricted string alias and `ProjectType.id`
is the filename-derived identifier. The GUI takes its initial selection
from the first registry entry and derives labels and welcome content from
each definition. Adding a packaged type needs no built-in-specific branch.
External directories accepted by `load_project_types(directory)` are for
explicit validation; they are not automatically registered or hot-loaded.

A project type does not name the source folder: every new project uses
`sources/`. See
[workspace source convention](project_workspaces.md#source-directory-convention)
for the layout and how existing projects keep theirs. A new type cannot bypass
the core's creation-only ownership rules or authorize reuse of an occupied
workspace. Empty projects are
supported for every type. Missing welcome content has generic fallbacks;
project-type prompts, tools and retrieval policies still control answers.

Each intent key names an operation. Its fields may supply a description,
shortcuts, wording `cues` that set the intent, `requires` patterns a
question must match for the intent, history policy, retrieval
mode/strategy, instruction, and a tool reference. `Intent` is the validated in-memory record; its values are not
always a literal one-to-one copy of keys in JSON because the loader applies
defaults. Supported history modes are `none`, `recent`, `relevant`; supported
retrieval modes are `none`, `search`, `document`, `reuse`. The implemented
retrieval strategy is `hybrid`. A missing tool name or an unregistered name
fails validation rather than importing a module by string.

Ordinary intents can also declare `thinking: true`, `false`, or `null`.
An omitted or null field inherits the configured chat setting; a boolean
overrides only the direct answer request. Tool-backed intents must leave
this field omitted or null, because their model calls are tool-owned.
The parser rejects explicit overrides on tools. See the complete
[thinking policy and precedence](configuration.md#per-intent-thinking-policy).

Adding a new project type requires checking its source-selection defaults,
router prompts, installed tool requirements, selected template resources,
and how the GUI lists it. Merely adding a type ID is not proof that the
entire pipeline supports its source formats or domain reasoning.

## Source reader/extractor contract

`sources/sources.py` dispatches by supported extension to modules under
`sources/extensions/`. A reader may implement `read_source_file(path,
relative_path)` to load non-UTF-8-backed content (the PDF reader does).
An extractor implements `extract_source_units(...)` using the signature
expected by the dispatcher. Returned records must use consistent relative
paths, exact text, inclusive source-coordinate ranges and valid symbol,
section and cell fields. Reading a file does not execute its code.

When adding a suffix:

1. Specify binary versus UTF-8 reading and optional dependencies.
2. Implement deterministic units with stable order and correct provenance.
3. Add validation/dispatch and tests for empty, malformed and unusually
   large content; do not assume a file named `.pdf` is parseable.
4. Ensure citation previews can interpret the coordinates (especially PDF
   page-local and notebook cell-local line numbers).
5. Recheck chunk identities, source manifests and incremental index reuse.
6. Update project-type suffix defaults **only if** the project type should select
   the new format automatically; explicit files follow different policy.

Code relationship extractors for Python and MATLAB live under
`retrievals/code_graph/extensions/`; their results are heuristic, not
execution or complete semantic analysis. Adding another language may also
require source-unit conventions, graph entity typing and retrieval tests.

## Source-card kind contract (PR3)

`retrievals/source_cards/` introduces an optional, currently inactive
contract for whole-source cards. A `SourceDescriptor` carries the file's
identity and existing chunk identifiers, without assigning a content kind
from a filename suffix. `SourceCard` holds immutable metadata, substantive
statements with original-chunk provenance, and source/generation/validation
fingerprints. `SourceCardResult` records `valid`, `not_applicable` or
`failed` separately from card content.

A trusted Python `SourceCardKind` supplies `identifier`, `schema_version`,
`validation_version`, `precheck(source)` and `validate(card)`.
`precheck` must be conservative: `False` means definitely ineligible, while
`True` does not establish eligibility. It is not a model inference call.
`SourceCardRegistry` refuses duplicate/unknown names and obsolete card
versions, verifies generic provenance, then delegates to the kind's own
schema checks. Kind registration must be explicit in source code; project
JSON must not contain executable import paths.

PR3 established the kind and provenance contracts. PR4 adds an independent
index for cards and their vectors, with separate checked persistence and
incremental reuse. Neither `paper_index` nor the active `papers` retrieval
path uses it yet. A future project type can select registered kinds without
loading arbitrary modules. Scientific-paper registration, generation and
automatic indexing remain separate PRs. See [Retrieval pipeline](retrieval_pipeline.md#generic-source-card-contracts-pr3).

## Implementing a built-in tool

![Validated project intents resolve registered built-in tools, which consume ToolRequest and ToolContext and return ToolResult for checkpointing.](figures/tool-contract.svg)

*Figure: The tool registry, request/context contract, optional confirmation callbacks and answer persistence are explicitly owned interfaces.*

The [standard tool layout](tool_packages.md#standard-tool-layout) specifies
the package, configuration and resource files. `tools/_registry.py` pairs a
canonical `TOOL_*` identifier with its parser, runner, optional `follows_up`
behavior and optional `workspace_resources` callback. Project-type JSON can
select an existing entry, not load another implementation by string.

Implement an extension in this order:

1. Create the package, validated settings/parser and `run()` implementation.
   Document every owned default and supported request in its README.
2. Reuse private shared services where semantics match; do not import an
   executable sibling tool to reuse its policy.
3. Register the parser/runner and optional resources, then select the tool
   in an intended project-type intent. Registration and selection differ.
4. Add deterministic tests under `tests/kilin/tools/<name>/test_tool.py`
   for parsing, invalid inputs, results, registry selection and optional
   resource declarations. Shared helpers have their own `_shared/` tests.
5. Include JSON, README and resources in `pyproject.toml` package data and
   verify an installed wheel. Source-checkout imports alone miss this step.

The request/result contract below defines the integration boundary;
[tool generation settings](tool_packages.md#generation-settings-for-tools)
define the supplied model services. Extend the smallest necessary boundary
rather than building a parallel settings, retry or filesystem subsystem.

### Request and context

`ToolRequest` includes the user's question, search-relevant user text,
optional guidance and referenced earlier turns. `ToolContext` supplies the
project root; all indexed chunks; the code graph; a ranking callback;
retrieval count; source-file and unsearchable-file queries; controlled router
and answering-model calls; streaming output; progress/done callbacks;
confirmation; trace and turn ID; optional PDF writer; and optional document
choice callback. These are **capabilities supplied by the caller**, not
license to bypass project file ownership.

`ToolResult` has four fields:

- `answer`: final Markdown text, with numeric citations if evidence exists;
- `evidence`: provenance records numbered to match the answer;
- `summary`: JSON-compatible tool metadata saved with the turn;
- `streamed`: whether the answer was already emitted through the context
  writer (avoid duplicate output).

`parse_settings` should reject unknown names and invalid types before a
model run. `run` must return a valid result or propagate an explicit error.
A tool that needs many model calls should produce a `ToolPlan` and follow
the configured confirmation behavior. `continue`, `quick` and `cancel`
are exact user-choice identifiers; Quick may intentionally be a smaller,
nonexhaustive operation. Headless callers may not supply a confirmation
callback, so document the default policy and cost.

### Available built-in tools

| Tool ID | Main implementation purpose | Authoritative reference |
| --- | --- | --- |
| `project_inventory` | Summarize selected files and unsearchable source coverage. | [README](../kilin/tools/project_inventory/README.md) |
| `scan_and_list` | Count and list literal mentions with source locations. | [README](../kilin/tools/scan_and_list/README.md) |
| `batch_classify` | Classify many mentions in model-assisted batches. | [README](../kilin/tools/batch_classify/README.md) |
| `batch_review` | Read one document in parts and answer from notes. | [README](../kilin/tools/batch_review/README.md) |
| `filter_results` | Filter items of an earlier answer against a follow-up. | [README](../kilin/tools/filter_results/README.md) |
| `find_fact` | Find one fact or a complete set across a document/corpus. | [README](../kilin/tools/find_fact/README.md) |
| `find_passages` | Locate relevant literal passages/sentences. | [README](../kilin/tools/find_passages/README.md) |
| `paper_overview` | Produce a structured overview of a paper. | [README](../kilin/tools/paper_overview/README.md) |
| `reference_check` | Check a paper's references against the project's files, find their citations, and judge each claim against the cited paper. | [README](../kilin/tools/reference_check/README.md) |
| `suggest_reference` | Suggest papers for a statement from model-assessed passages; include checked quotations when available. | [README](../kilin/tools/suggest_reference/README.md) |
| `show_source` | Display requested exact source context. | [README](../kilin/tools/show_source/README.md) |
| `generate_paper_summary` | Create template-driven reports and saved artifacts. | [README](../kilin/tools/generate_paper_summary/README.md) |

These tools do not have identical output formats, costs or confirmation
policies. Consult their own READMEs, `defaults.json` and tests before
assuming reusable behavior.

## Workspace resources are declared, not blindly copied

Register `workspace_resources(settings)` when a tool needs packaged files.
It returns `WorkspaceResource` declarations; the tool does not install them.
The [resource contract](project_workspaces.md#tool-resource-contract) owns
package containment, safe destinations, conflict detection and preservation
of existing files.

A new template must be requested by the configured intent/tool, not copied
to every project merely because it ships in a package. Test both fresh
creation and reopening after a user edits or deletes the installed resource.
Explicit repair remains distinct from ordinary opening.

## Shared private services

`tools/_shared/` currently groups document selection, note-taking/citations,
prior-turn matching, mention and term scanning, sentence splitting,
reference-list extraction, planning,
path disambiguation, safe output-file access and source statistics. These
modules are private implementation helpers. Prefer them when semantics match,
but do not silently move a tool's policy into a shared service unless
multiple tools really need the same contract.

Document-reading tools validate passage citations through
`tools/_shared/citations.py`. Valid individual references and bounded
ranges may contribute evidence; invalid range parts never contribute
partial evidence. Numeric missing references remain in
`invalid_citations`, while malformed or invalid ranges appear in
`invalid_citation_groups` when present. Citation validation checks
identifiers, not factual support. The maximum expansion of an individual
citation range is 100 passages.


The report-generation package has an additional `_report_template.py` and
`_report_format.py` to validate Markdown templates and report formatting.
The GUI offers PDF rendering through its optional `_report_pdf.py`; core
tools should still function without Qt (Markdown output remains available).

## Extending models or retrieval

This is source-code extension, not a configurable plugin point. A new model
backend must implement cache resolution, model/tokenizer loading, chat
formatting, generation, embedding normalization, explicit download control,
cleanup and tests on supported hardware. A new retrieval strategy must
preserve exact `Chunk` references, source positions, indexing identity and
fusion compatibility and be integrated with `RAGSession`; changing the
strategy can invalidate answer-ranking expectations even without changing
an index file format.

## Tests and review checklist

- Register explicitly; reject unknown tools/settings/intent references.
- Validate function return types and source provenance; handle malformed
  source/LLM output without hidden retries.
- Keep model access, source reading and filesystem work in core services.
- Check tool `summary` JSON serializability and citation/evidence numbering.
- Confirm resource destinations cannot traverse outside the project or
  overwrite edited user files.
- Preserve `kilin` imports without Qt and test the headless path.
- Test both project creation and reopened profiles; no unintended migration.
- Update tool README, local module docstrings and the module map.

Use [tool dependency checks](../tests/kilin/tools/test__architecture.py),
[resource contracts](../tests/kilin/tools/test__resources.py),
[project-type contracts](../tests/kilin/test__project_type_contract.py) and
the affected package's tests. Update the consuming intent, module docstrings,
local README, package-data declaration and [module map](module_map.md)
together. Source-format changes also require coordinate/citation and cache
checks, as described in [Source pipeline](source_pipeline.md#change-impact-and-verification).
For optional GUI adapters, run supported-host Qt smoke tests separately.
See [Development](development.md) for full verification.

## Related

[Tool packages](tool_packages.md),
[Answering pipeline](answering_pipeline.md),
[Source pipeline](source_pipeline.md),
[Configuration](configuration.md).
