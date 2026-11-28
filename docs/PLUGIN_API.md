# Python tool plugin API

## Scope and compatibility

The public authoring interface is `kilin.tools`. `PLUGIN_API_VERSION` is an
integer generation of that interface, independent of KíLín's release version.
Version 1 defines the descriptor and execution contract below. The registry
checks each descriptor's `api_version` before publishing the registry; values
of the wrong type or an unsupported generation are rejected. An incompatible
contract change requires a new API generation. Additions must preserve
existing documented behavior within a generation.

The runtime discovers distribution and user-level tool packages. Each
bundled descriptor declares its individual version and API explicitly.
New descriptors should do the same. Project types have a separate JSON
`schema_version` and a runtime catalog retaining missing-tool diagnostics;
see [Creating a project type](extension_interfaces.md#creating-a-project-type).

Tool Python code is trusted, in-process code. An API check establishes
compatibility, not safety. The supplied services and filesystem helpers
establish conventions and ownership boundaries; they are not a sandbox.

## Descriptor and settings

The supported authoring imports are available from `kilin.tools`, without
constructing the runtime registry, loading project-type definitions or starting
models or Qt. See the [tool contracts](../kilin/tools/_contract.py) for exact
signatures and attribute documentation.

```python
from dataclasses import dataclass

from kilin.tools import (
    Tool,
    ToolContext,
    ToolRequest,
    ToolResult,
    check_int,
    reject_unknown,
)


@dataclass(frozen=True)
class Settings:
    count: int


def parse_settings(data: dict, where: str) -> Settings:
    reject_unknown(where, data, ("count",))
    return Settings(check_int(where, data, "count", 1, 1, 10))


def run(
    settings: Settings, request: ToolRequest, context: ToolContext,
) -> ToolResult:
    return ToolResult(
        answer="\n".join([request.question] * settings.count),
        evidence=(),
        summary={"count": settings.count},
    )


PLUGIN = Tool(
    name="repeat_question",
    summary="Repeat a question a specified number of times.",
    version="1.0.0",
    api_version=1,
    parse_settings=parse_settings,
    run=run,
)
```

Save this example as `tools/repeat_question/__init__.py` under the user
plugin directory described below. Discovery requires no central registry
edit, build metadata, package installer or Python distribution.
The descriptor name must match the package directory and match
`[a-z][a-z0-9_]*`. Declare the API generation the code implements as a literal
integer, not the current host constant: otherwise an incompatible future host
could appear compatible. The descriptor has the following fields:

| Field | Contract |
| --- | --- |
| `name` | Stable, registered string identifier; use lowercase package-safe names. |
| `summary` | Human-readable explanation of the operation. |
| `version` | Individual tool version, independent of KíLín. |
| `api_version` | Integer version of the host API implemented by the descriptor. |
| `parse_settings(data, where)` | Validate a configuration mapping; return a settings dataclass or raise `ValueError` with the field/location. |
| `run(settings, request, context)` | Execute with validated settings and return `ToolResult`. |
| `follows_up` | Whether the tool always operates on an earlier answer; defaults to false. |
| `follows_up_when(question)` | Optional predicate for conditional follow-up behavior. |
| `routing(settings, project_root, others)` | Optional project-specific `ToolRouting` or `None`; `others` contains settings of other intents using the tool. |
| `workspace_resources(settings)` | Optional tuple of `WorkspaceResource` declarations for new workspaces. |

Settings must round-trip through `dataclasses.asdict` and JSON. Use strings,
booleans, finite numbers, `None`, lists/tuples and string-keyed dictionaries;
the parser must accept the JSON representation (tuples become lists). Reject
unknown keys, inappropriate types and out-of-range values. The supported
helpers are `reject_unknown`, `check_bool`, `check_int`, and `check_choice`.
Project-type intent values and workspace overrides are validated by the same
parser. Do not treat a required but missing setting as a successful default.
Bundled tools retain their `defaults.json` and `settings_info.json`. A minimal
plugin may omit either file. If `defaults.json` declares `SETTINGS`, registration
validates those defaults with the parser and checks dataclass/JSON compatibility.
Required settings without declared defaults are validated when configured.
An absent model-call policy inherits the active session; a supplied policy
must be valid. Missing help returns an empty mapping. Malformed supplied help
is rejected by the metadata interface for bundled tools; for user plugins
it produces a nonfatal warning and empty metadata. See
[tool packages](tool_packages.md) and [configuration](configuration.md).

## Execution context and data records

`ToolRequest` contains `question`, `text`, optional `guidance`, and `earlier`
(a tuple of `EarlierTurn` records containing number, question, evidence
paths/texts/lines, link status and answer). Use the supplied context rather
than constructing a RAG session, importing the GUI, or invoking another tool.
Treat indexed data and host-owned records as read-only.

| `ToolContext` field | Supported behavior |
| --- | --- |
| `project_root`, `source_root` | Workspace root and optional saved source root. |
| `chunks`, `graph`, `rank`, `retrieval_count` | Indexed `Chunk` records, `CodeGraph`, complete hybrid ranking and configured retrieval count. |
| `source_files()`, `unsearchable(paths)` | Selected source paths and sizes; files without indexed text. |
| `retrieve(request)` | Optional bounded retrieval returning `RetrievalResult`. |
| `card(path)` | Optional indexed metadata/statements mapping or `None`, without a new model call. |
| `ask_router(system, prompt, max_new_tokens)` | Controlled router call, greedy and without thinking. |
| `ask_model(system, prompt, max_new_tokens)` | Internal answering-model call under the tool's model policy. `None` for the budget inherits the session limit. |
| `ask_check(system, prompt, max_new_tokens)` | Optional short answering-model check, without thinking; otherwise use `ask_router`. |
| `write(system, prompt, max_new_tokens)` | Stream the final answer using session policy; return that reply and set `ToolResult.streamed=True`. |
| `model_id`, `model_attempt_count()` | Model identifier and optional cumulative internal-generation attempt count. |
| `progress(message, **details)`, `done(message, **details)` | Report progress and result status. |
| `confirm`, `choose` | Optional host callbacks; use the recorded interaction helpers below. |
| `trace`, `trace_turn_id` | Host-owned diagnostic sink and turn correlation ID. |
| `write_pdf(markdown, path, label=...)` | Optional PDF renderer; returns actual page count. |
| `suggestions()`, `fixed_suggestions` | Optional current suggested questions and fixed example requests. |

Optional callbacks may be `None` in notebooks or headless tests. A tool must
handle their absence according to the documented operation rather than assume
that a user or renderer is available.

The following record types are exported from the same public namespace. Their
fields and constructor semantics are part of the tool-facing contract.

| Record | Fields and semantics |
| --- | --- |
| `Chunk` | `identifier`, source-relative `path`, exact `text`, one-based `start_line`/`end_line`, `kind`, optional `symbol`, `section`, `cell`. |
| `HybridResult` | `score` (rank-fusion score), `chunk`, `channels`. |
| `CodeGraph` | `entities` and `relations`; `CodeEntity(name, kind, path)` and `CodeRelation(source, relation, target)`. |
| `EvidenceLimits` | `max_sources`, `passages_per_source`, `max_passages`, `max_tokens`, `coverage` (`bounded` or `exhaustive`). |
| `RetrievalRequest` | `query`, `limits`, `strategy` (`source_first` or `hybrid`), optional relative `scope`, `compare`, `card_kinds`. |
| `RetrievalResult` | Original `passages` in citation order, selected `sources`, and JSON-compatible `coverage`. Source selection does not imply exhaustive reading. |
| `Suggestion` | `intent`, `question`. |
| `ToolRouting` | Additional regex `cues` and router `description`. |
| `WorkspaceResource` | Package-owned file `source`, workspace-relative POSIX `destination`. |

See the [retrieval pipeline](retrieval_pipeline.md) and the [retrieval request
contracts](../kilin/retrievals/requests.py) for semantics of limits, source
coverage and evidence. A card is not original source evidence.

## Results, failures and interactions

Return `ToolResult(answer, evidence, summary, streamed=False)`. The Markdown
answer uses one-based `[n]` citations corresponding to `evidence` records:
`citation`, `path`, `start_line`, `end_line`, `symbol`, `text`, and `cell`.
Use a source-relative `pathlib.Path` for `path`; the host serializes it.
Evidence text must match original source lines. `summary` contains JSON-safe
facts saved with the turn. Use an empty evidence tuple when no evidence is
cited. Do not manufacture evidence from model responses or diagnostic data.

The host raises `IncompleteModelResponse` for rejected model completions;
its `reason`, `finish_reason`, `raw_text`, and `to_dict()` distinguish failure
from a result. The raw text is not a valid answer. Internal thinking retries
and token budgets are owned by the host. For independent items, use
`partial_result(result, failures, completed=True)` to retain verified work
and mark incomplete items; set `completed=False` if nothing was assessed.
`GenerationCancelled` inherits `BaseException`; let it propagate so Stop
cannot become a successful or partial answer. See the
[answering pipeline](answering_pipeline.md).

For user interactions, use `confirm_plan(plan, policy, large_batches,
context)` and `ask_choice(question, options, context)` rather than invoking
the raw callbacks. Both helpers preserve the host's interaction records.
`ToolPlan` has `tool`, `summary`, `scope`, `items`, `batches`, `batches_to_run`,
`estimated_seconds`, `quick_items`, `quick_seconds`, optional `details`,
`quick_label`, `quick_note`, and `operation`. Confirmation policies are
`always`, `when_large`, and `never`; choices are `continue`, `quick`, and
`cancel`. The confirmation helper returns `(choice, asked)` and continues
without asking when no callback exists. The choice helper returns a zero-based
option index or `None` for cancellation, invalid choice, or absent callback.
The tool owns the meaning and handling of its quick/cancel options.

`ToolTrace` describes the supported trace interface: `include_content`,
`emit(event_type, *, component, data=None, status="completed", turn_id=None)`
and `span(event_type, *, component, data=None, turn_id=None)`. A span is a
context manager yielding its ID; it propagates operation failures. Use
`component="tool"` and the supplied turn ID. Do not include source or prompt
content unless `include_content` is true. Do not close the trace, alter its
configuration, write its files or invent interaction events. See
[trace recording](trace_recording.md).

## Source and output ownership

Use `selected_paths(context)` to obtain the active selected-source inventory.
`selected_source_file(project_root, source_root, path, selected)` validates a
selected source-relative regular file and returns a contained path or `None`;
it does not search arbitrary folders or accept symlink traversal. Handle an
absent `source_root` explicitly. These helpers are not a security sandbox.
`save_output_file(project_root, name, text)` writes a text output under a safe,
unique filename in the workspace's outputs folder without overwriting an
existing file. Use relative imports for package-local code and locate owned assets relative
to `Path(__file__).parent`. The loader binds the actual package directory, not
a path claimed by the descriptor. Registered tools expose `resource_root`;
`model_calls` and `internal_defaults` resolve against it. These properties
require a registered descriptor. The core validates and copies declared
`WorkspaceResource` files when creating a workspace. Refer to the
[workspace contract](project_workspaces.md) for ownership and safe publication.

## Catalog access and initialization

`kilin.tools.registered_tools()` returns a read-only mapping of the runtime's
tool definitions. `kilin.tools.tool_choices()` returns available identifiers
in their established order. `kilin.project_type_ids()` returns the project
catalog's identifiers. Catalog queries initialize the needed definitions on
first use; ordinary contract imports do not. The runtime tool registry is created once per process. Public package
directories with `__init__.py` are discovered automatically; private
packages are excluded. Existing iteration and chooser orderings are retained,
with additional bundled names and then user names appended alphabetically.
An invalid bundled descriptor prevents publication; invalid user packages
are reported individually. Recursive initialization fails explicitly.
Project types are discovered after tools and retain individual diagnostics.
Explicit directory validation remains strict.

`ToolName` is an open `str` alias; it is not authorization to import arbitrary
Python paths. `parse_tool(where, data)` resolves a configured ID and validates
settings; `run_tool(use, request, context)` dispatches a validated `ToolUse`.

The older `TOOL_CHOICES`, `TOOL_REGISTRY`, and `PROJECT_TYPE_IDS` exports still
resolve on access for existing callers. The legacy registry dictionary remains
mutable for compatibility; do not mutate it from plugins. Use the explicit
read-only accessor for new code. Likewise, importing a legacy tool-specific
convenience export loads the corresponding executable package. Do not rely on
those exports or private modules when implementing independent plugins.

## User storage, discovery and validation

`kilin.plugins.plugin_directory()` returns the captured absolute location:

| Platform | Default user plugin directory |
| --- | --- |
| macOS | `~/Library/Application Support/kilin/plugins` |
| Linux | `$XDG_DATA_HOME/kilin/plugins`, or `~/.local/share/kilin/plugins` |
| Windows | `%LOCALAPPDATA%/kilin/plugins`, or `~/AppData/Local/kilin/plugins` |

An absolute `KILIN_PLUGINS_DIR` overrides this location. Set it before the
first catalog query or `plugin_directory()` call. No directories are created
by discovery. There is no workspace-level plugin store, registry database,
package installer or dependency installer. A missing directory is normal.
Create the following layout manually while all KíLín processes and notebook
kernels using the directory are closed:

```text
plugins/
    tools/
        repeat_question/
            __init__.py
    project_types/
        repetition.json
```

A tool package may also contain its own Python modules, `defaults.json`,
`settings_info.json` and resources. Use relative imports for package-local
code. The loader imports a user package in its own namespace, without
adding its directory to `sys.path`, and uses source rather than cached
bytecode. A package must consist of regular files and directories (no
symlinks, except ignored cache files). Dependencies must already be
available in KíLín's Python environment. Do not import other plugins.

Project types refer to tool IDs, not Python module paths. For example,
`repetition.json` can use the minimal tool above:

```json
{
  "schema_version": 1,
  "display_name": "Repetition",
  "description": "Repeat a question using a deterministic tool.",
  "extensions": [".txt", ".md"],
  "max_file_bytes": 4096,
  "retrieval_instruction": "Find relevant information.",
  "system_prompt": "Use the configured repetition tool.",
  "history": {"exclude_disliked_answers": true},
  "conversation_memory": {
    "enabled": false,
    "questions": "all",
    "answers": "selected",
    "max_hits": 4,
    "max_tokens": 2000
  },
  "intents": {
    "repeat": {
      "description": "Repeat a question.",
      "tool": {"name": "repeat_question", "count": 2}
    }
  }
}
```

Copying code into the tools directory is an explicit trust decision: the
next runtime will import it with the user's permissions. Review generated
code, particularly file access, network operations and dependencies, before
copying or enabling trusted validation. Do not change the directory contents
while a runtime is active: a plugin may read resources or import modules
lazily. Restart after manual changes; managed staging and activation are
not implemented yet. Bundled IDs take precedence over user IDs; conflicts
are visible as rejected candidates. An unavailable tool makes only types
that require it unavailable. A missing optional help file is harmless;
invalid user help is reported as a warning and omitted from settings help.

The headless inspection and validation interface is `kilin.plugins`:

```python
from pathlib import Path
from kilin import plugins

for info in plugins.list_tools():
    print(info.name, info.origin, info.available, info.error, info.warnings)
for info in plugins.list_project_types():
    print(info.type_id, info.origin, info.available, info.error)

candidate = Path("drafts/repeat_question").resolve()
structure = plugins.validate_tool(candidate)
print(structure.error, structure.sha256, structure.runtime_checked)
# Only after inspecting and trusting the code:
checked = plugins.validate_tool(candidate, trusted=True)
print(checked.available, checked.error)

kind = plugins.validate_project_type(Path("drafts/repetition.json"),
                                     bind_tools=False)
print(kind.error, kind.required_tools, kind.bound)
```

The default tool validation checks directory layout, source syntax and
JSON, without executing candidate code. Trusted validation imports the
candidate in a fresh interpreter, with a bounded 30-second worker timeout.
It checks the descriptor, version, defaults and optional help, without
replacing an already loaded plugin. It does not execute the tool runner or
prove its safety, and the worker is **not a security sandbox**. Structural
success is reported as `error is None` with `runtime_checked=False`; only
successfully runtime-checked candidates have `available=True`. Similarly,
project-type validation with `bind_tools=False` checks structure without
resolving tool references; `bound=False` means it is not execution-ready.
The default (`bind_tools=True`) resolves required tools against the current
catalog and validates their settings. It may initialize trusted installed
plugins. Validation is not installation and never activates the candidate.

The host validates settings dataclasses and JSON-compatible result data at
configuration and execution boundaries, but cannot prove semantic accuracy
or resource use in advance. Use deterministic unit tests for those behaviors.

## Execution identity

The immutable discovery records retain an origin, version (tool version and
API generation, or project-type schema version), and SHA-256 digest. Tool
identity hashes sorted relative file names and contents, excluding bytecode
and `__pycache__`; JSON identity hashes the captured definition bytes.
These portable identities do not include absolute user paths. The trace's
`turn.settings.snapshot` includes `project_type_plugin`, and the selected
`tool.run` span includes `plugin`. Persisted turn parameters include the
selected identities as `project_type_plugin` and `tool_plugin` when known.
These describe the package bytes captured during registration, not a proof
of the behavior of arbitrary code or external dependencies. Historical
records need not contain them. Replay/comparison integration is separate.

## Verification

The authoring boundary has fresh-interpreter tests in
[the plugin API regression suite](../tests/test__plugin_api.py). They check
import order, absence of catalog reads, one-time concurrent construction,
failed and recursive initialization, identifier validation, and descriptor
execution using the public interface. The
[registration tests](../tests/kilin/tools/test__registration.py) additionally
check bundled descriptors, optional resources, discovery and relocation.

For a stacked feature branch, run both complete repository test runners
(`python -m pytest` and `python -m tests`) against the exact head. Run the
existing Python tests and Distribution package workflows on that branch via
their manual dispatch if normal pull-request triggers do not cover the stack
base; verify the reported head SHA. Native and model-dependent checks remain
subject to their documented environments.

## Related

[Extension interfaces](extension_interfaces.md),
[Tool packages](tool_packages.md), [Development](development.md),
[Security](security.md).
