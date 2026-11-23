# KíLín technical documentation

This is the implementation-oriented reference for people and agents
maintaining the accompanying source tree. The executable code under
`kilin/`, its validated resources and actual call paths are authoritative.
These pages explain the architecture and contracts; they do not authorize
changing the implementation merely to make a description true. Suspected
runtime defects should be logged separately from documentation corrections.

The root [README](../README.md) is the installation/entry-point guide,
[tutorials](../tutorials/) demonstrate use, and Python docstrings specify
individual API signatures and validation. Each tool's own README describes
its settings, requests and outputs. These references complement rather than
replace one another. The supported environment is defined in
[Development](development.md#scope-and-supported-environment).

## Reading paths

| Task | First reference | Next boundary to inspect |
| --- | --- | --- |
| Understand the system | [Architecture](architecture.md) | [Module map](module_map.md), then the owning subsystem and source |
| Explain an answer | [Answering](answering_pipeline.md) | [Retrieval](retrieval_pipeline.md), [Traces](trace_recording.md) |
| Import/update local data | [Source pipeline](source_pipeline.md) | [Workspaces](project_workspaces.md), [index identity](retrieval_pipeline.md#index-persistence-and-identity) |
| Add a type, reader or tool | [Extension interfaces](extension_interfaces.md) | [Configuration](configuration.md), [tool services](tool_packages.md) |
| Change desktop behavior | [GUI architecture](gui_architecture.md) | [Protocol](gui_protocol.md), [conversation ownership](conversation_ownership.md) |
| Debug/test/release | [Development](development.md) | [macOS builds](release.md), [Performance](performance.md), [Security](security.md) |

## Contract ownership

Each shared rule has one detailed home. Other pages retain enough local
context to be understandable, then link here rather than repeat precedence,
mutable values or failure procedures. This table assigns documentation
responsibility, not authority over the source implementation.

| Primary reference | Detailed contract it owns |
| --- | --- |
| [Architecture](architecture.md) | Overall component boundaries, core/GUI dependency direction and state lifetimes. |
| [Project workspaces](project_workspaces.md) | Creation, source-directory convention, resources, explicit updates and filesystem failure residue. |
| [Source pipeline](source_pipeline.md) | Selection versus reading, source coordinates, extraction and chunk identity. |
| [Retrieval pipeline](retrieval_pipeline.md) | Ranking/fusion, context expansion, persisted index identity and empty/incremental indexes. |
| [Answering pipeline](answering_pipeline.md) | Routing/history, answer production, completion/recovery, partial results and turn commitment. |
| [Model management](model_management.md) | Backends, loading/cache/network behavior, chat/embedding sessions and model-specific limitations. |
| [Configuration](configuration.md) | Setting ownership, precedence, canonical identifiers and project overrides. |
| [Packaged defaults](package_defaults.md) | Validated core JSON keys, values, bounds and effects; not per-tool defaults. |
| [Tool packages](tool_packages.md) | Package boundaries, safe generated outputs and internal/final model-call policy. |
| [Extension interfaces](extension_interfaces.md) | Steps and interface obligations for adding types, readers, tools and other components. |
| [GUI architecture](gui_architecture.md) | Desktop process/lifecycle integration and presentation ownership. |
| [GUI protocol](gui_protocol.md) | Commands, events, payloads, interactive waits, cancellation and restart. |
| [Conversation ownership](conversation_ownership.md) | Writer leases, atomic files, switching, archival and cleanup guarantees. |
| [Conversation memory](conversation_memory.md) | Turn UUIDs, authoritative feedback, active branch origins, independent indexing and separate prompt provenance. |
| [Trace recording](trace_recording.md) | Diagnostic activation/content, trace metadata, event versus checkpoint failure policy. |
| [Security](security.md) | Trust boundaries, sensitive data, deployment responsibilities and explicit non-guarantees. |
| [Performance](performance.md) | Trace-derived measurements, reproducible workloads and limits of timing evidence. |
| [Development](development.md) | Coding/documentation process, test runners, packaging and PR verification. |
| [macOS builds](release.md) | Ad-hoc development builds, native artifact checks, limitations and manual testing. |
| [Module map](module_map.md) | Complete production Python inventory and the primary reference for each module. |
| [Architecture diagrams](diagrams.md) | Editable SVG inventory, ownership, accessibility and visual-maintenance rules. |

The [GUI README](../kilin/gui/README.md) is the desktop companion. Individual
[tool READMEs](../kilin/tools/) own tool-specific parameter values and outputs;
use the [tool index](extension_interfaces.md#available-built-in-tools) to
find one. Do not infer behavior or safety from a tool identifier alone.

## Change-impact navigator

These are starting points, not exhaustive test selections. Read the primary
contract, inspect the listed implementation and its callers, then expand the
regression set for the actual change. Owner pages explain design rationale,
invariants, failure boundaries and native/model checks that substitutes cannot
cover. The [verification procedure](development.md#pull-request-verification-procedure)
still applies after a focused subset passes.

| Proposed change | Contract to read first | Update/review together | Representative verification |
| --- | --- | --- | --- |
| Create/open projects or change their layout | [Workspaces](project_workspaces.md) | Reservation, profile publication, resources, GUI activation and saved-layout compatibility. | [Creation](../tests/kilin/test_project__creation.py), [transactions](../tests/kilin/test__project_transactions.py), [GUI creation](../tests/kilin/gui/test_engine__creation.py) |
| Import, synchronize or remove sources | [Source pipeline](source_pipeline.md), [workspace updates](project_workspaces.md#explicit-imports-and-source-selection) | Selection, provenance/version records, original-file protection and explicit index refresh. | [Source management](../tests/kilin/test_workspace__source_management.py), [source changes](../tests/kilin/test_workspace__source_changes.py) |
| Add a reader or change chunk boundaries | [Source extensions](extension_interfaces.md#source-readerextractor-contract) | Dispatch, exact text/coordinates, citation previews, chunk IDs and cache reuse. | [Readers](../tests/kilin/sources/), [chunking](../tests/kilin/test_chunking.py), [index provenance](../tests/kilin/test_rag__index_provenance.py) |
| Change retrieval, expansion or saved indexes | [Retrieval](retrieval_pipeline.md) | All channel depths/order, model identity, evidence labels, checksum and empty-state behavior. | [Retrieval](../tests/kilin/retrievals/), [RAG corrections](../tests/kilin/test_rag__retrieval_corrections.py), [empty projects](../tests/kilin/test_rag__empty_workspace.py) |
| Change answer completion or tool model calls | [Answering](answering_pipeline.md), [tool policy](tool_packages.md#generation-settings-for-tools) | Side-call isolation, thinking/budgets, retries, partial status, usage and committed history. | [Completion](../tests/kilin/test_rag__completion.py), [recovery](../tests/kilin/test_rag__tool_recovery.py), [budgets](../tests/kilin/test_rag__tool_budgets.py) |
| Add a project type, tool or template | [Extensions](extension_interfaces.md) | Parser/registry, intents, settings/resources, package data, local README and module map. | [Type contracts](../tests/kilin/test__project_type_contract.py), [tool boundaries](../tests/kilin/tools/test__architecture.py), [resources](../tests/kilin/tools/test__resources.py) |
| Change model loading or generation controls | [Models](model_management.md) | Backend/tokenizer, history, completion metadata, download policy, cleanup and saved configuration. | [Runtime](../tests/kilin/test_runtime.py), [chat units](../tests/kilin/test_chat__unit.py), [embedding units](../tests/kilin/test_embeddings__unit.py) |
| Change defaults or saved settings | [Configuration](configuration.md) | Owning JSON/validator, initialization versus reopening, GUI controls and cache invalidation. | [Core defaults](../tests/kilin/test_defaults.py), [tool settings](../tests/kilin/test_tool_settings.py), [documented values](../tests/setup/test__documentation_assets.py) |
| Change conversation fields or writer ownership | [Conversation ownership](conversation_ownership.md), [conversation memory](conversation_memory.md), [traces](trace_recording.md) | Store/logger/RAG/GUI writers, old-state fixtures, lock cleanup and trace consumers. | [Format](../tests/kilin/test_conversation_format.py), [ownership](../tests/kilin/test_conversation_ownership.py), [lifecycle](../tests/kilin/test__conversation_lifecycle.py) |
| Change GUI commands, events or Stop | [Protocol](gui_protocol.md), [GUI lifecycle](gui_architecture.md) | Bridge, input thread/dispatcher, interactive waits, page state and actual committed results. | [Protocol](../tests/kilin/gui/test__protocol.py), [resume](../tests/kilin/gui/test_engine__resume.py), [native smoke](../tests/kilin/gui/test__webengine_smoke.py) |
| Change tracing or performance analysis | [Traces](trace_recording.md), [measurements](performance.md) | Event producers, projection, checkpoint/failure policy, parser fixtures and privacy. | [Trace logger](../tests/kilin/test_trace_logging.py), [benchmark parser](../tests/test__trace_benchmark.py) |

## Coverage and limits

The [module map](module_map.md) assigns every production Python module to a
primary reference. The [diagrams](diagrams.md) are editable schematics of
boundaries and flows, not exhaustive API/state-machine specifications. Source
and owner-page conditions take precedence over a simplified arrow or caption.

Documentation checks verify links, inventories, selected source-backed
contracts and organization. They do not establish that every prose claim is
correct, that evidence is factually sufficient, or that native models and Qt
work on every host. Distinguish observed behavior, tested guarantees and
proposals in reviews; do not fill gaps with architectural intent.

## Related

[Integrated-documentation maintenance](development.md#maintaining-an-integrated-reference),
[PR verification](development.md#pull-request-verification-procedure),
[diagram maintenance](diagrams.md#maintaining-the-figures).
