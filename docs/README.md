# KíLín technical documentation

This directory is the implementation-oriented reference for developers who
maintain or extend KíLín. It describes **the code in this repository**, not a
future general-purpose agent platform. The root [`README.md`](../README.md)
remains the short installation and entry-point guide; the
[`tutorials/`](../tutorials/) notebooks demonstrate use, and Python docstrings
specify individual call signatures and validation rules.

These references describe the accompanying source tree. KíLín targets
Python 3.12; backend and desktop behavior also depend on optional
dependencies and the host operating system.

## Reading paths

| Reader or task | Start here | Then read |
| --- | --- | --- |
| New developer | [Architecture](architecture.md) | [Development](development.md), [Module map](module_map.md) |
| Import or update sources | [Source pipeline](source_pipeline.md) | [Workspaces](project_workspaces.md), [Retrieval](retrieval_pipeline.md) |
| Explain an answer | [Answering pipeline](answering_pipeline.md) | [Retrieval](retrieval_pipeline.md), [Traces](trace_recording.md) |
| Work with models | [Model management](model_management.md) | [Configuration](configuration.md) |
| Extend a project type or tool | [Extension interfaces](extension_interfaces.md) | [Tool packages](tool_packages.md), [Configuration](configuration.md) |
| Desktop behavior or recovery | [GUI architecture](gui_architecture.md) | [GUI protocol](gui_protocol.md), [Conversation ownership](conversation_ownership.md) |
| Debug, test, release | [Development](development.md) | [Performance](performance.md), [Security](security.md) |
| Inspect every module | [Module map](module_map.md) | Owner document and source docstring |

## Subsystem references

### Architecture and implementation

- [Architecture](architecture.md): boundaries, components, entry points,
  ownership and data/control flows.
- [Source pipeline](source_pipeline.md): scans, explicit imports, extraction,
  provenance, change detection and chunk formation.
- [Retrieval pipeline](retrieval_pipeline.md): indexes, embeddings, lexical
  retrieval, code graphs, fusion, document-scoped retrieval and cache reuse.
- [Answering pipeline](answering_pipeline.md): router, history policy, intent,
  retrieval, tools, generation, citations, and committed turns.
- [Model management](model_management.md): backend selection, cache,
  sessions, inference options, tokenizer/streaming differences.
- [Configuration](configuration.md): shipped defaults, project profiles,
  user preferences, scope and precedence.
- [GUI architecture](gui_architecture.md): Qt, child process, web page,
  operations and recovery.
- [Extension interfaces](extension_interfaces.md): project types, source
  extractors, retrieval strategies and registered tools.
- [Security](security.md): boundaries, sensitive data, filesystem and
  network limitations.
- [Development](development.md): development/test/release process and
  validation checklists.

### Detailed reference contracts

These documents specify individual contracts in greater detail:

- [Project workspaces and tool resources](project_workspaces.md)
- [Conversation-file ownership](conversation_ownership.md)
- [Desktop command/event protocol](gui_protocol.md)
- [Packaged configuration defaults](package_defaults.md)
- [Modular tool packages](tool_packages.md)
- [Diagnostic trace recording](trace_recording.md)
- [Performance analysis](performance.md)
- [Desktop GUI README](../kilin/gui/README.md)

The twelve registered tools each have their own README under
[`kilin/tools/`](../kilin/tools/), which is the authoritative guide to
that tool's settings, supported requests and outputs. Do not infer tool
behavior from its identifier alone.

## Coverage, accuracy and architecture diagrams

[Module map](module_map.md) assigns **every production Python module** to a
primary technical document and records its responsibility. This is an index,
not a substitute for reading source or its NumPy-style docstrings. Each
owner document includes a flow, key invariants, failure modes, configuration
and tests. Any implementation behavior not supported by source or tests
must be called out as unknown, rather than filled in by architectural intent.

Eleven [editable SVG architecture diagrams](diagrams.md) are embedded
in their primary technical documents. The diagrams show component and
ownership boundaries, the source-to-answer flow, conversation checkpoints
and security limitations. They are schematics, **not exhaustive API or
state-machine specifications**. The surrounding text remains authoritative
for exceptions, precedence and error behavior. SVGs use native text and
vector elements and are maintained under [`figures/`](figures/).

## Maintenance and accuracy

The implementation, validated configuration and tests define actual
behavior. Keep each contract in its owner document and link to it rather
than repeating mutable defaults, protocol inventories or configuration
precedence elsewhere. Update the [module map](module_map.md) when the
package structure changes. For repository verification, see
[Development](development.md); for SVG conventions, see
[Architecture diagrams](diagrams.md).
