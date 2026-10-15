# QAChat technical documentation

This directory is the implementation-oriented reference for developers who
maintain or extend QAChat. It describes **the code in this repository**, not a
future general-purpose agent platform. The root [`README.md`](../README.md)
remains the short installation and entry-point guide; the
[`tutorials/`](../tutorials/) notebooks demonstrate use, and Python docstrings
specify individual call signatures and validation rules.

**Baseline:** QAChat after the GUI-test consolidation PR (14 October 2026).
Documentation describes that snapshot. The package targets Python 3.12;
backend and desktop behavior also depend on optional dependencies and the
host operating system.

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

### Existing authoritative contracts

These focused references are retained rather than replicated:

- [Project workspaces and tool resources](project_workspaces.md)
- [Conversation-file ownership](conversation_ownership.md)
- [Desktop command/event protocol](gui_protocol.md)
- [Packaged configuration defaults](package_defaults.md)
- [Modular tool packages](tool_packages.md)
- [Diagnostic trace recording](trace_recording.md)
- [Performance analysis](performance.md)
- [Desktop GUI README](../qachat/gui/README.md)

The ten registered tools each have their own README under
[`qachat/tools/`](../qachat/tools/), which is the authoritative guide to
that tool's settings, supported requests and outputs. Do not infer tool
behavior from its identifier alone.

## Coverage, accuracy and future diagrams

[Module map](module_map.md) assigns **every production Python module** to a
primary technical document and records its responsibility. This is an index,
not a substitute for reading source or its NumPy-style docstrings. Each
owner document includes a flow, key invariants, failure modes, configuration
and tests. Any implementation behavior not supported by source or tests
must be called out as unknown, rather than filled in by architectural intent.

[Diagram plan](diagram_plan.md) identifies proposed editable SVG architecture
schematics and their source-of-truth modules. **The diagrams are planned, not
implemented, in PR9a**. They belong in a separate PR9b under `docs/figures/`
when the textual architecture has been reviewed.

## Maintaining these documents

1. Read the current implementation and tests **before** changing a contract.
2. Update the subsystem's owner document and nearby docstrings as needed.
3. Prefer links to one authoritative statement over copying mutable defaults
   or command inventories into several places.
4. Distinguish configured behavior, runtime-selected behavior and data saved
   in an existing project; changing a package default does not migrate saved
   profiles automatically.
5. Check every referenced module, path, API, JSON key and Markdown link.
6. Record known limitations; do not describe a proposed feature as implemented.

See [Development](development.md) for the verification procedure.
