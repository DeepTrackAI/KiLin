# Source selection, import, extraction and chunking

## Scope and owners

Source ingestion converts **local files chosen by the user** into text units
with traceable origin, then token-bounded chunks for retrieval. The four
operations should not be mistaken for one transaction:

![Directory scanning and explicit file selection feed two possible managed-source workflows, followed by file readers, source units and chunks.](figures/source-ingestion.svg)

*Figure: Source selection, the two managed-import APIs, format-specific extraction and chunk identity are distinct; original source files remain unchanged.*

| Step | Owner / main API | Output |
| --- | --- | --- |
| Preview policy | `repository.scan_repository()` | Scan report with entries, selected files, skip codes |
| Snapshot / source management | `workspace.prepare_sources_workspace()` or `project.prepare_project_workspace()` | Managed copies and source manifest |
| File content | `sources.read_repository_files()` | `SourceFile` records |
| Natural units | `sources.extract_source_units()` | `SourceUnit` records |
| Chunking | `chunking.chunk_source_units()` | `Chunk` records with stable identifiers |
| Change comparison | `repository.build_repository_manifest()`, `compare_repository_manifests()` | Added, removed, modified content state |

The complete workspace ownership model, including which APIs create
`sources.json` or `source_snapshot.json`, is documented in
[Project workspaces](project_workspaces.md).

## 1. Selecting sources is policy, not reading

`scan_repository(repository, *, extensions, include_paths, exclude_paths,
max_file_bytes, max_entries)` walks a resolved local root using `os.scandir`.
It rejects remote-looking addresses, invalid size limits, a home/root scan,
and malformed relative include/exclude paths. Directories are inspected
without following descendant symlinks/junctions. Entries are sorted for a
stable report; a scan can report `complete=False` when a non-root directory
could not be inspected. Exceeding `max_entries` raises rather than returning
a falsely complete inventory.

The report contains the resolved repository root, normalized suffixes,
include/exclude paths, byte/entry totals, `selected_files`, and entry rows.
Each row has `path`, `status`, a human-readable `reason`, and a stable
`reason_code`. The codes are defined in
[`repository_reason_codes.json`](../qachat/repository_reason_codes.json),
validated once by `repository_reason_codes.py`, and transmitted to the
frontend in its settings event. The only special GUI opt-in is a file skipped
for `file_too_large`; protected or linked files do not become selectable
merely because a display message changes. See [GUI protocol](gui_protocol.md).

Scanning **does not establish an access-control boundary**. Between a
filesystem check and subsequent opening another process may change a path.
The directory scanner's exclusions are project/source-selection defaults,
not a guarantee that every secret has been identified. Especially important:
**direct file selection has a separate rule from automatic directory scans**.
An explicitly imported regular file may have an excluded-looking name or
extension. The managed source manifest records the selection, and
`workspace.include_explicit_imports()` keeps eligible snapshots in the RAG
index scope; unsupported content can still fail extraction. Do not add a
security prohibition to the GUI by reinterpreting a scanner explanation.

Packaged `REPOSITORY` exclusions and default suffixes are documented in
[Package defaults](package_defaults.md). A project type and saved profile
also set selection rules; see [Configuration](configuration.md).

## 2. Snapshot and provenance alternatives

QAChat has two different, deliberately retained workflows:

- `prepare_sources_workspace(source_paths, kind, workspace_directory=...)`
  accumulates individually selected local files/folders into managed
  `sources/`, tracking origins and selection metadata in `sources.json`.
  Re-adding a source can update matching snapshots. Deselecting or deleting
  something at the original location does **not** implicitly remove earlier
  managed copies. `remove_workspace_sources()` removes only selected managed
  snapshots, never original user files.
- `prepare_project_workspace(source_directory, kind, workspace_root=...)`
  mirrors one source directory into a managed project and tracks its prior
  state with `source_snapshot.json`. It compares manifests before
  synchronizing. It is not a drop-in replacement for the cumulative route.

`initialize_project()` operates on an existing folder and chooses a valid
source location according to its profile rules; it does not perform an
unrequested import. The GUI delegates cumulative imports and deletions to
core functions and then requests index refresh. `project_files.py` manages
`outputs/` and `templates/` separately; those folders are not newly indexed
as ordinary source material.

Source additions/removals are **not** a multi-file atomic transaction. In the
cumulative route an earlier successful source remains even if a later import
fails. The source-provenance file is updated as successful operations are
completed. Project initialization has a separate rollback journal and
creates its profile last; see [Project workspaces](project_workspaces.md).

## 3. Reading files and retaining source coordinates

`read_repository_files(report)` consumes the report's `selected_files` in
order. It resolves each selected path again, rejects traversal, links,
non-files and paths escaping the selected root, then uses a suffix-specific
reader if one exists. Other inputs are decoded as UTF-8. Binary content is
not automatically made readable by giving it a textual suffix. A PDF uses
its dedicated selectable-text reader and may require installed PDF
extraction dependencies.

`SourceFile` contains a relative path and extracted text; `SourceUnit`
contains a path, text, inclusive line range, kind, optional symbol/section,
optional notebook cell, and optional child units. The `extract_source_units`
entry point chooses format-specific parsing and validates unit provenance.
It does **not execute source files or notebook cells**.

| Extension module | Implemented structural unit strategy |
| --- | --- |
| `sources_py.py` | Python AST declaration boundaries and subordinate regions |
| `sources_m.py` | MATLAB declaration/section patterns in text |
| `sources_md.py` | ATX Markdown headings and sections |
| `sources_ipynb.py` | Notebook JSON, code/Markdown cell text and cell coordinates |
| `sources_pdf.py` | Page-oriented selectable PDF text; page/section coordinates |
| `sources_txt.py` | Nonblank paragraph text units |

The implementation uses the actual suffix-to-reader dispatch in
`sources/sources.py` and its `extensions/` directory. An unsupported suffix
may be decoded as text, but it is not thereby semantically parsed. PDF image
OCR, exhaustive language parsers and executing notebooks are **not**
implicit features of this pipeline.

**Coordinate conventions:** `Chunk.start_line`/`end_line` are one-based
inclusive and refer to the source unit's coordinate system, not necessarily
a single physical file-wide line number. Notebook positions are cell-local;
PDF text is page-local with section provenance. Citations and source previews
must preserve these distinctions. Altering a reader's coordinate convention
is a compatibility-sensitive change; update tests and evidence consumers.

## 4. Chunk boundaries and identifiers

`chunk_source_units(units, *, count_tokens, target_tokens, max_tokens,
overlap_tokens)` uses the **selected embedding tokenizer** to estimate
length, not a universal character-to-token conversion. It honors natural
units where possible, subdivides oversized units, and can overlap adjacent
chunks according to the validated parameters. `max_tokens >= target_tokens`
and `0 <= overlap_tokens < target_tokens` are required. A callback reporting
a negative or non-integer count raises a validation error.

A `Chunk` preserves `identifier`, relative `path`, exact fragment `text`,
start/end line, `kind`, `symbol`, `section`, and `cell`. The identifier is
content/location-derived using an unambiguous JSON-array encoding. A change
in the identifier algorithm is an index-format compatibility change: old
indexes may need rebuilding. The generic `CHUNKING` default arguments are
**different** from `PROJECT_PROFILE` values persisted in newly created
projects; the active RAG pipeline passes its project's saved values.

## 5. Refreshing sources and avoiding unnecessary embeddings

A repository manifest records selected file fingerprints for change
comparison. `RAGSession.refresh_sources(project)` updates chunk/index state
after source changes. Where saved indexing settings and the embedding model
snapshot match, changed source content can be re-extracted and re-embedded
while unchanged chunk vectors are reused. Removed source files are dropped.
The lexical statistics and global code graph are rebuilt to represent the
new complete set of chunks. Incompatible settings, missing snapshot
identity, corrupt indexes or other mismatches trigger a full rebuild.

`retrievals.persistence` stores validated index data and a content
fingerprint/checksum; do not confuse source manifest equality with index
payload integrity. The exact reindex/reuse contract is described in
[Retrieval pipeline](retrieval_pipeline.md).

## Errors, boundaries and validation

- Failures to list or read a selected root, inconsistent scan reports,
  relative path traversal, or invalid source records propagate as explicit
  errors; callers must not treat a partial scan as a complete deletion list.
- File selection is a usability and privacy precaution, not a robust
  race-free isolation boundary or a comprehensive secret scanner.
- Explicit user import does not make an unreadable format extractable.
- Workspace deletion must be confined to managed source snapshots. Source
  originals should never be deleted by cleanup or reindexing operations.
- Changing extraction or chunk identity affects cache reuse, evidence
  coordinates and citation reproducibility; test these together.

## Relevant tests and cross-references

Start with the test files matching `repository`, `sources`, `chunking`,
`workspace`, `project` and `rag` under `tests/qachat/`. The GUI selection
contract is additionally tested under `tests/qachat/gui/`. The latter also
covers reason-code independence from human-readable messages.

Related: [Project workspaces](project_workspaces.md),
[Retrieval pipeline](retrieval_pipeline.md),
[GUI protocol](gui_protocol.md), [Security](security.md).
