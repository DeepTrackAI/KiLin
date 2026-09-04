# QAChat plan

## Goal

Rebuild the useful GenRAG workflow as a small standalone application. The
application logic comes first. Tutorials and, later, the Qt interface call the
same logic instead of implementing their own versions.

## Rules for every step

- Write the smallest clear implementation that works.
- Use plain names, specific type hints, NumPy-style docstrings, and absolute
  imports from the DTDV411 style guide.
- Format Python with Black and check a maximum of 79 characters with Ruff.
- Add `from __future__ import annotations` as the first import.
- Use built-in types such as `dict[str, object]` and `list[str]`.
- Add `unittest` checks and a short standalone tutorial for each capability.
- Run engine-dependent behavior through both MLX and PyTorch on Apple Silicon.
- Keep model weights, environments, caches, secrets, and generated output out
  of Git. Keep source, tests, clean tutorials, settings, and `uv.lock` in Git.
- Keep application behavior in `qachat/`. Tutorials and the future GUI are
  clients of that code.
- Do not require a GUI, notebook, or MCP server to use the application logic.

## Completed foundation

- [x] One-command Python 3.12 setup with `uv`.
- [x] Switchable MLX and PyTorch calculation check.
- [x] Tracked VS Code settings and portable debugger launchers.
- [x] DTDV411 formatting, typing, documentation, and unittest structure.
- [x] Hugging Face shared-cache discovery and normal download behavior.
- [x] `ChatSession` with model choice, engine choice, thinking mode, system
  prompt, temperature, conversation memory, reset, and close.
- [x] No QAChat-defined reply-length cap; a caller may set a positive token
  count, otherwise the remaining model context is used.
- [x] Local repository metadata preview without reading or executing source.
- [x] Independent tutorials for setup, model download, chat, and file choice.
- [x] Real tiny-model tests for both MLX and PyTorch without network downloads.

## Next backend steps

Each item adds shared code, tests, and one small tutorial.

1. Read supported text files safely. Keep filenames and line locations.
2. Split text into model-sized passages without silently cutting content.
3. Load an embedding model through Hugging Face and embed short text.
4. Embed repository passages and build a reusable local index.
5. Search the index by meaning and exact terms.
6. Return passages with file and line references.
7. Build a RAG prompt from the question, history, and retrieved passages.
8. Generate an answer through `ChatSession` and show its supporting sources.
9. Save and reopen repository indexes with enough metadata to detect an
   incompatible model or chunking change.
10. Add cancellation, progress, clear errors, and diagnostics.
11. Add temporary document attachments without changing the repository index.
12. Expose one small local application interface for prepare, ask, reset,
    cancel, status, and close.

## Frontend after the backend is standalone

1. Build repository selection, preparation, chat, source viewing, settings,
   progress, and error states.
2. Place that interface in the standalone Qt application.
3. Start and stop only the local services owned by that Qt application.
4. Test fresh setup, offline reuse, cancellation, saved indexes, and full user
   workflows on every supported operating system.

MCP is optional. Consider it only after the standalone application interface
is useful and stable; it is not required for local RAG or the Qt product.
