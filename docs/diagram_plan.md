# SVG architecture diagrams — proposed follow-up PR9b

**Status:** Planning inventory only. PR9a does not create diagram assets.
These diagrams should be **editable, native SVG** with embedded text/vector
elements (not flattened screenshots), stored in `docs/figures/` and linked
from their primary technical documents once reviewed.

## Diagram standards

- Each figure illustrates *verified current implementation*, not a planned
  capability. Use package/module or protocol names from source.
- Distinguish processes, trust boundaries, persisted files, transient
  in-memory state and optional calls with different visual conventions.
- Keep text editable as SVG text; use accessible contrast, reasonable font
  sizes, logical reading order and explanatory text in the parent document.
- Prefer diagrams with a single technical question to giant unreadable
  architecture maps. Show failure/cleanup paths where they are important.
- Treat arrows as actual data/control/ownership transfers; do not imply a
  network call or process isolation where none exists.
- Avoid embedding sensitive example prompts, source documents or local
  filesystem paths captured from user workspaces.

## Inventory and suggested sequence

| Figure (proposed basename) | Technical question | Owner document / source of truth |
| --- | --- | --- |
| `architecture.svg` | Which subsystems exist and which ones depend on the GUI? | [Architecture](architecture.md); `qachat/__init__.py`, `rag.py`, `gui/` |
| `workspace-ownership.svg` | Which paths belong to packaged defaults, user preferences and one project? | [Project workspaces](project_workspaces.md); `project.py`, `workspace.py`, `gui/_paths.py` |
| `source-ingestion.svg` | How do source selection, managed import, extraction and chunking differ? | [Source pipeline](source_pipeline.md); `repository.py`, `sources/`, `chunking.py` |
| `retrieval-indexes.svg` | How are semantic/lexical/code channels built and fused? | [Retrieval](retrieval_pipeline.md); `retrievals/strategies/`, `persistence.py` |
| `answer-sequence.svg` | How do router, retrieval, tool/model, checkpoint and GUI events sequence? | [Answering](answering_pipeline.md); `answering.py`, `rag.py`, `_registry.py` |
| `model-lifetimes.svg` | What is loaded, reused, switched and released? | [Models](model_management.md); `chat.py`, `embeddings.py`, `rag.py` |
| `gui-processes.svg` | What crosses QWebChannel and JSON-line subprocess boundaries? | [GUI architecture](gui_architecture.md); `window.py`, `engine.py`, `web/app.js` |
| `configuration-precedence.svg` | Which defaults are overridden by a saved project? | [Configuration](configuration.md); JSON/default/profile modules |
| `tool-contract.svg` | How does declarative intent map to one registered tool? | [Extension interfaces](extension_interfaces.md); `project_type.py`, `tools/_contract.py` |
| `conversation-ownership.svg` | How are a conversation writer and atomic checkpoint coordinated? | [Conversation ownership](conversation_ownership.md); `conversations.py`, `trace_logging.py`, `conversation_ownership.py` |
| `trust-boundaries.svg` | What is under user control, and where do downloads and trust change? | [Security](security.md); launcher, Hugging Face and file APIs |

## Verification and review sequence

1. Agree on each diagram's factual source lines and intended audience.
2. Sketch a small graph of only those entities needed to answer the question.
3. Check every arrow against current implementation and exceptional paths.
4. Create native SVG with stable IDs and editable text; do not render source
   code into a bitmap as the delivered artifact.
5. Render each SVG as an image at normal documentation display size and
   inspect for truncation, overlapping labels and legibility.
6. Link the figure from its authoritative page and add descriptive alternative
   text and a short caption.
7. Validate all SVG/XML syntax and references; keep PR9b documentation-only.

**Not in PR9b without separate approval:** changing runtime architecture to
make a diagram cleaner, inventing backend capabilities, or creating a new
interactive architectural application.
