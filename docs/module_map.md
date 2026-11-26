# Production module ownership and coverage map

This inventory maps the production Python modules in the accompanying
repository to their primary technical references. Responsibility summaries
reflect module docstrings. It is a **navigation and coverage matrix**, not
an exhaustive API reference or a claim that every module has a dedicated
unit test. Consult source docstrings and nearby tests for exact contracts.

## How to use the map

- **Source** links to the current implementation, including private
  modules that are not part of the package-level public API.
- **Primary owner** is the authoritative architectural explanation;
  other related documents are linked from that page.
- **Tests** links to the relevant family, not necessarily a test that
  imports that exact module. Test coverage must be verified per behavior.
- Modules under a registered tool package use that tool's own README as
  their primary reference. Their common contracts are covered by
  [Extension interfaces](extension_interfaces.md).

## Complete Python module inventory

| Module | Responsibility (source docstring) | Primary owner | Test family |
| --- | --- | --- | --- |
| [`kilin/__init__.py`](../kilin/__init__.py) | Expose the package-level KíLín API. | [Architecture](architecture.md) | [Core tests](../tests/kilin/) |
| [`kilin/_model_completion.py`](../kilin/_model_completion.py) | Validate completed model replies and diagnose incomplete output. | [Answering pipeline](answering_pipeline.md) | [Completion tests](../tests/kilin/test__model_completion.py) |
| [`kilin/_project_creation.py`](../kilin/_project_creation.py) | Reserve empty destinations and allocate independent projects exclusively. | [Project workspaces](project_workspaces.md) | [Creation tests](../tests/kilin/test_project__creation.py) |
| [`kilin/answer_timeline.py`](../kilin/answer_timeline.py) | Record lightweight, persistent timing for an answered conversation turn. | [Answering pipeline](answering_pipeline.md) | [Timeline tests](../tests/kilin/test_answer_timeline.py) |
| [`kilin/answering.py`](../kilin/answering.py) | Route questions by intent and select conversation history. | [Answering pipeline](answering_pipeline.md) | [Core tests](../tests/kilin/) |
| [`kilin/atomic_writes.py`](../kilin/atomic_writes.py) | Replace or exclusively publish text files atomically. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/kilin/) |
| [`kilin/chat.py`](../kilin/chat.py) | Manage one loaded language model and its in-memory conversation. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/chunking.py`](../kilin/chunking.py) | Split source units into token-budgeted retrieval chunks. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/kilin/) |
| [`kilin/cli.py`](../kilin/cli.py) | Launch KíLín's optional graphical interface from the command line. | [Architecture](architecture.md) | [Core tests](../tests/kilin/) |
| [`kilin/conversation_format.py`](../kilin/conversation_format.py) | Define the canonical persisted conversation-record structure in KíLín. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/kilin/) |
| [`kilin/conversation_memory/__init__.py`](../kilin/conversation_memory/__init__.py) | Keep project conversation memory distinct from research sources. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/conversation_memory/) |
| [`kilin/conversation_memory/models.py`](../kilin/conversation_memory/models.py) | Define durable turn identities and conversation feedback values. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/conversation_memory/) |
| [`kilin/conversation_memory/feedback.py`](../kilin/conversation_memory/feedback.py) | Persist project-local conversation feedback with independent ownership. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/conversation_memory/) |
| [`kilin/conversation_memory/index.py`](../kilin/conversation_memory/index.py) | Synchronize independently indexed project conversation messages. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/test__conversation_memory_index.py) |
| [`kilin/conversation_memory/inspection.py`](../kilin/conversation_memory/inspection.py) | Inspect active conversation messages without embedding or indexing. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/conversation_memory/test_inspection.py) |
| [`kilin/conversation_memory/persistence.py`](../kilin/conversation_memory/persistence.py) | Persist rebuildable conversation-memory vectors and occurrence records. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/test__conversation_memory_index.py) |
| [`kilin/conversation_memory/retrieval.py`](../kilin/conversation_memory/retrieval.py) | Retrieve conversation memory without promoting it to source evidence. | [Conversation memory](conversation_memory.md) | [Memory tests](../tests/kilin/test__conversation_memory_retrieval.py) |
| [`kilin/conversation_memory/recall.py`](../kilin/conversation_memory/recall.py) | Recognize narrow requests for verifiable conversation history. | [Conversation memory](conversation_memory.md) | [Recall tests](../tests/kilin/test_rag__historical_recall.py) |
| [`kilin/conversation_ownership.py`](../kilin/conversation_ownership.py) | Provide exclusive write ownership for one conversation JSON path. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/kilin/) |
| [`kilin/conversations.py`](../kilin/conversations.py) | Manage named conversation records in a project's trace directory. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/kilin/) |
| [`kilin/defaults.py`](../kilin/defaults.py) | Load and validate KíLín's packaged configuration defaults. | [Configuration](configuration.md) | [Core tests](../tests/kilin/) |
| [`kilin/embeddings.py`](../kilin/embeddings.py) | Load a compatible embedding model and produce normalized text vectors. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/gui/__init__.py`](../kilin/gui/__init__.py) | Provide the optional KíLín graphical-interface package. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/__main__.py`](../kilin/gui/__main__.py) | Start the optional GUI when invoked as ``python -m kilin.gui``. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/_activation.py`](../kilin/gui/_activation.py) | Ask for KíLín's passkey before the desktop window first opens. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/_paths.py`](../kilin/gui/_paths.py) | Resolve user-owned GUI data paths without importing Qt. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/_report_pdf.py`](../kilin/gui/_report_pdf.py) | Render formatted report Markdown to PDF through optional Qt. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/app.py`](../kilin/gui/app.py) | Create and run KíLín's Qt desktop application. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/engine.py`](../kilin/gui/engine.py) | Run repository and model operations behind the desktop interface. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/model_catalog.py`](../kilin/gui/model_catalog.py) | Describe the model presets displayed by the desktop interface. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/model_settings.py`](../kilin/gui/model_settings.py) | Keep GUI effort and search settings alongside core router settings. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/preferences.py`](../kilin/gui/preferences.py) | Validate GUI defaults and persist user-owned preference overrides. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/gui/window.py`](../kilin/gui/window.py) | Host KíLín's local web interface in a Qt desktop window. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/kilin/gui/) |
| [`kilin/huggingface.py`](../kilin/huggingface.py) | Inspect and reuse Hugging Face's model cache. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/model_defaults.py`](../kilin/model_defaults.py) | Define shared model purposes and default model identifiers. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/model_settings.py`](../kilin/model_settings.py) | Resolve and persist auxiliary model settings in the KíLín core. | [Model management](model_management.md) | [Core tests](../tests/kilin/test_model_settings.py) |
| [`kilin/project.py`](../kilin/project.py) | Create and open project workspaces independently of the GUI. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/kilin/) |
| [`kilin/project_files.py`](../kilin/project_files.py) | Manage project-owned outputs and templates independently of the GUI. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/kilin/) |
| [`kilin/settings_info.py`](../kilin/settings_info.py) | Read and validate shared and tool-owned setting descriptions without GUI dependencies. | [Configuration](configuration.md) | [Metadata tests](../tests/kilin/test_settings_info.py) |
| [`kilin/project_behavior.py`](../kilin/project_behavior.py) | Resolve and edit validated project-local behavior. | [Configuration](configuration.md) | [Core tests](../tests/kilin/test_project_behavior.py) |
| [`kilin/project_layout.py`](../kilin/project_layout.py) | Plan and materialize a project's core folders and tool resources. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/kilin/) |
| [`kilin/project_type.py`](../kilin/project_type.py) | Load project types from the JSON files in ``kilin/project_types/``. | [Configuration](configuration.md) | [Core tests](../tests/kilin/) |
| [`kilin/query_formats.py`](../kilin/query_formats.py) | Define the supported embedding-query formats and Qwen3 query formatter. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/question_suggestions.py`](../kilin/question_suggestions.py) | Write suggested questions about a project's own sources. | [Configuration](configuration.md) | [Core tests](../tests/kilin/test_question_suggestions.py) |
| [`kilin/rag.py`](../kilin/rag.py) | Coordinate retrieval, chat generation, and resumable project conversations. | [Answering pipeline](answering_pipeline.md) | [Core tests](../tests/kilin/) |
| [`kilin/repository.py`](../kilin/repository.py) | Select local source files and compare content manifests. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/kilin/) |
| [`kilin/repository_reason_codes.py`](../kilin/repository_reason_codes.py) | Load canonical repository-selection reasons shared with the GUI. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/kilin/) |
| [`kilin/retrievals/__init__.py`](../kilin/retrievals/__init__.py) | Expose KíLín's retrieval, code-graph, and index-persistence API. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/__init__.py`](../kilin/retrievals/code_graph/__init__.py) | Expose the code-graph records and builder. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/code_graph.py`](../kilin/retrievals/code_graph/code_graph.py) | Build a source-oriented graph of code entities and relationships. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/__init__.py`](../kilin/retrievals/code_graph/extensions/__init__.py) | Group language-specific code-relationship extraction hooks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/code_graph_html.py`](../kilin/retrievals/code_graph/extensions/code_graph_html.py) | Link an HTML page to the scripts it loads and the functions it calls. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/code_graph_js.py`](../kilin/retrievals/code_graph/extensions/code_graph_js.py) | Extract lightweight JavaScript relationships using textual patterns. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/code_graph_m.py`](../kilin/retrievals/code_graph/extensions/code_graph_m.py) | Extract lightweight MATLAB relationships using textual patterns. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/code_graph_py.py`](../kilin/retrievals/code_graph/extensions/code_graph_py.py) | Extract heuristic relationships from Python syntax trees. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/code_graph/extensions/code_graph_r.py`](../kilin/retrievals/code_graph/extensions/code_graph_r.py) | Extract lightweight R relationships using textual patterns. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/evidence_budget.py`](../kilin/retrievals/evidence_budget.py) | Apply deterministic passage, source, and evidence-token budgets. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/test_evidence_budget.py) |
| [`kilin/retrievals/document_reading.py`](../kilin/retrievals/document_reading.py) | Read one whole document or function as evidence. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/model_identity.py`](../kilin/retrievals/model_identity.py) | Identify the embedding model that produced a retrieval index. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/requests.py`](../kilin/retrievals/requests.py) | Define bounded retrieval requests and results for core and tools. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/source_coverage.py`](../kilin/retrievals/source_coverage.py) | Recognize questions requiring complete source coverage. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/test_source_coverage.py) |
| [`kilin/retrievals/source_descriptions.py`](../kilin/retrievals/source_descriptions.py) | Resolve papers a question describes instead of naming. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/test_source_descriptions.py) |
| [`kilin/retrievals/document_headers.py`](../kilin/retrievals/document_headers.py) | Extract document titles and section metadata for retrieval headers. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/persistence.py`](../kilin/retrievals/persistence.py) | Serialize retrieval indexes with source and payload integrity checks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/source_cards/__init__.py`](../kilin/retrievals/source_cards/__init__.py) | Expose generic source-card contracts and the trusted kind registry. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/descriptors.py`](../kilin/retrievals/source_cards/descriptors.py) | Build source descriptions from the existing selected files and chunks. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/generation.py`](../kilin/retrievals/source_cards/generation.py) | Run isolated, lazy source-card model generation with dedicated project settings. | [Model management](model_management.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/kinds/__init__.py`](../kilin/retrievals/source_cards/kinds/__init__.py) | Expose trusted scientific-paper card kinds and extensible interfaces. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/kinds/) |
| [`kilin/retrievals/source_cards/kinds/scientific_paper.py`](../kilin/retrievals/source_cards/kinds/scientific_paper.py) | Generate provenance-linked scientific-paper cards in one model call. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/kinds/) |
| [`kilin/retrievals/source_cards/index.py`](../kilin/retrievals/source_cards/index.py) | Validate and search separate source-card records and embeddings, and plan incremental reuse. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/indexing.py`](../kilin/retrievals/source_cards/indexing.py) | Reconcile source cards and vectors after normal project retrieval indexing. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/models.py`](../kilin/retrievals/source_cards/models.py) | Represent source cards independently of any particular source kind. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/persistence.py`](../kilin/retrievals/source_cards/persistence.py) | Atomically store source-card outcomes and bound vectors in separate, checksummed files. | [Retrieval pipeline](retrieval_pipeline.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/source_cards/registry.py`](../kilin/retrievals/source_cards/registry.py) | Register trusted source-card kinds without project-defined code imports. | [Tool extension contracts](extension_interfaces.md) | [Source-card tests](../tests/kilin/retrievals/source_cards/) |
| [`kilin/retrievals/context_expansion.py`](../kilin/retrievals/context_expansion.py) | Expand retrieval context from explicit source references in a question. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/__init__.py`](../kilin/retrievals/strategies/__init__.py) | Expose the concrete retrieval strategies and their result records. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/retrieval_code.py`](../kilin/retrievals/strategies/retrieval_code.py) | Retrieve chunks near code entities recognized in a question. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/retrieval_hybrid.py`](../kilin/retrievals/strategies/retrieval_hybrid.py) | Combine retrieval rankings through reciprocal-rank fusion. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/retrieval_lexical.py`](../kilin/retrievals/strategies/retrieval_lexical.py) | Build and query a sparse TF-IDF index over retrieval chunks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/retrieval_semantic.py`](../kilin/retrievals/strategies/retrieval_semantic.py) | Build and query an embedding-based chunk index. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/retrievals/strategies/retrieval_sources.py`](../kilin/retrievals/strategies/retrieval_sources.py) | Rank sources across four channels, then select original passages. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/kilin/retrievals/) |
| [`kilin/runtime.py`](../kilin/runtime.py) | Choose backend and device names without loading inference models. | [Model management](model_management.md) | [Core tests](../tests/kilin/) |
| [`kilin/source_card_policy.py`](../kilin/source_card_policy.py) | Resolve core-owned source-card indexing policy for one project. | [Configuration](configuration.md) | [Policy tests](../tests/kilin/test_source_card_policy.py) |
| [`kilin/sources/__init__.py`](../kilin/sources/__init__.py) | Expose source-file records and structured source extraction. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/_word_metadata.py`](../kilin/sources/_word_metadata.py) | Read optional title/author properties from Word files for citations and search. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/__init__.py`](../kilin/sources/extensions/__init__.py) | Group optional suffix-specific source readers and extractors. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_bib.py`](../kilin/sources/extensions/sources_bib.py) | Index BibTeX entries as individually searchable source units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_bbl.py`](../kilin/sources/extensions/sources_bbl.py) | Index compiled LaTeX bibliography entries as searchable source units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_doc.py`](../kilin/sources/extensions/sources_doc.py) | Read the body text of old binary Word `.doc` files. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_docx.py`](../kilin/sources/extensions/sources_docx.py) | Read Word `.docx` files as Markdown-style text with heading sections. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_html.py`](../kilin/sources/extensions/sources_html.py) | Read HTML files as page text, or as code in code projects. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_ipynb.py`](../kilin/sources/extensions/sources_ipynb.py) | Extract code and Markdown cell text from notebook JSON. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_js.py`](../kilin/sources/extensions/sources_js.py) | Partition JavaScript text at top-level declarations. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_json.py`](../kilin/sources/extensions/sources_json.py) | Partition JSON text at the keys of its top-level object. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_m.py`](../kilin/sources/extensions/sources_m.py) | Partition MATLAB text using recognizable declaration boundaries. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_md.py`](../kilin/sources/extensions/sources_md.py) | Partition Markdown text at ATX-style heading lines. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_pdf.py`](../kilin/sources/extensions/sources_pdf.py) | Extract selectable PDF text and expose page-scoped source units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_py.py`](../kilin/sources/extensions/sources_py.py) | Partition Python source using syntax-tree declaration boundaries. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_r.py`](../kilin/sources/extensions/sources_r.py) | Partition R text at its top-level functions and classes. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_tex.py`](../kilin/sources/extensions/sources_tex.py) | Partition LaTeX document text at sectioning commands. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/extensions/sources_txt.py`](../kilin/sources/extensions/sources_txt.py) | Split plain-text sources into nonblank paragraph units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/sources/source_kinds.py`](../kilin/sources/source_kinds.py) | Name the kinds of code passages the built-in readers produce. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/test_source_kinds.py) |
| [`kilin/sources/sources.py`](../kilin/sources/sources.py) | Read selected repository files and extract provenance-bearing text units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/kilin/sources/) |
| [`kilin/tools/__init__.py`](../kilin/tools/__init__.py) | Tools that intents in a project type can use. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_contract.py`](../kilin/tools/_contract.py) | Define what KíLín gives a tool and what the tool returns. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_defaults.py`](../kilin/tools/_defaults.py) | Read and validate defaults owned by individual KíLín tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_model_calls.py`](../kilin/tools/_model_calls.py) | Validate and resolve tool-owned internal thinking policies. | [Tool packages](tool_packages.md) | [Policy tests](../tests/kilin/tools/test__model_calls.py) |
| [`kilin/tools/_registry.py`](../kilin/tools/_registry.py) | Register KíLín's tools and resolve an intent's tool. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/__init__.py`](../kilin/tools/_shared/__init__.py) | Private services shared by registered KíLín tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/author_year.py`](../kilin/tools/_shared/author_year.py) | Find author-year citations and resolve them to reference-list entries. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/_shared/test_author_year.py) |
| [`kilin/tools/_shared/report_templates.py`](../kilin/tools/_shared/report_templates.py) | Read Markdown report templates and check reports against them. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/citations.py`](../kilin/tools/_shared/citations.py) | Validate bounded passage citations for document tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/_shared/test_citations.py) |
| [`kilin/tools/_shared/claims.py`](../kilin/tools/_shared/claims.py) | Judge whether a cited work supports the sentence that cites it. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/_rating.py`](../kilin/tools/_shared/_rating.py) | Rate citation suitability from checked evidence, not search rank. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/_shared/) |
| [`kilin/tools/_shared/choose_document.py`](../kilin/tools/_shared/choose_document.py) | Choose the source document a request is about, or ask the user. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/document_notes.py`](../kilin/tools/_shared/document_notes.py) | Shared reading, note-taking and citation checks for document tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/enough_evidence.py`](../kilin/tools/_shared/enough_evidence.py) | Ask whether the passages found for a question are enough to answer it. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/earlier_turns.py`](../kilin/tools/_shared/earlier_turns.py) | Select an earlier turn with listed supporting evidence. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/latex_text.py`](../kilin/tools/_shared/latex_text.py) | Reduce lines of LaTeX to the readable text they print. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/long_lists.py`](../kilin/tools/_shared/long_lists.py) | Offer to save a long list as a Markdown file instead of the chat. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/mentions.py`](../kilin/tools/_shared/mentions.py) | Reassemble and scan exact source mentions for multiple tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/model_results.py`](../kilin/tools/_shared/model_results.py) | Validate bounded model selections without false empty results. | [Tool extension contracts](extension_interfaces.md) | [Selection tests](../tests/kilin/tools/_shared/test_model_results.py) |
| [`kilin/tools/_shared/outputs.py`](../kilin/tools/_shared/outputs.py) | Read and write files in a project's ``outputs/`` folder. | [Tool packages](tool_packages.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/paths.py`](../kilin/tools/_shared/paths.py) | Name sources by the shortest unique ending of each path. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/planning.py`](../kilin/tools/_shared/planning.py) | Plan expensive tool runs and let the user confirm them or choose. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/references.py`](../kilin/tools/_shared/references.py) | Find a paper's reference lists and split one into entries, as printed. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/scan_settings.py`](../kilin/tools/_shared/scan_settings.py) | Validate shared mention-scanning fields with caller-owned defaults. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/source_access.py`](../kilin/tools/_shared/source_access.py) | Resolve source files only from an authorized project selection. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/_shared/) |
| [`kilin/tools/_shared/sentences.py`](../kilin/tools/_shared/sentences.py) | Split indexed text into distinct sentences with source positions. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/term_filter.py`](../kilin/tools/_shared/term_filter.py) | Find the passages that mention the exact terms of a question. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/_shared/terms.py`](../kilin/tools/_shared/terms.py) | Extract and validate exact search terms for registered tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/assistant_guide/__init__.py`](../kilin/tools/assistant_guide/__init__.py) | Expose the registered ``assistant_guide`` tool. | [Per-tool README](../kilin/tools/assistant_guide/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/assistant_guide/_configuration.py`](../kilin/tools/assistant_guide/_configuration.py) | Validate the defaults owned by the assistant_guide tool. | [Per-tool README](../kilin/tools/assistant_guide/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/assistant_guide/tool.py`](../kilin/tools/assistant_guide/tool.py) | Answer "What can you do for me?" with examples about the project. | [Per-tool README](../kilin/tools/assistant_guide/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_classify/__init__.py`](../kilin/tools/batch_classify/__init__.py) | Expose the registered ``batch_classify`` tool. | [Per-tool README](../kilin/tools/batch_classify/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_classify/_configuration.py`](../kilin/tools/batch_classify/_configuration.py) | Validate the defaults owned by the batch_classify tool. | [Per-tool README](../kilin/tools/batch_classify/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_classify/tool.py`](../kilin/tools/batch_classify/tool.py) | Tool ``batch_classify``: label every exact mention, in batches. | [Per-tool README](../kilin/tools/batch_classify/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_review/__init__.py`](../kilin/tools/batch_review/__init__.py) | Expose the registered ``batch_review`` tool. | [Per-tool README](../kilin/tools/batch_review/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_review/_configuration.py`](../kilin/tools/batch_review/_configuration.py) | Validate the defaults owned by the batch_review tool. | [Per-tool README](../kilin/tools/batch_review/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/batch_review/tool.py`](../kilin/tools/batch_review/tool.py) | Tool ``batch_review``: read a whole document in parts, then answer. | [Per-tool README](../kilin/tools/batch_review/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/filter_results/__init__.py`](../kilin/tools/filter_results/__init__.py) | Expose the registered ``filter_results`` tool. | [Per-tool README](../kilin/tools/filter_results/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/filter_results/_configuration.py`](../kilin/tools/filter_results/_configuration.py) | Validate the defaults owned by the filter_results tool. | [Per-tool README](../kilin/tools/filter_results/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/filter_results/tool.py`](../kilin/tools/filter_results/tool.py) | Judge the items of an earlier answer against a follow-up question. | [Per-tool README](../kilin/tools/filter_results/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_fact/__init__.py`](../kilin/tools/find_fact/__init__.py) | Expose the registered ``find_fact`` tool. | [Per-tool README](../kilin/tools/find_fact/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_fact/_configuration.py`](../kilin/tools/find_fact/_configuration.py) | Validate the defaults owned by the find_fact tool. | [Per-tool README](../kilin/tools/find_fact/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_fact/tool.py`](../kilin/tools/find_fact/tool.py) | Find a fact, or a complete list, in one paper or in every paper. | [Per-tool README](../kilin/tools/find_fact/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_passages/__init__.py`](../kilin/tools/find_passages/__init__.py) | Expose the registered ``find_passages`` tool. | [Per-tool README](../kilin/tools/find_passages/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_passages/_configuration.py`](../kilin/tools/find_passages/_configuration.py) | Validate the defaults owned by the find_passages tool. | [Per-tool README](../kilin/tools/find_passages/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/find_passages/tool.py`](../kilin/tools/find_passages/tool.py) | List every sentence, or every file, about a topic, judged by meaning. | [Per-tool README](../kilin/tools/find_passages/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/generate_paper_summary/__init__.py`](../kilin/tools/generate_paper_summary/__init__.py) | Produce a validated paper summary using project-owned templates. | [Per-tool README](../kilin/tools/generate_paper_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/generate_paper_summary/_configuration.py`](../kilin/tools/generate_paper_summary/_configuration.py) | Validate the paper-summary tool's packaged configuration defaults. | [Per-tool README](../kilin/tools/generate_paper_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/generate_paper_summary/_report_format.py`](../kilin/tools/generate_paper_summary/_report_format.py) | Format report Markdown into safe HTML for print rendering. | [Per-tool README](../kilin/tools/generate_paper_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/generate_paper_summary/_report_template.py`](../kilin/tools/generate_paper_summary/_report_template.py) | Read report templates and check reports with this tool's limits. | [Per-tool README](../kilin/tools/generate_paper_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/generate_paper_summary/tool.py`](../kilin/tools/generate_paper_summary/tool.py) | Tool ``generate_paper_summary``: turn one paper into a templated report. | [Per-tool README](../kilin/tools/generate_paper_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/paper_overview/__init__.py`](../kilin/tools/paper_overview/__init__.py) | Expose the registered ``paper_overview`` tool. | [Per-tool README](../kilin/tools/paper_overview/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/paper_overview/_configuration.py`](../kilin/tools/paper_overview/_configuration.py) | Validate the defaults owned by the paper_overview tool. | [Per-tool README](../kilin/tools/paper_overview/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/paper_overview/tool.py`](../kilin/tools/paper_overview/tool.py) | Summarize one paper, chosen first, from its own passages. | [Per-tool README](../kilin/tools/paper_overview/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/__init__.py`](../kilin/tools/project_summary/__init__.py) | Expose the registered ``project_summary`` tool. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/_configuration.py`](../kilin/tools/project_summary/_configuration.py) | Validate the defaults owned by the project_summary tool. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/_documents.py`](../kilin/tools/project_summary/_documents.py) | Choose a project's documents and the excerpt read from each. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/_mapping.py`](../kilin/tools/project_summary/_mapping.py) | Take one document's notes for a project report, and check them. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/_reduce.py`](../kilin/tools/project_summary/_reduce.py) | Write a project report from every document's notes and checks. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_summary/tool.py`](../kilin/tools/project_summary/tool.py) | Tool ``project_summary``: one report on all the documents of a project. | [Per-tool README](../kilin/tools/project_summary/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/__init__.py`](../kilin/tools/reference_check/__init__.py) | Expose the registered ``reference_check`` tool. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/_citations.py`](../kilin/tools/reference_check/_citations.py) | Find a paper's numbered in-text citations and their citing sentences. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/_configuration.py`](../kilin/tools/reference_check/_configuration.py) | Validate the defaults owned by the reference_check tool. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/_latex.py`](../kilin/tools/reference_check/_latex.py) | Read a LaTeX paper's citations and bibliography. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/_matching.py`](../kilin/tools/reference_check/_matching.py) | Match a paper's reference entries to the files of a project. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/_resolution.py`](../kilin/tools/reference_check/_resolution.py) | Resolve missing bibliography entries without implicit network access. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/reference_check/tool.py`](../kilin/tools/reference_check/tool.py) | Tool ``reference_check``: check one paper's references and citations. | [Per-tool README](../kilin/tools/reference_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_inventory/__init__.py`](../kilin/tools/project_inventory/__init__.py) | Expose the registered ``project_inventory`` tool. | [Per-tool README](../kilin/tools/project_inventory/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_inventory/_configuration.py`](../kilin/tools/project_inventory/_configuration.py) | Validate the defaults owned by the project_inventory tool. | [Per-tool README](../kilin/tools/project_inventory/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/project_inventory/tool.py`](../kilin/tools/project_inventory/tool.py) | Tool ``project_inventory``: exact facts about the project's files. | [Per-tool README](../kilin/tools/project_inventory/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/scan_and_list/__init__.py`](../kilin/tools/scan_and_list/__init__.py) | Expose the registered ``scan_and_list`` tool. | [Per-tool README](../kilin/tools/scan_and_list/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/scan_and_list/_configuration.py`](../kilin/tools/scan_and_list/_configuration.py) | Validate the defaults owned by the scan_and_list tool. | [Per-tool README](../kilin/tools/scan_and_list/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/scan_and_list/tool.py`](../kilin/tools/scan_and_list/tool.py) | Tool ``scan_and_list``: every exact mention, without model analysis. | [Per-tool README](../kilin/tools/scan_and_list/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/show_source/__init__.py`](../kilin/tools/show_source/__init__.py) | Expose the registered ``show_source`` tool. | [Per-tool README](../kilin/tools/show_source/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/show_source/_configuration.py`](../kilin/tools/show_source/_configuration.py) | Validate the defaults owned by the show_source tool. | [Per-tool README](../kilin/tools/show_source/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/show_source/tool.py`](../kilin/tools/show_source/tool.py) | Show the exact source text behind an earlier answer, and explain it. | [Per-tool README](../kilin/tools/show_source/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/suggest_reference/__init__.py`](../kilin/tools/suggest_reference/__init__.py) | Expose the registered ``suggest_reference`` tool. | [Per-tool README](../kilin/tools/suggest_reference/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/suggest_reference/_configuration.py`](../kilin/tools/suggest_reference/_configuration.py) | Validate the defaults owned by the suggest_reference tool. | [Per-tool README](../kilin/tools/suggest_reference/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/suggest_reference/tool.py`](../kilin/tools/suggest_reference/tool.py) | Tool ``suggest_reference``: find papers in the project to cite. | [Per-tool README](../kilin/tools/suggest_reference/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/__init__.py`](../kilin/tools/writing_check/__init__.py) | Expose the registered ``writing_check`` tool. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_checks.py`](../kilin/tools/writing_check/_checks.py) | Check a document's consistency and style in code, and build its glossary. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_configuration.py`](../kilin/tools/writing_check/_configuration.py) | Validate the defaults owned by the writing_check tool. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_document.py`](../kilin/tools/writing_check/_document.py) | Read a document into sections and paragraphs of prose. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_followup.py`](../kilin/tools/writing_check/_followup.py) | Rewrite the sentence a writing check flagged, when the user asks. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_review.py`](../kilin/tools/writing_check/_review.py) | Check the grammar and style of one paragraph with the model. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/_scope.py`](../kilin/tools/writing_check/_scope.py) | Choose the part of a document a question asks to check. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tools/writing_check/tool.py`](../kilin/tools/writing_check/tool.py) | Tool ``writing_check``: check the grammar and style of a document. | [Per-tool README](../kilin/tools/writing_check/README.md) | [Tool tests](../tests/kilin/tools/) |
| [`kilin/tool_settings.py`](../kilin/tool_settings.py) | List and change the tool settings of one KíLín project. | [Configuration](configuration.md) | [Core tests](../tests/kilin/) |
| [`kilin/system_testing/__init__.py`](../kilin/system_testing/__init__.py) | Expose read-only experiment inspection and validation records. | [System testing](system_testing.md) | [System tests](../tests/kilin/system_testing/) |
| [`kilin/system_testing/experiment.py`](../kilin/system_testing/experiment.py) | Represent and inspect a closed original reference workspace. | [System testing](system_testing.md) | [System tests](../tests/kilin/system_testing/) |
| [`kilin/system_testing/traces.py`](../kilin/system_testing/traces.py) | Decode original diagnostic events and compact span terminals. | [System testing](system_testing.md) | [System tests](../tests/kilin/system_testing/) |
| [`kilin/system_testing/timeline.py`](../kilin/system_testing/timeline.py) | Reconstruct chronological recorded user commands. | [System testing](system_testing.md) | [System tests](../tests/kilin/system_testing/) |
| [`kilin/system_testing/validation.py`](../kilin/system_testing/validation.py) | Validate recording completeness, sources, settings, and durable history. | [System testing](system_testing.md) | [System tests](../tests/kilin/system_testing/) |
| [`kilin/_trace_configuration.py`](../kilin/_trace_configuration.py) | Encode complete effective configurations and construct and verify compact references. | [Trace architecture](trace_architecture.md) | [Configuration trace tests](../tests/kilin/test_trace_logging__configuration.py) |
| [`kilin/_trace_project.py`](../kilin/_trace_project.py) | Coordinate project-wide event numbering and shared trace publication. | [Trace architecture](trace_architecture.md) | [Project trace tests](../tests/kilin/test__trace_project.py) |
| [`kilin/trace_logging.py`](../kilin/trace_logging.py) | Record structured execution events and resumable conversation state. | [Trace recording](trace_recording.md) | [Core tests](../tests/kilin/) |
| [`kilin/workspace.py`](../kilin/workspace.py) | Copy selected local sources into a managed multi-source workspace. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/kilin/) |
| [`kilin/workspace_resources.py`](../kilin/workspace_resources.py) | Represent packaged resources installed into a KíLín project workspace. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/kilin/) |

## Module owner summary

| Owner | Modules |
| --- | ---: |
| Answering pipeline | 4 |
| Architecture | 2 |
| Configuration | 7 |
| Conversation ownership | 4 |
| Conversation memory | 8 |
| GUI architecture | 11 |
| Model management | 8 |
| Per-tool README | 59 |
| Project workspaces | 6 |
| Retrieval pipeline | 32 |
| Source pipeline | 23 |
| System testing | 5 |
| Tool extension contracts | 27 |
| Tool packages | 2 |
| Trace architecture | 2 |
| Trace recording | 1 |

**Total production Python modules: 201.**

## Non-Python resources and UI assets

| Resource family | Implemented role | Primary reference |
| --- | --- | --- |
| [`kilin/settings_info.json`](../kilin/settings_info.json) | Public setting names and explanations, separate from default values | [Configuration](configuration.md) |
| [`kilin/kilin_defaults.json`](../kilin/kilin_defaults.json) | Validated, packaged core settings | [Package defaults](package_defaults.md) |
| [`kilin/repository_reason_codes.json`](../kilin/repository_reason_codes.json) | Shared scanner/GUI reason-code vocabulary | [Source pipeline](source_pipeline.md), [GUI protocol](gui_protocol.md) |
| [`kilin/project_types/`](../kilin/project_types/) | Validated packaged project type definitions | [Configuration](configuration.md), [Extension interfaces](extension_interfaces.md) |
| [`kilin/gui/defaults.json`](../kilin/gui/defaults.json) | Immutable GUI initial settings | [Configuration](configuration.md), [GUI README](../kilin/gui/README.md) |
| [`kilin/gui/web/`](../kilin/gui/web/) | HTML layout, styling and JavaScript state/events | [GUI architecture](gui_architecture.md), [GUI protocol](gui_protocol.md) |
| [`kilin/gui/icons/`](../kilin/gui/icons/) | Packaged native GUI icon assets | [GUI architecture](gui_architecture.md) |
| [`kilin/tools/`](../kilin/tools/) | Registered tools' READMEs, `defaults.json`, `settings_info.json` and templates | [Tool packages](tool_packages.md), [Extension interfaces](extension_interfaces.md) |
| [`tutorials/`](../tutorials/) | Eighteen executable numbered learning notebooks and reference SVGs | [Development](development.md) |
| [`assets/`](../assets/) | Maintenance, benchmark and notebook-validation assets | [Development](development.md), [Performance](performance.md) |
| [`pyproject.toml`](../pyproject.toml) | Python requirements, dependency extras, package data, tooling | [Development](development.md) |
| [`tests/`](../tests/) | Unit/integration suites and independent runner | [Development](development.md) |

## Completeness and maintenance

This table covers **all `.py` files under `kilin/` in the accompanying
repository**, including package initializers, private helpers, and
extension modules. It does **not** claim that non-Python resources are
executable modules, nor does it include generated bytecode, private
user workspaces or external installed dependencies.

When adding, moving or deleting a production module, update this matrix
with the same change. Ensure the owner document explains its
responsibility, dependency contracts, failure behavior and relevant tests.
Verify the inventory against the source tree rather than assuming it is
updated automatically.

Related: [Technical documentation index](README.md),
[Architecture](architecture.md), [Development](development.md).
