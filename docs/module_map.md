# Production module ownership and coverage map

**Source baseline:** QAChat after the GUI-test consolidation PR,
14 October 2026. The entries below were generated from the actual Python
module inventory and their opening module-docstring descriptions, then
assigned a primary technical owner. This is a **navigation and coverage
matrix**, not a claim that one page enumerates every function or that every
module has its own dedicated unit test. Consult each linked module docstring
and the nearby mirrored tests for precise call signatures.

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
| [`qachat/__init__.py`](../qachat/__init__.py) | Expose the package-level QAChat API. | [Architecture](architecture.md) | [Core tests](../tests/qachat/) |
| [`qachat/answering.py`](../qachat/answering.py) | Route questions by intent and select conversation history. | [Answering pipeline](answering_pipeline.md) | [Core tests](../tests/qachat/) |
| [`qachat/atomic_writes.py`](../qachat/atomic_writes.py) | Write local text files through isolated, atomic replacements. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/qachat/) |
| [`qachat/chat.py`](../qachat/chat.py) | Manage one loaded language model and its in-memory conversation. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/chunking.py`](../qachat/chunking.py) | Split source units into token-budgeted retrieval chunks. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/qachat/) |
| [`qachat/cli.py`](../qachat/cli.py) | Launch QAChat's optional graphical interface from the command line. | [Architecture](architecture.md) | [Core tests](../tests/qachat/) |
| [`qachat/conversation_format.py`](../qachat/conversation_format.py) | Define the canonical persisted conversation-record structure in QAChat. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/qachat/) |
| [`qachat/conversation_ownership.py`](../qachat/conversation_ownership.py) | Provide exclusive write ownership for one conversation JSON path. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/qachat/) |
| [`qachat/conversations.py`](../qachat/conversations.py) | Manage named conversation records in a project's trace directory. | [Conversation ownership](conversation_ownership.md) | [Core tests](../tests/qachat/) |
| [`qachat/defaults.py`](../qachat/defaults.py) | Load and validate QAChat's packaged configuration defaults. | [Configuration](configuration.md) | [Core tests](../tests/qachat/) |
| [`qachat/embeddings.py`](../qachat/embeddings.py) | Load a compatible embedding model and produce normalized text vectors. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/gui/__init__.py`](../qachat/gui/__init__.py) | Provide the optional QAChat graphical-interface package. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/__main__.py`](../qachat/gui/__main__.py) | Start the optional GUI when invoked as ``python -m qachat.gui``. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/_paths.py`](../qachat/gui/_paths.py) | Resolve user-owned GUI data paths without importing Qt. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/_report_pdf.py`](../qachat/gui/_report_pdf.py) | Render formatted report Markdown to PDF through optional Qt. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/app.py`](../qachat/gui/app.py) | Create and run QAChat's Qt desktop application. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/engine.py`](../qachat/gui/engine.py) | Run repository and model operations behind the desktop interface. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/model_catalog.py`](../qachat/gui/model_catalog.py) | Describe the model presets displayed by the desktop interface. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/model_settings.py`](../qachat/gui/model_settings.py) | Keep a project's effort levels, search depth, and router in profile.ini. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/preferences.py`](../qachat/gui/preferences.py) | Validate GUI defaults and persist user-owned preference overrides. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/gui/window.py`](../qachat/gui/window.py) | Host QAChat's local web interface in a Qt desktop window. | [GUI architecture](gui_architecture.md) | [GUI tests](../tests/qachat/gui/) |
| [`qachat/huggingface.py`](../qachat/huggingface.py) | Inspect and reuse Hugging Face's model cache. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/model_defaults.py`](../qachat/model_defaults.py) | Define shared model purposes and default model identifiers. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/project.py`](../qachat/project.py) | Create and open project workspaces independently of the GUI. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/qachat/) |
| [`qachat/project_files.py`](../qachat/project_files.py) | Manage project-owned outputs and templates independently of the GUI. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/qachat/) |
| [`qachat/project_layout.py`](../qachat/project_layout.py) | Plan and materialize a project's core folders and tool resources. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/qachat/) |
| [`qachat/project_type.py`](../qachat/project_type.py) | Load project types from the JSON files in ``qachat/project_types/``. | [Configuration](configuration.md) | [Core tests](../tests/qachat/) |
| [`qachat/query_formats.py`](../qachat/query_formats.py) | Define the supported embedding-query formats and Qwen3 query formatter. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/rag.py`](../qachat/rag.py) | Coordinate retrieval, chat generation, and resumable project conversations. | [Answering pipeline](answering_pipeline.md) | [Core tests](../tests/qachat/) |
| [`qachat/repository.py`](../qachat/repository.py) | Select local source files and compare content manifests. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/qachat/) |
| [`qachat/repository_reason_codes.py`](../qachat/repository_reason_codes.py) | Load canonical repository-selection reasons shared with the GUI. | [Source pipeline](source_pipeline.md) | [Core tests](../tests/qachat/) |
| [`qachat/retrievals/__init__.py`](../qachat/retrievals/__init__.py) | Expose QAChat's retrieval, code-graph, and index-persistence API. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/code_graph/__init__.py`](../qachat/retrievals/code_graph/__init__.py) | Expose the code-graph records and builder. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/code_graph/code_graph.py`](../qachat/retrievals/code_graph/code_graph.py) | Build a source-oriented graph of code entities and relationships. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/code_graph/extensions/__init__.py`](../qachat/retrievals/code_graph/extensions/__init__.py) | Group language-specific code-relationship extraction hooks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/code_graph/extensions/code_graph_m.py`](../qachat/retrievals/code_graph/extensions/code_graph_m.py) | Extract lightweight MATLAB relationships using textual patterns. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/code_graph/extensions/code_graph_py.py`](../qachat/retrievals/code_graph/extensions/code_graph_py.py) | Extract heuristic relationships from Python syntax trees. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/document_reading.py`](../qachat/retrievals/document_reading.py) | Read one whole document or function as evidence. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/model_identity.py`](../qachat/retrievals/model_identity.py) | Identify the embedding model that produced a retrieval index. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/persistence.py`](../qachat/retrievals/persistence.py) | Serialize retrieval indexes with source and payload integrity checks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/routing.py`](../qachat/retrievals/routing.py) | Expand retrieval context from explicit source references in a question. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/strategies/__init__.py`](../qachat/retrievals/strategies/__init__.py) | Expose the concrete retrieval strategies and their result records. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/strategies/retrieval_code.py`](../qachat/retrievals/strategies/retrieval_code.py) | Retrieve chunks near code entities recognized in a question. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/strategies/retrieval_hybrid.py`](../qachat/retrievals/strategies/retrieval_hybrid.py) | Combine retrieval rankings through reciprocal-rank fusion. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/strategies/retrieval_lexical.py`](../qachat/retrievals/strategies/retrieval_lexical.py) | Build and query a sparse TF-IDF index over retrieval chunks. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/retrievals/strategies/retrieval_semantic.py`](../qachat/retrievals/strategies/retrieval_semantic.py) | Build and query an embedding-based chunk index. | [Retrieval pipeline](retrieval_pipeline.md) | [Retrieval tests](../tests/qachat/retrievals/) |
| [`qachat/runtime.py`](../qachat/runtime.py) | Choose backend and device names without loading inference models. | [Model management](model_management.md) | [Core tests](../tests/qachat/) |
| [`qachat/sources/__init__.py`](../qachat/sources/__init__.py) | Expose source-file records and structured source extraction. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/__init__.py`](../qachat/sources/extensions/__init__.py) | Group optional suffix-specific source readers and extractors. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_ipynb.py`](../qachat/sources/extensions/sources_ipynb.py) | Extract code and Markdown cell text from notebook JSON. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_m.py`](../qachat/sources/extensions/sources_m.py) | Partition MATLAB text using recognizable declaration boundaries. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_md.py`](../qachat/sources/extensions/sources_md.py) | Partition Markdown text at ATX-style heading lines. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_pdf.py`](../qachat/sources/extensions/sources_pdf.py) | Extract selectable PDF text and expose page-scoped source units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_py.py`](../qachat/sources/extensions/sources_py.py) | Partition Python source using syntax-tree declaration boundaries. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/extensions/sources_txt.py`](../qachat/sources/extensions/sources_txt.py) | Split plain-text sources into nonblank paragraph units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/sources/sources.py`](../qachat/sources/sources.py) | Read selected repository files and extract provenance-bearing text units. | [Source pipeline](source_pipeline.md) | [Source tests](../tests/qachat/sources/) |
| [`qachat/tools/__init__.py`](../qachat/tools/__init__.py) | Tools that intents in a project type can use. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_contract.py`](../qachat/tools/_contract.py) | Define what QAChat gives a tool and what the tool returns. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_defaults.py`](../qachat/tools/_defaults.py) | Read and validate defaults owned by individual QAChat tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_registry.py`](../qachat/tools/_registry.py) | Register QAChat's tools and resolve an intent's tool. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/__init__.py`](../qachat/tools/_shared/__init__.py) | Private services shared by registered QAChat tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/choose_document.py`](../qachat/tools/_shared/choose_document.py) | Choose the source document a request is about, or ask the user. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/document_notes.py`](../qachat/tools/_shared/document_notes.py) | Shared reading, note-taking and citation checks for document tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/earlier_turns.py`](../qachat/tools/_shared/earlier_turns.py) | Select an earlier turn with listed supporting evidence. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/long_lists.py`](../qachat/tools/_shared/long_lists.py) | Offer to save a long list as a Markdown file instead of the chat. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/mentions.py`](../qachat/tools/_shared/mentions.py) | Reassemble and scan exact source mentions for multiple tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/outputs.py`](../qachat/tools/_shared/outputs.py) | Read and write files in a project's ``outputs/`` folder. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/paths.py`](../qachat/tools/_shared/paths.py) | Name sources by the shortest unique ending of each path. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/planning.py`](../qachat/tools/_shared/planning.py) | Plan expensive tool runs and let the user confirm them or choose. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/scan_settings.py`](../qachat/tools/_shared/scan_settings.py) | Validate shared mention-scanning fields with caller-owned defaults. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/sentences.py`](../qachat/tools/_shared/sentences.py) | Split indexed text into distinct sentences with source positions. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/term_filter.py`](../qachat/tools/_shared/term_filter.py) | Find the passages that mention the exact terms of a question. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/_shared/terms.py`](../qachat/tools/_shared/terms.py) | Extract and validate exact search terms for registered tools. | [Tool extension contracts](extension_interfaces.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_classify/__init__.py`](../qachat/tools/batch_classify/__init__.py) | Expose the registered ``batch_classify`` tool. | [Per-tool README](../qachat/tools/batch_classify/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_classify/_configuration.py`](../qachat/tools/batch_classify/_configuration.py) | Validate the defaults owned by the batch_classify tool. | [Per-tool README](../qachat/tools/batch_classify/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_classify/tool.py`](../qachat/tools/batch_classify/tool.py) | Tool ``batch_classify``: label every exact mention, in batches. | [Per-tool README](../qachat/tools/batch_classify/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_review/__init__.py`](../qachat/tools/batch_review/__init__.py) | Expose the registered ``batch_review`` tool. | [Per-tool README](../qachat/tools/batch_review/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_review/_configuration.py`](../qachat/tools/batch_review/_configuration.py) | Validate the defaults owned by the batch_review tool. | [Per-tool README](../qachat/tools/batch_review/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/batch_review/tool.py`](../qachat/tools/batch_review/tool.py) | Tool ``batch_review``: read a whole document in parts, then answer. | [Per-tool README](../qachat/tools/batch_review/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/filter_results/__init__.py`](../qachat/tools/filter_results/__init__.py) | Expose the registered ``filter_results`` tool. | [Per-tool README](../qachat/tools/filter_results/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/filter_results/_configuration.py`](../qachat/tools/filter_results/_configuration.py) | Validate the defaults owned by the filter_results tool. | [Per-tool README](../qachat/tools/filter_results/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/filter_results/tool.py`](../qachat/tools/filter_results/tool.py) | Judge the items of an earlier answer against a follow-up question. | [Per-tool README](../qachat/tools/filter_results/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_fact/__init__.py`](../qachat/tools/find_fact/__init__.py) | Expose the registered ``find_fact`` tool. | [Per-tool README](../qachat/tools/find_fact/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_fact/_configuration.py`](../qachat/tools/find_fact/_configuration.py) | Validate the defaults owned by the find_fact tool. | [Per-tool README](../qachat/tools/find_fact/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_fact/tool.py`](../qachat/tools/find_fact/tool.py) | Find a fact, or a complete list, in one paper or in every paper. | [Per-tool README](../qachat/tools/find_fact/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_passages/__init__.py`](../qachat/tools/find_passages/__init__.py) | Expose the registered ``find_passages`` tool. | [Per-tool README](../qachat/tools/find_passages/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_passages/_configuration.py`](../qachat/tools/find_passages/_configuration.py) | Validate the defaults owned by the find_passages tool. | [Per-tool README](../qachat/tools/find_passages/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/find_passages/tool.py`](../qachat/tools/find_passages/tool.py) | List every sentence, or every file, about a topic, judged by meaning. | [Per-tool README](../qachat/tools/find_passages/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/generate_paper_summary/__init__.py`](../qachat/tools/generate_paper_summary/__init__.py) | Produce a validated paper summary using project-owned templates. | [Per-tool README](../qachat/tools/generate_paper_summary/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/generate_paper_summary/_configuration.py`](../qachat/tools/generate_paper_summary/_configuration.py) | Validate the paper-summary tool's packaged configuration defaults. | [Per-tool README](../qachat/tools/generate_paper_summary/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/generate_paper_summary/_report_format.py`](../qachat/tools/generate_paper_summary/_report_format.py) | Format report Markdown into safe HTML for print rendering. | [Per-tool README](../qachat/tools/generate_paper_summary/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/generate_paper_summary/_report_template.py`](../qachat/tools/generate_paper_summary/_report_template.py) | Read Markdown report templates and check reports against them. | [Per-tool README](../qachat/tools/generate_paper_summary/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/generate_paper_summary/tool.py`](../qachat/tools/generate_paper_summary/tool.py) | Tool ``generate_paper_summary``: turn one paper into a templated report. | [Per-tool README](../qachat/tools/generate_paper_summary/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/paper_overview/__init__.py`](../qachat/tools/paper_overview/__init__.py) | Expose the registered ``paper_overview`` tool. | [Per-tool README](../qachat/tools/paper_overview/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/paper_overview/_configuration.py`](../qachat/tools/paper_overview/_configuration.py) | Validate the defaults owned by the paper_overview tool. | [Per-tool README](../qachat/tools/paper_overview/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/paper_overview/tool.py`](../qachat/tools/paper_overview/tool.py) | Summarize one paper, chosen first, from its own passages. | [Per-tool README](../qachat/tools/paper_overview/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/project_inventory/__init__.py`](../qachat/tools/project_inventory/__init__.py) | Expose the registered ``project_inventory`` tool. | [Per-tool README](../qachat/tools/project_inventory/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/project_inventory/_configuration.py`](../qachat/tools/project_inventory/_configuration.py) | Validate the defaults owned by the project_inventory tool. | [Per-tool README](../qachat/tools/project_inventory/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/project_inventory/tool.py`](../qachat/tools/project_inventory/tool.py) | Tool ``project_inventory``: exact facts about the project's files. | [Per-tool README](../qachat/tools/project_inventory/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/scan_and_list/__init__.py`](../qachat/tools/scan_and_list/__init__.py) | Expose the registered ``scan_and_list`` tool. | [Per-tool README](../qachat/tools/scan_and_list/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/scan_and_list/_configuration.py`](../qachat/tools/scan_and_list/_configuration.py) | Validate the defaults owned by the scan_and_list tool. | [Per-tool README](../qachat/tools/scan_and_list/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/scan_and_list/tool.py`](../qachat/tools/scan_and_list/tool.py) | Tool ``scan_and_list``: every exact mention, without model analysis. | [Per-tool README](../qachat/tools/scan_and_list/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/show_source/__init__.py`](../qachat/tools/show_source/__init__.py) | Expose the registered ``show_source`` tool. | [Per-tool README](../qachat/tools/show_source/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/show_source/_configuration.py`](../qachat/tools/show_source/_configuration.py) | Validate the defaults owned by the show_source tool. | [Per-tool README](../qachat/tools/show_source/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/tools/show_source/tool.py`](../qachat/tools/show_source/tool.py) | Show the exact source text behind an earlier answer, and explain it. | [Per-tool README](../qachat/tools/show_source/README.md) | [Tool tests](../tests/qachat/tools/) |
| [`qachat/trace_logging.py`](../qachat/trace_logging.py) | Record structured execution events and resumable conversation state. | [Trace recording](trace_recording.md) | [Core tests](../tests/qachat/) |
| [`qachat/workspace.py`](../qachat/workspace.py) | Copy selected local sources into a managed multi-source workspace. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/qachat/) |
| [`qachat/workspace_resources.py`](../qachat/workspace_resources.py) | Represent packaged resources installed into a QAChat project workspace. | [Project workspaces](project_workspaces.md) | [Core tests](../tests/qachat/) |

## Module owner summary

| Owner | Modules |
| --- | ---: |
| Answering pipeline | 2 |
| Architecture | 2 |
| Configuration | 2 |
| Conversation ownership | 4 |
| GUI architecture | 10 |
| Model management | 6 |
| Per-tool README | 32 |
| Project workspaces | 5 |
| Retrieval pipeline | 15 |
| Source pipeline | 12 |
| Tool extension contracts | 17 |
| Trace recording | 1 |

**Total production Python modules: 108.**

## Non-Python resources and UI assets

| Resource family | Implemented role | Primary reference |
| --- | --- | --- |
| [`qachat/qachat_defaults.json`](../qachat/qachat_defaults.json) | Validated, packaged core settings | [Package defaults](package_defaults.md) |
| [`qachat/repository_reason_codes.json`](../qachat/repository_reason_codes.json) | Shared scanner/GUI reason-code vocabulary | [Source pipeline](source_pipeline.md), [GUI protocol](gui_protocol.md) |
| [`qachat/project_types/`](../qachat/project_types/) | Built-in `code`/`literature` type definitions | [Configuration](configuration.md), [Extension interfaces](extension_interfaces.md) |
| [`qachat/gui/defaults.json`](../qachat/gui/defaults.json) | Immutable GUI initial settings | [Configuration](configuration.md), [GUI README](../qachat/gui/README.md) |
| [`qachat/gui/web/`](../qachat/gui/web/) | HTML layout, styling and JavaScript state/events | [GUI architecture](gui_architecture.md), [GUI protocol](gui_protocol.md) |
| [`qachat/gui/icons/`](../qachat/gui/icons/) | Packaged native GUI icon assets | [GUI architecture](gui_architecture.md) |
| [`qachat/tools/`](../qachat/tools/) | Ten tool READMEs, `defaults.json` files and summary templates | [Tool packages](tool_packages.md), [Extension interfaces](extension_interfaces.md) |
| [`tutorials/`](../tutorials/) | Sixteen executable numbered learning notebooks and reference SVGs | [Development](development.md) |
| [`assets/`](../assets/) | Maintenance, benchmark and notebook-validation assets | [Development](development.md), [Performance](performance.md) |
| [`pyproject.toml`](../pyproject.toml) | Python requirements, dependency extras, package data, tooling | [Development](development.md) |
| [`tests/`](../tests/) | Unit/integration suites and independent runner | [Development](development.md) |

## Completeness and maintenance

This table covers **all `.py` files under `qachat/` in the stated
baseline**, including package initializers, private helpers, and extension
modules. It does **not** claim that non-Python resources are executable
modules, nor does it include generated bytecode, private user workspaces
or external installed dependencies.

When adding, moving or deleting a production module, update this matrix
with the same PR. Ensure the owner document explains its responsibility,
dependency contracts, failure behavior and relevant tests. This is a
manually maintained checked-in reference after PR9a; do not assume the
original inventory-generation script is part of the distribution.

Related: [Technical documentation index](README.md),
[Architecture](architecture.md), [Development](development.md).
