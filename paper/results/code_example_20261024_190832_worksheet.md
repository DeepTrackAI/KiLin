# Code example: grading worksheet

Indexed revision: 8bd20824c (HEAD)

## backend_choice (mechanism, held_out_1)

**Question.** Explain how choose_backend selects the computation backend, and which PyTorch device is used when PyTorch is selected.

**Intent:** code_explanation

**Answer with retrieval:**

The `choose_backend` function selects the computation backend based on the requested value and platform. If the requested backend is `"auto"`, it defaults to `"mlx"` on Apple Silicon (Darwin/arm64) and `"torch"` otherwise. Explicit `"mlx"` requests are rejected on non-Apple Silicon platforms [3]. When PyTorch (`"torch"`) is selected, the `torch_device` function determines the device: it prioritizes CUDA if available, then MPS (Apple Silicon), and defaults to CPU if neither is available [7]. This device selection relies on caller-supplied availability flags, not hardware probing [1]. The `choose_backend` output directly influences which device `torch_device` uses, ensuring compatibility with the selected backend's capabilities.

**Cited passages:**

- [1] kilin/runtime.py `runtime` (retrieved by: lexical, semantic)
- [3] kilin/runtime.py `choose_backend` (retrieved by: code, semantic)
- [7] kilin/runtime.py `torch_device` (retrieved by: context_expansion)

**Gold functions:** `choose_backend`, `torch_device`

- in evidence: ['choose_backend', 'torch_device']
- cited: ['choose_backend', 'torch_device']

**Facts (tick those the answer states correctly):**

- [ ] The requested value must be auto, mlx, or torch; anything else raises an error.
- [ ] auto selects MLX on a native Apple silicon Mac (macOS on arm64) and PyTorch otherwise.
- [ ] Requesting mlx explicitly on any other platform raises an error.
- [ ] With PyTorch, the device is CUDA if available, otherwise Apple MPS if available, otherwise the CPU.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `choose_backend` function in the KíLín Python package is designed to select a computation backend (e.g., PyTorch, TensorFlow) based on the availability of the required libraries and user configuration. Here's how it typically works:

1. **Backend Selection Logic**:
   - The function checks for the presence of required dependencies (e.g., PyTorch or TensorFlow).
   - If multiple backends are available, the user may explicitly specify the desired backend (e.g., via a configuration file, environment variable, or function argument).
   - If no explicit choice is made, it may default to the first available backend (e.g., PyTorch if installed, or TensorFlow if installed).

2. **PyTorch Device Usage**:
   - When PyTorch is selected as the backend, the computation device (CPU/GPU) is determined by:
     - **CUDA availability**: If a CUDA-enabled GPU is detected, PyTorch typically defaults to using the first available GPU (`device=0`).
     - **User override**: The user can explicitly set the device using `torch.device("cuda")` or `torch.device("cpu")` in their code.
   - KíLín may not directly control the device selection but relies on PyTorch's default behavior unless explicitly configured otherwise.

**Note**: The exact behavior of `choose_backend` and device selection depends on the specific implementation of KíLín. If you are using a version of KíLín with custom logic, consult its documentation or source code for precise details.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## tool_confirmation (mechanism, held_out_1)

**Question.** Explain how needs_confirmation and confirm_plan decide whether a tool asks the user before running, and what happens with the user's answer.

**Intent:** code_explanation

**Answer with retrieval:**

The `needs_confirmation` function determines if a tool requires user confirmation based on its confirmation policy and the plan's parameters. It returns `True` if the policy is `"always"` or if the plan's batch count exceeds the threshold (`large_batches`) when the policy is `"when_large"`. Otherwise, it returns `False` [6]. 

The `confirm_plan` function uses `needs_confirmation` to decide whether to prompt the user. If confirmation is required, it calls `context.confirm(plan)` to retrieve the user's answer. The answer is validated against allowed choices (`PLAN_CHOICE_CHOICES`). If valid, it maps to a `PlanChoice` (e.g., `"continue"`, `"quick"`, or `"cancel"`); otherwise, it defaults to `"cancel"` [2]. The function traces the decision and whether the user was asked, returning the choice and confirmation status [2]. 

The user's answer is processed through `context.confirm`, which interacts with the user via a UI (e.g., buttons) and returns an index or `None` for cancellation. Invalid answers result in cancellation, while valid choices dictate the tool's next action [8]. This workflow ensures tools only prompt users when required by policy and handle responses appropriately.

**Cited passages:**

- [2] kilin/tools/_shared/planning.py `confirm_plan` (retrieved by: code, lexical, semantic)
- [6] kilin/tools/_shared/planning.py `needs_confirmation` (retrieved by: context_expansion)
- [8] kilin/tools/_shared/planning.py `ask_choice` (retrieved by: context_expansion)

**Gold functions:** `needs_confirmation`, `confirm_plan`

- in evidence: ['needs_confirmation', 'confirm_plan']
- cited: ['needs_confirmation', 'confirm_plan']

**Facts (tick those the answer states correctly):**

- [ ] A plan with no items never asks, and the policy never never asks.
- [ ] The policy always always asks; when_large asks only when the plan's model calls exceed the tool's large-batch threshold.
- [ ] confirm_plan asks only when a confirmation callback is available and needs_confirmation says so; otherwise it continues.
- [ ] The answer can be continue, quick, or cancel; any other answer is treated as cancel.
- [ ] The proposed plan and the decision are both recorded in the trace.

**Errors or unsupported statements:** 

**Answer without retrieval:**

In the KíLín Python package, the `needs_confirmation` and `confirm_plan` parameters work together to determine whether a tool requires user input before execution. Here's how they function:

1. **`needs_confirmation` (per-tool):**  
   This boolean flag specifies whether a tool inherently requires user confirmation before running. If set to `True`, the tool will prompt the user for input (e.g., "Proceed with this action?"). If `False`, the tool executes without user intervention.

2. **`confirm_plan` (global setting):**  
   This parameter controls whether the agent enforces confirmation for all steps in the plan. If `confirm_plan` is `True`, the agent will **always** ask for confirmation for every tool in the plan, regardless of the individual tool's `needs_confirmation` value. If `confirm_plan` is `False`, the agent will only ask for confirmation if the tool's `needs_confirmation` is explicitly `True`.

### Behavior Summary:
- **If `confirm_plan` is `True`:**  
  All tools in the plan will prompt the user for confirmation, even if their `needs_confirmation` is `False`.  
  - Example: A tool with `needs_confirmation=False` will still ask the user if `confirm_plan=True`.

- **If `confirm_plan` is `False`:**  
  The agent will only ask for confirmation if the tool's `needs_confirmation` is `True`.  
  - Example: A tool with `needs_confirmation=False` will execute without user input.

### User's Answer:
When a tool requires confirmation, the agent prompts the user (e.g., via a CLI or API). The user's response (e.g., "yes" or "no") determines whether the tool proceeds. If the user confirms, the tool runs; otherwise, it is skipped or the plan halts, depending on the agent's configuration. 

This mechanism allows fine-grained control over when user input is required, balancing safety and automation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## ask_choice_callers (dependency, held_out_1)

**Question.** Explain which functions call ask_choice, and what each of them asks the user to choose.

**Intent:** code_explanation

**Answer with retrieval:**

The `ask_choice` function is called by two functions: `offer_export` and `choose_document`.  

- **`offer_export`** asks the user to choose between saving a long list as a Markdown file or viewing a subset of it. The options are:  
  - "Save all [count] to outputs/ as Markdown"  
  - "Show the first [CHAT_LIST_LIMIT] here" [4].  

- **`choose_document`** asks the user to select a specific document (e.g., "Which paper should I turn into a research briefing?") from a list of options, though the exact options depend on the context [6].  

These calls use `ask_choice` to present users with choices, tracing interactions and handling user input according to the `ToolContext` configuration. No other functions explicitly call `ask_choice` in the provided evidence.

**Cited passages:**

- [4] kilin/tools/_shared/long_lists.py `offer_export` (retrieved by: code)
- [6] kilin/tools/_shared/choose_document.py `choose_document` (retrieved by: context_expansion)

**Gold functions:** `offer_export`, `choose_document`

- in evidence: ['offer_export', 'choose_document']
- cited: ['offer_export', 'choose_document']

**Facts (tick those the answer states correctly):**

- [ ] offer_export asks whether to save a long list as a Markdown file in outputs/ or to show only the first items in the chat.
- [ ] choose_document asks the user which of several candidate files the question refers to.
- [ ] If the user cancels the document choice, the tool stops and says which files matched.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package uses the `ask_choice` function to prompt users for selections during interactive data analysis tasks. While the exact implementation details may vary, here is a plausible explanation of functions that call `ask_choice` and the choices they present:

1. **`load_data()`**  
   - **Asks the user to choose a dataset format** (e.g., "CSV", "Excel", "JSON") or a specific file path.  
   - Example: "Select the file type to load: [CSV, Excel, JSON]".

2. **`analyze_data()`**  
   - **Asks the user to choose an analysis method** (e.g., "Descriptive Statistics", "Regression", "Clustering").  
   - Example: "Choose the analysis type: [Descriptive Statistics, Regression, Clustering]".

3. **`visualize_data()`**  
   - **Asks the user to select a visualization type** (e.g., "Bar Chart", "Line Chart", "Pie Chart").  
   - Example: "Select the visualization: [Bar Chart, Line Chart, Pie Chart]".

4. **`filter_data()`**  
   - **Asks the user to choose a filtering criterion** (e.g., "Date Range", "Value Range", "Category").  
   - Example: "Select the filter type: [Date Range, Value Range, Category]".

5. **`select_columns()`**  
   - **Asks the user to choose columns for analysis** from a list of available columns.  
   - Example: "Select the columns to include: [Column1, Column2, Column3]".

These functions use `ask_choice` to provide interactive prompts, allowing users to make selections that guide the data analysis workflow. The exact options and functions depend on the package's design, but the pattern of using `ask_choice` for user input is consistent with interactive tools like KíLín. Always refer to the package's documentation for precise details.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## readable_trace (mechanism, held_out_1)

**Question.** Explain when TraceLogger rewrites the human-readable trace file, and how it avoids leaving a partially written file.

**Intent:** code_explanation

**Answer with retrieval:**

TraceLogger rewrites the human-readable trace file when events are appended, specifically on progress updates (at most once per configured interval) or immediately for other events [8]. It avoids leaving a partially written file by using a temporary JSON file during the rewrite process. While the exclusive companion-file lock is held, the temporary file replaces the target file, ensuring the original file is only replaced after the write completes [4]. This lock prevents concurrent writes from other processes, and the temporary file guarantees the target file is fully written before replacement, avoiding partial writes [5]. The rewrite merges current events with a deep copy of the baseline, preserving prior data and preventing nested mappings from being altered [4].

**Cited passages:**

- [4] kilin/trace_logging.py `TraceLogger._write_readable_trace` (retrieved by: semantic)
- [5] kilin/trace_logging.py `trace_logging` (retrieved by: semantic)
- [8] kilin/trace_logging.py `TraceLogger._write_event` (retrieved by: context_expansion)

**Gold functions:** `TraceLogger._write_readable_trace`, `TraceLogger._write_event`, `TraceLogger.update_conversation`

- in evidence: ['TraceLogger._write_readable_trace', 'TraceLogger._write_event']
- cited: ['TraceLogger._write_readable_trace', 'TraceLogger._write_event']

**Facts (tick those the answer states correctly):**

- [ ] The file is rewritten after each recorded event; progress updates rewrite it at most every half second.
- [ ] It is also rewritten when the conversation data is updated.
- [ ] A new temporary file replaces the target in one step, while the exclusive lock on the companion file is held.
- [ ] Repeated writes replace this run's part of the file without duplicating it or overwriting earlier runs.
- [ ] If writing fails, the event stays in memory and the failure is handled rather than raised.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The **TraceLogger** in the **KíLín Python package** rewrites the human-readable trace file **when the logger is finalized or when the buffer reaches a threshold** (e.g., size or time-based rotation). To avoid leaving a partially written file, it employs a **temporary file strategy**:

1. **Rewriting Process**:  
   - The TraceLogger writes trace data to a **temporary file** (e.g., `trace_temp_12345.tmp`) instead of directly to the final destination.  
   - When the logger needs to finalize the trace (e.g., on shutdown, rotation, or buffer flush), it **moves the temporary file** to the final location (e.g., `trace.log`) using an **atomic rename operation**. This ensures the final file is only updated once all data is fully written.

2. **Avoiding Partial Writes**:  
   - The temporary file is **not accessible** until the rename operation completes, preventing partial writes.  
   - The rename operation is **atomic** on most filesystems, ensuring the final file is either fully written or remains in its previous state, avoiding corruption.  
   - Additionally, the TraceLogger may use **buffering** and **flushing** mechanisms to ensure all data is committed to the temporary file before the rename.

This approach guarantees the human-readable trace file is always complete and consistent, even if the process is interrupted during logging.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## workspace_sync (mechanism, held_out_1)

**Question.** Explain how synchronize_project_workspace keeps its copy of a source folder up to date with the original folder.

**Intent:** code_explanation

**Answer with retrieval:**

The `synchronize_project_workspace` function keeps its copy of a source folder up to date by comparing the current state of the source directory with a saved snapshot stored in the project workspace. It first validates the source and project directories, ensuring the source is outside the workspace [2]. It then reads a saved manifest (from `source_snapshot.json`) and scans the source directory to generate a current manifest [6]. 

If the manifests differ, the function identifies changes (added, removed, or modified files) and updates the workspace snapshot accordingly. It removes files no longer present in the source and copies new or modified files to the workspace, using safe copying routines to avoid overwriting existing snapshots unless explicitly allowed [19]. The process also ensures the snapshot matches the source's manifest, preserving file integrity [6]. By reconciling the current source state with the saved snapshot, the function maintains the workspace copy in sync with the original folder. This approach ensures the workspace reflects the latest state of the source while handling potential conflicts or errors during synchronization.

**Cited passages:**

- [2] kilin/project.py `synchronize_project_workspace` (retrieved by: code)
- [6] kilin/project.py `_synchronize_snapshot` (retrieved by: context_expansion)
- [19] kilin/workspace.py `_copy_file` (retrieved by: context_expansion)

**Gold functions:** `synchronize_project_workspace`, `_synchronize_snapshot`, `_snapshot_matches_manifest`

- in evidence: ['synchronize_project_workspace', '_synchronize_snapshot']
- cited: ['synchronize_project_workspace', '_synchronize_snapshot']

**Facts (tick those the answer states correctly):**

- [ ] It opens the existing workspace and refuses a source inside the workspace or a source that differs from the one recorded in source_snapshot.json; it never creates or changes the project profile.
- [ ] A manifest with SHA-256 hashes of the selected source files is built and compared with the saved manifest (source_snapshot.json) to find added, modified, and removed files.
- [ ] Before using that comparison, the copied files are rehashed and checked against the saved manifest; if the copy was changed, it is rebuilt from the original.
- [ ] Otherwise only added and modified files are copied and removed files deleted, then empty folders are removed.
- [ ] The new manifest, with the source folder's path, is saved for the next synchronization. The original folder is never modified.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `synchronize_project_workspace` function in the KíLín Python package ensures that its copy of a source folder remains up to date with the original by employing a combination of **file system monitoring** and **version control integration**. Here's how it works:

1. **File System Monitoring**:  
   KíLín likely uses tools like `inotify` (on Linux/macOS) or `ReadDirectoryChangesW` (on Windows) to monitor the source folder for changes (e.g., file additions, modifications, or deletions). When changes are detected, the function updates the corresponding files in the workspace copy to reflect these changes.

2. **Version Control Integration**:  
   If the source folder is under version control (e.g., Git), KíLín may synchronize the workspace by pulling the latest changes from the repository. This ensures the workspace matches the source folder's state, even if the source is managed via a version control system.

3. **Periodic Checks**:  
   In some cases, the function might periodically scan the source folder for discrepancies (e.g., comparing timestamps or checksums) and update the workspace accordingly, ensuring consistency without constant monitoring.

4. **Build/Task Runner Integration**:  
   KíLín might integrate with build systems (e.g., `Make`, `pip`, or `poetry`) or task runners (e.g., `invoke`, `nox`) to re-synchronize the workspace when dependencies or project structure changes, ensuring the workspace reflects the latest state of the source.

This approach ensures the workspace remains aligned with the source folder, either through real-time updates, version control, or periodic reconciliation. For precise details, refer to KíLín's documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## cached_model_callers (dependency, held_out_2)

**Question.** Explain which functions call cached_model_snapshot, and when each of them uses it instead of downloading the model.

**Intent:** code_explanation

**Answer with retrieval:**

The function `cached_model_snapshot` is called by two main functions:  
1. `ChatSession._load` (source [6]) uses it when `allow_download` is `False` and the model source is not a local directory. It resolves the cached snapshot without downloading.  
2. `_model_source` (source [8]) uses it when `allow_download` is `False`, ensuring the model is loaded from the cached snapshot instead of downloading.  

These functions prioritize using the cached model to avoid network requests, but only if the cached snapshot is complete (e.g., contains `config.json`). The test function `_cached_snapshot` (source [5]) also calls `cached_model_snapshot` during validation, but this is part of test logic rather than core functionality.  

No other functions explicitly call `cached_model_snapshot` based on the evidence.

**Cited passages:**

- [5] tests/kilin/test_chat__models.py `TestDefaultPretrainedChatModels._cached_snapshot` (retrieved by: code, semantic)
- [6] kilin/chat.py `ChatSession._load` (retrieved by: context_expansion)
- [8] kilin/embeddings.py `_model_source` (retrieved by: context_expansion)

**Gold functions:** `ChatSession._load`, `_model_source`

- in evidence: ['ChatSession._load', '_model_source']
- cited: ['ChatSession._load', '_model_source']

**Facts (tick those the answer states correctly):**

- [ ] ChatSession._load (chat model) and _model_source (embedding model) call it.
- [ ] Both use a local directory directly when the model ID is a folder that exists.
- [ ] Both call cached_model_snapshot only when downloads are not allowed and the model ID is not a local folder.
- [ ] When downloads are allowed, _model_source calls snapshot_download with local_files_only=False, and ChatSession._load passes the model ID to the backend loader, which may fetch missing files.
- [ ] cached_model_snapshot never downloads (local_files_only=True) and raises FileNotFoundError if the cached snapshot has no config.json.

**Errors or unsupported statements:** 

**Answer without retrieval:**

In the KíLín Python package, the `cached_model_snapshot` function is typically used by functions responsible for loading or retrieving models. Here's how it works:

1. **`load_model` or `get_model` Functions**:  
   These are the primary functions that call `cached_model_snapshot`. When you invoke them with a model identifier (e.g., a model name or path), the function first checks if the model is already cached locally.  
   - **If cached**: It uses `cached_model_snapshot` to retrieve the pre-downloaded model, avoiding redundant downloads.  
   - **If not cached**: It downloads the model from a remote source (e.g., a model repository), saves it as a snapshot, and then uses `cached_model_snapshot` to reference this cached version for subsequent calls.

2. **When `cached_model_snapshot` is Used Instead of Downloading**:  
   - **On Subsequent Calls**: After the first download, `cached_model_snapshot` is used to access the model directly from the local cache, bypassing the download step.  
   - **When Model Path is Local**: If the model is already stored locally (e.g., via a previous download or manual placement), `cached_model_snapshot` retrieves it without re-downloading.

This mechanism ensures efficient model management by minimizing redundant downloads and leveraging local storage for faster access. The exact implementation details may vary, but the core behavior aligns with standard caching patterns in machine learning libraries.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## file_selection (location, held_out_2)

**Question.** Where does KíLín decide which files of a source folder are indexed, and which files does it always leave out?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín determines which files are indexed by cross-referencing the **source manifest** (which lists all selected files and their sizes) and the **index** (which identifies files with searchable text) [1]. Files not in the index, such as scanned PDFs, are marked as unsearchable and excluded from indexing [3]. The selection process filters files based on **file types** (e.g., PDFs, markdown) and **top-level folders** specified in the query [5]. Files left out are those in the source manifest but not indexed, or excluded by path rules in `scan_repository` [13]. The decision is implemented in `project_inventory` [1], with unsearchable files explicitly listed [3].

**Cited passages:**

- [1] kilin/tools/project_inventory/tool.py `tool` (retrieved by: lexical)
- [3] kilin/rag.py `RAGSession._unsearchable_sources` (retrieved by: semantic)
- [5] kilin/tools/project_inventory/tool.py `run` (retrieved by: code)
- [13] kilin/repository.py `scan_repository` (retrieved by: context_expansion)

**Gold functions:** `scan_repository`, `_skip_reason`

- in evidence: ['scan_repository']
- cited: ['scan_repository']

**Facts (tick those the answer states correctly):**

- [ ] scan_repository walks the folder and asks _skip_reason about every entry, recording a reason code for each skipped one.
- [ ] Links are never followed.
- [ ] Environment, cache, and generated folders (such as .git, .venv, __pycache__, node_modules, build, logs, weights) are skipped.
- [ ] Potentially secret files (.env and .env.*, credentials.json, secrets.json, SSH keys) and model weights or key files by suffix (.safetensors, .pt, .gguf, .pem, .key, ...) are always skipped.
- [ ] Otherwise files are skipped when excluded by the user's selection, outside the included paths, not regular files, of an unselected file type, empty, or larger than the size limit.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín uses a configuration file (typically `kilin.yaml` or `kilin.json`) to determine which files in a source folder are indexed and which are excluded. The configuration allows you to specify:

1. **Source directories**: The folders to scan for files.
2. **Inclusion/exclusion patterns**: You can define patterns (e.g., using glob syntax) to include or exclude specific files or directories. For example, you might include `.py` files or exclude `.log` files.
3. **Default exclusions**: By default, KíLín may exclude common non-source files (e.g., hidden files like `.git`, directories like `node_modules`, or files with extensions like `.pyc` or `.bak`), but this behavior is customizable.

The exact rules depend on your configuration, so you can tailor the indexing behavior to your project's needs. If you don't specify a configuration, KíLín may use default settings, but it's recommended to explicitly define inclusion/exclusion rules for clarity.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## query_format (location, held_out_2)

**Question.** How is a question turned into the text that is embedded for semantic search, and does this depend on the project type?

**Intent:** code_explanation

**Answer with retrieval:**

A question is transformed into text for semantic search by being embedded using the `embedding_session.embed_query` method, with the exact formatting handled by the caller. The query text is passed directly to the embedding model, and model-specific formatting (e.g., adding instructions) is the caller's responsibility [1]. For example, the `format_qwen3_query` function structures the query with an instruction and question [9], but this formatting is not inherently tied to the project type. However, the project type may influence retrieval strategies (e.g., hybrid search) or how queries are contextualized [7]. The embedding process itself does not directly depend on the project type, as the `semantic_search` function uses a generic embedding session [3]. Thus, while the project type might affect higher-level retrieval strategies, the core text-to-embedding transformation is model-specific and not project-type dependent.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_semantic.py `semantic_search` (retrieved by: code)
- [3] kilin/retrievals/strategies/retrieval_semantic.py `semantic_search` (retrieved by: code)
- [7] kilin/project_type.py `project_type` (retrieved by: context_expansion)
- [9] kilin/query_formats.py `format_qwen3_query` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._format_query`, `format_qwen3_query`

- in evidence: ['format_qwen3_query']
- cited: ['format_qwen3_query']

**Facts (tick those the answer states correctly):**

- [ ] With the plain query format in the project profile, the question is embedded unchanged.
- [ ] Otherwise the text has two lines, 'Instruct: <instruction>' and 'Query: <question>', the format used by Qwen3 embedding models.
- [ ] The instruction is the project type's retrieval instruction, so it depends on the project type.
- [ ] The instruction must be a single nonempty line; surrounding whitespace is removed from both parts.
- [ ] Documents are embedded without this instruction.

**Errors or unsupported statements:** 

**Answer without retrieval:**

In the KíLín Python package, a question is converted into text for semantic search by generating **embeddings** using a pre-trained transformer model (e.g., BERT, RoBERTa, or similar). This process involves the following steps:

1. **Tokenization**: The question is tokenized into subwords using a tokenizer associated with the model.
2. **Embedding Generation**: The tokenized input is passed through the model to produce a contextualized embedding vector that represents the semantic meaning of the question.
3. **Normalization/Scaling**: The embedding may be normalized or scaled to ensure consistency with document embeddings.

This process **does depend on the project type** in the following ways:
- **Model Choice**: Different project types (e.g., document retrieval, chatbots, or classification) may use different pre-trained models or fine-tuned versions tailored to specific tasks.
- **Pipeline Configuration**: KíLín allows customization of the embedding pipeline (e.g., using `kilt.models` for different tasks) to align with the project's requirements (e.g., multilingual support, domain-specific knowledge, or efficiency).

For example, a document retrieval project might use a model optimized for similarity matching, while a chatbot might use a model fine-tuned for dialogue understanding. The core embedding generation step remains consistent, but the model and configuration adapt to the project's goals. 

If you're using KíLín, refer to its documentation for specific model and pipeline configurations relevant to your use case.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## conversation_delete (mechanism, held_out_2)

**Question.** Explain what happens to a saved conversation when ChatEngine.delete_conversation is called. Is the conversation file really deleted?

**Intent:** code_explanation

**Answer with retrieval:**

When `ChatEngine.delete_conversation` is called, the saved conversation is **archived** rather than immediately deleted. The conversation file is moved to a `logs/traces/deleted` directory, where it is renamed with a suffix (e.g., `_2`, `_3`) if a file with the same name already exists there [3]. This ensures the file remains recoverable. The conversation is removed from the active catalog, and if it was the active conversation, the most recently updated conversation becomes active [2]. The file is not permanently deleted but stored in a "deleted" folder, making it non-current but potentially recoverable. The process also detaches the conversation's trace to avoid rewriting the JSON file [4]. Thus, the conversation file is **not truly deleted** but moved to an archived state.

**Cited passages:**

- [2] kilin/gui/engine.py `ChatEngine.delete_conversation` (retrieved by: code, semantic)
- [3] kilin/conversations.py `ConversationStore.archive` (retrieved by: semantic)
- [4] kilin/gui/engine.py `ChatEngine._detach_project_trace` (retrieved by: code)

**Gold functions:** `ChatEngine.delete_conversation`, `ConversationStore.archive`

- in evidence: ['ChatEngine.delete_conversation', 'ConversationStore.archive']
- cited: ['ChatEngine.delete_conversation', 'ConversationStore.archive']

**Facts (tick those the answer states correctly):**

- [ ] The file is not deleted: ConversationStore.archive moves it into a deleted folder under logs/traces, so it can be recovered.
- [ ] If a file with the same name is already there, the moved file gets a _2, _3, ... suffix.
- [ ] The conversation's write lock is held while the file is moved, and the conversation is removed from the store's list.
- [ ] If it was the last conversation, a new empty one is created; if it was the active one, the most recently updated remaining conversation becomes active.
- [ ] The engine detaches the trace when the active conversation is deleted, switches to the replacement, and records a conversation.deleted event; on errors the list is refreshed and the error reported.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The behavior of `ChatEngine.delete_conversation` in the KíLín Python package depends on how the underlying storage system is implemented. Here's a general explanation:

1. **Intended Purpose**: The method is designed to remove a conversation from the active state, typically by marking it as deleted or removing it from the active conversation list.

2. **File Deletion**: If the conversation is stored in a file (e.g., a JSON or pickle file), the method may delete the file from the filesystem. However, this is not guaranteed unless explicitly stated in the documentation.

3. **Soft Delete vs. Hard Delete**: Some systems use a "soft delete" approach, where the conversation is marked as deleted in the database or storage (e.g., a flag in a file or database row) but the physical file remains. Others perform a "hard delete," permanently removing the file.

4. **KíLín-Specific Behavior**: Without explicit documentation or source code reference, the exact behavior of `delete_conversation` in KíLín is unclear. You should consult the [KíLín documentation](https://kiln.readthedocs.io/) or its source code for confirmation.

**Recommendation**: If you need to ensure data is permanently removed, verify the implementation details or test the method in your environment. Always back up important data before performing deletions.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## torch_embedding (mechanism, held_out_2)

**Question.** Explain how EmbeddingSession computes vectors with the PyTorch backend, which token's hidden state it uses, and how the vectors are checked before they are returned.

**Intent:** code_explanation

**Answer with retrieval:**

The `EmbeddingSession` computes vectors with the PyTorch backend by using the model's `last_hidden_state` output, which represents the final hidden states of the tokens after processing [7]. It identifies the relevant token by applying the `attention_mask` to determine the last unmasked token in each sequence. Specifically, it calculates the maximum position where the mask is active, effectively selecting the last token's hidden state [7]. 

Before returning, vectors are checked using `_check_vectors` [10], which verifies:  
1. The count matches the expected number of inputs.  
2. All vectors have consistent dimensions.  
3. All values are finite (not NaN or infinity).  
4. Each vector has at least one non-zero element (non-zero norm).  

These checks ensure validity and consistency of the output vectors. The PyTorch implementation also applies L2 normalization to the selected token's hidden state before validation [7].

**Cited passages:**

- [7] kilin/embeddings.py `EmbeddingSession._embed_torch` (retrieved by: context_expansion)
- [10] kilin/embeddings.py `_check_vectors` (retrieved by: context_expansion)

**Gold functions:** `EmbeddingSession._embed_torch`, `_check_vectors`, `EmbeddingSession.embed_documents`

- in evidence: ['EmbeddingSession._embed_torch', '_check_vectors']
- cited: ['EmbeddingSession._embed_torch', '_check_vectors']

**Facts (tick those the answer states correctly):**

- [ ] Texts are tokenized together with padding and without truncation, and the model runs in inference mode.
- [ ] The vector is the hidden state of each text's last non-padding token, found from the attention mask (last-token pooling).
- [ ] Vectors are L2-normalized.
- [ ] _check_vectors requires one vector per text, the same nonzero dimension for all, finite numeric values, and no all-zero vector; otherwise it raises RuntimeError.
- [ ] embed_documents rejects empty or blank texts and a closed session, and does not truncate inputs.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `EmbeddingSession` in the KíLín Python package, when using the PyTorch backend, computes vectors by leveraging a pre-trained language model (e.g., BERT, RoBERTa, etc.) and its associated tokenization. Here's a breakdown of the process:

1. **Vector Computation with PyTorch Backend**:
   - **Tokenization**: Input text is tokenized into subword units (e.g., using BPE or WordPiece). These tokens are converted into integer IDs.
   - **Model Forward Pass**: The tokenized input is passed through the PyTorch model. The model processes the tokens through its layers, generating hidden states for each token at each layer.
   - **Hidden State Extraction**: The `EmbeddingSession` typically uses the **[CLS] token's hidden state** (a special token added at the beginning of sequences) as the final embedding vector. This is common in models like BERT, where the [CLS] token's representation is used for classification tasks. Alternatively, it might use the **last token's hidden state** (e.g., for sentence-level embeddings) or the **average of all token hidden states** (for document-level embeddings), depending on the model's configuration.

2. **Vector Validation**:
   - **Normalization**: The computed vectors are often normalized to unit length (L2 normalization) to ensure consistency.
   - **NaN/Inf Checks**: The system checks for invalid values (e.g., `NaN` or `inf`) in the vectors and may raise errors or log warnings if detected.
   - **Shape Verification**: The vectors are validated to ensure they match the expected dimensionality (e.g., matching the model's embedding size).

If the exact implementation details of KíLín are not publicly documented, these steps are based on common practices in NLP libraries using PyTorch. For precise behavior, refer to KíLín's documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
