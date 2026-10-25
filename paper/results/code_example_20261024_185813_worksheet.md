# Code example: grading worksheet

Indexed revision: 8bd20824c (HEAD)

## notebook_extraction (location, held_out_3)

**Question.** Where does KíLín turn a Jupyter notebook into searchable text, and which parts of a notebook are left out?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín converts a Jupyter notebook into searchable text via the `_notebook_units` function in `kilin/sources/extensions/sources_ipynb.py` [2]. This function extracts **code** and **Markdown cells**, omitting stored outputs (e.g., execution results) and metadata (e.g., cell tags, execution counts) [4]. It processes each cell’s `source` content, joining multiline entries into a single string and stripping empty lines. For Markdown cells, it identifies section headers to associate text with logical sections. Cells of other types (e.g., raw, video) are skipped. The resulting `SourceUnit` objects represent the searchable text, excluding non-code/non-Markdown content and execution artifacts. The environment setup in `setup/notebook.py` [1] ensures the notebook uses the correct Python interpreter but does not directly handle content conversion.

**Cited passages:**

- [1] setup/notebook.py `notebook` (retrieved by: code)
- [2] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)
- [4] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)

**Gold functions:** `_notebook_units`, `extract_source_units`

- in evidence: ['_notebook_units']
- cited: ['_notebook_units']

**Facts (tick those the answer states correctly):**

- [ ] The notebook extension (sources_ipynb.py) reads the notebook JSON; extract_source_units returns the units built by _notebook_units.
- [ ] Only code and Markdown cells become source units; outputs, execution state, attachments, metadata, and raw cells are not indexed.
- [ ] Empty cells are skipped.
- [ ] Each kept cell is one unit with its one-based position in the original cell list (skipped cells included) and line numbers local to the cell; a Markdown cell's first heading becomes its section.
- [ ] A malformed notebook raises ValueError.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín converts a Jupyter notebook into searchable text by extracting **text content from markdown cells** and **code cells** (e.g., Python scripts). It processes these elements into a plain-text format, making them searchable. However, the following parts of the notebook are typically **left out** during this conversion:

1. **Output cells**: Results of code execution (e.g., printed outputs, plots, or error messages) are excluded, as they are dynamic and environment-dependent.
2. **Images and binary data**: Embedded images, figures, or other non-text content are not converted into searchable text.
3. **Metadata**: Fields like notebook creation time, author, or other metadata are omitted unless explicitly included in the processing configuration.
4. **Attachments**: Files attached to the notebook (e.g., external data files) are not included.

This approach ensures the resulting text is lightweight and focused on the static content of the notebook. For precise details, refer to the [KíLín documentation](https://kilin.readthedocs.io/).

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## atomic_files (mechanism, held_out_3)

**Question.** Explain how atomic_write_text and atomic_create_text write a file so that it is never left half-written, and how the two differ.

**Intent:** code_explanation

**Answer with retrieval:**

Both `atomic_write_text` and `atomic_create_text` use temporary files to ensure files are never left half-written by performing atomic operations that complete the write or fail cleanly. `atomic_write_text` creates a uniquely named temporary file in the destination directory, writes the content, and then atomically replaces the target file with the temporary one using `temporary.replace(path)`. If the write fails, the temporary file is deleted, preserving the prior destination [6]. `atomic_create_text` similarly uses a temporary file, writes content, and creates a hard link to the destination path (`os.link(temporary, path)`), ensuring the destination is fully written before the link is made. If the destination already exists, it raises an error, preventing overwrite [2]. 

The key difference is that `atomic_write_text` replaces an existing file, while `atomic_create_text` fails if the destination exists, ensuring no overwrite. Additionally, `atomic_create_text` uses a hard link for atomic publication, whereas `atomic_write_text` relies on a replace operation. Both avoid partial writes by completing the operation (write + link/replace) before modifying the destination, but `atomic_create_text` enforces exclusivity for new entries [1].

**Cited passages:**

- [1] kilin/atomic_writes.py `atomic_writes` (retrieved by: lexical, semantic)
- [2] kilin/atomic_writes.py `atomic_create_text` (retrieved by: code)
- [6] kilin/atomic_writes.py `atomic_write_text` (retrieved by: context_expansion)

**Gold functions:** `atomic_write_text`, `atomic_create_text`

- in evidence: ['atomic_write_text', 'atomic_create_text']
- cited: ['atomic_write_text', 'atomic_create_text']

**Facts (tick those the answer states correctly):**

- [ ] Both write the full UTF-8 text to a new uniquely named temporary file in the same folder (.<name>.*.tmp) and only then publish it.
- [ ] atomic_write_text replaces the destination with the temporary file (Path.replace), overwriting an existing file.
- [ ] atomic_create_text publishes with a hard link (os.link), so it fails with FileExistsError if the destination already exists and never overwrites; unsupported hard links fail instead of falling back.
- [ ] The temporary file is removed in all cases, including failures.
- [ ] Neither takes a lock nor forces the data to disk (no fsync).

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `atomic_write_text` and `atomic_create_text` functions in the KíLín Python package both ensure that a file is written atomically, preventing partial writes by using a temporary file and a rename operation. Here's how they work and differ:

### **Atomic Writing Process**
1. **Temporary File Creation**: Both functions create a temporary file in the same directory as the target file.
2. **Content Writing**: They write the provided content to the temporary file.
3. **Atomic Rename**: The temporary file is renamed to the target file path. This rename operation is atomic on most filesystems, meaning the original file remains unchanged until the rename is complete. If the process is interrupted, the original file is not corrupted, and the new content is only visible if the rename succeeds.

### **Key Differences**
- **`atomic_write_text`**:
  - **Overwrites the target file**: If the target file already exists, it is replaced by the new content.
  - **Behavior**: Ensures the target file is updated with the new content, even if it already exists.

- **`atomic_create_text`**:
  - **Creates a new file**: If the target file exists, the function may fail or behave differently (e.g., not overwrite). The exact behavior depends on the implementation, but it typically avoids overwriting existing files.
  - **Purpose**: Designed to create a new file without modifying existing ones, ensuring the original file remains untouched.

### **Summary**
Both functions prevent partial writes by using a temporary file and atomic renaming. The critical difference lies in how they handle the target file: `atomic_write_text` overwrites it, while `atomic_create_text` ensures a new file is created without overwriting existing content. This distinction is crucial for scenarios where data integrity or file existence checks are required.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## claim_passages_callers (dependency, held_out_3)

**Question.** Explain which functions call claim_passages, and what each of them uses the selected passages for.

**Intent:** code_explanation

**Answer with retrieval:**

The function `claim_passages` is called by `_verify` in `kilin/tools/reference_check/tool.py` [4] and by `run` in `kilin/tools/suggest_reference/tool.py` [9].  

In `_verify`, `claim_passages` is used to retrieve the most relevant passages from a cited file that relate to a claim. These passages are then passed to `verify_claim` to determine if the cited work supports, contradicts, or is unrelated to the claim [4]. The selected passages provide evidence for assessing the claim's validity.  

In `run`, `claim_passages` identifies passages from a paper that are relevant to a given statement. These passages are used by `judge_support` to evaluate whether the paper supports the statement, aiding in suggesting appropriate references [9]. The passages serve as evidence to determine the strength of the paper's support for the statement.  

Both uses rely on `claim_passages` to filter and prioritize passages based on their relevance, ensuring that only the most pertinent evidence is considered for verification or reference suggestions.

**Cited passages:**

- [4] kilin/tools/reference_check/tool.py `_verify` (retrieved by: code)
- [9] kilin/tools/suggest_reference/tool.py `run` (retrieved by: context_expansion)

**Gold functions:** `_verify`, `run`

- in evidence: ['_verify', 'run']
- cited: ['_verify', 'run']

**Facts (tick those the answer states correctly):**

- [ ] The reference_check tool's _verify calls it for each citing claim, with the citing sentence (citation marks removed) and what the reference is cited for, over the cited file's passages; verify_claim then has the model judge whether they support the claim.
- [ ] The suggest_reference tool's run calls it for each candidate paper with the user's statement; judge_support then has the model judge whether that paper supports the statement.
- [ ] claim_passages fuses the rankings of its texts by reciprocal rank, keeps at most count passages within a token budget (the best one is always kept), and returns them in reading order.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is designed for information retrieval and natural language processing tasks, including claim verification. While the exact implementation details may vary, here's a general explanation of how functions might interact with `claim_passages`:

1. **`verify_claim(claim, passages)`**  
   - **Purpose**: Verifies whether a given claim is supported by the provided passages.  
   - **Use of `claim_passages`**: This function likely uses the output of `claim_passages` (a list of relevant passages) to assess the claim's validity by checking if the passages contain evidence that supports or contradicts the claim.

2. **`evaluate_claim(claim, documents)`**  
   - **Purpose**: Evaluates the truthfulness of a claim against a set of documents.  
   - **Use of `claim_passages`**: This function might first call `claim_passages` to retrieve relevant passages from the documents, then use those passages to determine the claim's accuracy.

3. **`retrieval_pipeline(claim)`**  
   - **Purpose**: A pipeline that combines retrieval and verification steps.  
   - **Use of `claim_passages`**: This function could internally call `claim_passages` to fetch relevant passages, which are then passed to a verification function (e.g., `verify_claim`) for further analysis.

### Key Note:
If you are using a specific version of KíLín, the exact function names and workflows may differ. For precise details, consult the [KíLín documentation](https://kilin.readthedocs.io/) or its source code. If you have access to the package, inspecting its source or API reference would clarify the exact interactions.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## model_identity (location, held_out_3)

**Question.** How does KíLín identify the embedding model that built a saved index, so that it can tell when the model has changed?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín identifies the embedding model for a saved index by associating it with the selected computational backend (e.g., "mlx" or "torch") during project configuration [1]. The embedding model is stored separately for each backend in the `ProjectProfile` (e.g., `embedding_mlx` or `embedding_torch`) [1]. When loading an index, KíLín first resolves the backend from the project configuration, then selects the corresponding embedding model [3]. This linkage ensures that the model used to build the index matches the backend used during indexing, preventing mismatches between models and backends [4]. If the backend or model changes in the project configuration, KíLín detects this discrepancy by comparing the stored backend with the current configuration, ensuring consistency between the index and its associated model [1]. The model is explicitly stored in the project profile, allowing KíLín to validate and track changes when the project is reopened [5].

**Cited passages:**

- [1] kilin/project.py `ProjectProfile.embedding_model` (retrieved by: code)
- [3] tutorials/100_persistent_indexes.ipynb `None` (retrieved by: semantic)
- [4] tutorials/60_embeddings.ipynb `None` (retrieved by: semantic)
- [5] tutorials/150_project_types.ipynb `None` (retrieved by: lexical)

**Gold functions:** `embedding_model_identity`, `_hub_snapshot_commit`, `_directory_digest`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] A standard Hugging Face cache snapshot is identified by repository and commit (huggingface:<model>@<commit>).
- [ ] Any other model folder is identified by a SHA-256 digest over its files' relative names and content hashes (local-sha256:...).
- [ ] Per-file hashes are cached in a manifest outside the model folder and reused when file size, times, device, and inode are unchanged; force_verify rehashes every file.
- [ ] The identity is stored with the index; a different identity means the saved vectors are not reused.
- [ ] It is meant for index compatibility, not authentication.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín identifies the embedding model used to build a saved index by storing metadata about the model alongside the index. This metadata typically includes details such as the model's name, version, or a checksum of its parameters. When the index is loaded, KíLín compares this stored metadata with the current model's configuration to detect changes. If discrepancies are found (e.g., a different model version or parameters), it alerts the user that the model has changed, ensuring compatibility and preventing mismatches between the index and the embedding model. This mechanism relies on explicit tracking of the model's state during index creation and verification during loading.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## conversation_branch (mechanism, held_out_3)

**Question.** Explain how ConversationStore creates a branch of a conversation: what is copied from the parent, how the branch is named, and how it avoids overwriting an existing file.

**Intent:** code_explanation

**Answer with retrieval:**

ConversationStore creates a branch by copying non-metadata fields from the parent's resume state, such as turns and operational data, while recording the parent's identifier and checkpoint reference in the new branch's metadata [5]. The branch is named by appending " branch" and a sequential number to the parent's display name, ensuring uniqueness by checking against existing names in the registry [6]. To avoid overwriting existing files, the new branch's filename is selected through a process that locks and checks each candidate path before creation, raising an error if a collision is detected after selection [5]. This approach uses atomic writes and cross-process locks to prevent concurrent modifications, ensuring file names remain unique without manual intervention [2].

**Cited passages:**

- [2] kilin/conversations.py `ConversationStore.create` (retrieved by: lexical, semantic)
- [5] kilin/conversations.py `ConversationStore.create` (retrieved by: semantic)
- [6] kilin/conversations.py `ConversationStore._branch_name` (retrieved by: context_expansion)

**Gold functions:** `ConversationStore.create`, `ConversationStore._branch_name`

- in evidence: ['ConversationStore.create', 'ConversationStore._branch_name']
- cited: ['ConversationStore.create', 'ConversationStore._branch_name']

**Facts (tick those the answer states correctly):**

- [ ] With branch_from, the parent's saved resumable state is copied, except its metadata fields, while holding the parent's write lock; old trace operations are not copied.
- [ ] The branch records its parent conversation and the parent's checkpoint identifier.
- [ ] It is named '<parent name> branch N' with the first N not already used; a new independent conversation is named 'Conversation N'.
- [ ] A new file path is chosen, locked, and checked; if it already exists, FileExistsError is raised instead of overwriting.
- [ ] The new conversation is registered and becomes the active one.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `ConversationStore` in the KíLín Python package manages conversations and their branches by creating copies of the parent conversation's state while ensuring uniqueness and avoiding conflicts. Here's how it works:

1. **What is copied from the parent?**  
   When a branch is created, the `ConversationStore` copies the entire state of the parent conversation, including all messages, metadata (e.g., timestamps, user IDs), and any associated data structures. This ensures the branch starts with a complete snapshot of the parent's state.

2. **How is the branch named?**  
   The branch is typically named using a unique identifier, such as a UUID or a timestamp, appended to a base name (e.g., `branch/12345678-90ab-cdef-ghij-klmnopqrst`). This ensures the name is unique and avoids collisions. Alternatively, it might use a versioning scheme (e.g., `v1`, `v2`) if the parent conversation has a versioned structure.

3. **How does it avoid overwriting existing files?**  
   Before creating a new branch, the `ConversationStore` checks if a file with the proposed branch name already exists. If it does, it appends a suffix (e.g., `_1`, `_2`) or uses a hash to generate a new, unique name. This ensures the new branch's file does not overwrite existing data.

**Note:** The exact implementation details (e.g., naming conventions, file-checking logic) may vary depending on the specific version of KíLín and its internal design. For precise behavior, consult the package's documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
