# Code example: grading worksheet

Indexed revision: ec3f9d05e (HEAD)

## routing (mechanism, development)

**Question.** Explain how route_question assigns a question to an intent, and what happens when the router reply cannot be used.

**Intent:** code_explanation

**Answer with retrieval:**

The `route_question` function assigns a question to an intent by first checking if the question matches any intent's shortcut patterns [2]. If a match is found, it directly returns the intent and marks the question as standalone. If not, it constructs a prompt using the project's intents and previous conversation turns [9], then queries a router model to classify the question [6]. The router's reply is parsed as JSON [7], and if valid, the intent and related turns are determined. 

When the router reply cannot be used (e.g., malformed JSON, unknown intent, or errors during processing), the function defaults to the project's default intent [7]. This fallback ensures the question is still routed, even if the router fails. Additionally, if the intent is invalid or missing, the default intent is applied, and a fallback message explains the issue [8]. The function also checks if the question refers to previous turns, influencing intent assignment and related turn selection [3]. If no related turns are found, the question is marked as standalone [6]. This process ensures robust intent assignment while handling router failures gracefully.

**Cited passages:**

- [2] kilin/answering.py `route_question` (retrieved by: code, semantic)
- [3] kilin/answering.py `route_question` (retrieved by: code)
- [6] kilin/answering.py `route_question` (retrieved by: context_expansion)
- [7] kilin/answering.py `_parse_route` (retrieved by: context_expansion)
- [8] kilin/answering.py `_parse_route` (retrieved by: context_expansion)
- [9] kilin/answering.py `_router_prompt` (retrieved by: context_expansion)

**Gold functions:** `route_question`, `_parse_route`, `side_reply`

- in evidence: ['route_question', '_parse_route', 'side_reply']
- cited: ['route_question', '_parse_route']

**Facts (tick those the answer states correctly):**

- [ ] A message that fully matches an intent's shortcut pattern is routed without calling the model.
- [ ] Otherwise the router model is called through side_reply with a router prompt.
- [ ] The reply is parsed by _parse_route into a RouteDecision.
- [ ] If the router call fails or its reply is unusable, the project type's default intent is used.
- [ ] Related earlier turns proposed by the router are kept only if the question refers back or shares a distinctive term.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `route_question` function in the KíLín Python package is responsible for mapping a user's question to a specific intent. Here's how it works:

1. **Intent Assignment**:  
   - The function analyzes the input question and matches it to a predefined intent based on **keywords**, **patterns**, or **machine learning models** (e.g., trained classifiers).  
   - It evaluates the question against a set of intent definitions (e.g., "greet," "ask_weather," "request_help") and selects the most appropriate intent based on confidence scores or rule-based logic.

2. **Fallback Handling**:  
   - If the router (the component responsible for intent classification) cannot find a matching intent, KíLín typically uses a **fallback mechanism**. This could involve:  
     - Returning a predefined **fallback intent** (e.g., "unknown_intent") to handle ambiguous or unmatched queries.  
     - Triggering a **default response** (e.g., "I don't understand that. Can you rephrase?") to inform the user of the mismatch.  
     - Logging the unmatched question for further analysis or training.  

The exact behavior depends on the configuration of the KíLín pipeline, but the core idea is to ensure the system gracefully handles unclassified queries without crashing. For precise details, refer to the KíLín documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_reuse (mechanism, development)

**Question.** Explain how RAGSession decides whether to reuse the saved retrieval indexes or to rebuild them.

**Intent:** code_explanation

**Answer with retrieval:**

RAGSession decides to reuse or rebuild saved retrieval indexes by checking the validity of cached data against current project state. It uses a **fingerprint** to track inputs like chunk content, embedding model configuration, and chunking settings, ensuring the saved index matches these conditions [2]. If the fingerprint indicates the index is still valid (e.g., no changes to chunks or configuration), it reuses the saved index [4]. However, if the index is stale (e.g., chunks have changed or the embedding model differs), it rebuilds the index to ensure accuracy [2]. The `index_status` attribute in `RAGSession` records whether the index was reused or rebuilt [3]. Additionally, the system validates the embedding model's identity to ensure cached weights align with current settings, preventing mismatches between saved and active configurations [4]. This process ensures indexes are up-to-date while avoiding unnecessary recomputation when valid [1].

**Cited passages:**

- [1] tutorials/140_logging_trace.ipynb `None` (retrieved by: semantic)
- [2] tutorials/100_persistent_indexes.ipynb `None` (retrieved by: lexical)
- [3] kilin/rag.py `RAGSession` (retrieved by: code)
- [4] docs/retrieval_pipeline.md `None` (retrieved by: semantic)

**Gold functions:** `RAGSession._prepare_retrieval`, `load_indexes`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] The saved index is reused only if the index file, its metadata and the source manifest exist, the configuration is unchanged, and the sources match.
- [ ] The embedding-model identity of the saved index must match the current model.
- [ ] If the sources changed but the configuration did not, unchanged files keep their saved chunks and vectors and only new or changed files are embedded.
- [ ] Any problem with the saved index leads to embedding everything again.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `RAGSession` in the KíLín Python package determines whether to reuse saved retrieval indexes or rebuild them based on the following logic:

1. **Index Existence Check**: When initializing or using a `RAGSession`, it first checks if pre-existing retrieval indexes (e.g., vector stores or inverted indices) are available in the specified storage location (e.g., a directory or database).

2. **Timestamp/Version Comparison**: If indexes exist, the system compares their timestamps or version metadata with the current state of the data or model. If the indexes are up-to-date (e.g., the data hasn't changed, or the model hasn't been retrained), they are reused.

3. **Forced Rebuild Option**: If the user explicitly specifies a flag (e.g., `rebuild=True` in configuration) or the data/model has changed, the system will rebuild the indexes from scratch, overriding any existing ones.

4. **Configuration Parameters**: The behavior can be controlled via parameters like `use_saved_index` or `rebuild_index`, which allow users to force reuse, rebuild, or skip index creation based on their workflow needs.

This ensures efficiency by avoiding redundant computations while maintaining freshness of the retrieval components. If you're using KíLín, consult the documentation for specific parameter names and usage examples.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_storage (mechanism, development)

**Question.** Explain how build_semantic_index embeds the chunks, and how save_indexes and load_indexes protect the saved index.

**Intent:** code_explanation

**Answer with retrieval:**

The `build_semantic_index` function embeds chunks by processing their text content using an embedding model. It batches the chunks (controlled by `batch_size`), generates vector representations via `embedding_session.embed_documents`, and validates the resulting vectors for consistency (e.g., dimensions, finite values) [1][4]. These vectors are stored alongside the original chunks in a `SemanticIndex` object.  

`save_indexes` and `load_indexes` protect the saved index through validation and integrity checks. During saving, `save_indexes` verifies the index structure, ensures chunk identifiers are unique, and computes a checksum (`payload_sha256`) for the payload to detect accidental corruption [8][9]. During loading, `load_indexes` checks the checksum against a recomputed value, validates the embedding model identity, and ensures all data fields (chunks, vectors, code graph) are correctly formatted and consistent [6][7]. If any validation fails, errors are raised, preventing the use of corrupted or incompatible indexes. This ensures saved indexes remain trustworthy and reusable across sessions.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code, semantic)
- [4] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code)
- [6] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
- [7] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
- [8] kilin/retrievals/persistence.py `save_indexes` (retrieved by: context_expansion)
- [9] kilin/retrievals/persistence.py `save_indexes` (retrieved by: context_expansion)

**Gold functions:** `build_semantic_index`, `save_indexes`, `load_indexes`

- in evidence: ['build_semantic_index', 'save_indexes', 'load_indexes']
- cited: ['build_semantic_index', 'save_indexes', 'load_indexes']

**Facts (tick those the answer states correctly):**

- [ ] Chunk texts are embedded in batches, and the backend cache is cleared after each batch.
- [ ] The resulting vectors are validated (count, dimension, finite values, nonzero norm).
- [ ] save_indexes validates the records and the source fingerprint and stores a payload checksum.
- [ ] load_indexes verifies the checksum, the embedding-model identity and the fingerprint.
- [ ] The checksum detects accidental corruption; it is not an authenticity guarantee.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The **KíLín** Python package is designed for building and querying semantic indexes, typically using embeddings to represent text chunks. Here's how the mentioned functions work:

---

### **`build_semantic_index` and Embedding Chunks**
1. **Embedding Process**:
   - The `build_semantic_index` function takes a list of text chunks and converts them into **vector embeddings** using a pre-trained model (e.g., BERT, SentenceTransformer, or a custom model).
   - Each chunk is transformed into a numerical vector (embedding) that captures its semantic meaning. This is done via a model's `encode` method, which maps text to a fixed-dimensional space.

2. **Index Construction**:
   - After embedding, the function builds a **semantic index** (e.g., using FAISS, HNSW, or other vector similarity search structures) to store these embeddings. This index allows efficient retrieval of similar chunks based on query embeddings.

---

### **`save_indexes` and `load_indexes` for Protection**
1. **Saving the Index**:
   - The `save_indexes` function serializes the semantic index and saves it to disk (e.g., as a file or database). This ensures the index is stored in a format that can be later retrieved.
   - **Protection Mechanisms**:
     - **Encryption**: The index might be saved in an encrypted format to prevent unauthorized access.
     - **Permissions**: The saved files are stored in a directory with restricted permissions (e.g., only readable by the owner).
     - **Integrity Checks**: The function may include checksums or metadata to verify the index's integrity during saving.

2. **Loading the Index**:
   - The `load_indexes` function retrieves the saved index and reconstructs it in memory. It ensures:
     - **Consistency**: The loaded index matches the saved version (e.g., via versioning or schema validation).
     - **Security**: The index is loaded in a secure environment, and any tampering is detected (e.g., via checksum verification).

---

### **Key Protection Features**
- **Isolation**: Saved indexes are stored separately from the model or other data to prevent accidental modification.
- **Versioning**: The package may track versions of the index to ensure compatibility with the model used during embedding.
- **Secure Storage**: Files are saved in a secure location (e.g., a protected directory or encrypted volume).

---

### **Note**
If you're referring to specific implementation details (e.g., encryption methods or file formats), the exact behavior may depend on the version of KíLín or its integration with external libraries (e.g., FAISS, SentenceTransformer). Always consult the package's documentation for precise details.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## strip_thinking_callers (dependency, development)

**Question.** Explain which functions call strip_thinking, and why each of them needs to remove the model's reasoning.

**Intent:** code_explanation

**Answer with retrieval:**

The `strip_thinking` function is called by three key functions: `history_messages`, `side_reply`, and `turn_representation`.  

1. **`history_messages`** [6] uses `strip_thinking` to remove model reasoning and citations from turn answers, ensuring only the final answer is included in user/assistant message pairs. This is critical for maintaining a clean, human-readable conversation history without exposing internal processing.  

2. **`side_reply`** [7] calls `strip_thinking` to eliminate thinking text from the model’s reply, ensuring the output is a direct, final answer. This aligns with the side question’s purpose of generating concise, unambiguous responses without intermediate steps.  

3. **`turn_representation`** [5] and [8] uses `strip_thinking` to clean the answer’s summary, removing reasoning and collapsing whitespace. This ensures compact, standardized representations of turns for logging or analysis, focusing only on the final answer.  

Each function requires stripping reasoning to eliminate internal model steps, ensuring outputs are user-facing, concise, and free from unnecessary metadata.

**Cited passages:**

- [5] tests/kilin/test_answering.py `TestHistory.test_representation_strips_thinking_and_collapses_spaces` (retrieved by: semantic)
- [6] kilin/answering.py `history_messages` (retrieved by: context_expansion)
- [7] kilin/answering.py `side_reply` (retrieved by: context_expansion)
- [8] kilin/answering.py `turn_representation` (retrieved by: context_expansion)

**Gold functions:** `turn_representation`, `side_reply`, `history_messages`

- in evidence: ['turn_representation', 'side_reply', 'history_messages']
- cited: ['turn_representation', 'side_reply', 'history_messages']

**Facts (tick those the answer states correctly):**

- [ ] turn_representation removes reasoning (and citations) from an earlier answer to build the compact representation shown to the router.
- [ ] side_reply returns the reply of an auxiliary model call, such as the router's, without reasoning.
- [ ] history_messages removes reasoning from earlier answers before they are included as conversation history.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is designed to evaluate and debug large language models (LLMs), including tools to analyze their outputs. The `strip_thinking` function is used to remove the model's internal reasoning steps from its responses, typically to isolate the final answer. Here's an explanation of which functions might call `strip_thinking` and why:

---

### **1. `strip_thinking` Function**
- **Purpose**: Removes the model's internal reasoning (e.g., "Let me think...", "Step 1:...", etc.) from its output, leaving only the final answer.
- **Why it's used**: 
  - To extract the **direct answer** for evaluation or comparison with ground truth.
  - To eliminate **redundant or irrelevant steps** in the response (e.g., verbose explanations, self-talk).
  - To ensure the output is **concise** and user-friendly.

---

### **2. `process_model_output`**
- **Functionality**: Processes raw model outputs (e.g., from `generate()` or `predict()`) to clean and structure the response.
- **Why it calls `strip_thinking`**:
  - The model's output often includes internal reasoning (e.g., "Okay, I need to calculate..."). Stripping this ensures the final answer is clean and ready for use.
  - Example: If the model outputs `"Let me think... The answer is 42"`, `strip_thinking` would return `"The answer is 42"`.

---

### **3. `extract_answer`**
- **Functionality**: Extracts the final answer from a model's response, typically after removing reasoning steps.
- **Why it calls `strip_thinking`**:
  - To ensure the answer is **isolated** for tasks like automated testing, benchmarking, or comparison with reference answers.
  - For example, in a QA task, the model might output `"Step 1: Read the question. Step 2: Find the answer. Answer: 42"`. Stripping the reasoning leaves `"Answer: 42"`.

---

### **4. `evaluate_answer`**
- **Functionality**: Evaluates the correctness of a model's answer against a reference or ground truth.
- **Why it calls `strip_thinking`**:
  - The model's reasoning might contain errors or hallucinations. By removing it, the evaluation focuses only on the final answer, ensuring accuracy.
  - Example: If the model writes `"I think the answer is 42, but I'm not sure"`, stripping the reasoning leaves `"42"` for comparison.

---

### **5. `clean_response`**
- **Functionality**: Cleans up the model's output by removing unnecessary elements (e.g., markdown, formatting, or internal monologue).
- **Why it calls `strip_thinking`**:
  - To standardize the output format for downstream tasks (e.g., logging, summarization, or integration with other systems).
  - Ensures the response is **free of artifacts** that could interfere with analysis.

---

### **Why Each Function Needs to Remove Reasoning**
1. **Accuracy**: The model's reasoning might contain errors or hallucinations. Removing it ensures the final answer is evaluated independently.
2. **Conciseness**: Users or systems often only need the final answer, not the model's internal steps.
3. **Consistency**: Standardizing the output format (e.g., removing "Let me think...") ensures uniformity across evaluations.
4. **Automation**: Automated tools (e.g., benchmarking scripts) require clean, structured answers to function correctly.

---

### **Note**
If you're using KíLín, the exact functions and their usage may depend on the version of the package. Always refer to the latest documentation for precise details. If you don't have access to the documentation, the above explanation is based on common patterns in LLM evaluation tools.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## hybrid_callers (dependency, development)

**Question.** Explain which RAGSession methods call hybrid_search, and how each of them uses the fused ranking.

**Intent:** code_explanation

**Answer with retrieval:**

The `RAGSession` class calls `hybrid_search` in two methods: `_rank_everything` and `_retrieve`.  

1. **`_rank_everything`** [3]: This method calls `hybrid_search` to fuse semantic, lexical, and code rankings into a single list of `HybridResult` objects. It uses the fused ranking to prioritize chunks based on combined reciprocal rank scores, ensuring duplicates are removed and ties are broken by chunk identifiers. This fused ranking is used when the scope is a single document, ensuring its best passages are included even if they are not in the project-wide top results.  

2. **`_retrieve`** [11]: This method calls `hybrid_search` after performing semantic, lexical, and code searches. It uses the fused ranking (from `hybrid_search`) as the basis for context expansion and trace logging. The fused ranking is then passed to `expand_retrieval_context` to generate the final evidence tuple, which includes the most relevant chunks for the user's query.  

Both methods rely on `hybrid_search` to combine results from semantic, lexical, and code sources, applying reciprocal rank fusion to prioritize chunks based on their combined scores across all channels. The fused ranking is critical for subsequent steps like context expansion and trace reporting.

**Cited passages:**

- [3] kilin/rag.py `RAGSession._rank_everything` (retrieved by: lexical, semantic)
- [11] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._rank_everything`, `hybrid_search`

- in evidence: ['RAGSession._retrieve', 'RAGSession._rank_everything', 'hybrid_search']
- cited: ['RAGSession._retrieve', 'RAGSession._rank_everything']

**Facts (tick those the answer states correctly):**

- [ ] _retrieve fuses the semantic, lexical and code-graph rankings for the configured retrieval count to select the evidence for an answer.
- [ ] _rank_everything ranks every chunk with the same fused channels, for uses that need a complete ranking, such as tools.
- [ ] hybrid_search combines the rankings by reciprocal-rank fusion.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package's `RAGSession` class provides methods for building Retrieval-Augmented Generation (RAG) systems. The `hybrid_search` method is a core component of the retrieval phase, combining semantic and keyword-based search strategies. Here's how it is used:

### **Methods that Call `hybrid_search`**
1. **`search(query)`**  
   - This is the primary method for performing retrieval. It internally calls `hybrid_search` to combine results from two retrieval strategies:  
     - **Semantic search** (e.g., using embeddings for similarity).  
     - **Keyword-based search** (e.g., BM25 or TF-IDF).  
   - The `hybrid_search` method fuses these results by **fused ranking**, which typically involves combining scores from both strategies (e.g., averaging or weighted sum) to prioritize documents that are relevant to both semantic and keyword criteria.

2. **`retrieve(query)`**  
   - This method may also use `hybrid_search` directly, depending on the implementation. It focuses on retrieving documents but leverages the fused ranking to ensure high-quality, contextually relevant results.

### **How Fused Ranking Works**
- **Input**: Results from semantic and keyword-based searches (each with their own relevance scores).  
- **Process**:  
  - The `hybrid_search` method calculates a **combined score** for each document by integrating scores from both strategies.  
  - For example, it might use a weighted average:  
    $$
    \text{fused\_score} = \alpha \cdot \text{semantic\_score} + (1 - \alpha) \cdot \text{keyword\_score}
    $$  
    where $\alpha$ is a tunable parameter.  
- **Output**: A ranked list of documents sorted by the fused score, ensuring the most contextually relevant documents are prioritized.

### **Key Insight**
The fused ranking ensures that the retrieval process balances **semantic relevance** (for nuanced, context-aware queries) and **keyword precision** (for exact matches), leading to more robust and accurate results for the RAG pipeline.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## citation_check (location, development)

**Question.** Explain where KíLín checks that the citation numbers in an answer refer to the supplied evidence, and what happens when they do not.

**Intent:** code_explanation

**Answer with retrieval:**

KíLín checks citation validity in the `_invalid_citations` function [4], which compares citation numbers in the answer against the supplied evidence's available citations. It uses regex to extract citations from the answer and identifies numbers outside the range of valid evidence. When invalid citations are found, they are recorded as invalid identifiers in the output but do not trigger corrections or re-generations of the answer. This check is a diagnostic step, not a semantic verification of claims, and the answer may still contain unsupported claims even if all citations resolve [1]. The function returns a tuple of invalid identifiers in first-occurrence order, ensuring duplicates are omitted [4].

**Cited passages:**

- [1] docs/answering_pipeline.md `None` (retrieved by: lexical, semantic)
- [4] kilin/rag.py `_invalid_citations` (retrieved by: lexical)

**Gold functions:** `_invalid_citations`, `RAGSession.ask`

- in evidence: ['_invalid_citations']
- cited: ['_invalid_citations']

**Facts (tick those the answer states correctly):**

- [ ] _invalid_citations returns the citation numbers in the answer that do not match any supplied evidence.
- [ ] RAGSession.ask records them in the turn parameters as invalid_citations.
- [ ] The answer is neither modified nor regenerated.
- [ ] The check is syntactic: it does not verify that the cited evidence supports the claim.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package checks citation numbers in answers against supplied evidence during its **source verification phase**. Here's how it works:

1. **Citation Matching Process**:  
   - When an answer includes citations (e.g., `[1]`, `[2]`), KíLín maps these citation numbers to the corresponding evidence documents provided.  
   - It verifies that each citation number in the answer exists in the evidence list and that the referenced document supports the claim made in the answer.

2. **Location in the Package**:  
   - This validation is typically handled by the `kilian.checker` module, which processes the answer and evidence. The exact implementation may vary, but the core logic resides in the `check_citations` function (or similar) within the package's validation pipeline.

3. **What Happens if Citations Don’t Match**:  
   - If a citation number in the answer does not correspond to a valid evidence document, KíLín flags the discrepancy.  
   - This could result in an error message, a warning, or the rejection of the answer, depending on the configuration. The tool ensures that all claims are backed by the provided evidence, preventing hallucination or misattribution of sources.

This process ensures answers are grounded in the given evidence, enhancing reliability and accountability. If you need specifics about the code structure, refer to the KíLín documentation or source code for the exact implementation details.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## chunking (mechanism, added)

**Question.** Explain how chunk_source_units handles a source unit that is longer than the maximum chunk size, and how the overlap between consecutive chunks is kept.

**Intent:** code_explanation

**Answer with retrieval:**

The `chunk_source_units` function handles source units longer than `max_tokens` by splitting them into smaller chunks while respecting natural unit boundaries and maintaining overlap. If a `SourceUnit`'s text exceeds `max_tokens`, it recursively processes its children or splits the unit using `_split_unit`, ensuring each chunk does not surpass `max_tokens` [9]. Overlap between consecutive chunks is enforced by the `overlap_tokens` parameter, which dictates how much text is shared between adjacent chunks. During splitting, the function adjusts chunk boundaries so that each new chunk overlaps with the previous by `overlap_tokens` tokens, ensuring continuity and context retention [6]. This is validated during parameter checks, where `overlap_tokens` must be less than `target_tokens` [10]. The function uses the embedding tokenizer to estimate token counts, ensuring chunks align with the selected model's tokenization [1]. The resulting `Chunk` objects retain metadata like line ranges and identifiers, preserving source context [1].

**Cited passages:**

- [1] docs/source_pipeline.md `None` (retrieved by: context_expansion)
- [6] kilin/chunking.py `chunk_source_units` (retrieved by: code, semantic)
- [9] kilin/chunking.py `_chunk_source_unit` (retrieved by: semantic)
- [10] kilin/chunking.py `_check_sizes` (retrieved by: context_expansion)

**Gold functions:** `chunk_source_units`, `_chunk_source_unit`, `_split_unit`, `_overlap_start`

- in evidence: ['chunk_source_units', '_chunk_source_unit']
- cited: ['chunk_source_units', '_chunk_source_unit']

**Facts (tick those the answer states correctly):**

- [ ] A unit within the maximum size is kept whole as one chunk.
- [ ] An oversized unit is split through its children only when the children cover all of its non-whitespace text.
- [ ] Otherwise it is split along line boundaries, and within a single line if that line is too long.
- [ ] The target size guides where to split; the maximum size is the bound every chunk must respect.
- [ ] Overlap with the previous chunk is kept as whole source lines or a character suffix, up to the overlap token limit.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `chunk_source_units` function in the KíLín Python package handles source units longer than the maximum chunk size by splitting them into overlapping chunks. Here's how it works:

1. **Chunking Process**:  
   - If a source unit exceeds the maximum chunk size, it is divided into multiple chunks. Each chunk overlaps with the next by a predefined amount (e.g., 20% of the chunk size). This ensures continuity and avoids information loss at chunk boundaries.

2. **Overlap Management**:  
   - Consecutive chunks are created using a **sliding window** approach. For example, if the maximum chunk size is `N` and the overlap is `O`, each new chunk starts at a position `N - O` units after the previous chunk. This maintains the specified overlap between chunks.

3. **Handling Long Source Units**:  
   - The function calculates the number of required chunks based on the source unit's length, chunk size, and overlap. For a source unit of length `L`, the number of chunks is approximately `(L - chunk_size) / (chunk_size - overlap) + 1`. The last chunk is truncated to ensure it does not exceed the source unit's bounds.

4. **Use Case**:  
   - This method is critical for tasks requiring contextual analysis (e.g., sequence alignment, pattern detection) where overlapping chunks preserve critical information across boundaries.

If you need further details about specific parameters or implementation, let me know!

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## rank_everything_callers (dependency, added)

**Question.** Explain which RAGSession methods call _rank_everything, and why each of them needs a ranking of every chunk.

**Intent:** code_explanation

**Answer with retrieval:**

The `RAGSession._rank_everything` method is called by two primary methods: `_read_document` and `_retrieve`.  

1. **`_read_document`** calls `_rank_everything` to rank all chunks in the session when selecting evidence for a question. This ensures that even if a document is long, the most relevant parts are prioritized based on semantic, lexical, and code-graph rankings. This is critical for selecting chunks within the token budget, as it allows the system to focus on the most relevant content regardless of global ranking. [7]  

2. **`_retrieve`** calls `_rank_everything` when the query scope is limited to a single document. This ensures that all chunks of the specified document are ranked, even if they are not among the top results globally. This is necessary to find the best passages within the document, which might otherwise be overlooked during broader searches. [13]  

Both cases require ranking every chunk to avoid missing relevant information that might not appear in top results due to global scope or document-specific context. The method’s design ensures comprehensive evaluation of all chunks, enabling precise evidence selection. [1]

**Cited passages:**

- [1] kilin/rag.py `RAGSession._rank_everything` (retrieved by: code, semantic)
- [7] kilin/rag.py `RAGSession._read_document` (retrieved by: context_expansion)
- [13] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._run_tool`, `RAGSession._read_document`

- in evidence: ['RAGSession._retrieve', 'RAGSession._run_tool', 'RAGSession._read_document']
- cited: ['RAGSession._retrieve', 'RAGSession._read_document']

**Facts (tick those the answer states correctly):**

- [ ] _retrieve ranks every chunk when the question names exactly one document, and keeps that document's best passages, which may not be in the project-wide top results.
- [ ] _run_tool gives tools a rank function, so a tool can rank all chunks for any text.
- [ ] _read_document ranks all chunks to choose the document to read within the token budget.

**Errors or unsupported statements:** 

**Answer without retrieval:**

In the KíLín Python package, the `_rank_everything` method is typically called by RAGSession methods that require a **comprehensive ranking of all retrieved chunks** rather than just the top results. This is necessary in scenarios where the query's context or the system's design demands evaluating **every chunk** to ensure accuracy, completeness, or specific use cases. Here's a breakdown of such methods and their rationale:

---

### **1. `query(...)` or `generate(...)` Methods**
- **Why they call `_rank_everything`:**  
  These methods often retrieve a large set of candidate chunks (e.g., from a vector store or index) and then rank them. Ranking **every chunk** ensures that even less relevant or ambiguous chunks are considered, which can be critical for:
  - **Ambiguous queries**: When the user's query is vague or has multiple interpretations, ranking all chunks allows the system to explore all possible angles.
  - **Edge cases**: For example, if a query requires comparing multiple perspectives (e.g., "What are the pros and cons of X?"), ranking all chunks ensures no relevant information is missed.
  - **Consistency**: Some systems prioritize ranking all chunks to maintain uniformity in how information is processed, regardless of query specificity.

- **Why ranking every chunk is needed:**  
  By ranking all chunks, the system avoids missing subtle or context-dependent information that might not appear in the top results. This is particularly important for generating answers that are **comprehensive** or **nuanced**.

---

### **2. `rerank(...)` or `retrieval(...)` Methods**
- **Why they call `_rank_everything`:**  
  These methods might explicitly require re-ranking all chunks after initial retrieval, especially when:
  - **Custom ranking strategies** are applied (e.g., combining multiple criteria like relevance, recency, or source authority).
  - **Post-processing** is needed to refine results, such as filtering out low-quality chunks or adjusting scores based on additional metadata.

- **Why ranking every chunk is needed:**  
  Re-ranking all chunks ensures that the final set of chunks used for generation is **optimally ordered**, even if the initial retrieval was not perfect. This is crucial for maintaining high-quality outputs when the ranking strategy is complex or user-defined.

---

### **3. `batch_query(...)` or `batch_generate(...)` Methods**
- **Why they call `_rank_everything`:**  
  When handling multiple queries in parallel, these methods might rank all chunks for each query to avoid partial or inconsistent results. This ensures that each query's context is fully explored.

- **Why ranking every chunk is needed:**  
  Parallel processing could lead to overlapping or conflicting chunk rankings. Ranking all chunks ensures that each query's context is independently and thoroughly evaluated.

---

### **Why Ranking Every Chunk is Critical**
- **Comprehensiveness**: Ensures no relevant information is overlooked, even if it's not the top result.
- **Robustness**: Handles edge cases where the top results might be misleading or incomplete.
- **Customization**: Allows users to define ranking strategies that prioritize specific criteria (e.g., recency, source reliability) across all chunks.

---

### **Conclusion**
The `_rank_everything` method is called by RAGSession methods that require **full ranking of all retrieved chunks** to ensure the generation process has access to the most relevant and comprehensive information. This is essential for scenarios involving ambiguous queries, complex ranking strategies, or the need for consistent, high-quality outputs. Each method's reliance on `_rank_everything` reflects a design choice to prioritize **accuracy and completeness** over efficiency in critical cases.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## history_selection (location, added)

**Question.** Explain where KíLín decides which earlier turns of a conversation are given to the model as history, and what determines how many.

**Intent:** code_explanation

**Answer with retrieval:**

KíLín decides which earlier conversation turns are included as history in the `select_history` function [2]. This function uses a `HistoryPolicy` (e.g., `none`, `recent`, `relevant`) and the `recent_turns` parameter to determine the count. For `recent` policy, it returns the last `recent_turns` turns. If the router's decision marks the question as standalone, only the current turn is included; otherwise, it combines recent turns with related ones from the decision [2]. The maximum number of turns is controlled by `_MAX_EARLIER_TURNS` in `_earlier_turns` [5], which limits the returned history to avoid excessive context. The intent configuration in `ask` [4] also influences which turns are sent as history, depending on routing decisions. Thus, the count is determined by the policy, `recent_turns`, related turns, and a hard-coded maximum.

**Cited passages:**

- [2] kilin/answering.py `select_history` (retrieved by: lexical, semantic)
- [4] kilin/rag.py `RAGSession.ask` (retrieved by: lexical)
- [5] kilin/rag.py `RAGSession._earlier_turns` (retrieved by: code)

**Gold functions:** `select_history`, `RAGSession.ask`

- in evidence: ['select_history', 'RAGSession.ask']
- cited: ['select_history', 'RAGSession.ask']

**Facts (tick those the answer states correctly):**

- [ ] select_history chooses the earlier turns, and RAGSession.ask calls it for each question.
- [ ] The intent's history policy decides: none includes nothing, recent includes the last few turns, relevant also adds the turns the router marked as related.
- [ ] The number of recent turns comes from the project type (history.recent_turns).
- [ ] A question the router judged standalone still receives the previous turn, without its evidence.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python package for building conversational agents, manages conversation history by prioritizing the most recent interactions to stay within the model's context length. The number of earlier turns included as history is determined by two factors:

1. **Model Context Length**: The maximum number of tokens the underlying language model (e.g., Llama) can process. For example, if the model has a 2048-token limit, the conversation history is truncated to fit within this constraint.

2. **User Configuration**: KíLín allows users to explicitly set parameters (e.g., `max_history_length`) to control how many turns are retained. This ensures the history does not exceed the model's token capacity while preserving the most relevant context.

If the conversation exceeds the token limit, older messages are removed, typically starting from the oldest ones. The exact truncation logic depends on the implementation details of KíLín, which may prioritize recent interactions or use a sliding window approach. For precise details, refer to the package's documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## conversation_lock (mechanism, added)

**Question.** Explain how ConversationFileLock prevents two sessions from writing the same conversation file at the same time.

**Intent:** code_explanation

**Answer with retrieval:**

ConversationFileLock prevents concurrent writes by using an OS advisory lock on a companion `.lock` file, not the JSON inode, ensuring the lock persists through atomic JSON replacements [1]. When a session acquires the lock via `acquire()`, it blocks other sessions from acquiring it, raising `ConversationLockedError` immediately upon contention [2]. A process-local registry rejects duplicate owners before opening a file descriptor, ensuring only one writer exists [1]. The lock is nonblocking, meaning contention does not involve reading or modifying the JSON, and forked child processes inherit the lock but close it upon exit, preventing prolonged parent ownership [1]. Tests confirm that even aliases (e.g., `/var` vs. `/private/var`) are treated as the same file, and attempts to acquire the lock on the same path or alias fail, enforcing single-writer ownership [3][5]. This mechanism ensures no two sessions write the same conversation file simultaneously.

**Cited passages:**

- [1] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: semantic)
- [2] kilin/conversation_ownership.py `ConversationFileLock` (retrieved by: code)
- [3] tests/kilin/test_conversation_ownership.py `TestConversationFileLock.test_canonical_alias_conflicts_with_same_owner` (retrieved by: lexical)
- [5] tests/kilin/test_conversation_ownership.py `TestConversationFileLock.test_two_instances_conflict_without_modifying_json` (retrieved by: semantic)

**Gold functions:** `ConversationFileLock.acquire`, `conversation_write_ownership`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] The lock is an operating-system advisory lock on a companion .json.lock file, so it survives atomic replacement of the conversation file.
- [ ] Acquisition does not wait: if another writer holds the lock, ConversationLockedError is raised.
- [ ] A registry within the process rejects a second owner of the same path before the file is opened.
- [ ] The operating system releases the lock when the owning process exits.
- [ ] It coordinates KíLín writers only; programs that ignore the lock are not prevented from writing.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `ConversationFileLock` in the KíLín Python package prevents multiple sessions from writing to the same conversation file simultaneously by using a **file-based locking mechanism**. Here's how it works:

1. **Lock Acquisition**: When a session starts, it attempts to acquire the lock by creating or opening a dedicated lock file (e.g., a file named `.lock` or a temporary file) associated with the conversation. This lock file acts as a semaphore to control access.

2. **Exclusive Access**: The lock file is opened in **exclusive mode**, ensuring that only one session can hold the lock at a time. If another session tries to acquire the lock, it will block or raise an error until the lock is released.

3. **Conflict Resolution**: If a second session attempts to write to the conversation file while the lock is held, it will either:
   - **Wait** (e.g., retry after a delay) until the lock is released.
   - **Fail gracefully** (e.g., log an error or abort the operation) if the lock cannot be acquired immediately.

4. **Lock Release**: Once the session completes writing to the conversation file, it explicitly releases the lock by closing or deleting the lock file. This allows other sessions to acquire the lock and proceed.

This mechanism ensures that only one session can write to the conversation file at a time, preventing data corruption or overwrites caused by concurrent access. The exact implementation details (e.g., use of `fcntl`, `filelock`, or custom logic) may vary, but the core principle is to enforce exclusive access through a lock file.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
