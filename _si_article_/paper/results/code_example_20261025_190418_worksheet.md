# Code example: grading worksheet

Indexed revision: bd2515be3 (HEAD)

## routing (mechanism, development)

**Question.** Explain how route_question assigns a question to an intent, and what happens when the router reply cannot be used.

**Intent:** code_explanation

**Answer with retrieval:**

The `route_question` function assigns a question to an intent by first checking if the question matches any intent's shortcut patterns [1]. If a shortcut matches, it immediately returns a `RouteDecision` with the corresponding intent and marks the question as standalone. If no shortcut matches, it constructs a prompt for the router model using conversation history and the new question [10], then queries the router. 

If the router reply is valid (e.g., returns JSON with an intent and standalone status), it parses the result into a `RouteDecision` [9]. However, if the router reply is unusable (e.g., malformed JSON, exceptions during processing), the function falls back to the project's default intent, sets a `fallback` reason (e.g., "router failed: ..."), and returns a `RouteDecision` with the default intent [4]. 

When the router reply cannot be used, the fallback ensures the question is still routed to a valid intent, using the default intent and retaining conversation history metadata like `refers_back` and `related_turn_ids` [4]. This mechanism ensures robustness by avoiding interruptions in answering even if the router model fails.

**Cited passages:**

- [1] kilin/answering.py `route_question` (retrieved by: code, lexical, semantic)
- [4] kilin/answering.py `route_question` (retrieved by: code, semantic)
- [9] kilin/answering.py `_parse_route` (retrieved by: context_expansion)
- [10] kilin/answering.py `_router_prompt` (retrieved by: context_expansion)

**Gold functions:** `route_question`, `_parse_route`, `side_reply`

- in evidence: ['route_question', '_parse_route']
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

RAGSession decides to reuse saved retrieval indexes by comparing their metadata and model identity against the current project configuration. If the saved indexes' fingerprint (derived from source content and configuration) and embedding model identity match the current setup, they are reused [3]. This check is performed via `indexes_are_current` [4], which verifies the saved data against the current repository state. If mismatches occur (e.g., changed sources, updated models, or configuration changes), indexes are rebuilt [12]. The metadata, including embedding model settings and chunking parameters, is generated via `_index_metadata` [9] and compared against saved metadata during preparation. If discrepancies exist, the system triggers a rebuild, ensuring indexes align with the latest project state. This process avoids unnecessary recomputation when indexes are valid, but ensures correctness when changes occur.

**Cited passages:**

- [3] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: semantic)
- [4] kilin/retrievals/persistence.py `indexes_are_current` (retrieved by: lexical)
- [9] kilin/rag.py `RAGSession._index_metadata` (retrieved by: context_expansion)
- [12] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._prepare_retrieval`, `load_indexes`

- in evidence: ['RAGSession._prepare_retrieval', 'load_indexes']
- cited: ['RAGSession._prepare_retrieval']

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

The `build_semantic_index` function embeds chunks by using an `EmbeddingSession` to generate vectors for each chunk's text. It processes chunks in batches, as specified by the `batch_size` parameter, and calls `embedding_session.embed_documents` on each batch [2]. The resulting vectors are validated for correctness (e.g., non-finite components, zero norm) before being stored in a `SemanticIndex` alongside the original chunks [2]. This ensures embeddings are consistent and ready for semantic search.

The `save_indexes` and `load_indexes` functions protect the saved index through validation and integrity checks. `save_indexes` serializes the index data (including embeddings, chunk metadata, and checksums) into a JSON file, ensuring the source fingerprint matches the current repository state [12]. It also computes a checksum for the payload to detect accidental corruption. `load_indexes` verifies the checksum, validates the embedding model identity, and ensures the saved data aligns with the current repository’s chunks and code graph [10]. If any checks fail (e.g., invalid checksums, mismatched fingerprints), the index is rejected, preventing use of corrupted or outdated data. This workflow ensures saved indexes remain accurate and secure against tampering or version mismatches.

**Cited passages:**

- [2] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code, semantic)
- [10] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
- [12] kilin/retrievals/persistence.py `save_indexes` (retrieved by: context_expansion)

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

The functions that call `strip_thinking` are `history_messages`, `turn_representation`, and `side_reply`.  

- **`history_messages`** uses `strip_thinking` to remove model-generated reasoning and citations from answers, ensuring only the final answer is included in the message history for consistency and clarity [2].  
- **`turn_representation`** calls `strip_thinking` to extract concise, clean summaries of answers without internal reasoning, facilitating compact turn summaries for later use [4][6].  
- **`side_reply`** applies `strip_thinking` to eliminate reasoning steps from side-question replies, ensuring the output matches the expected format for direct use in user interactions [5].  

Each function requires this step to isolate the model’s final answer, avoid exposing internal logic, and maintain structured, clean outputs for downstream tasks like history logging or summarization.

**Cited passages:**

- [2] kilin/answering.py `history_messages` (retrieved by: code, lexical)
- [4] kilin/answering.py `turn_representation` (retrieved by: code, lexical)
- [5] kilin/answering.py `side_reply` (retrieved by: code, lexical)
- [6] kilin/answering.py `turn_representation` (retrieved by: code)

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

The `RAGSession` methods that call `hybrid_search` are `_rank_everything` [1] and `_retrieve` [10].  

- **`_rank_everything`** gathers semantic, lexical, and code search results (if applicable) and passes them to `hybrid_search` [1]. It uses the fused ranking by combining these results into `HybridResult` objects, which aggregate scores from all channels using reciprocal rank fusion. The fused ranking is sorted by score (descending) and chunk identifier (ascending) to prioritize higher-ranked chunks.  
- **`_retrieve`** directly calls `hybrid_search` with semantic, lexical, and code results [10]. It uses the fused ranking to combine all candidate chunks, then applies additional filtering (e.g., demoting documentation or test results based on user preference). The fused ranking ensures chunks are prioritized by their combined scores across semantic, lexical, and code channels.  

Both methods rely on `hybrid_search` to merge rankings from different sources into a unified list, where each chunk’s final rank reflects contributions from all active retrieval channels. The fused ranking is critical for determining which chunks are most relevant across heterogeneous search strategies.

**Cited passages:**

- [1] kilin/rag.py `RAGSession._rank_everything` (retrieved by: lexical, semantic)
- [10] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

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

KíLín checks citation validity in two key functions: `_invalid_citations` [1] and `answer_citations` [2]. 

1. **`_invalid_citations`**  
   - **Role**: Compares citation numbers in the answer against the `Evidence` objects provided. It collects all valid citation identifiers from `evidence` (via `item.citation`) and checks if any citation in the answer (parsed using regex patterns) matches these.  
   - **Mechanism**: If a citation number in the answer is not present in the evidence, it is recorded as invalid. The function returns a tuple of invalid identifiers (as strings or integers) in the order of their first occurrence.  
   - **Relation**: This function is part of the RAG pipeline, ensuring answers reference only valid evidence during prompt construction [7][10] and post-generation checks [5].

2. **`answer_citations`**  
   - **Role**: Validates citation numbers against the total number of passages (`passages`). It identifies citations outside the range [1, passages] as invalid.  
   - **Mechanism**: Returns a sorted list of invalid numbers. This ensures citations do not reference non-existent passages.  
   - **Relation**: Used in conjunction with `_invalid_citations` to flag both out-of-range and missing citations, ensuring answers align with the evidence provided.

**When citations do not match**:  
- `_invalid_citations` returns invalid identifiers, and `answer_citations` highlights out-of-range numbers. These results are likely used to flag answers for correction, though the exact downstream action (e.g., rejection, resubmission) is not explicitly stated in the evidence. The evidence confirms the checks are performed but does not detail post-check behavior.

**Cited passages:**

- [1] kilin/rag.py `_invalid_citations` (retrieved by: lexical, semantic)
- [2] kilin/tools/_shared/document_notes.py `answer_citations` (retrieved by: lexical)
- [5] kilin/rag.py `rag` (retrieved by: lexical)
- [7] kilin/rag.py `_answer_prompt` (retrieved by: semantic)
- [10] kilin/rag.py `_answer_prompt` (retrieved by: context_expansion)

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

## chunking (mechanism, development)

**Question.** Explain how chunk_source_units handles a source unit that is longer than the maximum chunk size, and how the overlap between consecutive chunks is kept.

**Intent:** code_explanation

**Answer with retrieval:**

The `chunk_source_units` function handles source units longer than `max_tokens` by splitting them into smaller chunks while preserving line boundaries and structural context. If a unit's text exceeds `max_tokens`, it recursively processes its children or splits the unit into line-based chunks using `_split_unit` [2]. This ensures chunks do not exceed the token limit. Overlap between consecutive chunks is maintained through the `_overlap_start` function, which calculates the starting position of the next chunk to ensure the overlap (repeated tokens) does not exceed `overlap_tokens` [13]. This binary search-based method efficiently determines the earliest valid start for the overlapping suffix, maintaining the specified overlap while splitting the unit. The overlap is enforced by checking token counts of overlapping text segments during chunk creation [10].

**Cited passages:**

- [2] kilin/chunking.py `_chunk_source_unit` (retrieved by: semantic)
- [10] kilin/chunking.py `_split_unit` (retrieved by: context_expansion)
- [13] kilin/chunking.py `_overlap_start` (retrieved by: context_expansion)

**Gold functions:** `chunk_source_units`, `_chunk_source_unit`, `_split_unit`, `_overlap_start`

- in evidence: ['chunk_source_units', '_chunk_source_unit', '_split_unit', '_overlap_start']
- cited: ['_chunk_source_unit', '_split_unit', '_overlap_start']

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

## rank_everything_callers (dependency, development)

**Question.** Explain which RAGSession methods call _rank_everything, and why each of them needs a ranking of every chunk.

**Intent:** code_explanation

**Answer with retrieval:**

The `RAGSession` methods that call `_rank_everything` are `_read_document`, `_retrieve`, and `_run_tool`.  

1. **`_read_document`** [4] calls `_rank_everything` to rank all chunks of a document when the question refers to a specific document (e.g., "in this paper"). This ensures the document’s most relevant passages are identified, even if they are not among the top results globally. Ranking every chunk allows the system to prioritize context within the document, improving accuracy for focused queries.  

2. **`_retrieve`** [11] uses `_rank_everything` when the question names a specific document. It ranks all chunks of that document to find the best passages, ensuring the document’s internal relevance is captured, even if the document itself isn’t the top result in the broader search. This avoids missing critical context within a named document.  

3. **`_run_tool`** [13] leverages `_rank_everything` as a lambda function to retrieve context for a tool’s answer. Ranking every chunk ensures the tool’s input is based on the most relevant evidence, even if the tool’s query requires deep, document-specific insights.  

Each method requires full chunk ranking to prioritize document-specific relevance, avoid missing critical context, and ensure accurate retrieval of the most pertinent information for the user’s query.

**Cited passages:**

- [4] kilin/rag.py `RAGSession._read_document` (retrieved by: lexical)
- [11] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)
- [13] kilin/rag.py `RAGSession._run_tool` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._run_tool`, `RAGSession._read_document`

- in evidence: ['RAGSession._retrieve', 'RAGSession._run_tool', 'RAGSession._read_document']
- cited: ['RAGSession._retrieve', 'RAGSession._run_tool', 'RAGSession._read_document']

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

## history_selection (location, development)

**Question.** Explain where KíLín decides which earlier turns of a conversation are given to the model as history, and what determines how many.

**Intent:** code_explanation

**Answer with retrieval:**

KíLín decides which earlier turns are included as history in the `select_history` function within `answering.py` [1]. This function uses a `HistoryPolicy` parameter (none, recent, or relevant) to determine the selection. If set to "recent," it includes the last `recent_turns` turns. If the router's decision marks the question as standalone, only the current turn is included. Otherwise, it combines recent turns with related turns identified by the router. The number of turns is constrained by `MAX_RELATED_TURNS` and `MAX_EARLIER_TURNS` constants, which limit how many earlier turns are considered [9]. The router's `RouteDecision` object, including `related_turn_ids`, guides which turns are linked, ensuring context is preserved while avoiding excessive history [7]. The final count depends on these policies, the router's judgment, and predefined limits.

**Cited passages:**

- [1] kilin/answering.py `select_history` (retrieved by: lexical, semantic)
- [7] kilin/rag.py `RAGSession._earlier_turns` (retrieved by: semantic)
- [9] kilin/rag.py `RAGSession._earlier_turns` (retrieved by: context_expansion)

**Gold functions:** `select_history`, `RAGSession.ask`

- in evidence: ['select_history', 'RAGSession.ask']
- cited: ['select_history']

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

## conversation_lock (mechanism, development)

**Question.** Explain how ConversationFileLock prevents two sessions from writing the same conversation file at the same time.

**Intent:** code_explanation

**Answer with retrieval:**

ConversationFileLock prevents concurrent writes by using an operating-system-level advisory lock on a sidecar file (e.g., `<path>.lock`) and maintaining a registry of active paths [3]. When a session attempts to acquire the lock via `acquire()`, it first checks if the target path is already in `_ACTIVE_PATHS` (a shared set across processes) [10]. If so, it raises `ConversationLockedError`, blocking the second session [12]. The lock is acquired nonblocking, ensuring immediate contention detection [6]. Additionally, the lock is tied to the conversation’s canonical path, ensuring atomic replacement of the JSON file without losing the lock [6]. The `conversation_write_ownership` context manager either reuses an existing token or acquires a new lock, ensuring exclusive access [9]. The `_REGISTRY_LOCK` synchronizes access to `_ACTIVE_PATHS` and `_OWNERS`, preventing race conditions [2]. This combination of OS-level locking, path tracking, and registry synchronization ensures only one session holds the lock for a given conversation file at a time.

**Cited passages:**

- [2] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: lexical, semantic)
- [3] kilin/conversation_ownership.py `ConversationFileLock` (retrieved by: code, semantic)
- [6] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: semantic)
- [9] kilin/conversation_ownership.py `conversation_write_ownership` (retrieved by: context_expansion)
- [10] kilin/conversation_ownership.py `ConversationFileLock.acquire` (retrieved by: context_expansion)
- [12] kilin/conversation_ownership.py `ConversationFileLock.acquire` (retrieved by: context_expansion)

**Gold functions:** `ConversationFileLock.acquire`, `conversation_write_ownership`

- in evidence: ['ConversationFileLock.acquire', 'conversation_write_ownership']
- cited: ['ConversationFileLock.acquire', 'conversation_write_ownership']

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

## backend_choice (mechanism, held_out_1)

**Question.** Explain how choose_backend selects the computation backend, and which PyTorch device is used when PyTorch is selected.

**Intent:** code_explanation

**Answer with retrieval:**

The `choose_backend` function selects the computation backend based on the requested value and platform. If the requested backend is `"auto"`, it defaults to `"mlx"` on Apple Silicon (Darwin/arm64) and `"torch"` otherwise. Explicit `"mlx"` requests are rejected on non-Apple Silicon platforms [3]. When PyTorch is selected (via `"auto"` or explicit `"torch"`), the `torch_device` function determines the device by checking availability flags: it prioritizes CUDA, then Apple MPS, and finally CPU, without probing hardware directly [9]. This device selection is used when PyTorch is the active backend. The logic for PyTorch device choice is mirrored in `choose_torch_device` [6], which follows the same priority order.

**Cited passages:**

- [3] kilin/runtime.py `choose_backend` (retrieved by: code, semantic)
- [6] setup/check_setup.py `choose_torch_device` (retrieved by: semantic)
- [9] kilin/runtime.py `torch_device` (retrieved by: context_expansion)

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

The `needs_confirmation` function determines if a tool should prompt the user based on the confirmation policy and plan details. It returns `True` if the policy is `"always"` or if the plan's `batches_to_run` exceeds the `large_batches` threshold under the `"when_large"` policy. Otherwise, it returns `False` [4]. 

The `confirm_plan` function uses `needs_confirmation` to decide whether to ask the user. If required, it calls `context.confirm(plan)` to capture the user's answer, which must be one of `PLAN_CHOICE_CHOICES` (`"continue"`, `"quick"`, or `"cancel"`). If the answer is valid, it sets the choice; otherwise, it defaults to `"cancel"` [1]. 

The user's answer is traced via `trace.emit`, and the final choice (e.g., `"continue"`, `"cancel"`) is returned alongside whether the user was asked. If the user cancels or provides an invalid response, the plan is aborted. This process ensures the tool respects the confirmation policy and handles user input according to predefined options.

**Cited passages:**

- [1] kilin/tools/_shared/planning.py `confirm_plan` (retrieved by: code, lexical, semantic)
- [4] kilin/tools/_shared/planning.py `needs_confirmation` (retrieved by: code, semantic)

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

The function `offer_export` in [5] calls `ask_choice` to ask the user whether to save a long list as a Markdown file, offering options like "Save all" or "Show the first". The function `choose_document` in [8] calls `ask_choice` to let the user select a file from a list, with options formatted as "filename (reason)". These are the only explicit callers of `ask_choice` in the evidence. [1] defines `ask_choice`'s parameters, but no other functions are explicitly linked to it in the provided sources.

**Cited passages:**

- [1] kilin/tools/_shared/planning.py `ask_choice` (retrieved by: code, lexical, semantic)
- [5] kilin/tools/_shared/long_lists.py `offer_export` (retrieved by: code)
- [8] kilin/tools/_shared/choose_document.py `choose_document` (retrieved by: semantic)

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

TraceLogger rewrites the human-readable trace file when events are emitted, particularly during progress updates or other critical events, as described in [9] and [8]. The `_write_readable_trace` method generates a compact, named hierarchy for inspection, using a temporary JSON file to replace the target file while an exclusive companion-file lock is held [8]. This lock ensures no other process can access the file during the rewrite, preventing partial writes. The temporary file is only swapped in once the write completes, ensuring the target file is fully written. Additionally, the lock is released after the rewrite, allowing other processes to access the file. This mechanism avoids partial writes by serializing access and completing the file write before releasing the lock.

**Cited passages:**

- [8] kilin/trace_logging.py `TraceLogger._write_readable_trace` (retrieved by: semantic)
- [9] kilin/trace_logging.py `TraceLogger._write_event` (retrieved by: context_expansion)

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

The `synchronize_project_workspace` function keeps its copy of a source folder up to date by comparing the current state of the source directory with a saved snapshot manifest and applying changes. It first reads the saved manifest from `source_snapshot.json` [11]. If the source directory has changed, it scans the repository, builds a new manifest, and compares it with the saved one [5]. Differences (added, modified, or removed files) are resolved by copying new/modified files to the project's snapshot directory and removing obsolete files [11]. The snapshot manifest is then updated to reflect the current state, ensuring the project's copy aligns with the original. This process avoids overwriting existing snapshots unless explicitly required, preserving edits made in the managed workspace [2]. The function relies on manifest comparisons and direct file copying to maintain synchronization.

**Cited passages:**

- [2] kilin/project.py `synchronize_project_workspace` (retrieved by: code, semantic)
- [5] kilin/project.py `_synchronize_snapshot` (retrieved by: semantic)
- [11] kilin/project.py `_synchronize_snapshot` (retrieved by: context_expansion)

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

The function `cached_model_snapshot` is called by `_model_source` (in [4]) and `ChatSession._load` (in [6] and [7]). 

- `_model_source` uses `cached_model_snapshot` when `allow_download=False`, bypassing downloads to resolve a cached snapshot. This occurs if the input `model_id` is not a local directory and downloading is explicitly disallowed.  
- `ChatSession._load` calls `cached_model_snapshot` when `allow_download=False` and the `model_id` is not a local directory. This ensures the model uses a cached snapshot instead of downloading, but only if the cached path exists and contains `config.json`.  

No other functions explicitly call `cached_model_snapshot` based on the evidence. The function is designed to avoid downloads by leveraging existing cached snapshots, but it relies on the presence of `config.json` to validate completeness.

**Cited passages:**

- [4] kilin/embeddings.py `_model_source` (retrieved by: code, lexical)
- [6] kilin/chat.py `ChatSession._load` (retrieved by: code)
- [7] kilin/chat.py `ChatSession._load` (retrieved by: code)

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

KíLín decides which files are indexed and which are excluded through a combination of configuration settings and scanning logic. The exclusion rules are defined in `SKIP_DIRECTORIES`, `SKIP_FILES`, and `BLOCKED_SUFFIXES` in `repository.py` [6], which specify name-based exclusions for file types, directories, and sensitive suffixes. These lists are loaded at import time and influence which files are considered during scans. 

The actual selection of files during indexing is handled by `scan_repository` in `repository.py` [10], which uses parameters like `extensions`, `exclude_paths`, and `max_file_bytes` to filter files. Excluded paths are determined by `source_scan_exclusions` in `project_layout.py` [3], which excludes managed project folders when the source folder is the project root. Additionally, the `RAGSession._source_files` method in `rag.py` [9] reads a manifest of indexed files, derived from these scans. Files left out are either explicitly excluded via `exclude_paths`, blocked by suffixes, or skipped due to size/selection rules. The decision is thus governed by these configuration lists, scanning parameters, and project layout exclusions.

**Cited passages:**

- [3] kilin/project_layout.py `source_scan_exclusions` (retrieved by: semantic)
- [6] kilin/repository.py `repository` (retrieved by: semantic)
- [9] kilin/rag.py `RAGSession._source_files` (retrieved by: context_expansion)
- [10] kilin/repository.py `scan_repository` (retrieved by: context_expansion)

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

The question is transformed into the text embedded for semantic search through the `_format_query` method in `RAGSession` [1]. This method checks the project's profile to determine the query format. If the format is not plain, it retrieves a retrieval instruction specific to the project type (e.g., for `qwen3`) and applies it to the question. This formatted query is then embedded using the embedding model. The project type influences this formatting step, as the instruction is derived from the project type's configuration. For example, a literature project might use a different instruction than a code repository. The semantic search module (`semantic_search`) [9] uses this formatted query text to generate an embedding vector, which is compared to stored chunk vectors. Thus, the transformation depends on the project type via its query format and retrieval instruction settings.

**Cited passages:**

- [1] kilin/rag.py `RAGSession._format_query` (retrieved by: semantic)
- [9] kilin/retrievals/strategies/retrieval_semantic.py `semantic_search` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._format_query`, `format_qwen3_query`

- in evidence: ['RAGSession._format_query']
- cited: ['RAGSession._format_query']

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

When `ChatEngine.delete_conversation` is called, the saved conversation is **not immediately deleted** but **archived**. The conversation file is moved to a `deleted` directory within the conversation store, retaining its content. Specifically, the file is renamed with a suffix (e.g., `_2`, `_3`, etc.) to avoid overwriting existing files in the `deleted` folder [3]. This ensures the file remains recoverable. 

The active conversation is replaced by another if the deleted conversation was active, and the file lock is released to prevent conflicts [1]. The original file is not deleted but stored in the `deleted` directory, making it inaccessible from the main catalog. The `ConversationStore.archive` method handles this by moving the file and updating the registry, though the operation is not part of a filesystem transaction [6]. 

Thus, the conversation file is **not truly deleted** but **archived and renamed**, allowing for potential recovery. The process ensures durability and recoverability, as noted in the documentation [3].

**Cited passages:**

- [1] kilin/gui/engine.py `ChatEngine.delete_conversation` (retrieved by: code, lexical, semantic)
- [3] kilin/conversations.py `ConversationStore.archive` (retrieved by: lexical, semantic)
- [6] kilin/conversations.py `ConversationStore.archive` (retrieved by: semantic)

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

The `EmbeddingSession` computes vectors with the PyTorch backend by using the **last unmasked hidden-state token** from the model's output. During inference, the model's `last_hidden_state` is extracted, and the attention mask identifies the final valid token positions. These positions are used to gather the corresponding hidden states, which are then normalized using L2 normalization [11]. 

Before returning, vectors are validated via the `_check_vectors` function [13], which ensures:  
1. The count of vectors matches the expected input count.  
2. All vectors have consistent dimensions.  
3. All values are finite (not NaN/inf) and non-zero.  
This validation occurs after vector generation and is part of the embedding process [8]. 

The PyTorch-specific logic is implemented in `_embed_torch` [11], which handles tokenization, model inference, and post-processing. The backend’s caching is cleared via `_clear_backend_cache` [12] to free resources.

**Cited passages:**

- [8] kilin/embeddings.py `EmbeddingSession.embed_documents` (retrieved by: code)
- [11] kilin/embeddings.py `EmbeddingSession._embed_torch` (retrieved by: context_expansion)
- [12] kilin/embeddings.py `EmbeddingSession._clear_backend_cache` (retrieved by: context_expansion)
- [13] kilin/embeddings.py `_check_vectors` (retrieved by: context_expansion)

**Gold functions:** `EmbeddingSession._embed_torch`, `_check_vectors`, `EmbeddingSession.embed_documents`

- in evidence: ['EmbeddingSession._embed_torch', '_check_vectors', 'EmbeddingSession.embed_documents']
- cited: ['EmbeddingSession._embed_torch', '_check_vectors', 'EmbeddingSession.embed_documents']

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

## notebook_extraction (location, held_out_3)

**Question.** Where does KíLín turn a Jupyter notebook into searchable text, and which parts of a notebook are left out?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín converts Jupyter notebooks into searchable text via the `_notebook_units` function in `kilin/sources/extensions/sources_ipynb.py` [2]. This function extracts **code** and **Markdown cells** as `SourceUnit` objects, preserving cell numbers and line ranges but excluding outputs, execution state, attachments, and raw cells [3]. 

Parts of the notebook left out include:  
- **Outputs** (e.g., cell results, figures)  
- **Metadata** (e.g., cell tags, execution counts)  
- **Attachments** (e.g., embedded files)  
- **Raw cells** (non-code/non-Markdown cells, like raw JSON)  

The searchable text includes only the `source` field of valid code or Markdown cells, stripped of formatting and execution artifacts [4]. Line numbers are cell-local, not notebook-wide, ensuring context is preserved within each cell [6]. The `cell` attribute in `Evidence` records the original notebook cell number for provenance but is not part of the text itself [1]. 

This process ensures only human-readable content is indexed, while execution-related data is excluded.

**Cited passages:**

- [1] kilin/rag.py `Evidence` (retrieved by: semantic)
- [2] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)
- [3] kilin/sources/extensions/sources_ipynb.py `sources_ipynb` (retrieved by: semantic)
- [4] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)
- [6] kilin/rag.py `Evidence` (retrieved by: semantic)

**Gold functions:** `_notebook_units`, `extract_source_units`

- in evidence: ['_notebook_units', 'extract_source_units']
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

`atomic_write_text` and `atomic_create_text` both use temporary files to ensure files are never left half-written, but they differ in their approach and use cases. 

`atomic_write_text` [7] creates a temporary file in the same directory as the target path, writes the content, and then atomically replaces the target file with the temporary one using `temporary.replace(path)`. This ensures the target file is only updated once the entire content is written, preventing partial writes. If the write fails, the temporary file is deleted. 

`atomic_create_text` [1] also uses a temporary file but creates a hard link to the target path using `os.link(temporary, path)`, which is atomic. However, it fails if the target file already exists, ensuring no replacement. This method avoids a "check-then-replace" race condition by directly linking the temporary file to the target. If the write fails, the temporary file is deleted. 

The key difference is that `atomic_write_text` replaces an existing file, while `atomic_create_text` publishes a new file and fails if the target exists. Both use temporary files to ensure atomicity, but `atomic_create_text` avoids overwriting existing entries, whereas `atomic_write_text` ensures the target is fully written before replacement.

**Cited passages:**

- [1] kilin/atomic_writes.py `atomic_create_text` (retrieved by: code, lexical, semantic)
- [7] kilin/atomic_writes.py `atomic_write_text` (retrieved by: semantic)

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

The function `claim_passages` is called by `_verify` in `reference_check/tool.py` [2] and by `run` in `suggest_reference/tool.py` [10].  

In `_verify`, `claim_passages` is used to identify the most relevant passages from a cited file that relate to a claim. These passages are then passed to `verify_claim` to determine if the claim is supported, contradicted, or not supported by the cited text. The selected passages serve as evidence for evaluating the claim's validity [2].  

In `run`, `claim_passages` retrieves passages from candidate papers that are most relevant to a given statement. These passages are used by `judge_support` to assess whether the statement is supported by the cited work, aiding in suggesting appropriate references. The passages provide the basis for determining the strength of the statement's support [10].  

Both uses rely on `claim_passages` to filter and prioritize relevant text segments, ensuring claims or statements are evaluated against the most pertinent parts of cited works.

**Cited passages:**

- [2] kilin/tools/reference_check/tool.py `_verify` (retrieved by: code, lexical, semantic)
- [10] kilin/tools/suggest_reference/tool.py `run` (retrieved by: context_expansion)

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

KíLín identifies the embedding model that built a saved index by using a combination of **Hugging Face repository commits** for immutable models and **file digests** for local models. For Hugging Face snapshots, it checks the repository structure and commit hash [5]. For local models, it calculates a SHA-256 digest of the directory's files, including weights and tokenizer data, and caches this digest in a manifest file outside the model directory [1][9]. The manifest path is derived from the model's resolved location, ensuring uniqueness [3]. If the model changes, the digest updates, signaling a change. The `force_verify` flag forces rehashing even if metadata matches, ensuring accuracy [2][9]. This approach allows KíLín to verify model consistency without reprocessing files, relying on metadata for optimization but prioritizing content verification when needed.

**Cited passages:**

- [1] kilin/retrievals/model_identity.py `model_identity` (retrieved by: lexical, semantic)
- [2] kilin/retrievals/model_identity.py `embedding_model_identity` (retrieved by: semantic)
- [3] kilin/retrievals/model_identity.py `_manifest_path` (retrieved by: lexical)
- [5] kilin/retrievals/model_identity.py `_hub_snapshot_commit` (retrieved by: lexical)
- [9] kilin/retrievals/model_identity.py `embedding_model_identity` (retrieved by: context_expansion)

**Gold functions:** `embedding_model_identity`, `_hub_snapshot_commit`, `_directory_digest`

- in evidence: ['embedding_model_identity', '_hub_snapshot_commit']
- cited: ['embedding_model_identity', '_hub_snapshot_commit']

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

ConversationStore creates a branch by copying non-metadata fields from the parent's resume state, excluding metadata fields like `ConversationInfo` attributes [1]. The branch name is generated by appending "branch" and a sequential number to the parent's display name, ensuring uniqueness by checking against existing names in the registry [5]. To avoid overwriting, the code selects a filename that avoids existing paths at creation time, locking and checking each candidate path. If a collision is detected after selection, it raises an error instead of overwriting [1]. This process ensures the new branch's file is uniquely named and safely created without conflicting with existing files.

**Cited passages:**

- [1] kilin/conversations.py `ConversationStore.create` (retrieved by: lexical, semantic)
- [5] kilin/conversations.py `ConversationStore._branch_name` (retrieved by: lexical, semantic)

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

## python_units (location, evaluation)

**Question.** How does KíLín divide a Python file into the parts it indexes, and what happens to a file that is not valid Python?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín divides a Python file into parts by parsing valid Python code using `ast.parse`, identifying top-level definitions (classes, functions, etc.) and splitting the file into units around these structures. Invalid Python files are treated as a single safe unit without parsing [7]. For valid files, the code extracts context surrounding each definition (e.g., module context) and creates `SourceUnit` objects, ensuring structured indexing. Non-Python files or invalid syntax are not processed further and remain as unsplit units. This approach ensures valid code is segmented for analysis while handling errors gracefully.

**Cited passages:**

- [7] kilin/sources/extensions/sources_py.py `_python_units` (retrieved by: lexical)

**Gold functions:** `_python_units`, `_python_class_unit`

- in evidence: ['_python_units']
- cited: ['_python_units']

**Facts (tick those the answer states correctly):**

- [ ] The file is parsed with Python's ast module; nothing is executed.
- [ ] Each top-level function becomes one unit, and the code between definitions (imports, constants) becomes module-context units.
- [ ] A class becomes one class unit whose children are its methods and the class-level code between them.
- [ ] A file that does not parse (SyntaxError) becomes a single python_file unit.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín divides a Python file into parts by parsing it into an **Abstract Syntax Tree (AST)** using Python's built-in `ast` module. This process breaks the code into structured elements like functions, classes, variables, and statements, which are then indexed for features like autocompletion or navigation. The division is based on the syntactic structure of the code, not line breaks.

For files that are **not valid Python** (e.g., syntax errors), KíLín will **fail to parse** the file. This results in the entire file being ignored or an error being raised, depending on how the parser handles the invalid input. KíLín does not attempt to recover or index partial content from invalid files.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## pdf_text (location, evaluation)

**Question.** How does KíLín get the text out of a PDF file, and what happens when a PDF contains no text?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín extracts text from a PDF using `pdfminer.six` layout analysis in `read_source_file` [11], which processes each page's text boxes and lines, sorting them by column order when applicable. If `pdfminer.six` fails to parse the PDF, it falls back to `pypdf`'s plain extraction [9]. The extracted text is cleaned of malformed Unicode and blank lines [10]. When a PDF contains no text, `read_source_file` raises a `ValueError` indicating the PDF has no extractable text and requires OCR [11]. If the PDF is encrypted or contains non-extractable content, similar errors are raised [11]. The `pdf_text` function also returns an empty string for non-PDF files or unreadable PDFs [6].

**Cited passages:**

- [6] kilin/tools/_shared/references.py `pdf_text` (retrieved by: semantic)
- [9] kilin/sources/extensions/sources_pdf.py `_layout_pages` (retrieved by: context_expansion)
- [10] kilin/sources/extensions/sources_pdf.py `_clean_extracted_text` (retrieved by: context_expansion)
- [11] kilin/sources/extensions/sources_pdf.py `read_source_file` (retrieved by: context_expansion)

**Gold functions:** `read_source_file`, `_layout_pages`, `extract_source_units`

- in evidence: ['read_source_file', '_layout_pages', 'extract_source_units']
- cited: ['read_source_file', '_layout_pages']

**Facts (tick those the answer states correctly):**

- [ ] Pages are extracted with pdfminer.six layout analysis; if that fails or gives a different page count, pypdf's plain extraction is used.
- [ ] Encrypted or unreadable PDFs raise an error.
- [ ] The text is cleaned (invalid Unicode surrogates, unmapped glyphs, trailing spaces, repeated blank lines).
- [ ] A PDF with no extractable text raises an error saying scanned PDFs need OCR.
- [ ] Pages are joined with a page break, and each non-empty page becomes one unit labelled with its page number.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín uses the **PyMuPDF** (also known as **fitz**) library to extract text from PDF files. Here's how it works:

1. **Text Extraction Process**:  
   - KíLín leverages PyMuPDF's ability to parse the PDF's internal text layers. It reads the text content from each page by accessing the PDF's "text" stream, which is part of the PDF's structure. This includes extracting text, fonts, and layout information.

2. **Handling PDFs with No Text**:  
   - If a PDF contains no text (e.g., it consists solely of images, vector graphics, or other non-text elements), PyMuPDF will return an **empty string** or `None` for the text extraction. KíLín will propagate this result directly, meaning the extracted text will be empty or absent.

3. **Fallback or Special Cases**:  
   - KíLín does not inherently add fallback text or guess content in such cases. If no text is present, the output will reflect this absence. However, if the PDF contains non-text elements (e.g., images), KíLín might still extract metadata (e.g., page numbers, author, title) if available, depending on the PDF's structure.

In summary, KíLín extracts text by parsing the PDF's text stream, and if no text exists, the result will be empty. For purely graphical PDFs, additional tools (e.g., OCR) would be required to extract content, but KíLín does not support OCR by default.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## call_relations (location, evaluation)

**Question.** How does KíLín work out which functions call which other functions in Python code, and what happens when a called name could refer to several functions?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín determines function calls by analyzing a **CodeGraph** built from Python code, which tracks relationships like `calls` between entities. The `_ranked_callees` function [1] uses this graph to identify callees, prioritizing those with names sharing words with the query, closer call steps, and functions that perform more work. The graph is constructed via AST traversal, collecting calls during code parsing [8]. When a called name refers to multiple functions, KíLín ranks them using the query text, call depth, and function relevance (e.g., functions calling more repository functions are preferred). Ambiguity is resolved by these criteria, ensuring the most contextually relevant functions are selected first. If no explicit calls are found, the system defaults to ranking based on shared keywords and call proximity.

**Cited passages:**

- [1] kilin/retrievals/context_expansion.py `_ranked_callees` (retrieved by: lexical, semantic)
- [8] kilin/retrievals/code_graph/extensions/code_graph_py.py `_Visitor` (retrieved by: semantic)

**Gold functions:** `extract_code_relations`, `_Visitor.visit_Call`, `_resolve_name`

- in evidence: []
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] Each Python file is parsed with ast and walked; the code is never imported or run, and a file with a syntax error contributes no relations.
- [ ] For every call, the called expression is turned into a dotted name (a plain name or attribute chain) and recorded as a calls relation from the enclosing function, method, class, or module.
- [ ] The name is resolved by its last component against the known entities: a unique match gives the qualified name.
- [ ] An ambiguous name is kept as written (unless only one candidate differs from the name itself), so no call is guessed; calls through non-static expressions are not recorded.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python code analysis tool, determines function call relationships through **static analysis** of the source code. Here's how it works and how it handles ambiguous references:

---

### **1. How KíLín Identifies Function Calls**
- **AST Parsing**: KíLín parses Python code into an **Abstract Syntax Tree (AST)**, which represents the code's structure. This allows it to identify function definitions and calls explicitly.
- **Call Graph Construction**: It traverses the AST to detect **function calls** (e.g., `func()` or `var()` where `var` refers to a function). For each call, it records the **caller** (the function or variable making the call) and the **callee** (the function being called).
- **Scope Resolution**: KíLín uses **scope information** (e.g., local, global, or module-level variables) to resolve names to their definitions. This helps disambiguate function references within nested scopes.

---

### **2. Handling Ambiguous References**
When a called name could refer to **multiple functions** (e.g., due to dynamic assignments or overlapping scopes), KíLín behaves as follows:
- **Ambiguity Detection**: If a name is assigned to multiple functions (e.g., `f = some_function()` and later `f = another_function()`), KíLín may flag this as a potential issue during analysis.
- **Contextual Resolution**: If the context (e.g., the scope or type) allows disambiguation, KíLín resolves the reference. For example, if a variable is assigned a function in a specific scope, it uses that scope to determine the callee.
- **Reporting Ambiguity**: If the ambiguity cannot be resolved statically (e.g., dynamic function calls like `func = getattr(obj, 'method')`), KíLín may report the ambiguity as a **potential bug** or **uncertain call path**, depending on its configuration.

---

### **3. Example Scenarios**
- **Direct Call**:  
  ```python
  def add(a, b):
      return a + b

  def multiply(a, b):
      return a * b

  result = add(2, 3)  # KíLín identifies `add` as the caller and `add` as the callee.
  ```
- **Ambiguous Reference**:  
  ```python
  def foo():
      print("Hello")

  def bar():
      print("World")

  func = foo  # func refers to foo
  func()      # KíLín resolves `func` to `foo` here.
  ```
  If `func` is reassigned later (e.g., `func = bar`), KíLín may flag the call as ambiguous unless the context is clear.

---

### **4. Limitations**
- **Dynamic Calls**: KíLín cannot resolve calls that depend on runtime values (e.g., `getattr(obj, 'method')` or `locals()['func']`), as these require dynamic execution.
- **Overlapping Scopes**: If multiple functions share the same name in different scopes (e.g., a function and a class with the same name), KíLín may report ambiguity unless the scope is explicitly resolved.

---

In summary, KíLín uses static analysis to map function calls and resolves ambiguities using scope and context. However, it may report unresolved ambiguities as potential issues, especially in cases involving dynamic or overlapping references.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## keyword_scoring (location, evaluation)

**Question.** How are passages scored in KíLín's keyword search, and how are code identifiers broken into searchable words?

**Intent:** code_explanation

**Answer with retrieval:**

Passages in KíLín's keyword search are scored using **TF-IDF cosine similarity** between query terms and indexed chunks, as implemented in `lexical_search` [11]. Query terms are derived from the input text via `lexical_terms`, which splits text into lowercase unigrams and bigrams, handling code identifiers by splitting camelCase and underscore-separated terms (e.g., `scanRepository` → `scan`, `repository`) [6]. Code identifiers are further processed in `_identifier_words` [9], which splits identifiers into lowercase words by inserting spaces before uppercase letters and replacing underscores, ensuring terms like `route_question` become `route` and `question`. This decomposition allows TF-IDF vectors to capture both plain text and code-specific terms, enabling ranking by cosine similarity. The scoring does not require a positive-score threshold, allowing zero-score matches if needed [11].

**Cited passages:**

- [6] kilin/retrievals/strategies/retrieval_lexical.py `lexical_terms` (retrieved by: lexical)
- [9] kilin/retrievals/context_expansion.py `_identifier_words` (retrieved by: context_expansion)
- [11] kilin/retrievals/strategies/retrieval_lexical.py `lexical_search` (retrieved by: context_expansion)

**Gold functions:** `lexical_terms`, `lexical_search`, `_tfidf_vector`

- in evidence: ['lexical_terms', 'lexical_search']
- cited: ['lexical_terms', 'lexical_search']

**Facts (tick those the answer states correctly):**

- [ ] Text is split into lowercase words of letters, digits, and underscores; identifiers also contribute their camelCase and snake_case parts longer than one character.
- [ ] Adjacent words also form two-word terms; there is no stemming or stop-word removal.
- [ ] Each passage and the query become TF-IDF vectors with sublinear term frequency (1 + log f) and smoothed inverse document frequency, normalized to unit length.
- [ ] Passages are ranked by cosine similarity, ties by chunk identifier, with no minimum score.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python package for searching through code, employs specific strategies for scoring keyword matches and tokenizing code identifiers. Here's a breakdown of the process:

### **Passage Scoring in Keyword Search**
1. **Relevance Scoring**: KíLín likely uses a relevance-based scoring system, where the score is determined by factors such as:
   - **Keyword Frequency**: The number of times a keyword appears in the code.
   - **Contextual Importance**: Keywords appearing in function names, variable names, or other syntactically significant locations may receive higher scores.
   - **Proximity**: Matches that are closer to each other or to specific code elements (e.g., function definitions) might be prioritized.
   - **Inverse Document Frequency (IDF)**: Less common keywords might be weighted more heavily to avoid trivial matches.

2. **Search Context**: The scoring may also consider the context in which keywords appear, such as within strings, comments, or code blocks, to differentiate between meaningful matches and accidental overlaps.

### **Code Identifiers to Searchable Words**
KíLín tokenizes code identifiers (e.g., variable names, function names) into searchable terms by:
1. **Handling Naming Conventions**:
   - **CamelCase** (e.g., `myVariable`): Split into `my` and `Variable` by inserting spaces between lowercase and uppercase letters.
   - **snake_case** (e.g., `my_variable`): Split into `my` and `variable` by splitting on underscores.
   - **PascalCase** (e.g., `MyClass`): Split into `My` and `Class` (or treat as a single term if contextually relevant).

2. **Tokenization Process**: 
   - A tokenizer identifies and splits identifiers into individual words, which are then indexed for search. This allows users to search for terms like `my` or `variable` even if they appear as part of a larger identifier.

3. **Case Sensitivity**: KíLín may normalize case (e.g., treating `MyClass` and `myClass` as the same) or preserve case depending on configuration, depending on the intended use case.

### **Key Considerations**
- **Customization**: Users may configure tokenization rules or scoring weights to adapt to specific codebases or search priorities.
- **Efficiency**: The tokenization and scoring mechanisms are optimized for performance, especially when dealing with large codebases.

For precise details, refer to KíLín's documentation or source code, as the exact implementation may vary.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## follow_up_query (mechanism, evaluation)

**Question.** Explain how refers_back decides that a question depends on the earlier conversation, and how search_text uses that to build the search query.

**Intent:** code_explanation

**Answer with retrieval:**

The `refers_back` function determines if a question depends on earlier conversation by checking for specific linguistic patterns, such as pronouns ("it"), ordinal references ("the second one"), pointers back ("you said"), or short questions (under a word limit) [2]. If these criteria are met, it returns `True`, indicating the question references prior turns. 

The `search_text` function uses this information to build a search query by appending relevant earlier conversation turns to the current question. If `refers_back` is `True` and there are `related_turn_ids`, it gathers the text from those earlier turns, oldest first, and combines them with the current question. This ensures the search query includes context from prior discussions, enhancing relevance [1]. If `refers_back` is `False`, the query remains just the current question. This mechanism allows the search to incorporate historical context when the question logically depends on it.

**Cited passages:**

- [1] kilin/answering.py `search_text` (retrieved by: lexical, semantic)
- [2] kilin/answering.py `refers_back` (retrieved by: lexical, semantic)

**Gold functions:** `refers_back`, `search_text`

- in evidence: ['refers_back', 'search_text']
- cited: ['refers_back', 'search_text']

**Facts (tick those the answer states correctly):**

- [ ] refers_back is true for messages of at most two words or when a configured pattern matches: pronouns (it, they, this), ordinal references (the second one), pointers back (you said, earlier, so far), or elliptical openings (and, what about).
- [ ] search_text returns the question unchanged unless the routing decision refers back and lists related turns.
- [ ] Otherwise it appends the questions of the related turns, with whitespace normalized, after the new question.
- [ ] Model-written text (earlier answers) is never added to the query.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `refers_back` function in the KíLín Python package determines if a question depends on earlier conversation by analyzing the current question's text for linguistic cues (e.g., pronouns, references to entities, or topics mentioned previously). It checks whether the question implicitly or explicitly connects to information from the conversation history, such as specific entities, topics, or prior answers. For example, if the user asks, "What's the weather like today?" and then follows up with "Is it going to rain tomorrow?" the second question refers back to the weather topic discussed earlier.

Once `refers_back` identifies that the question is context-dependent, the `search_text` function uses this information to build a more targeted search query. It combines the current question with relevant context from the conversation history to refine the search. This ensures the search is focused on the specific topic or entity mentioned earlier, improving the relevance of results. For instance, if the prior conversation was about a product, `search_text` might prioritize information related to that product, making the search more efficient and aligned with the user's ongoing dialogue. 

If you need precise implementation details, consult the KíLín documentation or source code, as the exact logic may vary based on the package's design.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## download_progress (mechanism, evaluation)

**Question.** Explain how download_model reports download progress to its caller, and where the downloaded model files are stored.

**Intent:** code_explanation

**Answer with retrieval:**

The `download_model` function reports download progress to its caller via a `progress_callback` parameter, which is a callable that receives completed and expected byte counts during the download. This callback is configured using a custom `tqdm`-like class (`_progress_tqdm`) that wraps the callback, ensuring progress updates are sent to the frontend. The progress is tracked by the Hugging Face `snapshot_download` utility, which aggregates byte counts and triggers the callback with `current` (bytes transferred) and `byte_total` (expected total bytes) [1][7][11]. 

Downloaded model files are stored in Hugging Face's shared cache, determined by the `huggingface_hub` library. The exact path is returned as a `Path` object by `download_model`, which delegates to `snapshot_download`. The cache location is not explicitly configurable in this function but is managed by Hugging Face's defaults. If a model is already cached, it reuses existing files without redownloading [1][12]. The returned `Path` points to the directory containing the model files, which includes necessary files like `config.json` for completeness [13].

**Cited passages:**

- [1] kilin/huggingface.py `download_model` (retrieved by: code, lexical, semantic)
- [7] kilin/huggingface.py `download_model` (retrieved by: code)
- [11] kilin/huggingface.py `_progress_tqdm` (retrieved by: context_expansion)
- [12] kilin/huggingface.py `cached_model_snapshot` (retrieved by: context_expansion)
- [13] kilin/huggingface.py `cached_model_snapshot` (retrieved by: context_expansion)

**Gold functions:** `download_model`, `_progress_tqdm`

- in evidence: ['download_model', '_progress_tqdm']
- cited: ['download_model', '_progress_tqdm']

**Facts (tick those the answer states correctly):**

- [ ] download_model calls huggingface_hub.snapshot_download, so files go to (or are reused from) Hugging Face's shared cache; KíLín does not copy them.
- [ ] With a progress callback, it passes a progress class built by _progress_tqdm; without one, the Hub shows its own progress.
- [ ] That class disables terminal output and reports only the aggregate byte bar as (completed bytes, total bytes).
- [ ] Because a disabled tqdm stops counting, the class keeps the count itself.
- [ ] It returns the snapshot path and does not load the model.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `download_model` function in the KíLín Python package reports download progress to its caller through a callback mechanism. It allows users to pass a custom callback function that is invoked periodically during the download process. This callback typically receives parameters such as the current download progress (e.g., percentage completed), the total size of the download, and the current download speed. This enables users to integrate progress tracking into their applications, such as updating a GUI or logging status messages.

By default, the downloaded model files are stored in a user-specific cache directory. The exact path is typically something like `~/.kilin/models` or a subdirectory within a system-wide cache location (e.g., `~/.cache/kilin/models`). The function may also accept an optional parameter to specify a custom download directory, allowing users to override the default storage location. The files are often saved with a unique identifier (e.g., a hash of the model name) to avoid conflicts and ensure proper versioning. 

If you need precise details, consult the KíLín documentation or source code for confirmation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## reference_matching (mechanism, evaluation)

**Question.** Explain how match_reference decides which project file a reference entry points to, and when a match is only probable.

**Intent:** code_explanation

**Answer with retrieval:**

The `match_reference` function determines which project file a reference entry points to by evaluating matches against candidate files (papers) using criteria like DOI, author, and content similarity. It uses the `_candidate` helper function [9][10] to assess each paper. A **found match** occurs if the entry contains a DOI matching the paper's DOI, or if the first author and first page match exactly. If the entry has too few words to compare, only DOI or citation matches are considered [9]. For content-based matches, the function calculates a score based on word overlap between the entry and the paper's title or first page. A **probable match** is returned if the score meets a lower threshold but falls short of a definitive match, or if the first author differs [10]. 

A match is marked **probable** when the score is high but not sufficient for a "found" status (e.g., partial word overlap or ambiguous authorship). If multiple entries match the same file, only the best match (by score and criteria) is retained as "found," while others are labeled "probable" with reasons like conflicting authors [6][8]. This ensures prioritization of exact matches while flagging potential ambiguities.

**Cited passages:**

- [6] kilin/tools/reference_check/_matching.py `match_reference` (retrieved by: code, lexical, semantic)
- [8] kilin/tools/reference_check/_matching.py `settle_duplicates` (retrieved by: semantic)
- [9] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)
- [10] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)

**Gold functions:** `match_reference`, `_candidate`

- in evidence: ['match_reference', '_candidate']
- cited: ['match_reference', '_candidate']

**Facts (tick those the answer states correctly):**

- [ ] A file's DOI found in the entry gives a found match.
- [ ] So does the entry's first author appearing in the file's authors or first page together with the same journal citation.
- [ ] Otherwise the score is the share of the file's title words found in the entry (or of the entry's words found on the first page when there is no title), compared with found and probable thresholds.
- [ ] A high score with a different first author, or a lower score with the same first author, is only probable.
- [ ] Found matches come before probable ones, then higher scores; other matches are listed as alternatives; an entry with too few distinctive words is unreadable, and one with no candidate is missing.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `match_reference` function in the KíLín Python package is designed to resolve which project file (e.g., a `.tex` file) a reference entry (e.g., a bibliography entry) points to. It uses a combination of exact and heuristic matching strategies to determine the correct file, and it can flag matches as "probable" when ambiguity exists. Here's how it works:

---

### **1. Exact Matching**
- **Key Fields**: KíLín compares the reference entry's metadata (e.g., title, author, year, journal, volume, etc.) with the metadata of project files (e.g., `.tex` files) in the project directory.
- **Exact Match**: If the reference entry's fields exactly match a project file's metadata, the match is considered **definite**. For example, if a reference entry has the title "The Art of Computer Programming" and a project file has the same title, KíLín will map the reference to that file.

---

### **2. Fuzzy/Heuristic Matching**
- **Ambiguity Handling**: When exact matches are not found, KíLín uses **fuzzy matching** or **heuristics** to find the best possible match. This might involve:
  - **Partial Matches**: Comparing substrings of titles, authors, or other fields.
  - **Similarity Scores**: Calculating a similarity score (e.g., using algorithms like Levenshtein distance) between the reference entry and project file metadata.
  - **Prioritization**: Prioritizing matches based on fields with higher importance (e.g., title over author) or other rules (e.g., most recent file, longest match).

---

### **3. Probable Matches**
- **Ambiguity Flag**: If multiple project files have similar metadata (e.g., two files with nearly identical titles), KíLín may flag the match as **probable**. This occurs when:
  - The similarity score is high but not 100%.
  - Multiple files could plausibly match the reference entry.
  - No definitive metadata fields (e.g., unique title or author) are present.
- **User Intervention**: In such cases, the user is typically prompted to manually verify or resolve the ambiguity (e.g., by selecting the correct file or providing additional metadata).

---

### **4. Contextual Factors**
- **Project File Structure**: KíLín may also consider the structure of the project files (e.g., whether a file is a main document, a bibliography, or a section) to improve accuracy.
- **Configuration**: The behavior of `match_reference` can be influenced by configuration settings, such as thresholds for similarity scores or prioritization rules.

---

### **When a Match is Only Probable**
A match is considered **only probable** when:
- There is **no exact match**.
- The **similarity score** is high but not definitive.
- **Multiple files** could plausibly match the reference entry.
- The **metadata is incomplete or ambiguous** (e.g., missing author or title).

In such cases, KíLín may suggest the match but require the user to confirm or refine the mapping.

---

### **Summary**
`match_reference` combines exact and heuristic matching to resolve references to project files. Matches are definitive when metadata aligns precisely, but they are marked as probable when ambiguity exists (e.g., due to partial matches or similar metadata). This ensures accuracy while allowing users to resolve uncertainties manually. For precise details, consult the KíLín documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## term_hits_callers (dependency, evaluation)

**Question.** Explain which functions call term_hits, and what each of them does with the passages it returns.

**Intent:** code_explanation

**Answer with retrieval:**

The functions that call `term_hits` are `batch_review.run`, `find_fact.run`, and `find_passages.run`.  

1. **`batch_review.run`** [7]: Calls `term_hits` to identify passages mentioning the question's exact terms. If terms are found, it uses the returned `hits` (indices of relevant passages) to structure a quick review of those passages, filtering the processing to only those containing the terms. This helps prioritize relevant content for the model.  

2. **`find_fact.run`** [15]: Uses `term_hits` to determine which passages mention the question's terms. If terms exist, `hits` are used to focus on those passages for fact-checking, improving efficiency by avoiding irrelevant text. If no terms are found, it defaults to ranked passages.  

3. **`find_passages.run`** [17]: Invokes `term_hits` to filter sentences or chunks containing the question's terms. If terms are present, it processes only those passages; otherwise, it uses ranked passages. This narrows the scope of analysis to terms explicitly mentioned in the question.  

Each function leverages `term_hits` to refine the set of passages processed, either for speed (quick versions) or precision (targeted analysis). The returned `hits` indices are used to subset the input `passages` or `chunks`, aligning with the tool's goal of focusing on relevant content.

**Cited passages:**

- [7] kilin/tools/batch_review/tool.py `run` (retrieved by: lexical)
- [15] kilin/tools/find_fact/tool.py `run` (retrieved by: context_expansion)
- [17] kilin/tools/find_passages/tool.py `run` (retrieved by: context_expansion)

**Gold functions:** `run`, `run`, `run`

- in evidence: ['run', 'run', 'run']
- cited: ['run', 'run', 'run']

**Facts (tick those the answer states correctly):**

- [ ] term_hits asks the router model for the question's exact terms and returns them with the passages that mention any of them (whole words, ignoring case).
- [ ] batch_review's run uses them for its quick version: only the passages that mention the terms, read in parts.
- [ ] find_fact's run uses them for its quick version: only the parts that contain a hit, per file.
- [ ] find_passages' run uses them for its quick version: the sentences of those passages, found without the model.
- [ ] Without terms, each falls back to the best-ranked passages.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is designed for information retrieval tasks, leveraging Elasticsearch under the hood. While specific implementation details may vary, here's a general explanation of how functions might interact with `term_hits` and what they do with the returned passages:

---

### **Functions That Call `term_hits`**
1. **`search` or `search_terms`**  
   - These are likely high-level functions that accept a query (e.g., a string or list of terms) and internally use `term_hits` to retrieve documents containing the specified terms.  
   - Example: `search("climate change")` might break the query into terms like "climate" and "change," then call `term_hits` for each term to gather matching documents.

2. **`term_hits` (lower-level function)**  
   - This function directly queries Elasticsearch for documents containing a specific term. It returns a list of "hits" (documents) that match the term.  
   - Example: `term_hits("climate")` would return all documents with the word "climate" in their content.

---

### **What Each Function Does with the Returned Passages**
1. **`term_hits`**  
   - **Input**: A term (e.g., "climate").  
   - **Output**: A list of documents (hits) that contain the term. Each hit typically includes metadata (e.g., document ID, score) and the raw text snippet matching the term.  
   - **Processing**: The function may return the raw hits or pre-processed snippets (e.g., highlighting the matched term).

2. **High-level functions (e.g., `search`)**  
   - **Aggregation**: Combines results from multiple terms (e.g., using `term_hits` for each term) and merges them into a unified list of documents.  
   - **Filtering/Sorting**: Removes duplicates, sorts documents by relevance (e.g., using Elasticsearch's scoring), or applies filters (e.g., limiting results to a specific number).  
   - **Formatting**: Returns structured results, such as a list of documents with metadata (e.g., title, score) and highlighted passages.

---

### **Example Workflow**
1. User calls `search("climate change")`.  
2. The function splits the query into terms: ["climate", "change"].  
3. For each term, `term_hits` is called to retrieve matching documents.  
4. Results are merged, duplicates are removed, and documents are sorted by relevance.  
5. The final output includes a list of documents with their matching passages and scores.

---

### **Notes**
- The exact behavior depends on the KíLín implementation. For precise details, consult the [KíLín documentation](https://kilin.readthedocs.io/).  
- If `term_hits` is not explicitly documented, it may be an internal utility function used by higher-level methods.  

If you have access to the package's source code or documentation, checking the `search` and `term_hits` functions directly would clarify the exact workflow.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## explicit_imports_callers (dependency, evaluation)

**Question.** Explain which functions call include_explicit_imports, and why each of them needs it.

**Intent:** code_explanation

**Answer with retrieval:**

The function `include_explicit_imports` is called by `RAGSession._prepare_source` [2] and `RAGSession._prepare_retrieval` [9].  

In `RAGSession._prepare_source`, it is called to refine the scan report by adding explicitly imported files, ensuring they are included in the code graph and embeddings even if they don’t meet default selection criteria (e.g., file extensions or size limits). This is critical for accurate indexing of dependencies.  

In `RAGSession._prepare_retrieval`, it is called to update the scan report before building the retrieval index, ensuring explicitly imported files are considered during indexing. This guarantees that files directly referenced in the project are prioritized, improving the relevance of search results.  

Both calls rely on `include_explicit_imports` to adjust the scan report by incorporating files explicitly recorded in `sources.json`, which are essential for reconstructing the project’s import structure and ensuring no critical dependencies are missed.

**Cited passages:**

- [2] kilin/rag.py `RAGSession._prepare_source` (retrieved by: lexical)
- [9] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._prepare_retrieval`, `RAGSession._prepare_source`

- in evidence: ['RAGSession._prepare_retrieval', 'RAGSession._prepare_source']
- cited: ['RAGSession._prepare_retrieval', 'RAGSession._prepare_source']

**Facts (tick those the answer states correctly):**

- [ ] Both scan the project's source folder with scan_repository and then call include_explicit_imports on the report.
- [ ] _prepare_retrieval uses the scan to build the source manifest and compare it with the saved index, to decide whether to reuse or rebuild it.
- [ ] _prepare_source uses it (when no report is supplied) to choose which files to read and split into chunks.
- [ ] include_explicit_imports adds files the user imported one by one (recorded in sources.json) even when their suffix or name would otherwise be skipped; size limits and link safeguards still apply, and imported folders keep the normal rules.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is not a widely known or standard library, and I do not have specific information about its internal implementation, including functions like `include_explicit_imports`. If this is a custom or niche package, its documentation or source code would need to be consulted for precise details.

In general, a function like `include_explicit_imports` might be used in code analysis tools to distinguish between **explicit** (directly written) and **implicit** (e.g., inferred or dynamic) imports. For example:

1. **`parse_imports()`**  
   - **Why it needs `include_explicit_imports`:** To filter or flag only the imports explicitly written in the code, excluding dynamically generated or inferred ones. This ensures accurate analysis of the code's structure.

2. **`generate_report()`**  
   - **Why it needs `include_explicit_imports`:** To include only explicitly declared imports in the report, avoiding misleading information from implicit imports (e.g., those resolved at runtime).

3. **`check_imports()`**  
   - **Why it needs `include_explicit_imports`:** To enforce coding standards, ensuring that all imports are explicitly stated rather than relying on implicit resolution (e.g., `from __future__` imports or dynamic `importlib` calls).

For exact details, refer to the KíLín package's documentation or source code. If you can provide more context or specific code snippets, I can offer a more tailored explanation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## extension_callers (dependency, evaluation)

**Question.** Explain which functions call normalize_extensions, and what each of them checks with it.

**Intent:** code_explanation

**Answer with retrieval:**

The function `normalize_extensions` is called by three functions: `_parse_project_type` [4][10], `_read_profile` [8][9], and `scan_repository` [3][7].  

- `_parse_project_type` checks that project type extensions (e.g., `.pdf`) are valid literal suffixes, ensuring they start with a period and conform to the regex pattern in `normalize_extensions` [4][10].  
- `_read_profile` validates project configuration extensions (e.g., `.py`) similarly, rejecting invalid formats and ensuring they match the required grammar [8][9].  
- `scan_repository` normalizes user-specified extensions (e.g., `['.md', '.txt']`) to ensure consistency across the repository scanner, filtering out invalid patterns like wildcards [3][7].  

Each caller ensures extensions adhere to the same validation rules, such as lowercase formatting, no whitespace, and proper suffix syntax, to maintain uniformity in file selection logic across the repository.

**Cited passages:**

- [3] kilin/repository.py `scan_repository` (retrieved by: code)
- [4] kilin/project_type.py `_parse_project_type` (retrieved by: lexical)
- [7] kilin/repository.py `scan_repository` (retrieved by: code)
- [8] kilin/project.py `_read_profile` (retrieved by: lexical)
- [9] kilin/project.py `_read_profile` (retrieved by: context_expansion)
- [10] kilin/project_type.py `_parse_project_type` (retrieved by: context_expansion)

**Gold functions:** `_read_profile`, `_parse_project_type`, `scan_repository`

- in evidence: ['_read_profile', '_parse_project_type', 'scan_repository']
- cited: ['_read_profile', '_parse_project_type', 'scan_repository']

**Facts (tick those the answer states correctly):**

- [ ] normalize_extensions lowercases the suffixes, adds a leading dot, rejects wildcards, whitespace, and non-text values, and removes duplicates.
- [ ] _read_profile checks the extensions setting of a project's profile.ini.
- [ ] _parse_project_type checks the extensions declared by a project-type JSON file.
- [ ] scan_repository checks the extensions it is asked to select before scanning.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `normalize_extensions` function in the KíLín Python package is used to standardize file extensions, ensuring consistency in how files are processed and checked. While the exact implementation details may vary, here's a general explanation of how it might be used in practice:

### Functions That Call `normalize_extensions`
1. **Main Checker Functions**  
   - Functions like `check_pep8` or `check_style` might call `normalize_extensions` to ensure file extensions (e.g., `.py`, `.PY`) are standardized before applying checks. This ensures that rules like "only check `.py` files" are applied consistently.

2. **Configuration Setup Functions**  
   - When parsing configuration files (e.g., `kilin.ini`), functions that process extension lists (e.g., `load_config`) might use `normalize_extensions` to standardize user-specified extensions (e.g., converting `.PY` to `.py`).

3. **File Filtering Functions**  
   - Functions that determine which files to include/exclude based on extensions (e.g., `filter_files`) might call `normalize_extensions` to ensure the file's actual extension matches the expected format.

### What Each Function Checks
- **Consistency of Extensions**: Ensures file extensions match expected formats (e.g., `.py` instead of `.PY` or `.PY`).
- **Relevance to Checkers**: Validates that files with normalized extensions are compatible with the rules being applied (e.g., only `.py` files are checked for PEP 8 compliance).
- **Configuration Alignment**: Ensures user-specified extensions in configuration files are normalized to match the internal representation used by KíLín.

If you're using KíLín, the `normalize_extensions` function is typically part of the internal logic to maintain consistency in how files are processed, but its exact usage depends on the package's implementation. For precise details, refer to the KíLín source code or documentation.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
