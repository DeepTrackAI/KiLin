# Code example: grading worksheet

Indexed revision: ba57213a1 (HEAD)

## routing (mechanism, development)

**Question.** Explain how route_question assigns a question to an intent, and what happens when the router reply cannot be used.

**Intent:** code_explanation

**Answer with retrieval:**

The `route_question` function assigns a question to an intent by first checking if the question matches any intent's predefined shortcuts. If a shortcut matches, it immediately returns a `RouteDecision` with that intent and marks it as standalone [1]. If no shortcut matches, it constructs a prompt using `_router_prompt`, which includes the project's intents and relevant conversation history, then queries a router model via `side_reply` to classify the question [3]. The model's response is parsed by `_parse_route`, which validates the JSON output and ensures the intent exists in the project. If the reply is invalid (e.g., malformed JSON or unknown intent), it falls back to the project's default intent, setting a `fallback` message explaining the failure [9]. When the router reply cannot be used, the function defaults to the project's default intent, ensuring the question is still addressed, with `fallback` indicating the router's failure [3]. This fallback mechanism prevents the system from breaking when the router model fails or returns unusable results.

**Cited passages:**

- [1] kilin/answering.py `route_question` (retrieved by: code, semantic)
- [3] kilin/answering.py `route_question` (retrieved by: code)
- [9] kilin/answering.py `_parse_route` (retrieved by: context_expansion)

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

The `RAGSession` decides to reuse or rebuild retrieval indexes by comparing saved metadata with current project configuration and source state. It first checks if the saved indexes' metadata (e.g., embedding model, chunking parameters, source files) matches the current setup [5]. If they match, indexes are reused [10]. If not, or if source files have changed, indexes are rebuilt [11]. The metadata is generated using `_index_metadata`, which captures configured settings like `embedding_model` and `query_format` [6]. Additionally, the system verifies the loaded model's identity against saved indexes to ensure compatibility [10]. If discrepancies exist (e.g., configuration changes, updated source files), indexes are rebuilt, and the status is set to "rebuilt" [5]. This ensures indexes reflect the current project state without unnecessary recomputation.

**Cited passages:**

- [5] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: semantic)
- [6] kilin/rag.py `RAGSession._index_metadata` (retrieved by: context_expansion)
- [10] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)
- [11] kilin/rag.py `RAGSession._source_files` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._prepare_retrieval`, `load_indexes`

- in evidence: ['RAGSession._prepare_retrieval']
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

The `build_semantic_index` function embeds chunks by using the provided `embedding_session` to generate vector representations. It processes chunks in batches, as specified by the `batch_size` parameter, and embeds each batch's text (either from the chunks or provided `texts`). The resulting vectors are validated for correctness (e.g., non-finite values, zero norms) and stored in a `SemanticIndex` object, which pairs chunks with their embeddings [1][2].  

The `save_indexes` and `load_indexes` functions protect the saved index through integrity checks. `save_indexes` serializes the index data (including checksums) to a JSON file, ensuring the source fingerprint matches the current repository state before saving [9]. `load_indexes` verifies the checksum, validates the structure of the saved data, and ensures the embedding model identity matches the current model, preventing incompatible or corrupted data from being loaded [7]. These steps ensure the saved index remains consistent and secure against tampering or version mismatches.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code, semantic)
- [2] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code)
- [7] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
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

The function `strip_thinking` is called by three functions: `history_messages` [3], `side_reply` [5], and `turn_representation` [4]. Each uses it to remove the model's reasoning steps from the output, ensuring only the final answer is retained. 

- `history_messages` strips thinking text to generate a clean history of user and assistant messages, excluding internal reasoning.  
- `side_reply` removes thinking text to deliver a direct reply without the model's internal process, as side questions use greedy sampling.  
- `turn_representation` eliminates thinking to focus on the answer's content for summaries, ensuring citations and sources are based solely on the final output.  

These calls ensure outputs are concise, free from model-specific artifacts, and aligned with user-facing requirements.

**Cited passages:**

- [3] kilin/answering.py `history_messages` (retrieved by: code)
- [4] kilin/answering.py `turn_representation` (retrieved by: code)
- [5] kilin/answering.py `side_reply` (retrieved by: code)

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

The `RAGSession` methods that call `hybrid_search` are `_rank_everything` [2] and `_retrieve` [6]. Both use the fused ranking generated by `hybrid_search` to combine semantic, lexical, and code results. 

In `_rank_everything`, `hybrid_search` is called with semantic, lexical, and code results derived from the session’s indexes, using a count equal to the total number of chunks. This fused ranking merges the three channels using reciprocal-rank fusion (RRF), where each channel contributes `1 / (rank_constant + rank)` to chunk scores, with duplicates removed within channels [1]. The results are sorted by descending score and chunk identifier [3].

In `_retrieve`, `hybrid_search` is called with semantic, lexical, and code results, with a count set to the maximum of the input lengths. The fused ranking is used to combine evidence, and additional filtering (e.g., demoting documentation or test results) is applied post-ranking [6]. Both methods rely on `hybrid_search` to aggregate and prioritize chunks across retrieval channels, ensuring the final output reflects the fused ranking’s weighted, deduplicated scores.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_hybrid.py `hybrid_search` (retrieved by: code, semantic)
- [2] kilin/rag.py `RAGSession._rank_everything` (retrieved by: lexical, semantic)
- [3] kilin/retrievals/strategies/retrieval_hybrid.py `hybrid_search` (retrieved by: code)
- [6] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._rank_everything`, `hybrid_search`

- in evidence: ['RAGSession._retrieve', 'RAGSession._rank_everything', 'hybrid_search']
- cited: ['RAGSession._retrieve', 'RAGSession._rank_everything', 'hybrid_search']

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

KíLín checks citation validity in two main places. First, the `_invalid_citations` function in `rag.py` [2] identifies citation numbers in the answer that are absent from the supplied evidence by comparing them against a set of valid citations from the evidence. It returns a tuple of invalid identifiers, ensuring the answer only references existing evidence. Second, `answer_citations` [3] checks if citations fall within the range of valid passage numbers, returning invalid numbers outside this range. When citations do not match, these functions flag them as invalid, preventing answers from referencing non-existent or out-of-range evidence. The `_evidence` function [5] prepares valid citations by numbering passages, which these checks then reference.

**Cited passages:**

- [2] kilin/rag.py `_invalid_citations` (retrieved by: lexical, semantic)
- [3] kilin/tools/_shared/document_notes.py `answer_citations` (retrieved by: lexical)
- [5] kilin/tools/find_passages/tool.py `_evidence` (retrieved by: code)

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

The `chunk_source_units` function handles source units longer than `max_tokens` by splitting them into smaller chunks while preserving their structure. If a unit's text exceeds `max_tokens`, the `_split_unit` function (called via `_chunk_source_unit`) divides it line-by-line, ensuring each chunk does not exceed `max_tokens` [4]. 

Overlap between consecutive chunks is maintained by calculating the starting position of the next chunk using `_overlap_start`, which ensures the new chunk overlaps with the previous one by `overlap_tokens` [8]. This prevents gaps and maintains contextual continuity. For example, if the last chunk ends at a certain line, the next chunk starts near that line, overlapping by the specified amount. This mechanism ensures adjacent chunks share a portion of their content, preserving coherence across splits. The overlap is enforced during chunk creation, directly tied to the `overlap_tokens` parameter.

**Cited passages:**

- [4] kilin/chunking.py `_chunk_source_unit` (retrieved by: semantic)
- [8] kilin/chunking.py `_split_unit` (retrieved by: context_expansion)

**Gold functions:** `chunk_source_units`, `_chunk_source_unit`, `_split_unit`, `_overlap_start`

- in evidence: ['chunk_source_units', '_chunk_source_unit', '_split_unit']
- cited: ['_chunk_source_unit', '_split_unit']

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

The `RAGSession._read_document` and `RAGSession._retrieve` methods call `_rank_everything`. 

- `_read_document` uses `_rank_everything` to rank all chunks of a single document, ensuring the most relevant passages are selected even if they are not among the top results globally. This is necessary when the document exceeds the evidence budget, allowing the system to prioritize critical sections [9]. 

- `_retrieve` invokes `_rank_everything` in a one-document scope to find the best passages within that document, independent of project-wide rankings. This ensures document-specific relevance, especially when the question references a specific file [10]. 

Both methods require full chunk ranking to combine semantic, lexical, and code-graph signals, enabling precise selection of relevant evidence while accounting for local document context. The comprehensive ranking ensures no critical chunk is overlooked, even if it ranks lower in broader searches.

**Cited passages:**

- [9] kilin/rag.py `RAGSession._read_document` (retrieved by: context_expansion)
- [10] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._run_tool`, `RAGSession._read_document`

- in evidence: ['RAGSession._retrieve', 'RAGSession._read_document']
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

## history_selection (location, development)

**Question.** Explain where KíLín decides which earlier turns of a conversation are given to the model as history, and what determines how many.

**Intent:** code_explanation

**Answer with retrieval:**

KíLín decides which earlier turns are included as history in two key steps. First, the `select_history` function [1] determines the selection based on the `HistoryPolicy` (none, recent, or relevant) and the `recent_turns` parameter, which specifies how many recent turns to include. If the policy is "recent," it returns the last `recent_turns` turns. If "relevant," it combines recent turns with related turns identified by the router's `RouteDecision`, ensuring standalone questions include the previous turn. Second, the `RAGSession._earlier_turns` method [2] and [6] gathers turns the new question refers back to, following related turns through the conversation. It ensures the last turn is always included and limits the total to `_MAX_EARLIER_TURNS`. The number of turns is thus determined by `recent_turns`, the router's decision (e.g., standalone or related turns), and the `_MAX_EARLIER_TURNS` cap. These mechanisms work together to balance context retention and relevance.

**Cited passages:**

- [1] kilin/answering.py `select_history` (retrieved by: lexical, semantic)
- [2] kilin/rag.py `RAGSession._earlier_turns` (retrieved by: code)
- [6] kilin/rag.py `RAGSession._earlier_turns` (retrieved by: context_expansion)

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

ConversationFileLock prevents concurrent writes by using an operating-system advisory lock on a companion `.lock` file and a registry to track active paths. When a session attempts to acquire a lock (via `acquire()` [7][9]), it first checks if the target path is already in `_ACTIVE_PATHS` [7]. If so, it raises `ConversationLockedError`, blocking the second session. The lock is enforced via `fcntl.flock` (POSIX) or `msvcrt.locking` (Windows) [9], ensuring only one process holds the lock at a time. The `_ACTIVE_PATHS` set [3] acts as a global registry, preventing duplicate ownership across threads, processes, or sessions. When a session releases the lock (via `release()` [8]), it removes the path from `_ACTIVE_PATHS` and unlocks the file, allowing others to acquire it. This combination of OS-level locking and process-local tracking ensures exclusive write access, preventing collisions [4][6].

**Cited passages:**

- [3] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: semantic)
- [4] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: semantic)
- [6] kilin/conversation_ownership.py `conversation_write_ownership` (retrieved by: context_expansion)
- [7] kilin/conversation_ownership.py `ConversationFileLock.acquire` (retrieved by: context_expansion)
- [8] kilin/conversation_ownership.py `ConversationFileLock.release` (retrieved by: context_expansion)
- [9] kilin/conversation_ownership.py `ConversationFileLock.acquire` (retrieved by: context_expansion)

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

The `choose_backend` function selects the computation backend based on the requested value and platform. If the requested backend is `"auto"`, it defaults to `"mlx"` on Apple Silicon (Darwin/arm64) and `"torch"` otherwise. Explicit `"mlx"` requests are rejected on non-Apple platforms [3]. When PyTorch is selected (via `"torch"` or auto-selection on non-Apple systems), the `torch_device` function determines the PyTorch device. It prioritizes CUDA if available, then MPS (Apple MPS), and defaults to CPU if neither is available [7]. The device selection relies on caller-supplied availability flags, not direct hardware probing [2]. Thus, PyTorch uses the best available device (cuda, mps, or cpu) based on these flags.

**Cited passages:**

- [2] kilin/runtime.py `runtime` (retrieved by: lexical, semantic)
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

The `needs_confirmation` function determines if a tool requires user confirmation based on its confirmation policy and the plan's parameters. It returns `False` if the policy is `"never"` or the plan has zero items. For `"always"`, it returns `True`. For `"when_large"`, it checks if the plan's `batches_to_run` exceeds the `large_batches` threshold [4]. 

The `confirm_plan` function uses `needs_confirmation` to decide whether to prompt the user. If confirmation is needed, it calls `context.confirm(plan)` to retrieve the user's answer. The answer is validated against allowed choices (`"continue"`, `"quick"`, `"cancel"`). If valid, it updates the plan's choice; otherwise, it defaults to `"cancel"` [1]. The user's response is traced, and the outcome (whether the user was asked and the choice made) is logged [1]. 

The user's answer directly influences the tool's execution: valid choices proceed with the plan, while invalid or cancellation responses halt it. The `confirm_plan` function ensures the user is only asked when required by policy, and the answer is properly interpreted and recorded.

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

The function `ask_choice` is called by `offer_export` [3] and `choose_document` [5].  

- **`offer_export`** asks the user to choose between saving all items of a long list as a Markdown file or showing a subset. The options are:  
  - "Save all [count] to outputs/ as Markdown"  
  - "Show the first [CHAT_LIST_LIMIT] here"  

- **`choose_document`** asks the user to select a file from a list of options, where each option includes the file name and a reason (e.g., "in this project"). The options are dynamically generated based on context, such as matching the user's question or preferences.  

These calls use `ask_choice` to prompt the user for a choice, with the selected index returned as an integer or `None` for cancellation. Other evidence does not show additional callers of `ask_choice`.

**Cited passages:**

- [3] kilin/tools/_shared/long_lists.py `offer_export` (retrieved by: code)
- [5] kilin/tools/_shared/choose_document.py `choose_document` (retrieved by: semantic)

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

TraceLogger rewrites the human-readable trace file when events are emitted, particularly progress updates, and immediately after other events [7]. It avoids leaving a partially written file by using a temporary JSON file during the rewrite process, replacing the target file only after the write is complete [6]. This ensures atomicity, as the temporary file is swapped in while an exclusive companion-file lock is held, preventing other processes from accessing the file during the operation [2]. The lock also serializes writes within a single logger, ensuring no concurrent modifications [2]. Additionally, the rewrite uses a fixed numbering offset to prevent duplicates and merges events into a deep copy to maintain consistency [6]. This approach guarantees the final file is fully written before being considered valid, avoiding partial writes.

**Cited passages:**

- [2] kilin/trace_logging.py `trace_logging` (retrieved by: semantic)
- [6] kilin/trace_logging.py `TraceLogger._write_readable_trace` (retrieved by: context_expansion)
- [7] kilin/trace_logging.py `TraceLogger._write_event` (retrieved by: context_expansion)

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

The `synchronize_project_workspace` function keeps its copy of a source folder up to date by comparing the current state of the original source directory with a saved snapshot manifest and applying changes to the project's mirrored copy. It first validates the source and project directories, ensuring the source is outside the project workspace [3]. It then scans the source directory, generating a current manifest of files and metadata [2]. This manifest is compared with a previously saved snapshot manifest (stored as `source_snapshot.json` in the project root) [9]. 

If differences are detected (e.g., new files, deletions, or modifications), the function updates the mirrored copy by removing outdated files, copying new/modified files, and rebuilding the snapshot [9]. The process ensures the mirrored copy reflects the original source's current state. The saved manifest is updated with the new state, allowing future synchronizations to detect further changes. This approach avoids overwriting user edits in the mirrored copy and ensures the snapshot aligns with the original source [1].

**Cited passages:**

- [1] kilin/project.py `synchronize_project_workspace` (retrieved by: code, semantic)
- [2] kilin/project.py `_synchronize_snapshot` (retrieved by: semantic)
- [3] kilin/project.py `synchronize_project_workspace` (retrieved by: code)
- [9] kilin/project.py `_synchronize_snapshot` (retrieved by: context_expansion)

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

The function `cached_model_snapshot` is called by two specific functions in the repository:  
1. `_model_source` in `kilin/embeddings.py` [7], which uses it when `allow_download=False` and the model identifier points to a cached snapshot (not a local directory).  
2. `ChatSession._load` in `kilin/chat.py` [12], which invokes it when `allow_download=False` and the model source is not a local directory, to resolve a cached Hugging Face snapshot instead of downloading.  

In both cases, `cached_model_snapshot` bypasses downloading by reusing an existing cached snapshot, ensuring no network interaction occurs. It relies on `huggingface_hub.snapshot_download` with `local_files_only=True` to locate the cached directory. If the snapshot lacks critical files like `config.json`, it raises an error [3]. These calls occur only when the model is already cached and downloading is explicitly disallowed.

**Cited passages:**

- [3] kilin/huggingface.py `cached_model_snapshot` (retrieved by: code, semantic)
- [7] kilin/embeddings.py `_model_source` (retrieved by: context_expansion)
- [12] kilin/chat.py `ChatSession._load` (retrieved by: context_expansion)

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

KíLín decides which files are indexed and which are left out through a combination of filtering rules and exclusion lists. The core logic for selecting files to index is in `tool.py:run` [4], where files are filtered by allowed types (e.g., PDFs, markdown) and named folders (e.g., "papers"). Files not matching these criteria are excluded. The exclusion of specific file types (e.g., scanned PDFs) is handled by `RAGSession._unsearchable_sources` [2], which identifies selected files not present in the index. Additionally, `source_scan_exclusions` [5] excludes managed project folders (like "templates") during scans, but this applies to folder-level exclusion, not individual files. Files are always left out if they exceed size limits or fail to meet inclusion criteria in `scan_repository` [7]. The exact exclusion list is determined by the intersection of these rules, with the final indexed files derived from the source manifest [6].

**Cited passages:**

- [2] kilin/rag.py `RAGSession._unsearchable_sources` (retrieved by: semantic)
- [4] kilin/tools/project_inventory/tool.py `run` (retrieved by: code)
- [5] kilin/project_layout.py `source_scan_exclusions` (retrieved by: semantic)
- [6] kilin/rag.py `RAGSession._source_files` (retrieved by: context_expansion)
- [7] kilin/repository.py `scan_repository` (retrieved by: context_expansion)

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

The question is transformed into the text embedded for semantic search through the `_format_query` method in `RAGSession._retrieve` [3], which formats the question according to the project's configuration (e.g., `query_format` in `project.profile`). This formatted text is then embedded using the `embedding_session.embed_query` method [4], producing a vector used for cosine similarity comparisons against indexed chunks. The project type may influence the formatting rules (e.g., `query_format` settings) but does not directly alter the embedding process itself. The semantic search function uses the preformatted query text, and the embedding model/backend (e.g., "loop" or "numpy") affects ranking but not the initial text transformation. Evidence does not explicitly link project type settings to the query formatting logic, so the dependency remains indirect [3][4][5].

**Cited passages:**

- [3] kilin/rag.py `RAGSession._retrieve` (retrieved by: semantic)
- [4] kilin/retrievals/strategies/retrieval_semantic.py `semantic_search` (retrieved by: code)
- [5] tutorials/140_logging_trace.ipynb `None` (retrieved by: lexical, semantic)

**Gold functions:** `RAGSession._format_query`, `format_qwen3_query`

- in evidence: []
- cited: []

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

When `ChatEngine.delete_conversation` is called, the saved conversation is **archived** rather than permanently deleted. The conversation file is moved to a `deleted` subdirectory within the logs/traces folder, retaining its original name with a suffix (e.g., `_2`, `_3`) if conflicts exist [3]. This process ensures the file remains recoverable. The conversation is removed from the active catalog, but its data persists in the `deleted` folder [2]. The method also updates the conversation store to reflect the archive, and if the deleted conversation was active, it switches to another conversation [4]. The file is not truly deleted but relocated, and the original path is no longer referenced in the active session [3]. Thus, the conversation file is not permanently erased but stored in a recoverable state.

**Cited passages:**

- [2] kilin/gui/engine.py `ChatEngine.delete_conversation` (retrieved by: code, semantic)
- [3] kilin/conversations.py `ConversationStore.archive` (retrieved by: semantic)
- [4] kilin/conversations.py `ConversationStore.archive` (retrieved by: semantic)

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

The `EmbeddingSession` computes vectors with the PyTorch backend by using the **last unmasked hidden-state token** from the model's output. Specifically, during the `_embed_torch` method [8], the model's `last_hidden_state` is retrieved, and the attention mask is applied to identify the final position of each token. The hidden state of the last valid token (i.e., the last token not masked by the attention mask) is selected for vectorization. These vectors are then normalized using L2 normalization [8]. 

Before returning, vectors are checked via the `_check_vectors` method [10], which verifies:  
1. The count matches the expected number of input texts.  
2. All vectors have consistent dimensions.  
3. All numeric values are finite and non-zero.  
4. Each vector contains at least one non-zero element.  

This ensures the returned vectors are valid, normalized, and dimensionally consistent. The PyTorch backend's computation relies on the model's `last_hidden_state` and attention mask to extract the final token's hidden state [8].

**Cited passages:**

- [8] kilin/embeddings.py `EmbeddingSession._embed_torch` (retrieved by: context_expansion)
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

## notebook_extraction (location, held_out_3)

**Question.** Where does KíLín turn a Jupyter notebook into searchable text, and which parts of a notebook are left out?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín converts a Jupyter notebook into searchable text via the `_notebook_units` function in `kilin/sources/extensions/sources_ipynb.py` [3]. This function extracts code and Markdown cells, stripping outputs and metadata, and structures their text into searchable `SourceUnit` objects [5]. It processes the notebook’s JSON content, skipping cells with types other than "code" or "markdown" [3], and excludes stored outputs, metadata, and empty cells. The resulting text includes line numbers and section headers from Markdown cells, enabling searchability. Parts left out include outputs (e.g., execution results), metadata (e.g., cell tags), and non-code/Markdown cells (e.g., raw cells). The `extract_source_units` function [6] acts as a wrapper for this process.

**Cited passages:**

- [3] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)
- [5] kilin/sources/extensions/sources_ipynb.py `_notebook_units` (retrieved by: lexical)
- [6] kilin/sources/extensions/sources_ipynb.py `extract_source_units` (retrieved by: context_expansion)

**Gold functions:** `_notebook_units`, `extract_source_units`

- in evidence: ['_notebook_units', 'extract_source_units']
- cited: ['_notebook_units', 'extract_source_units']

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

Atomic_write_text and atomic_create_text ensure files are never left half-written by using temporary files and atomic operations. Both create a temporary file with `NamedTemporaryFile`, write the content to it, and then atomically replace the destination file or create a new one. If writing fails, the temporary file is deleted, preventing partial writes [6][2]. 

**atomic_write_text** replaces an existing destination file by renaming the temporary file to the target path (`temporary.replace(path)`), ensuring the destination is only updated once the write is complete [6]. It does not check if the destination exists beforehand. 

**atomic_create_text** publishes a new file only if the destination does not exist, raising `FileExistsError` otherwise. It uses `os.link()` to create a hard link from the temporary file to the target path, avoiding a "check-then-replace" race condition [2]. This ensures the destination is either created anew or left untouched, preserving existing files. 

The key difference is: `atomic_write_text` replaces existing files, while `atomic_create_text` creates new entries without overwriting existing ones. Additionally, `atomic_create_text` uses hard links for atomic publication, whereas `atomic_write_text` uses file renaming. Both ensure temporary files are cleaned up on failure, avoiding partial writes [1][2][6].

**Cited passages:**

- [1] kilin/atomic_writes.py `atomic_writes` (retrieved by: lexical, semantic)
- [2] kilin/atomic_writes.py `atomic_create_text` (retrieved by: code, semantic)
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

VERDICT: The functions `_verify` in `reference_check/tool.py` and `run` in `suggest_reference/tool.py` call `claim_passages`.  
[2] `_verify` uses the selected passages to verify if a claim is supported by the cited work, determining verdicts like "supported" or "contradicted" via `verify_claim`.  
[11] `run` uses the passages to assess how well a statement is supported by papers, determining citation suggestions and verdicts like "background" or "not found" via `judge_support`.  
Each uses the passages to evaluate alignment between the claim/sentence and the cited work's content.

**Cited passages:**

- [2] kilin/tools/reference_check/tool.py `_verify` (retrieved by: code, lexical)
- [11] kilin/tools/suggest_reference/tool.py `run` (retrieved by: context_expansion)

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

KíLín identifies the embedding model for a saved index by associating it with the backend (e.g., "mlx" or "torch") and using a unique manifest file generated from the model's directory path. The `embedding_model` method in `project.py` [1] selects the appropriate model based on the backend, ensuring consistency between the index and the model. When a model is saved, its directory path is hashed into a unique identifier (via `_manifest_path` in `model_identity.py` [2]), creating a manifest file (e.g., a `.json` file). This manifest uniquely identifies the model's location, and any change to the model's files would alter the hash, signaling a change. The backend selection ensures the model is paired with the correct computational framework, preventing mismatches. Thus, KíLín detects model changes by comparing the saved manifest's hash against the current model's directory hash, leveraging both the backend configuration and the manifest's uniqueness.

**Cited passages:**

- [1] kilin/project.py `ProjectProfile.embedding_model` (retrieved by: code)
- [2] kilin/retrievals/model_identity.py `_manifest_path` (retrieved by: lexical)

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

ConversationStore creates a branch by copying non-metadata fields from the parent's resume state, excluding metadata fields like `ConversationInfo.__dataclass_fields__` [1]. The branch name is generated by appending "branch" and a sequential number to the parent's name, ensuring uniqueness by checking against existing registry names [6]. To avoid overwriting, the new filename is chosen to avoid existing paths at the moment of selection; each candidate path is locked and checked before creation, and collisions raise an error instead of overwriting [1]. The process uses atomic writes and file locks to prevent concurrent modifications [2].

**Cited passages:**

- [1] kilin/conversations.py `ConversationStore.create` (retrieved by: lexical, semantic)
- [2] kilin/conversations.py `conversations` (retrieved by: lexical, semantic)
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

## python_units (location, evaluation)

**Question.** How does KíLín divide a Python file into the parts it indexes, and what happens to a file that is not valid Python?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín divides a Python file into parts by splitting it into "natural units" such as functions, classes, or code blocks using the `extract_source_units` function [6]. This function processes the file's text line by line, passing the lines to `_python_units`, which likely identifies structural boundaries (e.g., def/ class statements) to segment the code. For invalid Python files, the evidence does not explicitly describe error handling. However, if a file lacks valid syntax (e.g., missing colons, incorrect indentation), `extract_source_units` might return an empty list or fail to parse, as the code does not include explicit syntax validation logic. The repository scanning tools [9][10] focus on file selection and metadata, not syntax checks, so invalid Python files would likely be treated as unprocessable content without specific error reporting.

**Cited passages:**

- [6] kilin/sources/extensions/sources_py.py `extract_source_units` (retrieved by: context_expansion)
- [9] kilin/repository.py `scan_repository` (retrieved by: context_expansion)
- [10] kilin/repository.py `scan_repository` (retrieved by: context_expansion)

**Gold functions:** `_python_units`, `_python_class_unit`

- in evidence: []
- cited: []

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

KíLín extracts text from a PDF using a combination of `pdfminer.six` and `pypdf`. The `read_source_file` function [8] first checks if the PDF is encrypted or invalid, then uses `_layout_pages` [6] to extract text via `pdfminer.six`'s layout analysis. If this fails, it falls back to `pypdf`'s plain text extraction. The extracted text is cleaned of malformed Unicode and normalized [7]. If no text is found after cleaning, the function raises an error indicating the PDF requires OCR [8]. When a PDF contains no text, KíLín explicitly rejects it, noting that scanned PDFs without extractable text cannot be processed without OCR. The text extraction process caches results [1] but does not handle empty files or non-PDFs, which return empty strings [4].

**Cited passages:**

- [1] kilin/tools/_shared/references.py `_pdf_text` (retrieved by: semantic)
- [4] kilin/tools/_shared/references.py `pdf_text` (retrieved by: semantic)
- [6] kilin/sources/extensions/sources_pdf.py `_layout_pages` (retrieved by: context_expansion)
- [7] kilin/sources/extensions/sources_pdf.py `_clean_extracted_text` (retrieved by: context_expansion)
- [8] kilin/sources/extensions/sources_pdf.py `read_source_file` (retrieved by: context_expansion)

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

KíLín determines function call relationships using a `CodeGraph` that explicitly tracks "calls" relations between functions [2]. When analyzing which functions call others, it queries this graph for direct and indirect call chains, up to a specified depth [2]. The graph is populated with function entities and their relationships, likely derived from static analysis of the codebase. 

When a called name could refer to multiple functions, KíLín resolves ambiguity by prioritizing matches that share words with the query text, prefer nearer call steps, and prioritize functions that call more repository functions (indicating higher relevance) [2]. For plain-word names, unambiguity is enforced by checking if the name is unique, not a method, and longer than two letters [3]. If ambiguity persists, the system may default to ranking by name order or exclude ambiguous candidates, though the evidence does not explicitly describe fallback mechanisms for fully ambiguous cases. The graph's structure ensures explicit tracking of call relationships, but disambiguation relies on textual and contextual heuristics.

**Cited passages:**

- [2] kilin/retrievals/context_expansion.py `_ranked_callees` (retrieved by: semantic)
- [3] kilin/retrievals/context_expansion.py `_explicit_functions` (retrieved by: semantic)

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

Passages in KíLín's keyword search are scored using **TF-IDF cosine similarity** between the query and indexed chunks. The `lexical_search` function computes this by:  
1. Extracting lowercase one-word and two-word terms from the query using `lexical_terms` [2].  
2. Creating a TF-IDF vector for the query based on the inverse document frequency (IDF) of terms in the lexical index [11].  
3. Calculating cosine similarity between the query vector and each chunk's vector, then ranking results by score and chunk identifier [10].  

Code identifiers are broken into searchable words via the `_identifier_words` function:  
1. Text is normalized by inserting spaces before uppercase letters and replacing underscores with spaces (e.g., `_parse_route` becomes `parse route`) [6].  
2. Words are extracted as lowercase tokens using `_WORD.findall`, ensuring letters and digits are split from punctuation [6].  

These processes are distinct: `lexical_search` focuses on general text, while `_identifier_words` handles code identifiers specifically. The scoring mechanism is tied to the TF-IDF framework, and identifier splitting uses regex-based normalization.

**Cited passages:**

- [2] kilin/retrievals/strategies/__init__.py `__init__` (retrieved by: semantic)
- [6] kilin/retrievals/context_expansion.py `_identifier_words` (retrieved by: context_expansion)
- [10] kilin/retrievals/strategies/retrieval_lexical.py `lexical_search` (retrieved by: context_expansion)
- [11] kilin/retrievals/strategies/retrieval_lexical.py `build_lexical_index` (retrieved by: context_expansion)

**Gold functions:** `lexical_terms`, `lexical_search`, `_tfidf_vector`

- in evidence: ['lexical_search']
- cited: ['lexical_search']

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

The `refers_back` function determines if a question depends on earlier conversation by checking for patterns like pronouns ("it"), ordinal references, or explicit pointers to prior turns [2]. It also considers short questions (≤ 2 words) as implicitly referencing history. If these conditions are met, it returns `True`, indicating the question relies on prior context. 

The `search_text` function uses this flag (`refers_back`) and the `related_turn_ids` from the routing decision to build a search query. If `refers_back` is `True` and there are related turns, it appends the text from those earlier turns (oldest first) to the current question [1]. This ensures the search query includes context from prior conversation, enabling the model to retrieve relevant information tied to the historical discussion. The `related_turn_ids` are determined during routing, where `refers_back` influences which turns are marked as relevant [5]. Thus, `refers_back` directly enables `search_text` to contextualize the query by incorporating prior dialogue.

**Cited passages:**

- [1] kilin/answering.py `search_text` (retrieved by: lexical, semantic)
- [2] kilin/answering.py `refers_back` (retrieved by: lexical, semantic)
- [5] tests/kilin/test_answering.py `TestRelatedTurnChecks.test_pointing_back_adds_related_questions_to_search` (retrieved by: lexical, semantic)

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

The `download_model` function reports progress to its caller via a callback mechanism. When invoked, it accepts a `progress_callback` parameter, which is a callable that receives completed and expected byte counts during the download [3]. This callback is linked to a custom `tqdm`-like progress bar (`_progress_tqdm`) that aggregates byte counts and forwards them to the callback [8]. The progress updates are then emitted to the frontend through the `ChatEngine`'s `send_progress` method, which sends events like `model_download_progress` to the GUI [6]. 

The downloaded model files are stored in the Hugging Face cache directory, determined by the `huggingface_hub` library. The exact path is returned as a `Path` object by `download_model`, which is derived from the snapshot download location [3]. This path is not copied or owned by KíLín; instead, it reuses existing cached files if available [3]. The files are stored in a directory managed by Hugging Face, with the specific location dependent on the user's cache configuration.

**Cited passages:**

- [3] kilin/huggingface.py `download_model` (retrieved by: code, lexical)
- [6] kilin/gui/engine.py `ChatEngine._download_with_progress` (retrieved by: context_expansion)
- [8] kilin/huggingface.py `_progress_tqdm` (retrieved by: context_expansion)

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

The `match_reference` function determines which project file a reference entry points to by evaluating matches against candidate files (papers) based on three criteria: DOI, author, and content (title or page). It uses the `rules` to set thresholds for what constitutes a "found" or "probable" match. 

1. **DOI Match**: If the entry's DOI matches the paper's DOI exactly (after normalization), it is marked as a **found** match with a score of 1.0 [9].  
2. **Author and Citation Match**: If the first author and journal citation match, it is also a **found** match [9].  
3. **Content Match**: If the entry's distinctive words (after filtering common words) overlap with the paper's title or first page, the score determines if it is **found** (if it meets the threshold) or **probable** [10].  

A match is **only probable** when:  
- The score is below the "found" threshold but meets the "probable" threshold, or  
- The author does not match, even if the content aligns [10].  

The function prioritizes DOI matches first, then author/citation, then content. If multiple entries match the same file, only the highest-scoring match is marked as **found**, while others are labeled **probable** with a reason [8]. This ensures the best possible match is selected, with uncertainty flagged when the match is not definitive.

**Cited passages:**

- [8] kilin/tools/reference_check/_matching.py `settle_duplicates` (retrieved by: context_expansion)
- [9] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)
- [10] kilin/tools/reference_check/_matching.py `_candidate` (retrieved by: context_expansion)

**Gold functions:** `match_reference`, `_candidate`

- in evidence: ['match_reference', '_candidate']
- cited: ['_candidate']

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

The functions that call `term_hits` are `find_fact` [5], `find_passages` [12], and `batch_review` [10]. Each uses the returned passage indices (`hits`) to filter or prioritize content:  
- **`find_fact`** uses `hits` to identify passages containing terms, then filters parts of files to focus on relevant sections [5].  
- **`find_passages`** uses `hits` to select only passages mentioning terms for quick review, bypassing non-matching content [12].  
- **`batch_review`** uses `hits` to split and process only the relevant passages into reviewable parts, improving efficiency [10].  

Each function leverages `term_hits` to narrow scope, ensuring model calls or processing targets only passages with exact terms, as per the question’s requirements.

**Cited passages:**

- [5] kilin/tools/find_fact/tool.py `run` (retrieved by: code)
- [10] kilin/tools/batch_review/tool.py `run` (retrieved by: context_expansion)
- [12] kilin/tools/find_passages/tool.py `run` (retrieved by: context_expansion)

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

The function `include_explicit_imports` is called by `RAGSession._prepare_source` [4] and `RAGSession._prepare_retrieval` [7]. Both functions need it to ensure explicitly imported files are included in the scan report, overriding default exclusion rules like file extensions or protected names. This is critical for building accurate code graphs and embeddings, as these files are directly referenced in the project’s source management. The test case in [5] also uses `include_explicit_imports` to validate its behavior, confirming its role in selecting explicitly imported files for indexing. The function adjusts the scan report to include these files, ensuring they are processed despite not meeting standard criteria.

**Cited passages:**

- [4] kilin/rag.py `RAGSession._prepare_source` (retrieved by: lexical)
- [5] tests/kilin/test_workspace_source_management.py `TestWorkspaceSourceManagement.test_explicit_suffix_and_protected_file_are_imported` (retrieved by: code, lexical, semantic)
- [7] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)

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

The function `normalize_extensions` is called by three main functions: `scan_repository` [2], `_read_profile` [3][7][8], and `_parse_project_type` [4]. Each uses it to validate and normalize file extensions according to specific rules.  

1. **`scan_repository`** [2] checks that extensions are valid literal suffixes (e.g., `.py`, `.md`), ensuring they are lowercase, lack wildcards, and follow the regex pattern `\.[a-z0-9_-]+(?:\.[a-z0-9_-]+)*`. It also ensures extensions are not empty or whitespace.  
2. **`_read_profile`** [3][7][8] validates extensions in project profiles, ensuring they start with a period and conform to the same regex pattern. It also checks for missing or malformed entries.  
3. **`_parse_project_type`** [4] ensures project type configurations have valid extensions, rejecting empty lists or entries without a leading period. It re-raises errors with context about invalid formats.  

Each caller uses `normalize_extensions` to enforce consistent, valid extension formats across configuration files, repository scans, and project type definitions.

**Cited passages:**

- [2] kilin/repository.py `scan_repository` (retrieved by: code)
- [3] kilin/project.py `_read_profile` (retrieved by: code)
- [4] kilin/project_type.py `_parse_project_type` (retrieved by: lexical)
- [7] kilin/project.py `_read_profile` (retrieved by: context_expansion)
- [8] kilin/project.py `_read_profile` (retrieved by: context_expansion)

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
