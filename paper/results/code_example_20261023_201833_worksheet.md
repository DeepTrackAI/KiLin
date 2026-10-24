# Code example: grading worksheet

Indexed revision: ab823ff78 (HEAD)

## routing (mechanism, development)

**Question.** Explain how route_question assigns a question to an intent, and what happens when the router reply cannot be used.

**Intent:** code_explanation

**Answer with retrieval:**

The `route_question` function assigns a question to an intent by first checking if the question matches any intent's predefined shortcuts [1]. If a shortcut matches, it immediately returns a `RouteDecision` with that intent. If no shortcut matches, it uses a router model to classify the question. The router generates a JSON response specifying an intent, whether the question is standalone, and related turns. 

If the router reply is invalid (e.g., malformed JSON, unknown intent, or an exception occurs during processing), the function falls back to the project's default intent [6][7]. During validation, the function ensures the intent exists in the project and checks if the standalone flag and related turns are valid. If issues arise, such as an invalid intent or malformed fields, the fallback is set, and the default intent is used [7][8]. 

When the router reply cannot be used (e.g., technical failure or invalid output), the system defaults to the project's default intent, ensuring the question is still routed to a valid intent without relying on the router's response [3][6]. This fallback mechanism guarantees robustness, even if the router model fails or returns unusable results.

**Cited passages:**

- [1] kilin/answering.py `route_question` (retrieved by: code, semantic)
- [3] kilin/answering.py `answering` (retrieved by: semantic)
- [6] kilin/answering.py `route_question` (retrieved by: context_expansion)
- [7] kilin/answering.py `_parse_route` (retrieved by: context_expansion)
- [8] kilin/answering.py `_parse_route` (retrieved by: context_expansion)

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

RAGSession decides to reuse or rebuild retrieval indexes by comparing the current project configuration and source state against saved metadata and manifests. When initializing, it checks the saved source manifest and index settings [2]. If the saved index's metadata (e.g., embedding model, chunking configuration, and model identity) matches the current setup, it reuses the index [5]. The `index_status` attribute records this decision as "reused" or "rebuilt" [1]. 

Key validations include comparing the current repository manifest with the saved one [7], ensuring the model identity and source match the loaded embedding session [8]. If the saved index is stale (e.g., due to changed sources, model settings, or chunking configurations), it rebuilds the index to ensure compatibility [5]. The `refresh_sources` method updates indexes when sources change, triggering a rebuild if necessary [1]. Model identity is resolved via `embedding_model_identity` to prevent reuse of vectors from incompatible models [8]. This process ensures indexes are only reused when valid, avoiding mixed vectors from different settings [5].

**Cited passages:**

- [1] kilin/rag.py `RAGSession` (retrieved by: code)
- [2] tutorials/140_logging_trace.ipynb `None` (retrieved by: semantic)
- [5] docs/retrieval_pipeline.md `None` (retrieved by: semantic)
- [7] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)
- [8] kilin/rag.py `RAGSession._prepare_retrieval` (retrieved by: context_expansion)

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

The `build_semantic_index` function embeds chunks by using the provided `embedding_session` to generate vector representations of each chunk's text. It processes chunks in batches (controlled by `batch_size`), calls `embedding_session.embed_documents` on each batch, and stores the resulting vectors alongside the original chunks in a `SemanticIndex` object [1][4]. Validation ensures vectors have valid dimensions, no non-finite values, and correct counts [4].  

`save_indexes` and `load_indexes` protect the saved index through integrity checks. During saving, `save_indexes` computes a checksum (`payload_sha256`) of the serialized data and stores it alongside the index. This checksum is verified during loading to detect accidental corruption [6][9]. `load_indexes` also checks that all required fields exist, validates the embedding model identity, and ensures the saved data matches the current repository's fingerprint, preventing mismatches between saved and active data [6][7]. These steps ensure the saved index remains consistent, secure, and usable for future queries.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code, semantic)
- [4] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code)
- [6] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
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

The `strip_thinking` function is called by three functions: `history_messages` [3], `turn_representation` [5], and `side_reply` [6]. Each uses it to remove the model's reasoning steps, ensuring outputs focus only on the final answer.  

- **`history_messages`** [3] strips thinking text from answers to exclude internal reasoning, preserving only the answer and citations. This keeps conversation history clean for user-assistant message pairs.  
- **`turn_representation`** [5] removes thinking and citations to generate concise summaries, focusing on the answer’s first sentence without extraneous details.  
- **`side_reply`** [6] eliminates thinking text from model replies to provide direct, unambiguous answers, aligning with the side question’s purpose of delivering a straightforward response.  

Each removal ensures outputs are user-focused, avoiding model-specific artifacts like reasoning steps or citation markers, which are handled separately.

**Cited passages:**

- [3] kilin/answering.py `history_messages` (retrieved by: code)
- [5] kilin/answering.py `turn_representation` (retrieved by: code)
- [6] kilin/answering.py `side_reply` (retrieved by: context_expansion)

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

The `RAGSession` class has two methods that call `hybrid_search`: `_rank_everything` and `_retrieve`. Both use the fused ranking generated by `hybrid_search` to combine results from semantic, lexical, and code-based retrieval channels.  

In `_rank_everything` [2], `hybrid_search` is called with all chunks ranked across semantic, lexical, and code-graph channels. The fused ranking merges these results using reciprocal-rank fusion (RRF), where each channel contributes a score based on the inverse of its rank (adjusted by a constant). The final output is sorted by descending fused scores, then by chunk identifier to resolve ties. This method ensures all chunks are evaluated before ranking.  

In `_retrieve` [7], `hybrid_search` is called with semantic, lexical, and code results, but the count is set to ensure sufficient candidates (max of retrieval count and sum of individual result counts). The fused ranking here combines these sources, prioritizing chunks with higher combined RRF scores. This allows the system to balance relevance across different retrieval types before selecting the top results.  

Both methods rely on `hybrid_search` to integrate ranked results from multiple channels, ensuring the fused ranking reflects a weighted combination of semantic, lexical, and code-based evidence.

**Cited passages:**

- [2] kilin/rag.py `RAGSession._rank_everything` (retrieved by: lexical, semantic)
- [7] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)

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

KíLín checks citation validity in the `_invalid_citations` function [5], which compares cited numbers in the answer against available evidence identifiers. It identifies invalid citations by parsing answer text for citation markers, normalizing numbers, and checking if they exist in the evidence’s `citation` fields [1]. When citations do not match, invalid identifiers are recorded in `invalid` as tuples of integers or strings [5], but the answer is not corrected or reranked. This check is diagnostic, not semantic, meaning unsupported claims may still persist [1]. The `answer_citations` function [3] aids in identifying invalid numbers by cross-referencing cited numbers against the total passage count. The GUI ensures citation previews use exact saved evidence, avoiding silent substitutions if sources change [1].

**Cited passages:**

- [1] docs/answering_pipeline.md `None` (retrieved by: lexical, semantic)
- [3] kilin/tools/_shared/document_notes.py `answer_citations` (retrieved by: lexical)
- [5] kilin/rag.py `_invalid_citations` (retrieved by: lexical)

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

The `chunk_source_units` function handles oversized source units by splitting them into smaller chunks while respecting natural line boundaries and maintaining overlap as specified. If a `SourceUnit`'s text exceeds `max_tokens`, it is divided into smaller parts. The function uses line-based splitting, ensuring chunks do not exceed `max_tokens` [11]. For example, it iterates through lines, calculating remaining text and determining chunk boundaries to stay within token limits. Overlap between consecutive chunks is enforced by adjusting the start position of subsequent chunks using `_overlap_start`, which ensures each new chunk overlaps the previous one by `overlap_tokens` [12]. This maintains a consistent overlap, as validated during parameter checks [10]. The function also preserves structural metadata (e.g., line ranges, identifiers) for each chunk [13]. If a unit has nested children, they are processed recursively [9]. This approach balances token constraints with natural unit boundaries, ensuring retrievable chunks while preserving context through overlap.

**Cited passages:**

- [9] kilin/chunking.py `_chunk_source_unit` (retrieved by: semantic)
- [10] kilin/chunking.py `_check_sizes` (retrieved by: context_expansion)
- [11] kilin/chunking.py `_split_unit` (retrieved by: context_expansion)
- [12] kilin/chunking.py `_split_unit` (retrieved by: context_expansion)
- [13] kilin/chunking.py `_make_chunk` (retrieved by: context_expansion)

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

The `RAGSession` methods that call `_rank_everything` are `_read_document` [8], `_retrieve` [9], and `_run_tool` [12]. Each requires ranking every chunk to ensure comprehensive and contextually accurate retrieval. 

- `_read_document` uses `_rank_everything` to rank all chunks within a document, even if they are not among the project-wide top results, ensuring the most relevant passages are selected for the evidence budget [8].  
- `_retrieve` calls `_rank_everything` to generate a ranked list of chunks, allowing focused retrieval of a single document’s best parts, even if they are not globally top-ranked [9].  
- `_run_tool` leverages `_rank_everything` to provide the tool with a ranked chunk list, ensuring the tool’s context is based on the most relevant information, regardless of global ranking [12].  

Each method requires full chunk ranking to avoid missing critical context that might be locally significant within a document but overlooked in broader rankings. This ensures robustness in evidence selection and tool execution.

**Cited passages:**

- [8] kilin/rag.py `RAGSession._read_document` (retrieved by: context_expansion)
- [9] kilin/rag.py `RAGSession._retrieve` (retrieved by: context_expansion)
- [12] kilin/rag.py `RAGSession._run_tool` (retrieved by: context_expansion)

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

KíLín decides which earlier conversation turns are included as history in the `select_history` function within `answering.py` [2]. This function uses the `policy` (set via project configuration) and the router's `decision` to determine the history. The `policy` can be `none`, `recent`, or `relevant`. If `recent`, it includes the last `recent_turns` turns (a configurable number). If `relevant`, it combines recent turns with those related to the current question, as determined by the router. The number of turns is limited by `recent_turns` (a project-wide setting) and `_MAX_EARLIER_TURNS` (a constant in `rag.py` [6]). The router's `decision` also ensures the last turn is always included, even if unrelated. Thus, the count is determined by both configurable parameters and internal limits, with the router's logic influencing which specific turns are selected.

**Cited passages:**

- [2] kilin/answering.py `select_history` (retrieved by: lexical, semantic)
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

ConversationFileLock prevents concurrent writes by using an OS-level advisory lock on a companion `.lock` file and a process-local registry to track active owners [1]. When a session attempts to acquire the lock, it first checks if the target path is already in `_ACTIVE_PATHS`, a registry of currently owned paths. If so, it raises `ConversationLockedError` immediately, blocking the second writer without modifying the JSON [1][5]. The lock is applied to the companion file, ensuring it survives atomic JSON replacements [1]. Additionally, the lock is non-blocking: contention results in an error rather than waiting, ensuring no delays [1]. The registry also prevents duplicate owners by rejecting acquisitions for the same path in the same process or others [1][6]. When a session releases the lock, it removes its entry from `_ACTIVE_PATHS` and the OS lock, allowing another session to acquire it [9]. This combination of OS locks and registry checks ensures only one session can write to a conversation file at a time.

**Cited passages:**

- [1] kilin/conversation_ownership.py `conversation_ownership` (retrieved by: semantic)
- [5] docs/conversation_ownership.md `None` (retrieved by: semantic)
- [6] kilin/conversation_ownership.py `ConversationFileLock.acquire` (retrieved by: context_expansion)
- [9] kilin/conversation_ownership.py `ConversationFileLock.release` (retrieved by: context_expansion)

**Gold functions:** `ConversationFileLock.acquire`, `conversation_write_ownership`

- in evidence: ['ConversationFileLock.acquire']
- cited: ['ConversationFileLock.acquire']

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

## backend_choice (mechanism, held_out)

**Question.** Explain how choose_backend selects the computation backend, and which PyTorch device is used when PyTorch is selected.

**Intent:** code_explanation

**Answer with retrieval:**

The `choose_backend` function selects the computation backend based on the requested value and platform constraints. If "mlx" is requested, it only selects it on Apple Silicon (Darwin/arm64) [6]. If "auto" is requested, it defaults to "mlx" on Apple Silicon and "torch" otherwise [6]. When "torch" is selected, the `torch_device` function determines the PyTorch device: it优先 uses "cuda" if available, then "mps" (Apple MPS), and finally "cpu" if neither is available [9]. This ensures the best available device for PyTorch execution.

**Cited passages:**

- [6] kilin/runtime.py `choose_backend` (retrieved by: code, semantic)
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

## tool_confirmation (mechanism, held_out)

**Question.** Explain how needs_confirmation and confirm_plan decide whether a tool asks the user before running, and what happens with the user's answer.

**Intent:** code_explanation

**Answer with retrieval:**

The `needs_confirmation` function determines if a tool requires user confirmation based on its confirmation policy and the plan's parameters. If the policy is `"never"`, confirmation is skipped. If it's `"always"`, confirmation is required. For `"when_large"`, confirmation occurs only if the plan's `batches_to_run` exceeds the `large_batches` threshold [6]. 

The `confirm_plan` function uses this decision to prompt the user. If confirmation is needed, it traces the plan, waits for the user's input via `context.confirm`, and validates the response against allowed choices (`"continue"`, `"quick"`, `"cancel"`). Invalid answers default to `"cancel"` [2]. The user's choice is then recorded, and the plan proceeds or is canceled based on the response. This ensures the tool adheres to its confirmation policy while handling user input appropriately.

**Cited passages:**

- [2] kilin/tools/_shared/planning.py `confirm_plan` (retrieved by: code, lexical, semantic)
- [6] kilin/tools/_shared/planning.py `needs_confirmation` (retrieved by: context_expansion)

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

## ask_choice_callers (dependency, held_out)

**Question.** Explain which functions call ask_choice, and what each of them asks the user to choose.

**Intent:** code_explanation

**Answer with retrieval:**

The function `choose_document` calls `ask_choice` when the user needs to select a document from a list of candidates, such as "Which paper should I turn into a research briefing?" [3]. The options presented are up to four candidate files derived from earlier turns or best matches for the question. 

The function `offer_export` calls `ask_choice` to prompt the user whether to save a long list as a Markdown file, asking "Save all [X] to outputs/ as Markdown" or "Show the first [Y] here" [7]. 

These are the only functions explicitly documented as calling `ask_choice` in the evidence. Other potential callers are not identified in the provided sources.

**Cited passages:**

- [3] kilin/tools/_shared/choose_document.py `choose_document` (retrieved by: context_expansion)
- [7] kilin/tools/_shared/long_lists.py `offer_export` (retrieved by: code)

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

## readable_trace (mechanism, held_out)

**Question.** Explain when TraceLogger rewrites the human-readable trace file, and how it avoids leaving a partially written file.

**Intent:** code_explanation

**Answer with retrieval:**

TraceLogger rewrites the human-readable trace file when new events are appended, particularly during progress updates or at configured intervals [8]. It avoids leaving a partially written file by using a temporary JSON file during the rewrite process. While the exclusive companion-file lock is held, the temporary file replaces the target file, ensuring the original is only updated after the entire write completes [4]. This prevents partial writes by serializing the operation and holding the lock until the rewrite is finalized [5]. The temporary file approach guarantees atomicity, as other writers cannot access the path until the logger releases ownership, maintaining file integrity [4].

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

## workspace_sync (mechanism, held_out)

**Question.** Explain how prepare_project_workspace keeps its copy of a source folder up to date with the original folder.

**Intent:** code_explanation

**Answer with retrieval:**

The `prepare_project_workspace` function keeps its copy of a source folder up to date by comparing the current snapshot with the original source using SHA-256 manifests [1]. It scans the source directory, applying repository rules to select files, and builds a manifest of their hashes [4]. When the workspace is reopened, it compares this current manifest with a saved one. If discrepancies are found (e.g., file additions, modifications, or deletions), it rebuilds the snapshot from the original, ensuring the managed copy matches [2]. For incremental updates, it synchronizes only changed files, removing outdated ones and copying new/modified files while preserving directory structure [2]. The process also verifies the snapshot’s integrity against the manifest, rebuilding it entirely if mismatches persist [12]. This ensures the workspace copy always reflects the original source’s state without altering it.

**Cited passages:**

- [1] kilin/project.py `prepare_project_workspace` (retrieved by: code, lexical, semantic)
- [2] kilin/project.py `prepare_project_workspace` (retrieved by: code)
- [4] kilin/project.py `prepare_project_workspace` (retrieved by: code)
- [12] kilin/project.py `_snapshot_matches_manifest` (retrieved by: context_expansion)

**Gold functions:** `prepare_project_workspace`, `_snapshot_matches_manifest`, `compare_repository_manifests`

- in evidence: ['prepare_project_workspace', '_snapshot_matches_manifest']
- cited: ['prepare_project_workspace', '_snapshot_matches_manifest']

**Facts (tick those the answer states correctly):**

- [ ] A manifest with SHA-256 hashes of the selected source files is built and compared with the saved manifest (source_snapshot.json) to find added, modified, and removed files.
- [ ] Before using that comparison, the copied files are checked against their saved hashes; if the copy was changed, it is rebuilt from the original.
- [ ] Otherwise only added and modified files are copied and removed files deleted, then empty folders are removed.
- [ ] The new manifest, with the source folder's path, is saved for the next opening.
- [ ] The original folder is never modified.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `prepare_project_workspace` function in the KíLín Python package is designed to create and maintain a synchronized copy of a source folder within a project workspace. While the exact implementation details may vary, the general approach typically involves the following mechanisms to keep the copied folder up to date with the original:

1. **Version Control Integration**: If the original source folder is under version control (e.g., Git), the function might pull the latest changes from the repository to ensure the copied folder reflects the current state of the source.

2. **File System Monitoring**: The function could use tools like `inotify` (on Linux) or `watchmedo` (via Python) to monitor the original folder for changes (e.g., file additions, modifications, or deletions). When changes are detected, the copied folder is automatically updated.

3. **Build or Sync Tools**: It might leverage build systems (e.g., `make`, `pip`, or `poetry`) or custom scripts to regenerate the copied folder based on the original, ensuring consistency during development or testing.

4. **Symlinks or Hard Links**: In some cases, the function might create symbolic links or hard links to the original files, allowing the copied folder to reflect changes in real time without duplicating data.

5. **Periodic Checks**: The function could periodically compare the original and copied folders, updating the copy if discrepancies are found (e.g., via checksums or timestamps).

For precise behavior, refer to the KíLín documentation or its source code, as the exact method depends on the package's design. If you don't have access to this information, consider checking the package's GitHub repository or documentation for details.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
