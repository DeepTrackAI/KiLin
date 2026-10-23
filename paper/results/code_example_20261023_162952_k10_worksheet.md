# Code example: grading worksheet

Indexed revision: ae96853f3 (HEAD)

## routing (mechanism)

**Question.** Explain how route_question assigns a question to an intent, and what happens when the router reply cannot be used.

**Intent:** code_explanation

**Answer with retrieval:**

`route_question` assigns a question to an intent by first checking if the question matches any intent's predefined shortcuts [3]. If a shortcut matches, it immediately returns the intent and marks the question as standalone. If no shortcut matches, it uses a router model to generate a JSON response indicating the intent and related turns [7]. 

When the router reply cannot be used (e.g., invalid JSON, model failure, or no valid related turns), `route_question` falls back to the project's default intent [8]. Additionally, if the intent requires evidence reuse or is a tool that depends on prior answers, it defaults to the last relevant turn that isn’t small talk [11]. This ensures the system avoids arbitrary tool names and prioritizes usable history. The fallback mechanism ensures the conversation remains functional even if the router fails or the question lacks relevant context.

**Cited passages:**

- [3] kilin/answering.py `route_question` (retrieved by: code, semantic)
- [7] kilin/answering.py `answering` (retrieved by: semantic)
- [8] kilin/answering.py `route_question` (retrieved by: code)
- [11] kilin/answering.py `route_question` (retrieved by: context_expansion)

**Gold functions:** `route_question`, `_parse_route`, `side_reply`

- in evidence: ['route_question']
- cited: ['route_question']

**Facts (tick those the answer states correctly):**

- [ ] A message that fully matches an intent's shortcut pattern is routed without calling the model.
- [ ] Otherwise the router model is called through side_reply with a router prompt.
- [ ] The reply is parsed by _parse_route into a RouteDecision.
- [ ] If the router call fails or its reply is unusable, the project type's default intent is used.
- [ ] Related earlier turns proposed by the router are kept only if the question refers back or shares a distinctive term.

**Errors or unsupported statements:** 

**Answer without retrieval:**

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_reuse (mechanism)

**Question.** Explain how RAGSession decides whether to reuse the saved retrieval indexes or to rebuild them.

**Intent:** code_explanation

**Answer with retrieval:**

RAGSession decides to reuse or rebuild saved retrieval indexes by comparing the saved index's fingerprint and embedding model identity with the current repository state. When opening a session, it checks if the saved index matches the current source content, chunking configuration, and embedding model snapshot [2]. If the **fingerprint** (a checksum of source chunks and metadata) matches, and the embedding model identity aligns, unchanged chunks reuse cached vectors, while modified content is re-embedded [5]. However, if there are incompatible settings, missing snapshots, or corrupted indexes, a full rebuild is triggered [2]. The `indexes_are_current` function verifies this match by comparing the saved index's fingerprint with the current repository's fingerprint, ensuring consistency between saved data and the latest sources [10]. This process avoids unnecessary recomputation, reusing valid indexes while rebuilding only when necessary to maintain accuracy.

**Cited passages:**

- [2] docs/source_pipeline.md `None` (retrieved by: lexical, semantic)
- [5] docs/retrieval_pipeline.md `None` (retrieved by: semantic)
- [10] kilin/retrievals/persistence.py `indexes_are_current` (retrieved by: lexical)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_storage (mechanism)

**Question.** Explain how build_semantic_index embeds the chunks, and how save_indexes and load_indexes protect the saved index.

**Intent:** code_explanation

**Answer with retrieval:**

The `build_semantic_index` function embeds chunks by using an `EmbeddingSession` to generate vectors for each chunk's text. It processes chunks in batches, as specified by the `batch_size` parameter, and calls `embedding_session.embed_documents` on each batch. The resulting vectors are validated for correctness (e.g., non-zero norms, finite values) and stored in a `SemanticIndex` object alongside the original chunks [1][6]. This ensures embeddings are precomputed and ready for retrieval.

The `save_indexes` and `load_indexes` functions protect the saved index through integrity checks. `save_indexes` computes a checksum (`payload_sha256`) of the index data before saving, ensuring accidental modifications are detected [13][14]. It also validates the source fingerprint to confirm consistency between the saved index and the current repository state. `load_indexes` verifies the checksum, model identity, and data structure during loading, rejecting corrupted or incompatible files [11][12]. These steps prevent unauthorized alterations and ensure the saved index remains valid and aligned with the current data.

**Cited passages:**

- [1] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code, semantic)
- [6] kilin/retrievals/strategies/retrieval_semantic.py `build_semantic_index` (retrieved by: code)
- [11] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
- [12] kilin/retrievals/persistence.py `load_indexes` (retrieved by: context_expansion)
- [13] kilin/retrievals/persistence.py `save_indexes` (retrieved by: context_expansion)
- [14] kilin/retrievals/persistence.py `save_indexes` (retrieved by: context_expansion)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## strip_thinking_callers (dependency)

**Question.** Explain which functions call strip_thinking, and why each of them needs to remove the model's reasoning.

**Intent:** code_explanation

**Answer with retrieval:**

The `strip_thinking` function is called by three functions: `history_messages`, `side_reply`, and `turn_representation`.  

- **`history_messages`** uses `strip_thinking` to remove model reasoning from answers before including them in user/assistant message pairs. This ensures the history only contains clean, final answers without internal reasoning steps [4].  
- **`side_reply`** calls `strip_thinking` to eliminate thinking text from side-question replies, ensuring the output is a direct, polished answer without model-generated reasoning [5].  
- **`turn_representation`** applies `strip_thinking` to extract concise summaries of answers, stripping away reasoning to focus on the core content and improve readability [6].  

Each function removes reasoning to ensure user-facing outputs are clean, focused on final answers, and free from model-specific artifacts that are irrelevant to the end user.

**Cited passages:**

- [4] kilin/answering.py `history_messages` (retrieved by: code, lexical)
- [5] kilin/answering.py `side_reply` (retrieved by: code, lexical)
- [6] kilin/answering.py `turn_representation` (retrieved by: code, lexical)

**Gold functions:** `turn_representation`, `side_reply`, `history_messages`

- in evidence: ['turn_representation', 'side_reply', 'history_messages']
- cited: ['turn_representation', 'side_reply', 'history_messages']

**Facts (tick those the answer states correctly):**

- [ ] turn_representation removes reasoning (and citations) from an earlier answer to build the compact representation shown to the router.
- [ ] side_reply returns the reply of an auxiliary model call, such as the router's, without reasoning.
- [ ] history_messages removes reasoning from earlier answers before they are included as conversation history.

**Errors or unsupported statements:** 

**Answer without retrieval:**

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## hybrid_callers (dependency)

**Question.** Explain which RAGSession methods call hybrid_search, and how each of them uses the fused ranking.

**Intent:** code_explanation

**Answer with retrieval:**

The `RAGSession._rank_everything` method calls `hybrid_search` to fuse semantic, lexical, and code rankings into a single fused ranking. It uses the results from `semantic_search`, `lexical_search`, and `code_search` (if applicable) as inputs to `hybrid_search`, which combines them using reciprocal-rank fusion (RRF). Each channel contributes a score of `1 / (rank_constant + rank)` for each unique chunk identifier, with duplicates within a channel ignored. The fused ranking is sorted by descending combined score and chunk identifier, ensuring the highest-ranked chunks are prioritized [3]. 

No other `RAGSession` methods explicitly call `hybrid_search` based on the evidence. The fused ranking is central to `_rank_everything`, as it merges evidence from multiple retrieval channels into a unified list of `HybridResult` objects, which are then used to inform the final answer generation.

**Cited passages:**

- [3] kilin/rag.py `RAGSession._rank_everything` (retrieved by: lexical, semantic)

**Gold functions:** `RAGSession._retrieve`, `RAGSession._rank_everything`, `hybrid_search`

- in evidence: ['RAGSession._rank_everything', 'hybrid_search']
- cited: ['RAGSession._rank_everything']

**Facts (tick those the answer states correctly):**

- [ ] _retrieve fuses the semantic, lexical and code-graph rankings for the configured retrieval count to select the evidence for an answer.
- [ ] _rank_everything ranks every chunk with the same fused channels, for uses that need a complete ranking, such as tools.
- [ ] hybrid_search combines the rankings by reciprocal-rank fusion.

**Errors or unsupported statements:** 

**Answer without retrieval:**

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## citation_check (location)

**Question.** Explain where KíLín checks that the citation numbers in an answer refer to the supplied evidence, and what happens when they do not.

**Intent:** code_explanation

**Answer with retrieval:**

KíLín checks citation validity in the `_invalid_citations` function [3], which compares citation numbers in the answer against the supplied evidence's identifiers. If a citation number does not match any evidence, it is recorded as an invalid identifier in the output [1]. This check is purely syntactic and does not verify whether the cited evidence supports the claim. When citations are invalid, they are logged but not corrected, and the answer remains unchanged [1]. Additionally, the `answer_citations` function [5] filters out citation numbers outside the range of available passages, further ensuring only valid references are considered. Invalid citations are reported but do not trigger auto-correction or re-generation of the answer.

**Cited passages:**

- [1] docs/answering_pipeline.md `None` (retrieved by: lexical, semantic)
- [3] kilin/rag.py `_invalid_citations` (retrieved by: lexical, semantic)
- [5] kilin/tools/_shared/document_notes.py `answer_citations` (retrieved by: lexical)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
