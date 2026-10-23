# Code example: grading worksheet

Indexed revision: ec3f9d05e (HEAD)

## routing (mechanism)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_reuse (mechanism)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## index_storage (mechanism)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## strip_thinking_callers (dependency)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## hybrid_callers (dependency)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## citation_check (location)

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

(none)

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
