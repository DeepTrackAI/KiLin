# Packaged configuration defaults

`kilin/kilin_defaults.json` contains package-wide starting values used by
KíLín core. This document defines **every setting** in that file and the
behavior it controls. The JSON remains valid JSON (without comments); the
explanations belong here and in the corresponding Python module docstrings.

`kilin.defaults` reads the JSON once per process. Each consumer validates
its own settings as it initializes. A missing setting or an invalid type
causes a descriptive error rather than silently substituting another value.
Changing the packaged file requires **restarting KíLín**; it is not a
live-reloadable project-preferences file. A setting already persisted in a
project profile is not overwritten by changing a packaged default.

## `DEFAULT_MODELS` — initial model identifiers

Owned by `kilin/model_defaults.py`. These strings identify recommended
starting models, **not an allowlist**. KíLín can load other compatible model
identifiers supplied by a project or caller. The identifiers use the form
`organisation/model-name`.

| Key | Current value | Meaning |
| --- | --- | --- |
| `DEFAULT_CHAT_MODEL_MLX` | `Qwen/Qwen3-8B-MLX-4bit` | Initial chat model identifier for the MLX backend. |
| `DEFAULT_CHAT_MODEL_TORCH` | `Qwen/Qwen3-8B` | Initial chat model identifier for the PyTorch backend. |
| `DEFAULT_EMBEDDING_MODEL_MLX` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Initial embedding model identifier for MLX. |
| `DEFAULT_EMBEDDING_MODEL_TORCH` | `Qwen/Qwen3-Embedding-0.6B` | Initial embedding model identifier for PyTorch. |

All four values must be nonempty strings. The selected model and backend may
be overridden in a project profile. These defaults do not automatically
download a model.

## `REPOSITORY` — source-file selection

Owned by `kilin/repository.py`. The values are JSON arrays of nonempty
strings. The loaded module exposes immutable or set-like collections for its
selection policy. File and directory exclusions remain in effect even if a
file would otherwise match an allowed extension.

| Key | Meaning |
| --- | --- |
| `DEFAULT_EXTENSIONS` | Allowed literal file suffixes for repository scans that do not explicitly supply extension filters (for example `.py`, `.md`, `.ipynb` and `.pdf`). This is not a content-type detector. |
| `SKIP_DIRECTORIES` | Directory names not traversed, including VCS metadata, virtual environments, caches, generated output, and model weights. |
| `SKIP_FILES` | Filenames skipped regardless of extension, including typical credentials and secret files. |
| `BLOCKED_SUFFIXES` | Suffixes rejected even if included elsewhere, particularly model binaries and private-key formats. |

Entries in these lists are configuration *values*, not separate Python
constants. Their spelling and membership are visible directly in the JSON.
These defaults are **not a security sandbox**: users remain responsible for
reviewing which files are made accessible to KíLín.

## `ANSWERING` — routing and conversation heuristics

Owned by `kilin/answering.py`. The values below are package-wide and are
used both for newly started and resumed conversations. They do not currently
have per-project overrides.

| Key | Current value | Meaning |
| --- | --- | --- |
| `MAX_RELATED_TURNS` | `3` | Maximum earlier-turn identifiers retained from the router's decision. When more are returned, only the newest valid related identifiers remain. |
| `MAX_LISTED_TURNS` | `30` | Maximum eligible earlier-turn summaries shown in the router prompt. KíLín takes the newest eligible turns but lists them in chronological order. |
| `ROUTER_MAX_TOKENS` | `64` | Maximum number of tokens requested for the router's short JSON reply. It is not the answer-generation limit. |
| `REPRESENTATION_MAX_CHARACTERS` | `200` | Maximum number of characters kept in each question and answer-summary field of a compact turn representation. |
| `REPRESENTATION_MAX_SOURCES` | `5` | Maximum distinct source labels included in a compact turn representation. |
| `SHORT_QUESTION_MAX_WORDS` | `2` | Maximum number of words in a question automatically treated as potentially referring to earlier turns, even without an explicit reference phrase. |
| `REFERS_BACK_PATTERN` | Regex string | Case-insensitive expression matching pronouns, earlier-turn references, follow-up wording, and elliptical openings. Used by `refers_back()`. |
| `COMMON_WORDS` | List of words | Lowercase vocabulary ignored when matching distinctive terms between a new question and stored turn summaries. |

All six numerical settings must be **positive integers** (not booleans).
In particular, zero is invalid: slicing with a zero limit could select all
turns rather than none. The loader validates the regular expression at import
time. `COMMON_WORDS` must be a nonempty JSON array of nonempty lowercase
words without leading, trailing, or interior whitespace. The vocabulary is
converted to an immutable `frozenset` for lookup.

**Editing the pattern:** JSON requires escaping regular-expression
backslashes. For example, the regex token `\b` must be written as `\\b`
inside the JSON string. Retain the existing pattern unless deliberately
changing routing behavior. The configured pattern is compiled with
`re.IGNORECASE`. Changing the pattern or the common-word vocabulary affects
how a resumed conversation is routed after restart, not the text already
stored in its history.

`MAX_LISTED_TURNS`, `MAX_RELATED_TURNS`, and the representation limits
control router context size. Larger values increase prompt length and may
change routing decisions. They are different from project retrieval limits
and from the history policies configured in project types.

## `RAG` — evidence retention, indexing, and progress reporting

Owned by `kilin/rag.py`. These settings affect RAG execution and the number
of evidence items retained when reusing or supplementing earlier answers.
They are package-level policies, not retrieval counts stored in a project
profile. All four evidence limits and the embedding batch size must be
positive integers.

| Key | Current value | Meaning |
| --- | --- | --- |
| `EMBEDDING_PROGRESS_BATCH_SIZE` | `64` | Number of texts submitted in one embedding batch while creating an index. Larger batches may require more memory. |
| `MAX_REUSED_EVIDENCE` | `40` | Maximum evidence items retained when reusing an earlier turn, with older items discarded first. |
| `MAX_RELATED_EVIDENCE` | `10` | Maximum extra items from related answers added to newly retrieved evidence. |
| `UNCITED_RELATED_EVIDENCE` | `3` | Number of items taken from a related turn when its answer cites none of its evidence. |
| `MAX_EARLIER_TURNS` | `6` | Most earlier turns inspected when a tool request refers to earlier conversation content; this does not configure individual tools. |
| `PROGRESS_MIN_INTERVAL_SECONDS` | `0.15` | Minimum interval between ordinary user-facing progress notifications; must be a finite positive number. |
| `PROGRESS_TRACE_PERCENT_STEP` | `5.0` | Percentage-point increment used to throttle progress trace events; must be finite, positive, and no greater than 100. |

Progress throttling does not impose a checkpoint or event-retention policy.
The tracing system retains its single enabled trace-detail behavior.

## `PROJECT_PROFILE` — values for newly created projects

Owned by `kilin/project.py`. These settings are rendered into a new
`profile.ini` only when that profile does not yet exist. **Existing profiles
are not rewritten.** The generic chunking defaults in `CHUNKING` are
intentionally different from the new-project profile settings.

| Key | Current value | Meaning |
| --- | --- | --- |
| `ALLOW_DOWNLOAD` | `true` | Whether new projects initially permit downloading model files. Must be a JSON boolean. |
| `TARGET_TOKENS` | `300` | Preferred number of tokens in a new project's source chunks. |
| `MAX_TOKENS` | `400` | Maximum tokens allowed per chunk for a new project. |
| `OVERLAP_TOKENS` | `40` | Maximum overlap of neighboring chunks in the new profile; can be zero. |
| `RETRIEVAL_COUNT` | `5` | Number of search results requested by default in newly created project profiles. |
| `THINKING` | `true` | Initial thinking-mode flag in the new project profile. Must be a JSON boolean. |
| `TEMPERATURE` | `0.0` | Initial model generation temperature; must be a finite, nonnegative number. |
| `MAX_NEW_TOKENS` | `8192` | Initial cap for generated answer tokens, not for router decisions. |

The token targets, maximum, retrieval count, and new-token limit are
positive exact integers. Overlap is a nonnegative exact integer and must be
strictly smaller than `TARGET_TOKENS`; `MAX_TOKENS` must be at least
`TARGET_TOKENS`. The current `models.query_format` and model backend still
come from their existing canonical owners. There is no packaged default
query-format setting.

## `CHUNKING` — generic chunking API defaults

Owned by `kilin/chunking.py`. These are default arguments for a direct call
to `chunk_source_units()` without explicit size arguments; KíLín projects
normally pass the separately persisted chunking values from `profile.ini`.

| Key | Current value | Meaning |
| --- | --- | --- |
| `TARGET_TOKENS` | `500` | Preferred chunk length when splitting source units. |
| `MAX_TOKENS` | `700` | Maximum length of an individual chunk. |
| `OVERLAP_TOKENS` | `80` | Maximum repeated tokens between adjacent chunks; may be zero. |

`TARGET_TOKENS` and `MAX_TOKENS` must be positive integers, with maximum at
least as large as target. Overlap must be a nonnegative integer smaller than
target. Values are loaded once when the module is imported.

## `PROJECT_TYPES` — fallback settings for project-type JSON

Owned by `kilin/project_type.py`. These are used **only when fields are
omitted** from an individual `project_types/*.json` definition; explicit
project-type fields take precedence. They do not overwrite project profiles.

| Key | Current value | Meaning |
| --- | --- | --- |
| `DEFAULT_RECENT_TURNS` | `2` | Recent conversation turns included when a project type omits `history.recent_turns`. |
| `DOCUMENT_MAX_TOKENS` | `30000` | Estimated-token budget used when a document-reading intent omits `retrieval.max_tokens`. |

The recent-turn default is a nonnegative exact integer no greater than 100,
and may be zero. The document budget is a positive exact integer between
1,000 and 200,000, matching the project-type parser's validation bounds.

## `RETRIEVAL` — ranking and document-selection heuristics

Owned by retrieval modules in `kilin/retrievals/`. These settings affect
algorithm defaults rather than the number of search results selected by a
project profile. A caller's explicit function arguments override the
corresponding function defaults.

| Key | Current value | Meaning |
| --- | --- | --- |
| `RRF_RANK_CONSTANT` | `60` | Smoothing constant used in reciprocal-rank fusion when `hybrid_search()` receives no explicit `rank_constant`. |
| `CODE_MAX_DISTANCE` | `1` | Maximum graph hops used by default in `code_search()`; zero permits only directly matched entities. |
| `DOCUMENT_CHARACTERS_PER_TOKEN` | `4` | Approximate number of characters per token in document-budget estimation; not a tokenizer. |
| `MIN_DOCUMENT_STEM_LENGTH` | `4` | Shortest file stem recognized as a named-document candidate. |
| `CALLEE_EXPANSION_DEPTH` | `2` | Call steps followed from a named function when collecting its callees (1 = direct callees only). Zero disables callee expansion. |
| `CALLEE_EXPANSION_MAX` | `4` | Most callees added to the evidence when a question names a function without asking about relationships; callees sharing words with the question come first, then nearer ones. Zero disables callee expansion. |
| `CLASS_METHOD_EXPANSION_MAX` | `4` | Methods added when a question names a class: all of them if the class has at most this many, otherwise this many of those sharing most words with the question. Zero disables class expansion. |
| `RELATED_CHUNKS_MAX` | `2` | Most chunks kept from one long related function (a callee, neighbor, or class method): those mentioning a named function first, then those sharing most words with the question. Zero keeps every chunk. |
| `RELATION_PHRASES` | List of phrases | Lowercase phrases identifying questions about relationships, dependencies, or calls for deterministic context expansion. |

The RRF constant and both document heuristics require positive exact
integers. `CODE_MAX_DISTANCE` and the four expansion limits are
nonnegative exact integers.
`RELATION_PHRASES` must be a nonempty list of nonempty, lowercase strings
without leading or trailing whitespace; entries may contain spaces.
The original vocabulary is preserved in the JSON, and the code-graph entity
kinds and source parsing contracts remain defined in Python.

## Settings intentionally defined elsewhere

Supported query formats (`plain` and `qwen3`) are implemented in
`kilin/query_formats.py`. Actual query-format selection belongs to the
saved project profile and, when relevant, GUI preferences. There is **no**
`DEFAULT_QUERY_FORMAT` key in the packaged defaults.

Machine states, conversation roles, trace identifiers, and persistence-format
versions remain Python contracts, not runtime preferences. GUI presentation
defaults remain under `kilin/gui/defaults.json`. Tool parameters remain
under their current owners; they are intentionally **not** consolidated in
this refactor. A future independently configurable tool may have its own
defaults file.
