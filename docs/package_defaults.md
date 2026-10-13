# Packaged configuration defaults

`qachat/qachat_defaults.json` contains package-wide starting values used by
QAChat core. This document defines **every setting** in that file and the
behavior it controls. The JSON remains valid JSON (without comments); the
explanations belong here and in the corresponding Python module docstrings.

`qachat.defaults` reads the JSON once per process. Each consumer validates
its own settings as it initializes. A missing setting or an invalid type
causes a descriptive error rather than silently substituting another value.
Changing the packaged file requires **restarting QAChat**; it is not a
live-reloadable project-preferences file. A setting already persisted in a
project profile is not overwritten by changing a packaged default.

## `DEFAULT_MODELS` — initial model identifiers

Owned by `qachat/model_defaults.py`. These strings identify recommended
starting models, **not an allowlist**. QAChat can load other compatible model
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

Owned by `qachat/repository.py`. The values are JSON arrays of nonempty
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
reviewing which files are made accessible to QAChat.

## `ANSWERING` — routing and conversation heuristics

Owned by `qachat/answering.py`. The values below are package-wide and are
used both for newly started and resumed conversations. They do not currently
have per-project overrides.

| Key | Current value | Meaning |
| --- | --- | --- |
| `MAX_RELATED_TURNS` | `3` | Maximum earlier-turn identifiers retained from the router's decision. When more are returned, only the newest valid related identifiers remain. |
| `MAX_LISTED_TURNS` | `30` | Maximum eligible earlier-turn summaries shown in the router prompt. QAChat takes the newest eligible turns but lists them in chronological order. |
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

## Settings intentionally defined elsewhere

Supported query formats (`plain` and `qwen3`) are implemented in
`qachat/query_formats.py`. Actual query-format selection belongs to the
saved project profile and, when relevant, GUI preferences. There is **no**
`DEFAULT_QUERY_FORMAT` key in the packaged defaults.

Machine states, conversation roles, trace identifiers, and persistence-format
versions remain Python contracts, not runtime preferences. GUI presentation
defaults remain under `qachat/gui/defaults.json`. Tool parameters remain
under their current owners; they are intentionally **not** consolidated in
this refactor. A future independently configurable tool may have its own
defaults file.
