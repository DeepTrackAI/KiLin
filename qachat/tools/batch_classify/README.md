# batch_classify

This package owns the `batch_classify` registered tool. Its `defaults.json` is independent of `qachat_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CASE_SENSITIVE` (default `false`): Whether letter case must match.
- `WHOLE_WORDS` (default `true`): Whether a term must not continue into adjacent letters or digits.
- `CONTEXT_LINES` (default `2`): Lines around each mention that the model sees (0-5). Allowed range: 0–5.
- `MAX_LISTED` (default `100`): Mentions listed with citations (1-1000); counts include all. Allowed range: 1–1000.
- `CANDIDATES` (default `"mentions"`): How items are collected; only ``mentions``. Allowed values: mentions.
- `CATEGORIES` (default `[]`): Fixed labels (two to eight distinct names of at most 40 characters). Empty lets the router suggest labels, which are shown in the plan before execution. Project types may override this list.
- `BATCH_TOKENS` (default `2500`): Estimated tokens of mentions per model call (500-8000). Allowed range: 500–8000.
- `MAX_MODEL_CALLS` (default `30`): Most model calls one answer may make (1-200); mentions beyond them are reported as not classified. Allowed range: 1–200.
- `CONFIRMATION` (default `"when_large"`): When to ask before running. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `2`): With ``when_large``, plans with more calls than this ask first (0-100). Allowed range: 0–100.

### `ESTIMATES`

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_ITEM` (default `25`): Reply tokens per item used by this tool during execution. Allowed range: 0–100000.
- `PROMPT_OVERHEAD_TOKENS` (default `350`): Prompt overhead tokens used by this tool during execution. Allowed range: 0–100000.
- `REPLY_TOKEN_LIMIT_PER_ITEM` (default `60`): Reply token limit per item used by this tool during execution. Allowed range: 1–100000.
- `RETRY_GROUP` (default `8`): Maximum number of missing labels included in a classification retry. Allowed range: 1–10000.

