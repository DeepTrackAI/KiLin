# batch_review

This package owns the `batch_review` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `PART_TOKENS` (default `6000`): Estimated tokens of document per part (1000-16000). Allowed range: 1000–16000.
- `NOTE_TOKENS` (default `400`): Longest notes for one part, in tokens (100-1500). Allowed range: 100–1500.
- `ANSWER_TOKENS` (default `1024`): Longest final answer, in tokens (200-4000). Allowed range: 200–4000.
- `MAX_MODEL_CALLS` (default `20`): Most part calls one answer may make (1-200); later parts are reported as not read. Allowed range: 1–200.
- `CONFIRMATION` (default `"when_large"`): When to ask before running. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `3`): With ``when_large``, plans with more parts than this ask first (0-100). Allowed range: 0–100.

### `ESTIMATES`

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `PROMPT_OVERHEAD_TOKENS` (default `300`): Prompt overhead tokens used by this tool during execution. Allowed range: 0–100000.


## Model completion

Model calls must finish with usable final text. A token-limit stop, unfinished
thinking block or missing final answer makes the operation incomplete; it is
not a zero-match result. The coordinator saves an explicit failure notice
and completion metadata without adding retries. Internal parsers never
receive thinking text.
