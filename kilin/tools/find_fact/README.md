# find_fact

This package owns the `find_fact` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `PART_TOKENS` (default `6000`): Estimated tokens of passages per reading part (1000-16000). Allowed range: 1000–16000.
- `NOTE_TOKENS` (default `300`): Longest notes for one part, in tokens (100-1500). Allowed range: 100–1500.
- `ANSWER_TOKENS` (default `1024`): Longest answer, in tokens (100-4000). Allowed range: 100–4000.
- `MAX_MODEL_CALLS` (default `200`): Most reading parts (1-500). Allowed range: 1–500.
- `CONFIRMATION` (default `"always"`): When to ask before reading; ``always`` by default, because the user chooses between a thorough and a quick answer. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `0`): With ``when_large``, plans with more parts than this ask first. Allowed range: 0–100.

### `ESTIMATES`

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `NOTES_PROMPT_TOKENS` (default `300`): Notes prompt tokens used by this tool during execution. Allowed range: 0–100000.
- `CALL_SECONDS` (default `1.0`): Fixed per-call overhead, seconds, for planning estimates. Allowed range: 0–1000000000.0.
- `QUICK_ANSWER_SECONDS` (default `2.0`): Quick answer seconds used by this tool during execution. Allowed range: 0–1000000000.0.

