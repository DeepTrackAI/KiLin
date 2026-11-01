# find_passages

This package owns the `find_passages` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `BATCH_TOKENS` (default `2500`): Estimated tokens of sentences per model call (500-8000). Allowed range: 500–8000.
- `MAX_MODEL_CALLS` (default `200`): Most batches (1-500). Allowed range: 1–500.
- `QUICK_PASSAGES` (default `40`): Best-ranked passages read by the quick sample (5-200). Allowed range: 5–200.
- `MAX_LISTED` (default `100`): Most sentences listed in the answer (10-1000); all are counted. Allowed range: 10–1000.
- `CONFIRMATION` (default `"when_large"`): When to ask before running. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `3`): With ``when_large``, plans with more batches than this ask first. Allowed range: 0–100.

### `HEURISTICS`

- `MIN_WORDS` (default `4`): Minimum words for a candidate sentence. Allowed range: 1–100.

### `ESTIMATES`

- `TOPIC_REPLY_TOKENS` (default `128`): Topic reply tokens used by this tool during execution. Allowed range: 1–100000.
- `SENTENCE_NUMBER_TOKENS` (default `2`): Sentence number tokens used by this tool during execution. Allowed range: 0–100000.
- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `JUDGE_PROMPT_TOKENS` (default `150`): Judge prompt tokens used by this tool during execution. Allowed range: 0–100000.
- `JUDGE_REPLY_ESTIMATE_TOKENS` (default `15`): Judge reply estimate tokens used by this tool during execution. Allowed range: 0–100000.
- `CALL_SECONDS` (default `1.0`): Fixed per-call overhead, seconds, for planning estimates. Allowed range: 0–1000000000.0.


## Model completion

Model calls must finish with usable final text. A token-limit stop, unfinished
thinking block or missing final answer makes the operation incomplete; it is
not a zero-match result. The coordinator saves an explicit failure notice
and completion metadata without adding retries. Internal parsers never
receive thinking text.

A selector must return item numbers or the explicit `NONE` sentinel. Empty
or malformed text cannot establish that no items matched.

## Generation settings

Output allowances come from the active session, including per-question GUI
or API settings. They are ceilings for each model call, not target lengths
or a total budget for a multi-call tool. Prose length is controlled by the
prompt or report template; input passage budgets remain separate.

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"off"`).
It accepts `inherit`, `on`, or `off` for internal `context.ask_model` calls.
It never infers thinking from a token count. `context.write` always inherits
main-session thinking. Router calls remain nonthinking and separately bounded.

Thinking requests use the user's full allowance. Only nonthinking requests
may use an explicit smaller cap, never exceeding the user's maximum.
Changing `THINKING` to `inherit` resets this tool to the main setting; `on`
forces thinking and bypasses local short-response caps. Restart after editing
cached defaults. See the [shared policy](../../../docs/configuration.md#internal-tool-thinking)
for precedence, the retained exceptions, and migration of obsolete keys.
