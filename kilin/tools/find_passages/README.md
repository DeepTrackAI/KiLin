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

- `TOPIC_REPLY_TOKENS` (default `80`): Topic reply tokens used by this tool during execution. Allowed range: 1–100000.
- `JUDGE_REPLY_TOKENS` (default `120`): Judge reply tokens used by this tool during execution. Allowed range: 1–100000.
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

## Internal model thinking

`MODEL_CALLS` in this tool's `defaults.json` controls only
`ToolContext.ask_model`, identically in the notebook and GUI.

This tool selects structured numbers or labels, so its internal requests default to nonthinking inference regardless of batch size.

| Key | Packaged value | Meaning |
| --- | --- | --- |
| `THINKING` | `off` | `inherit` follows the session subject to the short-reply rule; `on` or `off` forces all internal requests and takes precedence. |
| `SHORT_REPLY_THINKING` | `inherit` | `inherit`, `on`, or `off` for bounded short requests, only when `THINKING` is `inherit`. |
| `SHORT_REPLY_MAX_TOKENS` | `0` | Inclusive requested output-token bound; integer 0-100000. Zero disables the rule. `tokens=None` never counts as short. |

To restore full inheritance, set `THINKING` and `SHORT_REPLY_THINKING` to
`inherit` (or set the short bound to zero). To force thinking for every
internal request, set `THINKING` to `on`. Restart KíLín or the notebook kernel
after editing these cached packaged defaults. Existing projects need no
reindexing; saved answers are not changed. Project-type tool settings do not
override this section.

These controls do not change the selected model, temperature, output cap,
or number of model calls. They do not affect `context.write()` final answers
or the nonthinking router. A small output budget can still be exhausted when
thinking is forced on; the completion guard reports an incomplete operation
instead of a false result. Models must support the existing thinking control.
See [configuration policy](../../../docs/configuration.md#internal-tool-thinking).
