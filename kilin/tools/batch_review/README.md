# batch_review

This package owns the `batch_review` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `PART_TOKENS` (default `6000`): Estimated tokens of document per part (1000-16000). Allowed range: 1000–16000.
- `MAX_MODEL_CALLS` (default `20`): Most part calls one answer may make (1-200); later parts are reported as not read. Allowed range: 1–200.
- `CONFIRMATION` (default `"when_large"`): When to ask before running. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `3`): With ``when_large``, plans with more parts than this ask first (0-100). Allowed range: 0–100.

### `ESTIMATES`

- `ANSWER_REPLY_TOKENS` (default `512`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.
- `NOTE_REPLY_TOKENS` (default `256`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `PROMPT_OVERHEAD_TOKENS` (default `300`): Prompt overhead tokens used by this tool during execution. Allowed range: 0–100000.


## Model completion

Model calls must finish with usable final text. A token-limit stop, unfinished
thinking block or missing final answer makes the operation incomplete; it is
not a zero-match result. The coordinator saves an explicit failure notice
and completion metadata without adding retries. Internal parsers never
receive thinking text.


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

## Partial completion

Independent reading parts continue after incomplete model results. Failed
parts are omitted from notes rather than represented as `NOTHING`; the final
synthesis receives a coverage warning. Useful completed work is returned with
`status: partial` and `incomplete_items`. No usable notes means an incomplete
result without synthesis. The shared internal adapter may retry unfinished,
length-limited thinking once without thinking; final `write` is not retried.
