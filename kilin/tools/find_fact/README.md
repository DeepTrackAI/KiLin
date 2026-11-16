# find_fact

This package owns the `find_fact` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `PART_TOKENS` (default `6000`): Estimated tokens of passages per reading part (1000-16000). Allowed range: 1000–16000.
- `MAX_MODEL_CALLS` (default `200`): Most reading parts (1-500). Allowed range: 1–500.
- `CONFIRMATION` (default `"always"`): When to ask before reading; ``always`` by default, because the user chooses between a thorough and a quick answer. Allowed values: always, when_large, never.
- `LARGE_BATCHES` (default `0`): With ``when_large``, plans with more parts than this ask first. Allowed range: 0–100.
- `EXTRA_SEARCHES` (default `1`): Maximum extra ranked pages in a quick answer when the answering model clearly says the first selection is insufficient; `0` disables checking. Range: 0–5.
- `MAX_CHECK_CHARACTERS` (default `12000`): Maximum characters in a complete evidence-check prompt. Oversized checks are skipped without dropping the passages. Range: 128–100000.

### `ESTIMATES`

- `ANSWER_REPLY_TOKENS` (default `512`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.
- `NOTE_REPLY_TOKENS` (default `256`): expected reply tokens for
  duration estimates only; never a generation cap. Range: 1-131072.

- `PROMPT_TOKENS_PER_SECOND` (default `700.0`): Estimated prompt-processing throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `REPLY_TOKENS_PER_SECOND` (default `25.0`): Estimated model generation throughput, tokens per second; planning only. Allowed range: 0.001–1000000000.0.
- `NOTES_PROMPT_TOKENS` (default `300`): Notes prompt tokens used by this tool during execution. Allowed range: 0–100000.
- `CALL_SECONDS` (default `1.0`): Fixed per-call overhead, seconds, for planning estimates. Allowed range: 0–1000000000.0.
- `QUICK_ANSWER_SECONDS` (default `2.0`): Quick answer seconds used by this tool during execution. Allowed range: 0–1000000000.0.


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

`MODEL_CALLS` contains one key: `THINKING` (packaged value `"inherit"`).
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

Independent reading parts continue after incomplete model results. A wholly
failed document is omitted from the synthesis notes, never labelled as having
no facts. Valid notes from other parts remain usable with an explicit coverage
warning. Partial results identify failed parts and exclude their passages from
result evidence. With no usable notes the dependent answer is not generated.
Only the shared internal-call adapter owns the one nonthinking retry.
