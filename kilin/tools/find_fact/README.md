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


## Model completion

Model calls must finish with usable final text. A token-limit stop, unfinished
thinking block or missing final answer makes the operation incomplete; it is
not a zero-match result. The coordinator saves an explicit failure notice
and completion metadata without adding retries. Internal parsers never
receive thinking text.

## Internal model thinking

`MODEL_CALLS` in this tool's `defaults.json` controls only
`ToolContext.ask_model`, identically in the notebook and GUI.

Bounded internal replies of at most 512 requested output tokens default to nonthinking inference. Longer analysis requests and requests without an explicit output cap inherit the main setting.

| Key | Packaged value | Meaning |
| --- | --- | --- |
| `THINKING` | `inherit` | `inherit` follows the session subject to the short-reply rule; `on` or `off` forces all internal requests and takes precedence. |
| `SHORT_REPLY_THINKING` | `off` | `inherit`, `on`, or `off` for bounded short requests, only when `THINKING` is `inherit`. |
| `SHORT_REPLY_MAX_TOKENS` | `512` | Inclusive requested output-token bound; integer 0-100000. Zero disables the rule. `tokens=None` never counts as short. |

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
