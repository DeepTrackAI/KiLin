# filter_results

This package owns the `filter_results` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `BATCH_TOKENS` (default `2500`): Estimated tokens of items per model call (500-8000). Allowed range: 500–8000.
- `ITEM_CHARACTERS` (default `400`): Longest item text shown to the model and in the answer (100-2000). Allowed range: 100–2000.

### `ESTIMATES`

- `ITEM_NUMBER_TOKENS` (default `4`): Item number tokens used by this tool during execution. Allowed range: 0–100000.
- `REPLY_TOKENS` (default `120`): Reply tokens used by this tool during execution. Allowed range: 1–100000.


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
