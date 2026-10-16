# filter_results

This package owns the `filter_results` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `BATCH_TOKENS` (default `2500`): Estimated tokens of items per model call (500-8000). Allowed range: 500–8000.
- `ITEM_CHARACTERS` (default `400`): Longest item text shown to the model and in the answer (100-2000). Allowed range: 100–2000.

### `ESTIMATES`

- `ITEM_NUMBER_TOKENS` (default `4`): Item number tokens used by this tool during execution. Allowed range: 0–100000.
- `REPLY_TOKENS` (default `120`): Reply tokens used by this tool during execution. Allowed range: 1–100000.

