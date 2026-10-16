# show_source

This package owns the `show_source` registered tool. Its `defaults.json` is independent of `kilin_defaults.json`. Project-type tool settings override the keys in `SETTINGS`. Other sections are internal execution parameters. All values are read and validated on import.

## Defaults

### `SETTINGS`

- `CONTEXT_SENTENCES` (default `2`): Sentences shown before and after the cited sentence (0-10). Allowed range: 0–10.
- `MAX_PASSAGES` (default `3`): Most cited passages shown (1-10). Allowed range: 1–10.
- `ANSWER_TOKENS` (default `400`): Longest explanation, in tokens (100-2000). Allowed range: 100–2000.

### `HEURISTICS`

- `MIN_WORDS` (default `4`): Minimum number of words required when splitting cited passages into sentences. Allowed range: 1–100.
