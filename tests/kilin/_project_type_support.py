"""Provide isolated project-type definitions for core contract tests.

Keep shared fixture construction independent of discoverable test modules.
The factory returns fresh nested data so callers can alter one candidate
without changing another test's inputs or the packaged project types.

Key Features
------------
- Minimal JSON-compatible definitions with one search intent.
- Independent dictionaries and lists on every factory call.
- No filesystem writes, package-cache changes, or model dependencies.

Module Structure
----------------
Functions:

- `_type_data() -> dict[str, object]`
    Build one valid custom definition for loader and cross-module tests.

Examples
--------
>>> from tests.kilin._project_type_support import _type_data
>>> first = _type_data()
>>> second = _type_data()
>>> first["intents"]["explain"]["description"] = "Changed for one test."
>>> second["intents"]["explain"]["description"]
'Explain a document.'

"""

from __future__ import annotations


__all__: list[str] = []


def _type_data() -> dict[str, object]:
    """Create an independent minimal valid custom project-type definition.

    Returns
    -------
    dict[str, object]
        JSON-compatible project type with a single search intent.

    """

    return {
        "display_name": "Experimental type",
        "description": "A local collection.",
        "extensions": [".TXT", ".md"],
        "max_file_bytes": 4096,
        "retrieval_instruction": "Find relevant information.",
        "system_prompt": "Ground answers in the documents.",
        "intents": {
            "explain": {
                "description": "Explain a document.",
                "system_prompt": "Explain with evidence.",
            },
        },
    }
