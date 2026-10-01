"""Group language-specific code-relationship extraction hooks.

The graph builder discovers ``code_graph_<suffix>`` modules by the source
file's final suffix. An absent extension leaves only the structural graph.
Extension implementations return ``CodeRelation`` records and must not execute
the analyzed code. This initializer registers no plugins and imports no
language implementations.

Key Features
------------

- **Naming convention**

    Use suffix-based discovery rather than a global mutable registration table.

- **Optional language hooks**

    Keep graph-specific heuristics separate from source-unit extraction.

Module Structure
----------------

This module is an entry point or namespace marker and does not define a
separate exported API. The related implementation modules are described above.

Examples
--------
>>> import qachat.retrievals.code_graph.extensions

Normal callers should use ``qachat.build_code_graph(files, units)`` rather
than invoking a language hook directly.

"""

__all__: list[str] = []
