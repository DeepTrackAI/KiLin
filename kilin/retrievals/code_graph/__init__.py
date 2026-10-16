"""Expose the code-graph records and builder.

The implementation in ``code_graph`` combines structural source units with
language-specific heuristics. A graph can contain references to external
names without corresponding entity records; it is not a complete static
program analysis or an executable dependency model.

Key Features
------------

- **Graph records**

    Expose immutable entity, relation, and graph containers.

- **Language extensions**

    Delegate supported relationship extraction to suffix-specific modules.

Module Structure
----------------

- ``CodeEntity``

    One named source-code element.

- ``CodeGraph``

    Entities and deterministic code relationships.

- ``CodeRelation``

    One directed relationship between code entities.

- ``build_code_graph``

    Build a code graph without relying on directory roles.

Examples
--------
>>> import kilin
>>> graph = kilin.CodeGraph(entities=(), relations=())

"""

from .code_graph import (
    CodeEntity,
    CodeGraph,
    CodeRelation,
    build_code_graph,
)


__all__ = [
    "CodeEntity",
    "CodeGraph",
    "CodeRelation",
    "build_code_graph",
]
