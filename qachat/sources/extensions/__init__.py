"""Group optional suffix-specific source readers and extractors.

``qachat.sources.sources`` discovers a module named ``sources_<suffix>`` for
the final file suffix. An extension may provide ``read_source_file`` for
specialized input decoding and ``extract_source_units`` for structural
partitioning. Missing hooks use the core fallback; dependency failures inside
an existing hook propagate rather than silently changing the extraction.

Key Features
------------

- **Reader hooks**

    Convert specialized file formats into a common ``SourceFile`` record.

- **Extractor hooks**

    Return provenance-bearing ``SourceUnit`` records without changing the
    shared source API.

Module Structure
----------------

This module is an entry point or namespace marker and does not define a
separate exported API. The related implementation modules are described above.

Examples
--------
>>> import qachat.sources.extensions

Normal callers use ``qachat.read_repository_files`` and
``qachat.extract_source_units`` rather than importing individual hooks.

"""

__all__: list[str] = []
