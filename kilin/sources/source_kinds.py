"""Name the kinds of code passages the built-in readers produce.

Every source unit has a ``kind``, such as ``"python_function"`` or
``"js_method"``. The readers that produce these kinds and the parts that
interpret them (the code graph and retrieval's context expansion) share the
names defined here, so a new language is described once. The values are the
strings already saved in indexes and traces; they never change.

The groups describe what a kind is for, not just which kinds exist: a
callable can be named in a question and followed through calls; a method
belongs to a class, so its plain name alone does not name it; a module
context holds a file's own top-level code. Kinds outside these groups, such
as those of a custom reader, still work everywhere; they only miss the
behaviour a group gives.

Key Features
------------

- **Shared names**

    One constant per built-in code kind, used by readers and consumers.

- **Groups by role**

    Callables, methods, classes and module contexts across Python, MATLAB,
    JavaScript and R.

Module Structure
----------------

- ``CALLABLE_KINDS``

    Functions, methods and constructors.

- ``METHOD_KINDS``

    Callables that belong to a class.

- ``CLASS_KINDS``

    Classes.

- ``MODULE_CONTEXT_KINDS``

    A file's own top-level code outside its declarations.

- ``CODE_KIND_PREFIXES``

    Kind prefixes of source code, as opposed to prose and notebooks.

Examples
--------
>>> from kilin.sources.source_kinds import CALLABLE_KINDS, JS_METHOD
>>> JS_METHOD in CALLABLE_KINDS
True

"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CALLABLE_KINDS",
    "CLASS_KINDS",
    "CODE_KIND_PREFIXES",
    "METHOD_KINDS",
    "MODULE_CONTEXT_KINDS",
]

PYTHON_FILE: Final[str] = "python_file"
PYTHON_MODULE_CONTEXT: Final[str] = "python_module_context"
PYTHON_FUNCTION: Final[str] = "python_function"
PYTHON_CLASS: Final[str] = "python_class"
PYTHON_CLASS_CONTEXT: Final[str] = "python_class_context"
PYTHON_METHOD: Final[str] = "python_method"

MATLAB_SCRIPT: Final[str] = "matlab_script"
MATLAB_FILE_CONTEXT: Final[str] = "matlab_file_context"
MATLAB_FUNCTION: Final[str] = "matlab_function"
MATLAB_CLASS: Final[str] = "matlab_class"
MATLAB_CLASS_CONTEXT: Final[str] = "matlab_class_context"
MATLAB_METHOD: Final[str] = "matlab_method"
MATLAB_CONSTRUCTOR: Final[str] = "matlab_constructor"

JS_MODULE_CONTEXT: Final[str] = "js_module_context"
JS_FUNCTION: Final[str] = "js_function"
JS_CLASS: Final[str] = "js_class"
JS_CLASS_CONTEXT: Final[str] = "js_class_context"
JS_METHOD: Final[str] = "js_method"

R_SCRIPT: Final[str] = "r_script"
R_FUNCTION: Final[str] = "r_function"
R_CLASS: Final[str] = "r_class"
R_CLASS_CONTEXT: Final[str] = "r_class_context"
R_METHOD: Final[str] = "r_method"

METHOD_KINDS: Final[frozenset[str]] = frozenset(
    {PYTHON_METHOD, MATLAB_METHOD, MATLAB_CONSTRUCTOR, JS_METHOD, R_METHOD}
)
CALLABLE_KINDS: Final[frozenset[str]] = METHOD_KINDS | {
    PYTHON_FUNCTION,
    MATLAB_FUNCTION,
    JS_FUNCTION,
    R_FUNCTION,
}
CLASS_KINDS: Final[frozenset[str]] = frozenset(
    {PYTHON_CLASS, MATLAB_CLASS, JS_CLASS, R_CLASS}
)
MODULE_CONTEXT_KINDS: Final[frozenset[str]] = frozenset(
    {PYTHON_MODULE_CONTEXT, MATLAB_FILE_CONTEXT, JS_MODULE_CONTEXT, R_SCRIPT}
)
# Units that name the code around them rather than an entity of their own:
# their symbol is the enclosing class or file.
CONTEXT_KINDS: Final[frozenset[str]] = frozenset(
    {
        PYTHON_CLASS_CONTEXT,
        PYTHON_MODULE_CONTEXT,
        MATLAB_CLASS_CONTEXT,
        MATLAB_FILE_CONTEXT,
    }
)
CODE_KIND_PREFIXES: Final[tuple[str, ...]] = (
    "python_",
    "matlab_",
    "js_",
    "r_",
)
