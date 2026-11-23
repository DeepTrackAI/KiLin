"""Shared Node.js adapter for frontend unit tests.

This private test utility extracts top-level ``app.js`` functions and runs
small JavaScript harnesses without Qt or a browser. It intentionally retains
the source delimiter used by the existing tests; it is not a JavaScript
parser or a substitute for the native WebEngine smoke tests.

"""

from __future__ import annotations

import shutil
import subprocess
from typing import Mapping, Sequence


def extract_function(source: str, name: str) -> str:
    """Extract a top-level JavaScript function using the existing delimiter.

    Parameters
    ----------
    source: str
        Complete JavaScript source containing the declaration.
    name: str
        Function identifier following the ``function`` keyword.

    Returns
    -------
    str
        Source from the declaration through its closing brace and newline.

    Raises
    ------
    ValueError
        If the declaration or its expected terminator is missing.

    Notes
    -----
    The source must use a standalone closing ``}`` on its own line. This
    deliberately matches the existing test adapters; structural frontend
    refactoring will require a separate decision about JS module testing.

    """

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


def run_node(
    script: str,
    *,
    args: Sequence[str] = (),
    timeout: float = 15,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an isolated Node harness and return its output and exit status.

    Parameters
    ----------
    script: str
        JavaScript evaluated from standard input in ``node -e``.
    args: Sequence[str]
        Optional arguments appended after the script, accessible in
        Node through ``process.argv[1:]``. Defaults to no arguments.
    timeout: float
        Maximum duration in seconds, default 15.
    env: Mapping[str, str] | None
        Optional subprocess environment, otherwise inherited.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed process with captured text output and return code.

    Raises
    ------
    subprocess.TimeoutExpired
        If Node exceeds ``timeout``.
    OSError
        If the Node executable cannot be started.

    Notes
    -----
    The JavaScript is passed on stdin rather than the command line to
    support long frontend harnesses on Windows. ``node -e`` retains the
    existing ``process.argv`` indexing for optional script arguments.
    The helper does not skip tests when Node is absent. Tests retain their
    existing ``unittest.skipUnless`` gates and result assertions. Using
    ``check=False`` ensures the individual test sees the failing stderr.

    """

    executable = shutil.which("node")
    if executable is None:
        raise FileNotFoundError("Node.js is not available")
    return subprocess.run(
        [
            executable, "-e",
            "eval(require('node:fs').readFileSync(0, 'utf8'))",
            *args,
        ],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
