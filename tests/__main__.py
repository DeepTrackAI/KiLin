"""Entry point for the KíLín unit test suite.

Run the complete test suite using::

    uv run --no-sync python -m tests

On developer machines this also refreshes ``README.pypi.md`` before test
collection. CI environments skip regeneration to detect stale committed files.

"""

import os
from pathlib import Path
import unittest

from setup.generate_pypi_readme import main as generate_pypi_readme


def main() -> int:
    """Prepare the local README, then discover and run KíLín tests.

    The ``CI`` environment variable disables regeneration so automated
    checks cannot silently repair an outdated committed description.

    Returns
    -------
    int
        Exit code: 0 if preparation and tests pass, 1 otherwise.

    """

    ci = os.environ.get("CI", "").strip().lower()
    if ci in ("", "0", "false"):
        if generate_pypi_readme([]) != 0:
            return 1

    root = Path(__file__).resolve().parent

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root),
        pattern="test_*.py",
        top_level_dir=str(root.parent),
    )

    result = unittest.TextTestRunner(verbosity=2).run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
