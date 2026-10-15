"""Entry point for the QAChat unit test suite.

Run the complete test suite using::

    uv run --no-sync python -m tests

"""

from pathlib import Path
import unittest


def main():
    """Discover and run all QAChat unit tests.

    Returns
    -------
    int
        Exit code: 0 if all tests pass, 1 otherwise.

    """

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
