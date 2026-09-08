"""Run every test with python -m tests."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

suite = unittest.defaultTestLoader.discover(
    start_dir=str(ROOT / "tests"),
    top_level_dir=str(ROOT),
)
result = unittest.TextTestRunner(verbosity=2).run(suite)

raise SystemExit(0 if result.wasSuccessful() else 1)
