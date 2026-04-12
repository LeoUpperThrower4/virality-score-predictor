"""Regression snapshot test for the full /api/analyze pipeline.

Skipped when fixtures are missing — see fixtures/README.md to bootstrap.
"""

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.mp4"
EXPECTED = FIXTURES / "sample_expected.json"

# Scalar fields whose numeric values may jitter slightly between runs.
# Snapshot asserts structural equality + exact match on categorical fields;
# numeric fields get a looser tolerance.
NUMERIC_TOLERANCE = 1e-3


def _approx_equal(a, b, path="") -> list[str]:
    """Return a list of human-readable diff messages. Empty = match."""
    if type(a) is not type(b):
        return [f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})"]
    if isinstance(a, dict):
        diffs = []
        if set(a) != set(b):
            return [f"{path}: key set mismatch (extra_a={set(a) - set(b)}, extra_b={set(b) - set(a)})"]
        for k in a:
            diffs.extend(_approx_equal(a[k], b[k], f"{path}.{k}"))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: list length mismatch ({len(a)} vs {len(b)})"]
        diffs = []
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_approx_equal(x, y, f"{path}[{i}]"))
        return diffs
    if isinstance(a, float):
        if abs(a - b) > NUMERIC_TOLERANCE:
            return [f"{path}: {a} != {b} (tolerance {NUMERIC_TOLERANCE})"]
        return []
    if a != b:
        return [f"{path}: {a!r} != {b!r}"]
    return []


@pytest.mark.skipif(
    not (SAMPLE.exists() and EXPECTED.exists()),
    reason="Snapshot fixtures not present — see fixtures/README.md",
)
def test_analyze_matches_snapshot():
    from tribe_analyzer import TribeAnalyzer

    analyzer = TribeAnalyzer()
    analyzer.load_model()
    actual = analyzer.analyze_video(str(SAMPLE))
    expected = json.loads(EXPECTED.read_text())

    diffs = _approx_equal(actual, expected)
    assert not diffs, "Snapshot mismatch:\n  " + "\n  ".join(diffs[:20])
