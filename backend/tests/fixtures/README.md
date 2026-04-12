# Snapshot test fixtures

The regression snapshot test in `backend/tests/test_snapshot.py` compares
`/api/analyze` output against a committed baseline. The baseline is not
checked in by default because it requires TRIBE v2 weights (HF token + GPU
strongly recommended) to generate.

## Capturing the baseline

1. Place a representative Reel (≤ 30s, mp4) at `backend/tests/fixtures/sample.mp4`.
2. From `backend/`, with `HF_TOKEN` set and the venv active:

   ```bash
   python -m tests.capture_snapshot
   ```

3. This writes `backend/tests/fixtures/sample_expected.json`.
4. Commit both files.

## Running the test

```bash
cd backend && pytest tests/test_snapshot.py -v
```

The test is skipped when either fixture is missing.
