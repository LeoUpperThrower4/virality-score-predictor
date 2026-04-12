"""Run sample.mp4 through the pipeline and dump the response JSON baseline."""

import json
from pathlib import Path

from tribe_analyzer import TribeAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.mp4"
EXPECTED = FIXTURES / "sample_expected.json"


def main() -> None:
    if not SAMPLE.exists():
        raise SystemExit(f"Missing sample video: {SAMPLE}")

    analyzer = TribeAnalyzer()
    analyzer.load_model()
    result = analyzer.analyze_video(str(SAMPLE))

    EXPECTED.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {EXPECTED}")


if __name__ == "__main__":
    main()
