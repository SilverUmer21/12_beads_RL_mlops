"""Model quality gate for CI: re-check the served champion on the puzzle suite.

CI pulls the serving bundle (``models/serve``) and the puzzle suite with DVC,
then this script:

1. checks that the champion in the bundle is the version the registry stage
   recorded in ``reports/registry.json`` (no stale or swapped model), and
2. re-measures its puzzle accuracy and fails if it is below
   ``gate.min_puzzle_accuracy`` or more than ``gate.max_drop`` below the
   accuracy recorded at registration (the result must reproduce).

    python -m barahai.rl.gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..inference import OnnxAgent
from ..mlops import load_params, read_json
from .evaluate import puzzle_accuracy

BUNDLE = Path("models/serve")


def main() -> int:
    params = load_params()["gate"]
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    registry = read_json("reports/registry.json")
    champion = manifest["champion"]
    registered = str(registry["serving"]["champion"]["version"])
    recorded = float(registry["candidate_gate"]["puzzle_accuracy"]) if registry["alias"] == "champion" else None

    acc, n = puzzle_accuracy(OnnxAgent(str(BUNDLE / champion["file"])))
    print(f"champion v{champion['version']}: puzzle accuracy {acc:.4f} on {n} puzzles")
    failures = []
    if str(champion["version"]) != registered:
        failures.append(f"bundle champion v{champion['version']} != registered v{registered}")
    if acc < params["min_puzzle_accuracy"]:
        failures.append(f"puzzle accuracy {acc:.4f} < minimum {params['min_puzzle_accuracy']}")
    if recorded is not None and acc < recorded - params["max_drop"]:
        failures.append(f"puzzle accuracy {acc:.4f} dropped from recorded {recorded:.4f}")
    for f in failures:
        print("GATE FAILED:", f)
    if not failures:
        print("gate passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
