"""CI gate: fail if the champion's puzzle accuracy drops.

Replays the puzzle suite with the DVC-tracked model (``models/policy.onnx``, the
champion: get it with ``dvc pull``) and compares the accuracy with a baseline
``eval.json``. Exits with status 1 if accuracy < baseline - tolerance.

The baseline is the ``puzzle_accuracy`` recorded by the evaluate stage. CI uses
the base branch's ``reports/eval.json`` on pull requests (a new model must not be
worse) and the current one on pushes (code changes must not break the champion).
The tolerance (``gate.puzzle_tolerance`` in params.yaml) absorbs float
differences between machines; one puzzle is about 0.0034.

    python -m barahai.rl.gate [--baseline reports/eval.json] [--tolerance 0.01]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..mlops import load_params, read_json
from .evaluate import MODEL, PUZZLES, puzzle_accuracy
from .policy import OnnxAgent


def passes(accuracy: float, baseline: float, tolerance: float) -> bool:
    return accuracy >= baseline - tolerance - 1e-9


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default="reports/eval.json", help="eval.json holding the baseline puzzle_accuracy")
    ap.add_argument("--tolerance", type=float, default=None, help="default: gate.puzzle_tolerance in params.yaml")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--puzzles", default=str(PUZZLES))
    args = ap.parse_args(argv)

    tol = args.tolerance if args.tolerance is not None else load_params()["gate"]["puzzle_tolerance"]
    baseline = read_json(args.baseline)["puzzle_accuracy"]
    acc, n = puzzle_accuracy(OnnxAgent(args.model), Path(args.puzzles))
    ok = passes(acc, baseline, tol)

    report = (
        f"Puzzle gate {'PASSED' if ok else 'FAILED'}: accuracy {acc:.4f} on {n} puzzles, "
        f"baseline {baseline:.4f} ({args.baseline}), tolerance {tol}"
    )
    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
