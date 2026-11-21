"""Write a serving bundle with small untrained models, for tests and CI image checks.

    python scripts/make_test_bundle.py models/serve

The real bundle comes from the ``register`` pipeline stage (``dvc pull models/serve``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from barahai.rl.policy import build_policy, export_onnx


def main(out: str = "models/serve") -> None:
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    for role, seed in (("champion", 0), ("shadow", 1)):
        torch.manual_seed(seed)
        export_onnx(build_policy({"net_arch": [32]}), str(path / f"{role}.onnx"))
    manifest = {
        "model_name": "barahai-policy",
        "champion": {"version": "test-1", "file": "champion.onnx"},
        "shadow": {"version": "test-2", "file": "shadow.onnx"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"test bundle written to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
