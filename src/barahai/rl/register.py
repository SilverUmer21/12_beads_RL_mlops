"""Stage 7: register the model in the MLflow Model Registry.

* Logs ``models/policy.onnx`` to the training run and registers it as a new
  version of ``register.model_name``.
* Tags the version with where it came from: git commit, data hash, training
  settings and evaluation scores.
* Promotion rule: the version gets the ``champion`` alias if its gate score
  (score against ``register.gate_opponent`` plus puzzle accuracy) beats the
  current champion's, or if there is no champion yet. Otherwise it gets the
  ``challenger`` alias.

Load the champion anywhere with ``mlflow.onnx.load_model("models:/barahai-policy@champion")``.

    python -m barahai.rl.register
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx

from ..mlops import file_md5, git_commit, load_params, read_json, setup_mlflow, write_json

MODEL = Path("models/policy.onnx")
OUT = Path("reports/registry.json")


def gate_score(metrics: dict, opponent: str) -> float:
    return round(metrics[f"{opponent.replace('-', '_')}_score"] + metrics["puzzle_accuracy"], 4)


def main() -> None:
    params = load_params()
    r = params["register"]
    ev = read_json("reports/eval.json")
    train = read_json("reports/train.json")
    score = gate_score(ev, r["gate_opponent"])

    mlflow = setup_mlflow()
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_id=train["mlflow_run_id"]):
        info = mlflow.onnx.log_model(
            onnx.load(str(MODEL)),
            name="model",
            registered_model_name=r["model_name"],
            input_example=None,
        )
    version = str(info.registered_model_version)

    tags = {
        "git_commit": git_commit(),
        "data_md5": file_md5("data/processed"),
        "puzzles_md5": file_md5("data/puzzles.jsonl"),
        "train_init": params["train"]["init"],
        "train_timesteps": str(train["timesteps"]),
        "gate_opponent": r["gate_opponent"],
        "gate_score": str(score),
        "puzzle_accuracy": str(ev["puzzle_accuracy"]),
    }
    for opp in params["evaluate"]["opponents"]:
        key = opp.replace("-", "_")
        tags[f"score_{key}"] = str(ev[f"{key}_score"])
    for k, v in tags.items():
        client.set_model_version_tag(r["model_name"], version, k, v)

    try:
        champ = client.get_model_version_by_alias(r["model_name"], "champion")
        champ_score = float(champ.tags.get("gate_score", "-inf"))
        champ_version = champ.version
    except Exception:  # no champion yet
        champ_score, champ_version = float("-inf"), None

    promoted = score > champ_score
    alias = "champion" if promoted else "challenger"
    client.set_registered_model_alias(r["model_name"], alias, version)
    client.update_model_version(
        r["model_name"],
        version,
        description=f"Maskable PPO policy ({params['train']['init']} start). Gate score {score} "
        f"vs previous champion {champ_score if champ_version else 'none'}.",
    )

    result = {
        "model_name": r["model_name"],
        "version": version,
        "alias": alias,
        "gate_score": score,
        "previous_champion": champ_version,
        "previous_champion_score": None if champ_version is None else champ_score,
        "run_id": train["mlflow_run_id"],
    }
    write_json(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
