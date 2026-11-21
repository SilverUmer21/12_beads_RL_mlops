"""Stage: register the model in the MLflow Model Registry, with a champion/challenger gate.

* Logs ``models/policy.onnx`` to the training run and registers it as a new
  version of ``register.model_name``, tagged with where it came from (git
  commit, data hashes, settings) and its evaluation scores.
* **Promotion rule (head to head on the same test):** the current champion is
  downloaded from the registry and measured with exactly the same gate as the
  new version: search score against ``register.gate_opponent`` on the fixed
  opening book, plus puzzle accuracy. The new version becomes ``champion`` only
  if its gate score is higher by at least ``register.min_improvement``;
  otherwise it becomes ``challenger``.
* Writes the **serving bundle** ``models/serve/``: ``champion.onnx`` (answers
  every request), ``shadow.onnx`` (scored silently on every request: the
  challenger, or the previous champion right after a promotion) and
  ``manifest.json``. DVC tracks the bundle, and the deploy pipeline bakes it
  into the server image.

Load the champion anywhere with ``mlflow.onnx.load_model("models:/barahai-policy@champion")``.

    python -m barahai.rl.register
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx

from ..mlops import file_md5, git_commit, load_params, read_json, setup_mlflow, write_json
from .evaluate import gate_metrics

MODEL = Path("models/policy.onnx")
OUT = Path("reports/registry.json")
SERVE = Path("models/serve")


def write_bundle(champion: tuple[str, Path], shadow: tuple[str, Path] | None, name: str) -> None:
    """Copy the serving models into models/serve with a manifest."""
    if SERVE.exists():
        shutil.rmtree(SERVE)
    SERVE.mkdir(parents=True)
    manifest: dict = {"model_name": name}
    for role, entry in (("champion", champion), ("shadow", shadow)):
        if entry is None:
            manifest[role] = None
            continue
        version, path = entry
        shutil.copyfile(path, SERVE / f"{role}.onnx")
        manifest[role] = {"version": version, "file": f"{role}.onnx"}
    (SERVE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def gate_score(m: dict) -> float:
    return round(m["search_score"] + m["puzzle_accuracy"], 4)


def champion_model(client, mlflow, name: str):
    """(version, local onnx path) of the current champion, or (None, None)."""
    try:
        v = client.get_model_version_by_alias(name, "champion")
    except Exception:
        return None, None
    local = mlflow.artifacts.download_artifacts(f"models:/{name}@champion", dst_path="models/champion")
    path = next(Path(local).rglob("*.onnx"))
    return v.version, str(path)


def main() -> None:
    params = load_params()
    r, ev_params = params["register"], params["evaluate"]
    ev = read_json("reports/eval.json")
    train = read_json("reports/train.json")

    mlflow = setup_mlflow()
    client = mlflow.MlflowClient()
    candidate = gate_metrics(str(MODEL), ev_params, params["seed"], r["gate_opponent"])
    champ_version, champ_path = champion_model(client, mlflow, r["model_name"])
    champion = gate_metrics(champ_path, ev_params, params["seed"], r["gate_opponent"]) if champ_path else None

    with mlflow.start_run(run_id=train["mlflow_run_id"]):
        info = mlflow.onnx.log_model(onnx.load(str(MODEL)), name="model", registered_model_name=r["model_name"])
        mlflow.log_metrics({f"gate_candidate_{k}": v for k, v in candidate.items()})
        if champion:
            mlflow.log_metrics({f"gate_champion_{k}": v for k, v in champion.items()})
    version = str(info.registered_model_version)

    score = gate_score(candidate)
    champ_score = gate_score(champion) if champion else None
    promoted = champ_score is None or score >= champ_score + r["min_improvement"]
    alias = "champion" if promoted else "challenger"
    client.set_registered_model_alias(r["model_name"], alias, version)
    if promoted and champ_version:
        client.set_registered_model_alias(r["model_name"], "previous", champ_version)
    if promoted:
        write_bundle((version, MODEL), (champ_version, Path(champ_path)) if champ_path else None, r["model_name"])
    else:
        write_bundle((champ_version, Path(champ_path)), (version, MODEL), r["model_name"])

    tags = {
        "git_commit": git_commit(),
        "data_md5": file_md5("data/labelled"),
        "puzzles_md5": file_md5("data/puzzles.jsonl"),
        "net_arch": str(params["model"]["net_arch"]),
        "train_init": params["train"]["init"],
        "train_timesteps": str(train["timesteps"]),
        "gate_opponent": r["gate_opponent"],
        "gate_score": str(score),
        "puzzle_accuracy": str(ev["puzzle_accuracy"]),
        "search_puzzle_accuracy": str(ev["search_puzzle_accuracy"]),
        "outputs": "logits,value",
    }
    for key, m in ev["matches"].items():
        tags[f"elo_{key}"] = str(m["elo"])
    for k, v in tags.items():
        client.set_model_version_tag(r["model_name"], version, k, v)
    client.update_model_version(
        r["model_name"],
        version,
        description=f"Policy+value network ({params['model']['net_arch']}), pretrained on depth-5 search labels, "
        f"then KL-anchored Maskable PPO. Gate {score} vs champion v{champ_version}: {champ_score}.",
    )

    result = {
        "model_name": r["model_name"],
        "version": version,
        "alias": alias,
        "gate_score": score,
        "candidate_gate": candidate,
        "previous_champion": champ_version,
        "previous_champion_gate_score": champ_score,
        "previous_champion_gate": champion,
        "run_id": train["mlflow_run_id"],
        "serving": json.loads((SERVE / "manifest.json").read_text(encoding="utf-8")),
    }
    write_json(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
