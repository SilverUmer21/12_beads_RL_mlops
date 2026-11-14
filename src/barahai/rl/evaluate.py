"""Stage 6: evaluate the trained policy (the exported ONNX model, as it will be served).

* Matches: ``games`` per opponent, half as each colour. Every game starts with a
  few random moves from a fixed seed, because the opening is so narrow that
  deterministic players would otherwise replay the same game.
* Puzzles: share of puzzle positions where the policy picks one of the best moves.
* ``score`` per opponent = (wins + draws / 2) / games.

Metrics go to ``reports/eval.json`` (tracked by DVC) and to the training run in
MLflow.

    python -m barahai.rl.evaluate
"""

from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from ..agents import make_agent
from ..engine import GameState
from ..mlops import load_params, read_json, setup_mlflow, write_json
from .policy import OnnxAgent

MODEL = Path("models/policy.onnx")
PUZZLES = Path("data/puzzles.jsonl")
METRICS = Path("reports/eval.json")

_agent: OnnxAgent | None = None


def _init(model_path: str) -> None:
    global _agent
    _agent = OnnxAgent(model_path, threads=1)


def play(job: tuple[str, int, int, tuple[int, int]]) -> tuple[str, str, int]:
    """One game; returns (opponent, outcome for the model, turns)."""
    opp_spec, game_no, seed, opening = job
    rng = random.Random(f"{seed}:{opp_spec}:{game_no}")
    opp = make_agent(opp_spec, seed=rng.randrange(1 << 30))
    me = game_no % 2
    s = GameState.new()
    for _ in range(rng.randint(*opening)):
        if s.is_terminal:
            break
        s = s.apply(rng.choice(s.legal_actions()))
    while not s.is_terminal:
        s = s.apply((_agent if s.turn == me else opp).act(s))
    w = s.result.winner
    outcome = "draw" if w is None else ("win" if w == me else "loss")
    return opp_spec, outcome, s.turn_no


def puzzle_accuracy(agent: OnnxAgent, path: Path = PUZZLES) -> tuple[float, int]:
    solved = total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            z = json.loads(line)
            s = GameState.new()
            for a in z["actions"]:
                s = s.apply(a)
            solved += agent.act(s) in z["best"]
            total += 1
    return (solved / total if total else 0.0), total


def main() -> None:
    params = load_params()
    p = params["evaluate"]
    t0 = time.perf_counter()
    jobs = [(o, g, params["seed"], tuple(p["opening_plies"])) for o in p["opponents"] for g in range(p["games"])]
    results: dict[str, list] = {o: [] for o in p["opponents"]}
    with Pool(p["workers"], initializer=_init, initargs=(str(MODEL),)) as pool:
        for opp, outcome, turns in pool.imap_unordered(play, jobs, chunksize=2):
            results[opp].append((outcome, turns))

    metrics: dict = {}
    for opp, rs in results.items():
        n = len(rs)
        w = sum(o == "win" for o, _ in rs)
        d = sum(o == "draw" for o, _ in rs)
        key = opp.replace("-", "_")
        metrics[f"{key}_win"] = round(w / n, 4)
        metrics[f"{key}_draw"] = round(d / n, 4)
        metrics[f"{key}_loss"] = round((n - w - d) / n, 4)
        metrics[f"{key}_score"] = round((w + d / 2) / n, 4)
        metrics[f"{key}_avg_turns"] = round(sum(t for _, t in rs) / n, 1)

    acc, n_puzzles = puzzle_accuracy(OnnxAgent(str(MODEL)))
    metrics["puzzle_accuracy"] = round(acc, 4)
    metrics["puzzles"] = n_puzzles
    metrics["games_per_opponent"] = p["games"]
    metrics["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(METRICS, metrics)

    run_id = read_json("reports/train.json")["mlflow_run_id"]
    mlflow = setup_mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(METRICS))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
