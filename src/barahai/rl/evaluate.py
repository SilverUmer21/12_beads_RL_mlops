"""Stage: evaluate the trained model (the exported ONNX file, as it is served).

Two ways of playing are measured:

* ``policy``: the network alone (pick the highest-scoring legal move)
* ``search``: the network guiding a ``search_simulations``-walk tree search

Protocol (fairer than Phase 1):

* A fixed **opening book**: ``openings`` distinct start positions, each reached
  by a few random moves from a fixed seed. Every opening is played **twice with
  colours swapped**, so neither side benefits from a lucky opening.
* **Score** = (wins + draws / 2) / games, with a 95% confidence interval from the
  per-opening pair scores, converted to an **Elo difference** against each bot.
* **Puzzles**: share of puzzle positions where the move is one of the best.

Results go to ``reports/eval.json`` (tracked by DVC) and to the training run in
MLflow.

    python -m barahai.rl.evaluate
"""

from __future__ import annotations

import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

from ..agents import make_agent
from ..engine import GameState
from ..inference import OnnxAgent
from ..mlops import load_params, read_json, setup_mlflow, write_json
from ..search import SearchAgent

MODEL = Path("models/policy.onnx")
PUZZLES = Path("data/puzzles.jsonl")
METRICS = Path("reports/eval.json")

_players: dict = {}


def opening_book(n: int, plies: tuple[int, int], seed: int) -> list[list[int]]:
    """``n`` distinct, still-running start positions as action prefixes."""
    rng = random.Random(f"openings:{seed}")
    seen, book = set(), []
    while len(book) < n:
        s, prefix = GameState.new(), []
        for _ in range(rng.randint(*plies)):
            if s.is_terminal:
                break
            a = rng.choice(s.legal_actions())
            prefix.append(a)
            s = s.apply(a)
        if not s.is_terminal and s.key not in seen:
            seen.add(s.key)
            book.append(prefix)
    return book


def _init(model_path: str, simulations: int) -> None:
    net = OnnxAgent(model_path, threads=1)
    _players["policy"] = net
    _players["search"] = SearchAgent(net, simulations=simulations)


def play(job: tuple[str, str, list[int], int, int]) -> tuple[str, str, int, float]:
    """One game from an opening; returns (player, opponent, opening index, score for the model)."""
    who, opp_spec, prefix, opening_no, colour = job
    opp = make_agent(opp_spec, seed=opening_no * 2 + colour)
    me = colour
    s = GameState.new()
    for a in prefix:
        s = s.apply(a)
    agent = _players[who]
    while not s.is_terminal:
        s = s.apply((agent if s.turn == me else opp).act(s))
    w = s.result.winner
    return who, opp_spec, opening_no, 0.5 if w is None else (1.0 if w == me else 0.0)


def elo(score: float) -> float:
    s = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def summarise(pairs: dict[int, list[float]]) -> dict[str, float]:
    """Score, 95% CI and Elo from games grouped by opening (colour-swapped pairs)."""
    pair_scores = [sum(v) / len(v) for v in pairs.values()]
    n = len(pair_scores)
    mean = sum(pair_scores) / n
    var = sum((x - mean) ** 2 for x in pair_scores) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    return {
        "score": round(mean, 4),
        "score_ci95": round(half, 4),
        "elo": round(elo(mean), 1),
        "elo_low": round(elo(mean - half), 1),
        "elo_high": round(elo(mean + half), 1),
        "games": sum(len(v) for v in pairs.values()),
    }


def puzzle_accuracy(agent, path: Path = PUZZLES) -> tuple[float, int]:
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


def run_matches(model: str, players: list[str], opponents: list[str], p: dict, seed: int) -> dict:
    book = opening_book(p["openings"], tuple(p["opening_plies"]), seed)
    jobs = [(who, o, pre, i, c) for who in players for o in opponents for i, pre in enumerate(book) for c in (0, 1)]
    results: dict = {}
    with Pool(p["workers"], initializer=_init, initargs=(model, p["search_simulations"])) as pool:
        for who, opp, i, sc in pool.imap_unordered(play, jobs, chunksize=4):
            results.setdefault((who, opp), {}).setdefault(i, []).append(sc)
    return {f"{who}_{opp.replace('-', '_')}": summarise(r) for (who, opp), r in results.items()}


def gate_metrics(model: str, p: dict, seed: int, opponent: str) -> dict:
    """The numbers the promotion rule compares (used for both the candidate and the champion)."""
    m = run_matches(model, ["search"], [opponent], p, seed)
    net = OnnxAgent(model)
    return {
        "search_score": m[f"search_{opponent.replace('-', '_')}"]["score"],
        "puzzle_accuracy": puzzle_accuracy(net)[0],
        "search_puzzle_accuracy": puzzle_accuracy(SearchAgent(net, p["search_simulations"]))[0],
    }


def main() -> None:
    params = load_params()
    p = params["evaluate"]
    t0 = time.perf_counter()
    matches = run_matches(str(MODEL), ["policy", "search"], p["opponents"], p, params["seed"])

    net = OnnxAgent(str(MODEL))
    acc, n_puzzles = puzzle_accuracy(net)
    search_acc, _ = puzzle_accuracy(SearchAgent(net, p["search_simulations"]))
    metrics: dict = {"matches": matches}
    metrics["puzzle_accuracy"] = round(acc, 4)
    metrics["search_puzzle_accuracy"] = round(search_acc, 4)
    metrics["puzzles"] = n_puzzles
    metrics["games_per_opponent"] = 2 * p["openings"]
    metrics["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(METRICS, metrics)

    flat = {f"eval_{k}_{m}": v for k, d in matches.items() for m, v in d.items() if m in ("score", "elo")}
    flat |= {f"eval_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    run_id = read_json("reports/train.json")["mlflow_run_id"]
    mlflow = setup_mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(flat)
        mlflow.log_artifact(str(METRICS))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
