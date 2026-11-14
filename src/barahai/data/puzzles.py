"""Stage 3: build the puzzle suite, a fixed test of move quality.

Candidate positions come from **validation** games only, so the model never
trained on them. A deeper alpha-beta search scores every legal move; a position
becomes a puzzle when the best move beats the second best by at least
``margin`` centi-beads (or wins outright). The answer is the set of moves that
share the best score.

Each puzzle stores the action prefix that reaches it, so replaying it gives the
exact state including the draw-rule counters.

    python -m barahai.data.puzzles
"""

from __future__ import annotations

import json
import random
from multiprocessing import Pool
from pathlib import Path

from ..agents.alphabeta import INF, AlphaBetaAgent
from ..engine import GameState
from ..mlops import load_params, write_json
from .preprocess import load_games, split_ids

OUT = Path("data/puzzles.jsonl")
STATS = Path("reports/puzzles.json")


def score_moves(s: GameState, depth: int) -> dict[int, int]:
    """Search score of every legal action, from the mover's point of view."""
    ag = AlphaBetaAgent(depth=depth, seed=0)
    ag._deadline = None
    scores = {}
    for a in s.legal_actions():
        child = s.apply(a)
        if child.turn == s.turn:  # chain continues: same player, no depth used
            scores[a] = ag._negamax(child, depth, -INF, INF, 1)
        else:
            scores[a] = -ag._negamax(child, depth - 1, -INF, INF, 1)
    return scores


def solve(job: tuple[list[int], int, int]) -> dict | None:
    prefix, depth, margin = job
    s = GameState.new()
    for a in prefix:
        s = s.apply(a)
    scores = score_moves(s, depth)
    ranked = sorted(scores.values(), reverse=True)
    best, second = ranked[0], ranked[1]
    if best - second < margin:
        return None
    return {
        "actions": prefix,
        "best": sorted(a for a, v in scores.items() if v == best),
        "best_score": best,
        "gap": best - second,
        "legal": len(scores),
    }


def candidates(games: list[dict], n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pool = []
    for g in games:
        s = GameState.new()
        for i, a in enumerate(g["actions"]):
            if s.chain < 0 and len(s.legal_actions()) >= 2:
                pool.append(g["actions"][:i])
            s = s.apply(a)
    rng.shuffle(pool)
    return pool[:n]


def main() -> None:
    params = load_params()
    p = params["puzzles"]
    games = load_games()
    val_ids = split_ids([g["id"] for g in games], params["preprocess"]["val_fraction"], params["seed"])
    jobs = [(c, p["depth"], p["margin"]) for c in candidates([g for g in games if g["id"] in val_ids],
                                                            p["candidates"], params["seed"])]
    with Pool(p["workers"]) as pool:
        found = [r for r in pool.imap(solve, jobs, chunksize=8) if r is not None]
    puzzles = found[: p["count"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for z in puzzles:
            f.write(json.dumps(z, separators=(",", ":")) + "\n")
    stats = {
        "candidates": len(jobs),
        "decisive": len(found),
        "puzzles": len(puzzles),
        "avg_legal": round(sum(z["legal"] for z in puzzles) / max(1, len(puzzles)), 2),
        "single_answer_share": round(sum(len(z["best"]) == 1 for z in puzzles) / max(1, len(puzzles)), 3),
    }
    write_json(STATS, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
