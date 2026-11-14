"""Stage 1: generate the raw dataset of expert games.

Two alpha-beta searchers play each other. The opening is narrow (only 4, 5, 6,
12, 19 lines at depths 1-5), so every game starts with a few random moves, and
a small fraction of later moves are random too. Those random moves are marked
``expert = 0`` and are not used as training labels; they only add variety.

Games are spread over a process pool: one game per task, ordered by game id in
the output so the file is identical for the same parameters.

    python -m barahai.data.generate
"""

from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from ..agents import AlphaBetaAgent
from ..engine import GameState
from ..mlops import load_params, write_json

OUT = Path("data/raw/games.jsonl")
STATS = Path("reports/generate.json")


def play_game(job: tuple[int, int, int, tuple[int, int], float]) -> dict:
    game_id, seed, depth, opening, epsilon = job
    rng = random.Random(seed * 1_000_003 + game_id)
    agents = (
        AlphaBetaAgent(depth=depth, seed=rng.randrange(1 << 30)),
        AlphaBetaAgent(depth=depth, seed=rng.randrange(1 << 30)),
    )
    s = GameState.new()
    actions: list[int] = []
    expert: list[int] = []
    n_random = rng.randint(*opening)
    while not s.is_terminal:
        legal = s.legal_actions()
        if len(actions) < n_random or (len(legal) > 1 and rng.random() < epsilon):
            a, is_expert = rng.choice(legal), 0
        else:
            a, is_expert = agents[s.turn].act(s), 1
        actions.append(a)
        expert.append(is_expert)
        s = s.apply(a)
    return {
        "id": game_id,
        "actions": actions,
        "expert": expert,
        "winner": s.result.winner,
        "reason": s.result.reason,
        "turns": s.turn_no,
    }


def main() -> None:
    params = load_params()
    p = params["generate"]
    jobs = [(i, params["seed"], p["depth"], tuple(p["opening_plies"]), p["epsilon"]) for i in range(p["games"])]
    t0 = time.perf_counter()
    games = []
    with Pool(p["workers"]) as pool:
        for n, g in enumerate(pool.imap_unordered(play_game, jobs, chunksize=4), 1):
            games.append(g)
            if n % 250 == 0:
                print(f"  {n}/{len(jobs)} games  {time.perf_counter() - t0:6.0f}s", flush=True)
    games.sort(key=lambda g: g["id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for g in games:
            f.write(json.dumps(g, separators=(",", ":")) + "\n")

    n = len(games)
    stats = {
        "games": n,
        "actions": sum(len(g["actions"]) for g in games),
        "expert_actions": sum(sum(g["expert"]) for g in games),
        "avg_turns": round(sum(g["turns"] for g in games) / n, 2),
        "p0_win_rate": round(sum(g["winner"] == 0 for g in games) / n, 4),
        "p1_win_rate": round(sum(g["winner"] == 1 for g in games) / n, 4),
        "draw_rate": round(sum(g["winner"] is None for g in games) / n, 4),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    write_json(STATS, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
