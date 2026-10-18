"""Play built-in agents against each other and print a result table.

    python scripts/arena.py --games 40 random greedy alphabeta-d2 alphabeta-d4

Each pairing plays both colours equally. Also reports engine speed.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time

from barahai.agents import make_agent
from barahai.engine import GameState


def play(a, b, seed: int) -> tuple[int | None, int]:
    """Agent a is player 0. Returns (winner, turns)."""
    a.reset()
    b.reset()
    s = GameState.new()
    while not s.is_terminal:
        s = s.apply((a, b)[s.turn].act(s))
    return s.result.winner, s.turn_no


def engine_speed(seconds: float = 2.0) -> float:
    rng = random.Random(0)
    actions = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        s = GameState.new()
        while not s.is_terminal:
            s = s.apply(rng.choice(s.legal_actions()))
            actions += 1
    return actions / seconds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("agents", nargs="+")
    p.add_argument("--games", type=int, default=20, help="games per pairing (split across colours)")
    args = p.parse_args()

    print(f"engine: {engine_speed():,.0f} random actions/s")
    for x, y in itertools.combinations(args.agents, 2):
        score = {"x": 0, "y": 0, "draw": 0}
        lengths = []
        t0 = time.perf_counter()
        for g in range(args.games):
            ax, ay = make_agent(x, seed=g), make_agent(y, seed=g + 1000)
            if g % 2 == 0:
                winner, turns = play(ax, ay, g)
                key = {0: "x", 1: "y", None: "draw"}[winner]
            else:
                winner, turns = play(ay, ax, g)
                key = {0: "y", 1: "x", None: "draw"}[winner]
            score[key] += 1
            lengths.append(turns)
        dt = time.perf_counter() - t0
        print(
            f"{x:>16} {score['x']:3d} - {score['y']:3d} {y:<16} draws {score['draw']:3d}"
            f"   avg {sum(lengths) / len(lengths):5.1f} turns   {dt:6.1f}s"
        )


if __name__ == "__main__":
    main()
