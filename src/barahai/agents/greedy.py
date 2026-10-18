"""Rule-based baseline: grab the most beads this turn without hanging your own."""

from __future__ import annotations

import random

from ..engine import GameState
from .base import Agent

WIN = 1000


def best_capture_run(state: GameState) -> int:
    """Most beads the side to move can take this turn (0 if no capture)."""
    best = 0
    for a in state.legal_actions():
        if not state.is_capture(a):
            continue
        child = state.apply(a)
        gain = 1 + (best_capture_run(child) if child.turn == state.turn and not child.is_terminal else 0)
        best = max(best, gain)
    return best


def _finish_turn(state: GameState, player: int, taken: int):
    """Yield (end-of-turn state, beads taken) for every way to finish this turn."""
    if state.is_terminal or state.turn != player:
        yield state, taken
        return
    for a in state.legal_actions():
        yield from _finish_turn(state.apply(a), player, taken + state.is_capture(a))


class GreedyAgent(Agent):
    """Scores each action by beads won minus beads the opponent can win back next turn.

    ``epsilon`` mixes in random moves, which makes a gentler opponent.
    """

    name = "greedy"

    def __init__(self, seed: int | None = None, epsilon: float = 0.0) -> None:
        self._seed = seed
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self._seed)

    def score(self, state: GameState, action: int) -> float:
        player = state.turn
        child = state.apply(action)
        best = -float("inf")
        for end, taken in _finish_turn(child, player, int(state.is_capture(action))):
            if end.result is not None:
                value = 0 if end.result.winner is None else (WIN if end.result.winner == player else -WIN)
            else:
                value = taken - best_capture_run(end)
            best = max(best, value)
        return best

    def act(self, state: GameState) -> int:
        actions = state.legal_actions()
        if len(actions) == 1:
            return actions[0]
        if self.epsilon and self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        scored = [(self.score(state, a), self.rng.random(), a) for a in actions]
        return max(scored)[2]
