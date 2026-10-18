"""Built-in opponents: a random mover, a rule-based greedy bot and an alpha-beta searcher."""

from .alphabeta import AlphaBetaAgent, evaluate
from .base import Agent, RandomAgent
from .greedy import GreedyAgent


def make_agent(spec: str, seed: int | None = 0) -> Agent:
    """Build an agent from a short spec: ``random``, ``greedy``, ``alphabeta-d4``, ``alphabeta-t1.5``."""
    if spec == "random":
        return RandomAgent(seed)
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec.startswith("alphabeta-d"):
        return AlphaBetaAgent(depth=int(spec.removeprefix("alphabeta-d")), seed=seed)
    if spec.startswith("alphabeta-t"):
        return AlphaBetaAgent(depth=64, time_limit=float(spec.removeprefix("alphabeta-t")), seed=seed)
    raise ValueError(f"unknown agent spec {spec!r}")


__all__ = ["Agent", "AlphaBetaAgent", "GreedyAgent", "RandomAgent", "evaluate", "make_agent"]
