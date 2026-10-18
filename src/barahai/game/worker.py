"""Search worker process.

Searching on a thread starves the UI: every pygame blit releases the GIL and
then waits up to a full switch interval to get it back from the busy search,
which adds up to frames of 300 ms or more. A separate process has its own
interpreter, so the window stays at 60 fps while the computer thinks.

Agents live in the worker between moves (keyed per game and seat) so their
transposition tables survive. Cancelling uses a shared watermark: every job id
at or below it is stale and aborts at its next node check.
"""

from __future__ import annotations

from ..agents import Agent, AlphaBetaAgent, GreedyAgent
from ..agents.alphabeta import SearchCancelled
from ..engine import GameState

_watermark = None
_agents: dict[tuple, Agent] = {}


def init(watermark) -> None:
    global _watermark
    _watermark = watermark


def build(level: str, seed: int) -> Agent:
    if level == "Naya":
        return GreedyAgent(seed=seed, epsilon=0.3)
    if level == "Khiladi":
        return AlphaBetaAgent(depth=3, seed=seed)
    if level == "Ustad":
        return AlphaBetaAgent(depth=64, time_limit=1.0, seed=seed, name="ustad")
    if level == "hint":
        return AlphaBetaAgent(depth=64, time_limit=0.7, seed=seed, name="hint")
    raise ValueError(f"unknown level {level!r}")


def think(job: int, key: tuple, level: str, seed: int, state: GameState) -> int | None:
    """Return the chosen action, or None if the job was cancelled."""
    if _watermark is not None and job <= _watermark.value:
        return None
    agent = _agents.get(key)
    if agent is None:
        if len(_agents) > 6:
            _agents.clear()
        agent = _agents[key] = build(level, seed)
    if isinstance(agent, AlphaBetaAgent) and _watermark is not None:
        agent.stop_check = lambda: job <= _watermark.value
    try:
        return agent.act(state)
    except SearchCancelled:
        return None


def warm_up() -> bool:
    return True
