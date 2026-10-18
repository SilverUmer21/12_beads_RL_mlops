from __future__ import annotations

import random
from abc import ABC, abstractmethod

from ..engine import GameState


class Agent(ABC):
    """Picks one atomic action. Mid-chain the agent is simply asked again."""

    name: str = "agent"

    @abstractmethod
    def act(self, state: GameState) -> int: ...

    def reset(self) -> None:
        """Called at the start of each game."""


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self._seed)

    def act(self, state: GameState) -> int:
        return self.rng.choice(state.legal_actions())
