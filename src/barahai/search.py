"""Network-guided search at play time (PUCT Monte Carlo tree search, as in AlphaZero).

The trained network gives two things for any position: a probability for each
legal move (the policy) and an estimate of who is winning (the value). The
search uses both to look ahead: it repeatedly walks down the tree choosing moves
that are promising (high policy prior) or proven good so far (high average
value), evaluates the new leaf with the network, and backs the value up. After
``simulations`` walks it plays the most-visited move.

No retraining is needed, and the strength grows with the number of simulations.
A chain capture keeps the same player to move, so values only flip sign when
the turn actually passes to the opponent.
"""

from __future__ import annotations

import math

import numpy as np

from .agents.base import Agent
from .engine import GameState
from .inference import OnnxAgent
from .rl.features import canonical_mask, from_canonical


def terminal_value(s: GameState) -> float:
    """Result for the player whose turn it is in a finished position."""
    w = s.result.winner
    return 0.0 if w is None else (1.0 if w == s.turn else -1.0)


class Node:
    __slots__ = ("state", "actions", "priors", "visits", "values", "children", "expanded")

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.expanded = False
        self.children: dict[int, Node] = {}

    def expand(self, logits: np.ndarray) -> None:
        mask = canonical_mask(self.state)
        idx = np.flatnonzero(mask)
        z = logits[idx] - logits[idx].max()
        p = np.exp(z)
        p /= p.sum()
        self.actions = [from_canonical(int(a), self.state.turn) for a in idx]
        self.priors = p
        self.visits = np.zeros(len(idx))
        self.values = np.zeros(len(idx))  # summed value, from this node's mover's view
        self.expanded = True


class SearchAgent(Agent):
    def __init__(self, net: OnnxAgent, simulations: int = 100, c_puct: float = 1.5, name: str = "search") -> None:
        self.net = net
        self.simulations = simulations
        self.c_puct = c_puct
        self.name = name
        self.last_value = 0.0
        self.last_visits: dict[int, int] = {}

    def _simulate(self, node: Node) -> float:
        """Run one walk from ``node``; returns the value for ``node``'s player to move."""
        s = node.state
        if s.is_terminal:
            return terminal_value(s)
        if not node.expanded:
            logits, value = self.net.evaluate(s)
            node.expand(logits)
            return value
        total = node.visits.sum()
        q = np.divide(node.values, node.visits, out=np.zeros_like(node.values), where=node.visits > 0)
        u = self.c_puct * node.priors * math.sqrt(total + 1) / (1 + node.visits)
        i = int(np.argmax(q + u))
        a = node.actions[i]
        child = node.children.get(a)
        if child is None:
            child = node.children[a] = Node(s.apply(a))
        v = self._simulate(child)
        if child.state.is_terminal or child.state.turn != s.turn:
            v = -v
        node.visits[i] += 1
        node.values[i] += v
        return v

    def search(self, state: GameState) -> Node:
        root = Node(state)
        for _ in range(self.simulations):
            self._simulate(root)
        return root

    def act(self, state: GameState) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        root = self.search(state)
        best = int(np.argmax(root.visits))
        self.last_value = float(root.values[best] / max(1, root.visits[best]))
        self.last_visits = {a: int(n) for a, n in zip(root.actions, root.visits, strict=True)}
        return root.actions[best]
