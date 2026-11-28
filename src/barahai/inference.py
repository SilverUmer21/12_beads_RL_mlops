"""Playing with an exported model. Needs only numpy and onnxruntime (no PyTorch).

This is everything the inference server needs: load the ONNX model, encode a
position, and pick a move either straight from the policy or with search
(``barahai.search``).
"""

from __future__ import annotations

import random

import numpy as np

from .agents.base import Agent
from .engine import GameState
from .rl.features import canonical_mask, encode, from_canonical


def choose(logits: np.ndarray, mask: np.ndarray, rng: random.Random | None = None, temperature: float = 1.0) -> int:
    """Best legal action, or a sample from the masked softmax when ``rng`` is given."""
    masked = np.where(mask, logits, -np.inf)
    if rng is None:
        return int(np.argmax(masked))
    z = masked / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return int(rng.choices(range(len(p)), weights=p)[0])


def move_probabilities(logits: np.ndarray, state: GameState) -> dict[int, float]:
    """Engine action -> probability under the policy, for the legal moves only."""
    mask = canonical_mask(state)
    idx = np.flatnonzero(mask)
    z = logits[idx] - logits[idx].max()
    p = np.exp(z)
    p /= p.sum()
    return {from_canonical(int(a), state.turn): float(pa) for a, pa in zip(idx, p, strict=True)}


class OnnxAgent(Agent):
    """Plays with an exported ONNX model (outputs ``logits`` and optionally ``value``)."""

    def __init__(self, path: str, threads: int = 1, name: str = "neural") -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.has_value = len(self.session.get_outputs()) > 1
        self.name = name

    def logits(self, state: GameState) -> np.ndarray:
        return self.session.run(None, {self.input: encode(state)[None, :]})[0][0]

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """(policy logits, value in [-1, 1] for the player to move; 0 if the model has no value head)."""
        out = self.session.run(None, {self.input: encode(state)[None, :]})
        value = float(np.clip(out[1][0][0], -1.0, 1.0)) if self.has_value else 0.0
        return out[0][0], value

    def act(self, state: GameState) -> int:
        return from_canonical(choose(self.logits(state), canonical_mask(state)), state.turn)
