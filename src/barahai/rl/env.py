"""Gymnasium environment: the agent plays one colour, an opponent plays the other.

* The colour is random each episode, and the board is always shown from the
  agent's side (see ``features``), so one network learns both colours.
* The opponent is drawn from a weighted mix each episode (random, greedy,
  alpha-beta, or ``self``: one of the last ``snapshot_ring`` frozen copies of
  the agent, so it keeps beating its past selves rather than only the latest).
* A turn can be several actions: after a jump that can continue, the same
  player acts again. The environment just hands control back to whoever is to
  move, so the agent sees each step of its own chain.
* ``action_masks()`` gives the legal actions; MaskablePPO never picks an illegal
  one. If an illegal action arrives anyway (e.g. from ``check_env``) the episode
  ends as a loss instead of raising.

Rewards: +1 win, -1 loss, ``draw_reward`` for a draw, plus ``capture_reward``
per bead of material swing during the episode (small, to speed up learning).
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np

from ..agents import Agent, make_agent
from ..engine import STANDARD, GameState, Rules
from .features import canonical_mask, encode, from_canonical
from .policy import ACTION_SPACE, OBS_SPACE, TorchAgent, build_policy

SELF = "self"


class BeadsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponents: dict[str, float] | None = None,
        draw_reward: float = 0.0,
        capture_reward: float = 0.0,
        opening_plies: tuple[int, int] = (0, 0),
        model_params: dict | None = None,
        snapshot_ring: int = 1,
        rules: Rules = STANDARD,
        render_mode: str | None = None,
    ) -> None:
        self.observation_space = OBS_SPACE
        self.action_space = ACTION_SPACE
        self.opponents = dict(opponents or {"random": 1.0})
        self.draw_reward = draw_reward
        self.capture_reward = capture_reward
        self.opening_plies = tuple(opening_plies)
        self.rules = rules
        self.render_mode = render_mode
        self._snapshots: list = []
        self._ring = max(1, snapshot_ring)
        self._model_params = model_params
        if SELF in self.opponents:
            import torch

            torch.set_num_threads(1)
            if model_params is None:
                raise ValueError("self-play needs model_params")
        self.me = 0
        self.opponent: Agent | None = None
        self.opponent_name = ""
        self.state = GameState.new(rules)
        self._rng = random.Random()

    # --- self-play -------------------------------------------------------------

    def set_snapshot(self, state_dict: dict) -> None:
        """Add a frozen copy of the agent to the self-play ring (oldest dropped)."""
        if self._model_params is None:
            return
        policy = build_policy(self._model_params)
        policy.load_state_dict(state_dict)
        policy.eval()
        self._snapshots = (self._snapshots + [policy])[-self._ring :]

    def _make_opponent(self, spec: str) -> Agent:
        seed = self._rng.randrange(1 << 30)
        if spec == SELF:
            if not self._snapshots:
                raise ValueError("no self-play snapshot")
            return TorchAgent(self._rng.choice(self._snapshots), sample=True, seed=seed, name=SELF)
        return make_agent(spec, seed=seed)

    # --- gym API -----------------------------------------------------------------

    def _material(self) -> int:
        return self.state.count(self.me) - self.state.count(1 - self.me)

    def _advance(self) -> None:
        """Let the opponent move until it is the agent's turn or the game ends."""
        s = self.state
        while not s.is_terminal and s.turn != self.me:
            s = s.apply(self.opponent.act(s))
        self.state = s

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None or not hasattr(self, "_seeded"):
            self._rng = random.Random(int(self.np_random.integers(1 << 62)))
            self._seeded = True
        specs, weights = zip(*self.opponents.items(), strict=False)
        self.opponent_name = self._rng.choices(specs, weights=weights)[0]
        self.opponent = self._make_opponent(self.opponent_name)
        self.me = self._rng.randrange(2)
        s = GameState.new(self.rules)
        for _ in range(self._rng.randint(*self.opening_plies)):
            if s.is_terminal:
                break
            s = s.apply(self._rng.choice(s.legal_actions()))
        self.state = s
        self._advance()
        if self.state.is_terminal:  # extremely rare: opening ended the game
            return self.reset()
        return encode(self.state), {"opponent": self.opponent_name}

    def step(self, action):
        s = self.state
        a = from_canonical(int(action), s.turn)
        if not s.is_legal(a):
            return encode(s), -1.0, True, False, {"illegal": True, "outcome": "loss", "opponent": self.opponent_name}
        before = self._material()
        self.state = s.apply(a)
        self._advance()
        reward = self.capture_reward * (self._material() - before)
        info: dict = {"opponent": self.opponent_name}
        terminated = self.state.is_terminal
        if terminated:
            w = self.state.result.winner
            if w is None:
                reward += self.draw_reward
                info["outcome"] = "draw"
            elif w == self.me:
                reward += 1.0
                info["outcome"] = "win"
            else:
                reward -= 1.0
                info["outcome"] = "loss"
        return encode(self.state), float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        return canonical_mask(self.state)

    def render(self):
        return str(self.state)
