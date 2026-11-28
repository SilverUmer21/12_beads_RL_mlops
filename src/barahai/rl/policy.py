"""The policy network and agents that play with it.

The network is Stable-Baselines3's ``MaskableActorCriticPolicy`` (an MLP with a
policy head over the 201 actions and a value head). Pretraining, PPO and
self-play all use this one definition, so weights move between stages as a
plain ``state_dict``.

For play we only need the policy head. ``ActorLogits`` wraps it as a small
``nn.Module`` (observation -> logits); the same module is exported to ONNX.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from ..agents.base import Agent
from ..engine import N_ACTIONS, GameState
from ..inference import OnnxAgent, choose  # noqa: F401  (re-exported for older imports)
from .features import OBS_SIZE, canonical_mask, encode, from_canonical

OBS_SPACE = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
ACTION_SPACE = spaces.Discrete(N_ACTIONS)


def policy_kwargs(model_params: dict) -> dict:
    arch = list(model_params["net_arch"])
    return {"net_arch": {"pi": arch, "vf": arch}}


def build_policy(model_params: dict) -> MaskableActorCriticPolicy:
    return MaskableActorCriticPolicy(OBS_SPACE, ACTION_SPACE, lambda _: 0.0, **policy_kwargs(model_params))


class ActorLogits(nn.Module):
    """observation batch (N, 106) -> action logits (N, 201). No masking."""

    def __init__(self, policy: MaskableActorCriticPolicy) -> None:
        super().__init__()
        self.features = policy.pi_features_extractor
        self.mlp = policy.mlp_extractor.policy_net
        self.head = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(self.features(obs)))



class TorchAgent(Agent):
    """Plays with an in-memory policy (used for self-play opponents)."""

    def __init__(self, policy: MaskableActorCriticPolicy, sample: bool = False, seed: int | None = None,
                 name: str = "neural") -> None:
        self.net = ActorLogits(policy).eval()
        self.sample = sample
        self.rng = random.Random(seed) if sample else None
        self.name = name

    def act(self, state: GameState) -> int:
        with torch.no_grad():
            logits = self.net(torch.as_tensor(encode(state)).unsqueeze(0))[0].numpy()
        return from_canonical(choose(logits, canonical_mask(state), self.rng), state.turn)


class PolicyValue(nn.Module):
    """observation batch (N, 106) -> (action logits (N, 201), value (N, 1)). Exported to ONNX."""

    def __init__(self, policy: MaskableActorCriticPolicy) -> None:
        super().__init__()
        self.actor = ActorLogits(policy)
        self.vf_features = policy.vf_features_extractor
        self.vf_mlp = policy.mlp_extractor.value_net
        self.vf_head = policy.value_net

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.vf_head(self.vf_mlp(self.vf_features(obs)))


def export_onnx(policy: MaskableActorCriticPolicy, path: str) -> None:
    """Export logits and value. Outputs: ``logits`` (N, 201) and ``value`` (N, 1)."""
    net = PolicyValue(policy).eval()
    dummy = torch.zeros(1, OBS_SIZE, dtype=torch.float32)
    torch.onnx.export(
        net,
        (dummy,),
        path,
        input_names=["obs"],
        output_names=["logits", "value"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
