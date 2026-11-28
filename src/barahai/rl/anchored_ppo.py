"""MaskablePPO with a KL anchor to the pretrained policy and a critic warm-up.

Plain PPO fine-tuning of a pretrained policy tends to forget what pretraining
taught before reward learning catches up. Following VPT, PIRLNav and AlphaStar,
the loss gets an extra term

    kl_coef(t) * KL(pi_current || pi_pretrained)

on the states in each rollout. ``kl_coef`` decays linearly from ``kl_start`` to
``kl_end`` over training, so the anchor is strong early and loose later.

``critic_warmup_updates``: for the first few updates only the value head
trains (policy loss and KL are switched off), so early advantages are not
dominated by a value head that was fitted to a different return.

The loop below is MaskablePPO.train() from sb3-contrib 2.9.0 with these two
additions; everything else is unchanged.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class AnchoredMaskablePPO(MaskablePPO):
    def __init__(self, *args, kl_start: float = 0.0, kl_end: float = 0.0, critic_warmup_updates: int = 0, **kwargs):
        self.kl_start = kl_start
        self.kl_end = kl_end
        self.critic_warmup_updates = critic_warmup_updates
        self.anchor = None
        self.kl_coef_history: list[float] = []
        super().__init__(*args, **kwargs)

    def set_anchor(self) -> None:
        """Freeze a copy of the current policy as the KL anchor (call after loading pretrained weights)."""
        self.anchor = copy.deepcopy(self.policy).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)

    def _kl_coef(self) -> float:
        frac = 1.0 - self._current_progress_remaining
        return self.kl_start + (self.kl_end - self.kl_start) * frac

    def _anchor_kl(self, obs: th.Tensor, masks) -> th.Tensor:
        cur = self.policy.get_distribution(obs, action_masks=masks).distribution
        with th.no_grad():
            ref = self.anchor.get_distribution(obs, action_masks=masks).distribution
        legal = th.as_tensor(masks, dtype=th.bool, device=obs.device)
        logp, logq = cur.logits, ref.logits  # normalised log-probs, illegal moves ~ -1e8
        kl = th.where(legal, logp.exp() * (logp - logq), th.zeros_like(logp)).sum(-1)
        return kl.mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        warmup = self._n_updates < self.critic_warmup_updates * self.n_epochs
        kl_coef = 0.0 if (warmup or self.anchor is None) else self._kl_coef()
        entropy_losses, pg_losses, value_losses, clip_fractions, kl_losses = [], [], [], [], []
        continue_training = True

        for _epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                if warmup:
                    loss = self.vf_coef * value_loss
                else:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    if kl_coef > 0:
                        kl = self._anchor_kl(rollout_data.observations, rollout_data.action_masks)
                        kl_losses.append(kl.item())
                        loss = loss + kl_coef * kl

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/anchor_kl", float(np.mean(kl_losses)) if kl_losses else 0.0)
        self.logger.record("train/anchor_kl_coef", kl_coef)
        self.kl_coef_history.append(kl_coef)
        self.logger.record("train/critic_warmup", float(warmup))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
