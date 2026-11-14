"""Stage 5: improve the policy with Maskable PPO.

* Starts from the pretrained weights (``train.init: pretrained``) or from
  scratch (``scratch``), so the two can be compared in MLflow.
* Opponents come from a weighted mix; ``self`` is a frozen copy of the agent
  that is refreshed every ``snapshot_every`` steps (self-play).
* Environments run in parallel worker processes (``vec_env: subproc``). Alpha-beta
  opponents make each step expensive, so spreading environments over the CPU
  cores is much faster than stepping them all in one process.
* Logs rollout win/draw/loss rates per opponent to MLflow, exports the final
  policy to ONNX, and writes the MLflow run id for the evaluate and register
  stages.

    python -m barahai.rl.train
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from ..mlops import file_md5, flatten, git_commit, load_params, read_json, seed_everything, setup_mlflow, write_json
from .env import SELF, BeadsEnv
from .policy import export_onnx, policy_kwargs

OUT_DIR = Path("models")
METRICS = Path("reports/train.json")


def make_env(t: dict, model_params: dict) -> BeadsEnv:
    return BeadsEnv(
        opponents=t["opponents"],
        draw_reward=t["draw_reward"],
        capture_reward=t["capture_reward"],
        opening_plies=tuple(t["opening_plies"]),
        model_params=model_params,
    )


def make_vec_env(t: dict, model_params: dict, seed: int):
    fns = [partial(make_env, t, model_params) for _ in range(t["n_envs"])]
    vec = SubprocVecEnv(fns, start_method="spawn") if t["vec_env"] == "subproc" else DummyVecEnv(fns)
    vec.seed(seed)
    return VecMonitor(vec)


class TrainingCallback(BaseCallback):
    """Logs outcomes per opponent to MLflow and refreshes the self-play snapshot."""

    def __init__(self, mlflow, snapshot_every: int, self_play: bool, window: int = 500) -> None:
        super().__init__()
        self.mlflow = mlflow
        self.snapshot_every = snapshot_every
        self.self_play = self_play
        self.outcomes: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.next_snapshot = snapshot_every
        self.t0 = time.perf_counter()

    def _push_snapshot(self) -> None:
        state = {k: v.detach().cpu() for k, v in self.model.policy.state_dict().items()}
        self.training_env.env_method("set_snapshot", state)

    def _on_training_start(self) -> None:
        if self.self_play:
            self._push_snapshot()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "outcome" in info:
                self.outcomes[info["opponent"]].append(info["outcome"])
        if self.self_play and self.num_timesteps >= self.next_snapshot:
            self._push_snapshot()
            self.next_snapshot += self.snapshot_every
        return True

    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        metrics = {"fps": step / max(1e-9, time.perf_counter() - self.t0)}
        for opp, q in self.outcomes.items():
            if q:
                n = len(q)
                key = opp.replace("-", "_")
                metrics[f"rollout_win_{key}"] = sum(o == "win" for o in q) / n
                metrics[f"rollout_draw_{key}"] = sum(o == "draw" for o in q) / n
                metrics[f"rollout_loss_{key}"] = sum(o == "loss" for o in q) / n
        for k in ("rollout/ep_rew_mean", "rollout/ep_len_mean", "train/entropy_loss", "train/approx_kl",
                  "train/value_loss", "train/policy_gradient_loss", "train/clip_fraction"):
            v = self.logger.name_to_value.get(k)
            if v is not None:
                metrics[k.replace("/", "_")] = float(v)
        self.mlflow.log_metrics(metrics, step=step)


def main() -> None:
    params = load_params()
    t = params["train"]
    seed_everything(params["seed"])
    torch.set_num_threads(t.get("threads", 4))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vec = make_vec_env(t, params["model"], params["seed"])
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        learning_rate=t["learning_rate"],
        n_steps=t["n_steps"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        gamma=t["gamma"],
        gae_lambda=t["gae_lambda"],
        ent_coef=t["ent_coef"],
        clip_range=t["clip_range"],
        policy_kwargs=policy_kwargs(params["model"]),
        seed=params["seed"],
        device="cpu",
        verbose=0,
    )
    if t["init"] == "pretrained":
        model.policy.load_state_dict(torch.load("models/pretrained.pt", weights_only=True))

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name=f"train-{t['init']}") as run:
        mlflow.set_tags({"stage": "train", "init": t["init"], "git_commit": git_commit()})
        if t["init"] == "pretrained" and Path("reports/pretrain.json").exists():
            mlflow.set_tag("pretrain_run_id", read_json("reports/pretrain.json").get("mlflow_run_id", ""))
        mlflow.log_params(flatten({"train": t, "model": params["model"]}))
        mlflow.set_tag("data_md5", file_md5("data/processed") if Path("data/processed").exists() else "none")
        t0 = time.perf_counter()
        cb = TrainingCallback(mlflow, t["snapshot_every"], SELF in t["opponents"])
        model.learn(total_timesteps=t["total_timesteps"], callback=cb, progress_bar=False)
        seconds = time.perf_counter() - t0

        model.save(OUT_DIR / "policy.zip")
        torch.save(model.policy.state_dict(), OUT_DIR / "policy.pt")
        export_onnx(model.policy, str(OUT_DIR / "policy.onnx"))
        vec.close()

        metrics = {
            "timesteps": int(model.num_timesteps),
            "seconds": round(seconds, 1),
            "steps_per_second": round(model.num_timesteps / seconds, 1),
            "mlflow_run_id": run.info.run_id,
        }
        for opp, q in cb.outcomes.items():
            if q:
                metrics[f"final_rollout_win_{opp}"] = round(sum(o == "win" for o in q) / len(q), 4)
        mlflow.log_metrics({"train_seconds": seconds, "steps_per_second": metrics["steps_per_second"]})
        mlflow.log_artifact(str(OUT_DIR / "policy.zip"))
        write_json(METRICS, metrics)
        print(metrics)


if __name__ == "__main__":
    main()
