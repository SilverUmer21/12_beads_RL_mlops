"""Stage: pretrain the policy and value network from the search teacher.

Supervised learning on ``data/labelled`` (positions scored by a depth-5 search):

* policy head: cross-entropy against the teacher's soft move distribution
  (``soft_weight``) mixed with its single best move (1 - ``soft_weight``)
* value head: mean squared error against the teacher's value target
* 8-fold symmetry augmentation: every batch is rotated/mirrored by a random
  board symmetry, so the network sees 8x more distinct positions
* the epoch with the lowest validation loss is kept

"Accuracy" is the share of held-out positions (from games never trained on)
where the network's top move is one the depth-5 teacher scores as best. The weights become the starting point for PPO and are also exported so
the pretrained network can be evaluated on its own.

    python -m barahai.rl.pretrain
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..mlops import flatten, load_params, seed_everything, setup_mlflow, write_json
from .features import symmetry_tables
from .policy import ActorLogits, build_policy, export_onnx

DATA = Path("data/labelled")
OUT = Path("models/pretrained.pt")
ONNX_OUT = Path("models/pretrained.onnx")
METRICS = Path("reports/pretrain.json")


def load_split(name: str) -> dict[str, torch.Tensor]:
    d = np.load(DATA / f"{name}.npz")
    return {
        "obs": torch.from_numpy(d["obs"]),
        "mask": torch.from_numpy(d["mask"]),
        "policy": torch.from_numpy(d["policy"].astype(np.float32)),
        "action": torch.from_numpy(d["action"].astype(np.int64)),
        "value": torch.from_numpy(d["value"].astype(np.float32)),
    }


class Augmenter:
    """Applies one random board symmetry per batch."""

    def __init__(self, generator: torch.Generator) -> None:
        obs_src, act_src, act_map = symmetry_tables()
        self.obs_src = torch.from_numpy(obs_src)
        self.act_src = torch.from_numpy(act_src)
        self.act_map = torch.from_numpy(act_map)
        self.gen = generator

    def __call__(self, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        k = int(torch.randint(0, 8, (1,), generator=self.gen))
        if k == 0:
            return b
        return {
            "obs": b["obs"][:, self.obs_src[k]],
            "mask": b["mask"][:, self.act_src[k]],
            "policy": b["policy"][:, self.act_src[k]],
            "action": self.act_map[k][b["action"]],
            "value": b["value"],
        }


def losses(policy, actor: ActorLogits, b: dict, soft_weight: float, value_coef: float):
    logits = actor(b["obs"]).masked_fill(~b["mask"], -1e9)
    logp = F.log_softmax(logits, dim=-1)
    soft = -(b["policy"] * logp).sum(-1).mean()
    hard = F.nll_loss(logp, b["action"])
    pi_loss = soft_weight * soft + (1 - soft_weight) * hard
    values = policy.predict_values(b["obs"]).squeeze(-1)
    v_loss = F.mse_loss(values, b["value"])
    # A move counts as correct when the teacher scores it as (one of) the best: several moves
    # often share the top score, so "equals the stored best move" would be a noisy measure.
    pred = logits.argmax(-1)
    target = b["policy"]
    picked = target.gather(1, pred.unsqueeze(1)).squeeze(1)
    correct = (picked >= target.max(-1).values - 1e-4).float().sum()
    return pi_loss + value_coef * v_loss, pi_loss, v_loss, correct


@torch.no_grad()
def evaluate(policy, actor, data: dict, p: dict, batch_size: int = 8192) -> dict[str, float]:
    policy.eval()
    n = len(data["action"])
    tot = pi = v = correct = 0.0
    for i in range(0, n, batch_size):
        b = {k: t[i : i + batch_size] for k, t in data.items()}
        loss, pl, vl, c = losses(policy, actor, b, p["soft_weight"], p["value_coef"])
        k = len(b["action"])
        tot, pi, v, correct = tot + loss.item() * k, pi + pl.item() * k, v + vl.item() * k, correct + c.item()
    policy.train()
    return {"loss": tot / n, "policy_loss": pi / n, "value_loss": v / n, "accuracy": correct / n}


def main() -> None:
    params = load_params()
    p = params["pretrain"]
    seed_everything(params["seed"])
    torch.set_num_threads(p.get("threads", 8))

    train, val = load_split("train"), load_split("val")
    policy = build_policy(params["model"])
    actor = ActorLogits(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    n = len(train["action"])
    steps = p["epochs"] * ((n + p["batch_size"] - 1) // p["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["learning_rate"], total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(params["seed"])
    augment = Augmenter(gen) if p["augment"] else (lambda b: b)

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="pretrain") as run:
        mlflow.set_tag("stage", "pretrain")
        mlflow.log_params(flatten({"pretrain": p, "model": params["model"], "relabel": params["relabel"]}))
        mlflow.log_params({"data.train_positions": n, "data.val_positions": len(val["action"])})
        t0 = time.perf_counter()
        best = None
        for epoch in range(1, p["epochs"] + 1):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, p["batch_size"]):
                idx = perm[i : i + p["batch_size"]]
                b = augment({k: t[idx] for k, t in train.items()})
                loss, *_ = losses(policy, actor, b, p["soft_weight"], p["value_coef"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                sched.step()
            tr = evaluate(policy, actor, {k: t[:50_000] for k, t in train.items()}, p)
            va = evaluate(policy, actor, val, p)
            mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in va.items()}, step=epoch)
            print(f"  epoch {epoch:2d}  train acc {tr['accuracy']:.3f}  val acc {va['accuracy']:.3f}"
                  f"  val loss {va['loss']:.3f}  {time.perf_counter() - t0:5.0f}s", flush=True)
            if best is None or va["loss"] < best[1]["loss"]:
                best = (epoch, va, {k: v.clone() for k, v in policy.state_dict().items()})

        epoch, va, state = best
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, OUT)
        policy.load_state_dict(state)
        export_onnx(policy, str(ONNX_OUT))
        metrics = {
            "best_epoch": epoch,
            "val_accuracy": round(va["accuracy"], 4),
            "val_policy_loss": round(va["policy_loss"], 4),
            "val_value_loss": round(va["value_loss"], 4),
            "seconds": round(time.perf_counter() - t0, 1),
            "mlflow_run_id": run.info.run_id,
        }
        mlflow.log_metrics({"best_val_accuracy": va["accuracy"], "best_epoch": epoch})
        mlflow.log_artifact(str(OUT))
        write_json(METRICS, metrics)
        print(metrics)


if __name__ == "__main__":
    main()
