"""Stage 4: pretrain the policy by copying the expert moves (behaviour cloning).

Supervised learning on ``data/processed``:

* policy head: cross-entropy between the masked policy and the expert's action
* value head: mean squared error against the game result for the mover

The weights become the starting point for PPO, which then improves on the
experts by playing. Everything is logged to MLflow.

    python -m barahai.rl.pretrain
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..mlops import flatten, load_params, seed_everything, setup_mlflow, write_json
from .policy import ActorLogits, build_policy

DATA = Path("data/processed")
OUT = Path("models/pretrained.pt")
METRICS = Path("reports/pretrain.json")


def load_split(name: str) -> dict[str, torch.Tensor]:
    d = np.load(DATA / f"{name}.npz")
    return {
        "obs": torch.from_numpy(d["obs"]),
        "mask": torch.from_numpy(d["mask"]),
        "action": torch.from_numpy(d["action"].astype(np.int64)),
        "value": torch.from_numpy(d["value"].astype(np.float32)),
    }


def losses(policy, actor: ActorLogits, batch: dict, value_coef: float):
    logits = actor(batch["obs"]).masked_fill(~batch["mask"], -1e9)
    pi_loss = F.cross_entropy(logits, batch["action"])
    values = policy.predict_values(batch["obs"]).squeeze(-1)
    v_loss = F.mse_loss(values, batch["value"])
    correct = (logits.argmax(-1) == batch["action"]).float().sum()
    return pi_loss + value_coef * v_loss, pi_loss, v_loss, correct


@torch.no_grad()
def evaluate(policy, actor, data: dict, value_coef: float, batch_size: int = 8192) -> dict[str, float]:
    policy.eval()
    n = len(data["action"])
    tot = pi = v = correct = 0.0
    for i in range(0, n, batch_size):
        b = {k: t[i : i + batch_size] for k, t in data.items()}
        loss, pl, vl, c = losses(policy, actor, b, value_coef)
        k = len(b["action"])
        tot += loss.item() * k
        pi += pl.item() * k
        v += vl.item() * k
        correct += c.item()
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
    opt = torch.optim.Adam(policy.parameters(), lr=p["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
    n = len(train["action"])
    gen = torch.Generator().manual_seed(params["seed"])

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="pretrain") as run:
        mlflow.set_tag("stage", "pretrain")
        mlflow.log_params(flatten({"pretrain": p, "model": params["model"]}))
        mlflow.log_params({"data.train_positions": n, "data.val_positions": len(val["action"])})
        t0 = time.perf_counter()
        best = None
        for epoch in range(1, p["epochs"] + 1):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, p["batch_size"]):
                idx = perm[i : i + p["batch_size"]]
                b = {k: t[idx] for k, t in train.items()}
                loss, *_ = losses(policy, actor, b, p["value_coef"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
            sched.step()
            tr = evaluate(policy, actor, {k: t[:50_000] for k, t in train.items()}, p["value_coef"])
            va = evaluate(policy, actor, val, p["value_coef"])
            mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in va.items()}, step=epoch)
            print(f"  epoch {epoch:2d}  train acc {tr['accuracy']:.3f}  val acc {va['accuracy']:.3f}"
                  f"  val loss {va['loss']:.3f}  {time.perf_counter() - t0:5.0f}s", flush=True)
            if best is None or va["loss"] < best[1]["loss"]:
                best = (epoch, va, {k: v.clone() for k, v in policy.state_dict().items()})

        epoch, va, state = best
        OUT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, OUT)
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
