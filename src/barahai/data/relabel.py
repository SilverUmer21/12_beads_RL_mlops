"""Stage: relabel positions with a deeper search (the "teacher" for pretraining).

Phase 1 trained on the single move a depth-3 searcher played. That caps the
student near depth-3 strength. This stage replays the same expert games and,
for each decision with two or more legal moves, scores **every** legal move
with a deeper alpha-beta search. That gives three targets per position:

    policy   (201,) float16   soft target: softmax of the move scores / temperature
    action   int16            the best move (hard label, used for accuracy)
    value    float16          tanh(best score / value_scale), blended with the game result

Soft targets carry more information than one move (how much worse each
alternative is), the same idea as AlphaZero's visit-count targets and Gumbel
MuZero's completed-Q targets.

Positions keep the train/val split of ``preprocess`` (by game). Training
positions can be subsampled with ``train_fraction`` to bound the search time.

    python -m barahai.data.relabel
"""

from __future__ import annotations

import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..engine import N_ACTIONS, GameState
from ..mlops import load_params, write_json
from ..rl.features import OBS_SIZE, canonical_mask, encode, to_canonical
from .preprocess import load_games, split_ids
from .puzzles import score_moves

OUT = Path("data/labelled")
STATS = Path("reports/relabel.json")
SCORE_CLIP = 2000  # centi-beads; forced wins (~100,000) are clipped to this


def soft_target(scores: dict[int, int], turn: int, temperature: float) -> np.ndarray:
    target = np.zeros(N_ACTIONS, dtype=np.float32)
    acts = list(scores)
    z = np.clip(np.array([scores[a] for a in acts], dtype=np.float64), -SCORE_CLIP, SCORE_CLIP) / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    for a, pa in zip(acts, p, strict=True):
        target[to_canonical(a, turn)] = pa
    return target


def label_game(job: tuple[dict, float, int, float, float, float]) -> dict[str, list]:
    game, keep, depth, temperature, value_scale, outcome_weight = job
    rng = random.Random(game["id"])
    out: dict[str, list] = {k: [] for k in ("obs", "mask", "policy", "action", "value", "game")}
    s = GameState.new()
    w = game["winner"]
    for a, is_expert in zip(game["actions"], game["expert"], strict=True):
        if is_expert and len(s.legal_actions()) > 1 and rng.random() < keep:
            scores = score_moves(s, depth)
            best = max(scores, key=scores.get)
            outcome = 0.0 if w is None else (1.0 if w == s.turn else -1.0)
            search_value = math.tanh(max(-SCORE_CLIP, min(SCORE_CLIP, scores[best])) / value_scale)
            out["obs"].append(encode(s))
            out["mask"].append(canonical_mask(s))
            out["policy"].append(soft_target(scores, s.turn, temperature))
            out["action"].append(to_canonical(best, s.turn))
            out["value"].append((1 - outcome_weight) * search_value + outcome_weight * outcome)
            out["game"].append(game["id"])
        s = s.apply(a)
    return out


def merge(parts: list[dict[str, list]]) -> dict[str, np.ndarray]:
    cat = {k: [x for p in parts for x in p[k]] for k in parts[0]}
    n = len(cat["action"])
    return {
        "obs": np.asarray(cat["obs"], dtype=np.float32).reshape(n, OBS_SIZE),
        "mask": np.asarray(cat["mask"], dtype=bool).reshape(n, N_ACTIONS),
        "policy": np.asarray(cat["policy"], dtype=np.float16).reshape(n, N_ACTIONS),
        "action": np.asarray(cat["action"], dtype=np.int16),
        "value": np.asarray(cat["value"], dtype=np.float16),
        "game": np.asarray(cat["game"], dtype=np.int32),
    }


def main() -> None:
    params = load_params()
    p = params["relabel"]
    games = load_games()
    val_ids = split_ids([g["id"] for g in games], params["preprocess"]["val_fraction"], params["seed"])
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {"depth": p["depth"]}
    for split, keep in (("val", p["val_fraction"]), ("train", p["train_fraction"])):
        subset = [g for g in games if (g["id"] in val_ids) == (split == "val")]
        jobs = [(g, keep, p["depth"], p["temperature"], p["value_scale"], p["outcome_weight"]) for g in subset]
        parts = []
        with Pool(p["workers"]) as pool:
            for n, part in enumerate(pool.imap_unordered(label_game, jobs, chunksize=2), 1):
                parts.append(part)
                if n % 500 == 0:
                    print(f"  {split}: {n}/{len(jobs)} games  {time.perf_counter() - t0:6.0f}s", flush=True)
        parts.sort(key=lambda q: q["game"][0] if q["game"] else -1)
        data = merge([q for q in parts if q["action"]])
        np.savez_compressed(OUT / f"{split}.npz", **data)
        stats[f"{split}_positions"] = int(len(data["action"]))
        top = data["policy"].astype(np.float32).max(1)
        stats[f"{split}_mean_top_prob"] = round(float(top.mean()), 4)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(STATS, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
