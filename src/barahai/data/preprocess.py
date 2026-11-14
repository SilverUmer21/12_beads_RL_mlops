"""Stage 2: turn the raw games into training arrays.

Every position where an expert chose among two or more legal actions becomes one
example:

    obs     (106,) float32   board from the mover's side (see rl.features)
    mask    (201,) bool      legal actions in the same frame
    action  int16            the expert's action in the same frame (the label)
    value   int8             final result for the mover: +1 win, 0 draw, -1 loss
    game    int32            game id

The split into train and validation is made **by game**, never by position:
positions from one game are strongly correlated, so splitting by position
would leak near-copies of validation positions into training.

    python -m barahai.data.preprocess
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from ..engine import GameState
from ..mlops import load_params, write_json
from ..rl.features import OBS_SIZE, canonical_mask, encode, to_canonical

RAW = Path("data/raw/games.jsonl")
OUT = Path("data/processed")
STATS = Path("reports/preprocess.json")


def load_games(path: Path = RAW) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_ids(ids: list[int], val_fraction: float, seed: int) -> set[int]:
    """Game ids that go to validation."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return set(ids[: max(1, round(len(ids) * val_fraction))])


def encode_games(games: list[dict]) -> dict[str, np.ndarray]:
    obs, mask, action, value, game = [], [], [], [], []
    for g in games:
        s = GameState.new()
        for a, is_expert in zip(g["actions"], g["expert"], strict=True):
            m = canonical_mask(s)
            if is_expert and m.sum() > 1:
                obs.append(encode(s))
                mask.append(m)
                action.append(to_canonical(a, s.turn))
                w = g["winner"]
                value.append(0 if w is None else (1 if w == s.turn else -1))
                game.append(g["id"])
            s = s.apply(a)
    return {
        "obs": np.asarray(obs, dtype=np.float32).reshape(-1, OBS_SIZE),
        "mask": np.asarray(mask, dtype=bool).reshape(len(obs), -1),
        "action": np.asarray(action, dtype=np.int16),
        "value": np.asarray(value, dtype=np.int8),
        "game": np.asarray(game, dtype=np.int32),
    }


def main() -> None:
    params = load_params()
    games = load_games()
    val_ids = split_ids([g["id"] for g in games], params["preprocess"]["val_fraction"], params["seed"])
    train = encode_games([g for g in games if g["id"] not in val_ids])
    val = encode_games([g for g in games if g["id"] in val_ids])

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "train.npz", **train)
    np.savez_compressed(OUT / "val.npz", **val)

    stats = {
        "train_games": len(games) - len(val_ids),
        "val_games": len(val_ids),
        "train_positions": int(len(train["action"])),
        "val_positions": int(len(val["action"])),
        "avg_legal_actions": round(float(train["mask"].sum(1).mean()), 3),
        "value_win_share": round(float((train["value"] == 1).mean()), 4),
        "value_draw_share": round(float((train["value"] == 0).mean()), 4),
    }
    write_json(STATS, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
