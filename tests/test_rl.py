"""Tests for the RL side: observation encoding, the environment and the data stages."""

import random

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from barahai.engine import N_ACTIONS, STOP, GameState  # noqa: E402
from barahai.engine.board import ROT180, transform_bb  # noqa: E402
from barahai.rl.features import (  # noqa: E402
    OBS_SIZE,
    canonical_mask,
    encode,
    from_canonical,
    repetition_count,
    to_canonical,
)


def random_states(n_games: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n_games):
        s = GameState.new()
        while not s.is_terminal:
            yield s
            s = s.apply(rng.choice(s.legal_actions()))


def test_action_rotation_is_an_involution():
    for a in range(N_ACTIONS):
        assert from_canonical(to_canonical(a, 1), 1) == a
        assert to_canonical(a, 0) == a
    assert to_canonical(STOP, 1) == STOP


def test_mask_matches_legal_actions_in_both_frames():
    for s in random_states(20, seed=1):
        mask = canonical_mask(s)
        legal = s.legal_actions()
        assert mask.sum() == len(legal)
        assert {from_canonical(int(a), s.turn) for a in np.flatnonzero(mask)} == set(legal)


def test_encoding_shape_and_range():
    for s in random_states(10, seed=2):
        obs = encode(s)
        assert obs.shape == (OBS_SIZE,) and obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        assert np.all(obs[:25] + obs[25:50] + obs[50:75] == 1.0)


def test_colour_swap_gives_the_same_view():
    """A position with player 1 to move looks identical to its rotated, colour-swapped twin."""
    for s in random_states(10, seed=3):
        if s.mid_chain:
            continue
        p0, p1 = s.bb
        twin = GameState.from_position(transform_bb(p1, ROT180), transform_bb(p0, ROT180), turn=1 - s.turn)
        same = GameState.from_position(p0, p1, turn=s.turn)
        assert np.array_equal(encode(same), encode(twin))


def test_repetition_count_handles_mid_chain():
    # a1 jumps b2 and can continue: the mid-chain state has an empty history.
    from barahai.engine import parse_square

    def bits(*names):
        return sum(1 << parse_square(n) for n in names)

    s = GameState.from_position(bits("a1"), bits("b2", "c4", "e5"), turn=0)
    s = s.apply(s.legal_actions()[0])
    assert s.mid_chain
    assert repetition_count(s) == 0
    encode(s)  # must not raise


def test_env_passes_gymnasium_checks_and_plays_to_the_end():
    from gymnasium.utils.env_checker import check_env

    from barahai.rl.env import BeadsEnv

    env = BeadsEnv(opponents={"random": 1.0, "greedy": 1.0}, opening_plies=(0, 3))
    check_env(env, skip_render_check=True)
    rng = np.random.default_rng(0)
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            mask = env.action_masks()
            assert mask.any()
            obs, reward, done, _, info = env.step(int(rng.choice(np.flatnonzero(mask))))
            assert env.observation_space.contains(obs)
        assert info["outcome"] in {"win", "draw", "loss"}


def test_env_illegal_action_ends_as_loss():
    from barahai.rl.env import BeadsEnv

    env = BeadsEnv()
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    _, reward, done, _, info = env.step(illegal)
    assert done and reward == -1.0 and info["illegal"]


def test_self_play_env_uses_snapshot():
    from barahai.rl.env import BeadsEnv
    from barahai.rl.policy import build_policy

    model = {"net_arch": [16]}
    env = BeadsEnv(opponents={"self": 1.0}, model_params=model)
    env.set_snapshot(build_policy(model).state_dict())
    obs, info = env.reset(seed=0)
    assert info["opponent"] == "self"


def test_generate_and_preprocess_split_by_game():
    from barahai.data.generate import play_game
    from barahai.data.preprocess import encode_games, split_ids

    games = [play_game((i, 0, 1, (2, 4), 0.1)) for i in range(6)]
    for g in games:
        assert len(g["actions"]) == len(g["expert"])
        assert sum(g["expert"][:2]) == 0  # the random opening is never a label
    val = split_ids([g["id"] for g in games], 0.34, seed=0)
    tr = encode_games([g for g in games if g["id"] not in val])
    va = encode_games([g for g in games if g["id"] in val])
    assert set(tr["game"]).isdisjoint(set(va["game"]))
    assert tr["obs"].shape[1] == OBS_SIZE
    assert np.all(tr["mask"][np.arange(len(tr["action"])), tr["action"]])  # labels are legal


def test_onnx_export_matches_torch(tmp_path):
    pytest.importorskip("onnxruntime")
    import torch

    from barahai.rl.policy import ActorLogits, OnnxAgent, build_policy, export_onnx

    policy = build_policy({"net_arch": [32, 32]})
    path = tmp_path / "p.onnx"
    export_onnx(policy, str(path))
    agent = OnnxAgent(str(path))
    for s in random_states(2, seed=4):
        expected = ActorLogits(policy)(torch.as_tensor(encode(s))[None]).detach().numpy()[0]
        assert np.allclose(agent.logits(s), expected, atol=1e-4)
        assert s.is_legal(agent.act(s))


def test_puzzle_gate_passes_and_fails(tmp_path):
    """The CI gate exits 0 when accuracy holds and 1 when it drops below baseline - tolerance."""
    pytest.importorskip("onnxruntime")
    import json

    from barahai.rl.gate import main, passes
    from barahai.rl.policy import build_policy, export_onnx

    assert passes(0.64, 0.6392, 0.0) and passes(0.63, 0.6392, 0.01) and not passes(0.62, 0.6392, 0.01)

    model = tmp_path / "p.onnx"
    export_onnx(build_policy({"net_arch": [32, 32]}), str(model))
    # Every legal move counts as best, so any model solves all 10 puzzles.
    rng = random.Random(5)
    lines = []
    for _ in range(10):
        s, actions = GameState.new(), []
        for _ in range(rng.randint(0, 8)):
            a = rng.choice(s.legal_actions())
            actions.append(a)
            s = s.apply(a)
            if s.is_terminal:
                break
        if not s.is_terminal:
            lines.append(json.dumps({"actions": actions, "best": sorted(s.legal_actions())}))
    puzzles = tmp_path / "puzzles.jsonl"
    puzzles.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(baseline: float) -> int:
        b = tmp_path / "eval.json"
        b.write_text(json.dumps({"puzzle_accuracy": baseline}), encoding="utf-8")
        return main(["--baseline", str(b), "--tolerance", "0.0", "--model", str(model), "--puzzles", str(puzzles)])

    assert run(1.0) == 0
    # Same puzzles, now with no correct answer: accuracy 0.0 drops below the baseline.
    puzzles.write_text("\n".join(json.dumps({**json.loads(z), "best": []}) for z in lines) + "\n", encoding="utf-8")
    assert run(0.5) == 1
