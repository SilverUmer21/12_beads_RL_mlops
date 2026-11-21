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


def test_symmetry_tables_preserve_legality_and_labels():
    from barahai.engine.board import SYM_SQ, transform_bb
    from barahai.rl.features import symmetry_tables

    obs_src, act_src, act_map = symmetry_tables()
    for k in range(8):
        assert sorted(act_map[k]) == list(range(N_ACTIONS))  # a permutation of the actions
        assert act_map[k][STOP] == STOP
    # Transforming a player-0 position must transform its observation and its legal moves alike.
    for s in random_states(5, seed=7):
        if s.turn != 0 or s.mid_chain:
            continue
        for k in range(8):
            t = GameState.from_position(transform_bb(s.bb[0], k), transform_bb(s.bb[1], k), turn=0)
            base = GameState.from_position(*s.bb, turn=0)
            assert np.array_equal(encode(base)[obs_src[k]][:100], encode(t)[:100])
            assert np.array_equal(canonical_mask(base)[act_src[k]], canonical_mask(t))
        assert SYM_SQ[0][3] == 3


def _tiny_onnx(tmp_path):
    from barahai.rl.policy import build_policy, export_onnx

    path = tmp_path / "pv.onnx"
    export_onnx(build_policy({"net_arch": [16]}), str(path))
    return str(path)


def test_onnx_exports_logits_and_value(tmp_path):
    from barahai.rl.policy import OnnxAgent

    agent = OnnxAgent(_tiny_onnx(tmp_path))
    logits, value = agent.evaluate(GameState.new())
    assert logits.shape == (N_ACTIONS,) and -1.0 <= value <= 1.0


def test_search_plays_legal_moves_and_finds_a_capture(tmp_path):
    from barahai.engine import parse_square
    from barahai.rl.policy import OnnxAgent
    from barahai.search import SearchAgent

    agent = SearchAgent(OnnxAgent(_tiny_onnx(tmp_path)), simulations=30)
    for s in random_states(1, seed=8):
        assert s.is_legal(agent.act(s))

    def bits(*names):
        return sum(1 << parse_square(n) for n in names)

    # Player 0 can capture the last enemy bead and win at once.
    s = GameState.from_position(bits("c3", "a1"), bits("c4"), turn=0, rules=__import__("barahai.engine").engine.Rules(
        forced_capture=False))
    a = SearchAgent(OnnxAgent(_tiny_onnx(tmp_path)), simulations=60).act(s)
    assert s.apply(a).is_terminal and s.apply(a).result.winner == 0


def test_anchored_ppo_learns_a_few_steps():
    from barahai.rl.anchored_ppo import AnchoredMaskablePPO
    from barahai.rl.env import BeadsEnv
    from barahai.rl.policy import policy_kwargs

    env = BeadsEnv(opponents={"random": 1.0})
    model = AnchoredMaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1,
                                policy_kwargs=policy_kwargs({"net_arch": [16]}), kl_start=0.2, kl_end=0.0,
                                critic_warmup_updates=1, device="cpu", seed=0)
    model.set_anchor()
    model.learn(192)
    assert model.kl_coef_history[0] == 0.0 and max(model.kl_coef_history) > 0  # warm-up, then anchored


def test_opening_book_and_elo():
    from barahai.rl.evaluate import elo, opening_book, summarise

    book = opening_book(20, (4, 8), seed=1)
    assert len(book) == 20 and len({tuple(b) for b in book}) == 20
    assert elo(0.5) == 0 and elo(0.75) > 0 > elo(0.25)
    s = summarise({0: [1.0, 0.5], 1: [0.5, 0.5], 2: [1.0, 1.0]})
    assert s["games"] == 6 and 0.5 < s["score"] < 1
