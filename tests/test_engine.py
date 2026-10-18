import random

import pytest

from barahai.engine import STOP, GameRecord, GameState, IllegalMoveError, Rules, parse_square
from barahai.engine.board import (
    EDGES,
    INVERSE_SYM,
    N_SYMMETRIES,
    NEIGHBORS,
    STRONG,
    SYM_DIR,
    SYM_SQ,
    transform_bb,
)
from barahai.engine.state import N_SQUARES, encode_action

from .reference import grid_from, legal_moves

VARIANTS = [
    Rules(),
    Rules(forced_capture=False),
    Rules(chain_capture=False),
    Rules(forced_capture=False, chain_capture=False),
]


def bits(*names: str) -> int:
    return sum(1 << parse_square(n) for n in names)


def engine_moves(s: GameState) -> set:
    return {(m.src, m.dst, m.captured) for m in map(s.describe, s.legal_actions()) if m.action != STOP}


def random_states(rules: Rules, games: int, seed: int):
    rng = random.Random(seed)
    for _ in range(games):
        s = GameState.new(rules)
        while not s.is_terminal:
            yield s
            s = s.apply(rng.choice(s.legal_actions()))
        yield s


# --- geometry ----------------------------------------------------------------


def test_board_geometry():
    # 40 orthogonal segments + one diagonal per each of the 16 cells.
    assert len(EDGES) == 56
    assert sorted(len(n) for n in NEIGHBORS)[:4] == [3, 3, 3, 3]  # weak edge midpoints
    assert len(NEIGHBORS[12]) == 8 and len(NEIGHBORS[7]) == 4  # centre strong, c4 weak
    assert STRONG.bit_count() == 13


def test_start_position():
    s = GameState.new()
    assert s.count(0) == s.count(1) == 12
    assert s.piece_at(12) == -1
    # Opening: only the four beads around the centre can move, all into it.
    dsts = {m.dst for m in map(s.describe, s.legal_actions())}
    assert dsts == {12} and len(s.legal_actions()) == 4


@pytest.mark.parametrize("k", range(N_SYMMETRIES))
def test_symmetries_preserve_lines(k):
    edges = set(EDGES)
    mapped = {tuple(sorted((SYM_SQ[k][a], SYM_SQ[k][b]))) for a, b in EDGES}
    assert mapped == edges
    assert transform_bb(STRONG, k) == STRONG
    assert transform_bb(transform_bb(0b1011001, k), INVERSE_SYM[k]) == 0b1011001


# --- differential testing against the naive reference ------------------------


@pytest.mark.parametrize("rules", VARIANTS, ids=lambda r: f"forced={r.forced_capture},chain={r.chain_capture}")
def test_matches_reference(rules):
    checked = 0
    for s in random_states(rules, games=150, seed=7):
        if s.is_terminal:
            continue
        expected = legal_moves(grid_from(*s.bb), s.turn, rules.forced_capture, s.chain)
        assert engine_moves(s) == expected, str(s)
        assert s.can_stop == (s.chain >= 0 and not rules.forced_capture)
        checked += 1
    assert checked > 5000


@pytest.mark.parametrize("rules", VARIANTS[:2])
def test_moves_commute_with_symmetry(rules):
    """The rules are orientation-free, so rotating a position rotates its moves."""
    for i, s in enumerate(random_states(rules, games=20, seed=3)):
        if s.is_terminal or s.chain >= 0:
            continue
        k = i % N_SYMMETRIES
        t = GameState.from_position(transform_bb(s.bb[0], k), transform_bb(s.bb[1], k), s.turn, rules)
        mapped = {(SYM_SQ[k][a], SYM_SQ[k][b], None if c is None else SYM_SQ[k][c]) for a, b, c in engine_moves(s)}
        assert engine_moves(t) == mapped


def test_action_ids_are_direction_aware():
    s = GameState.new()
    for a in s.legal_actions():
        d, sq = divmod(a, N_SQUARES)
        m = s.describe(a)
        mapped = GameState.from_position(transform_bb(s.bb[0], 1), transform_bb(s.bb[1], 1), s.turn)
        assert mapped.is_legal(encode_action(SYM_DIR[1][d], SYM_SQ[1][sq]))
        assert m.src == sq


# --- specific rules ----------------------------------------------------------


def test_forced_capture_hides_steps():
    p0 = bits("c3", "a1")
    p1 = bits("c4", "e5")
    s = GameState.from_position(p0, p1, turn=0)
    moves = engine_moves(s)
    assert moves == {(parse_square("c3"), parse_square("c5"), parse_square("c4"))}
    optional = GameState.from_position(p0, p1, turn=0, rules=Rules(forced_capture=False))
    assert len(engine_moves(optional)) > 1


def test_chain_capture_can_turn_corners_and_must_continue():
    # a1 jumps b2 -> c3 (diagonal), then c4 -> c5 (straight up).
    p0 = bits("a1")
    p1 = bits("b2", "c4", "e5")
    s = GameState.from_position(p0, p1, turn=0)
    a = s.action_for(parse_square("a1"), parse_square("c3"))
    s = s.apply(a)
    assert s.turn == 0 and s.chain == parse_square("c3") and not s.can_stop
    assert engine_moves(s) == {(parse_square("c3"), parse_square("c5"), parse_square("c4"))}
    s = s.apply(s.legal_actions()[0])
    assert s.turn == 1 and s.count(1) == 1 and s.quiet == 0


def test_optional_chain_allows_stop():
    rules = Rules(forced_capture=False)
    s = GameState.from_position(bits("a1"), bits("b2", "c4", "e5"), turn=0, rules=rules)
    s = s.apply(s.action_for(parse_square("a1"), parse_square("c3")))
    assert s.can_stop and STOP in s.legal_actions()
    s = s.apply(STOP)
    assert s.turn == 1 and s.count(1) == 2


def test_weak_points_have_no_diagonals():
    # b1 is weak: it cannot jump the bead on c2 diagonally.
    s = GameState.from_position(bits("b1"), bits("c2", "e5"), turn=0)
    assert all(m[2] is None for m in engine_moves(s))
    assert (parse_square("b1"), parse_square("c2"), None) not in engine_moves(s)


def test_blocked_player_loses():
    # Player 1's only bead on a5 is boxed in with no jump (landing points occupied).
    p1 = bits("a5")
    p0 = bits("a4", "b5", "b4", "a3", "c5", "c3")
    s = GameState.from_position(p0, p1, turn=1)
    assert s.result is not None and s.result.winner == 0 and s.result.reason == "blocked"


def test_capture_all_wins():
    s = GameState.from_position(bits("c3"), bits("c4"), turn=0)
    s = s.apply(s.legal_actions()[0])
    assert s.result.winner == 0 and s.result.reason == "captured_all"


def test_quiet_limit_draw():
    rules = Rules(quiet_limit=6, repetition_limit=0)
    s = GameState.from_position(bits("a1"), bits("e5"), turn=0, rules=rules)
    rng = random.Random(1)
    while not s.is_terminal:
        s = s.apply(rng.choice(s.legal_actions()))
    assert s.result.reason in ("quiet", "captured_all")


def test_threefold_repetition():
    s = GameState.from_position(bits("a1"), bits("e5"), turn=0, rules=Rules(quiet_limit=0))
    shuffle = [("a1", "a2"), ("e5", "e4"), ("a2", "a1"), ("e4", "e5")]
    for _ in range(2):
        for src, dst in shuffle:
            assert not s.is_terminal
            s = s.apply(s.action_for(parse_square(src), parse_square(dst)))
    assert s.result is not None and s.result.reason == "repetition"


def test_illegal_actions_rejected():
    s = GameState.new()
    with pytest.raises(IllegalMoveError):
        s.apply(0)
    with pytest.raises(IllegalMoveError):
        s.apply(STOP)
    with pytest.raises(IllegalMoveError):
        s.apply(-5)


def test_every_random_game_terminates_and_conserves_beads():
    for rules in VARIANTS:
        rng = random.Random(11)
        for _ in range(60):
            s = GameState.new(rules)
            while not s.is_terminal:
                before = s.count(0) + s.count(1)
                a = rng.choice(s.legal_actions())
                captured = s.is_capture(a)
                s = s.apply(a)
                assert s.count(0) + s.count(1) == before - captured
            assert s.turn_no <= rules.max_turns


def test_record_round_trip(tmp_path):
    rng = random.Random(5)
    s = GameState.new()
    rec = GameRecord()
    while not s.is_terminal:
        a = rng.choice(s.legal_actions())
        rec.actions.append(a)
        s = s.apply(a)
    path = tmp_path / "game.json"
    rec.save(path)
    loaded = GameRecord.load(path)
    assert loaded.actions == rec.actions
    assert loaded.replay()[-1].result == s.result
    turns = loaded.turns()
    assert sum(len(t.moves) for t in turns) == len(rec.actions)
    assert all(t.notation() for t in turns)


def test_perft_regression():
    """Leaf counts from the start position; any rule-engine change shows up here."""

    def perft(s, depth):
        if depth == 0 or s.is_terminal:
            return 1
        return sum(perft(s.apply(a), depth - 1) for a in s.legal_actions())

    s = GameState.new()
    # Forced capture keeps the opening narrow: after any first move into the
    # centre, the reply must be a capture.
    assert [perft(s, d) for d in range(1, 7)] == [4, 5, 6, 12, 19, 64]
