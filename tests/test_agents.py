import pytest

from barahai.agents import AlphaBetaAgent, GreedyAgent, RandomAgent, make_agent
from barahai.engine import GameState, parse_square


def bits(*names: str) -> int:
    return sum(1 << parse_square(n) for n in names)


@pytest.mark.parametrize("spec", ["random", "greedy", "alphabeta-d2"])
def test_agents_only_play_legal_moves(spec):
    agent = make_agent(spec, seed=1)
    opponent = RandomAgent(2)
    s = GameState.new()
    while not s.is_terminal:
        a = (agent if s.turn == 0 else opponent).act(s)
        assert s.is_legal(a)
        s = s.apply(a)


def test_greedy_prefers_the_double_capture():
    # Optional rules so the greedy bot actually has a choice: d1 can jump c2
    # then b3 (two beads), or step elsewhere.
    from barahai.engine import Rules

    s = GameState.from_position(bits("e1", "a1"), bits("d2", "c4", "a5"), turn=0, rules=Rules(forced_capture=False))
    a = GreedyAgent(seed=0).act(s)
    assert s.is_capture(a)


def test_greedy_avoids_hanging_a_bead():
    # Several quiet moves walk into a capture by c3; greedy must pick a safe one.
    from barahai.agents.greedy import best_capture_run

    s = GameState.from_position(bits("a1", "e1"), bits("c3", "e5"), turn=0)
    a = GreedyAgent(seed=0).act(s)
    child = s.apply(a)
    assert best_capture_run(child) == 0


def test_alphabeta_finds_win_in_two_turns():
    # Three beads against a lone bead cornered on a5: the search must see the
    # forced win (a mate-like score), not just the material edge.
    s = GameState.from_position(bits("b3", "c4", "a2"), bits("a5"), turn=0)
    agent = AlphaBetaAgent(depth=4)
    a = agent.act(s)
    assert s.is_legal(a)
    assert agent.last_score > 50_000  # a forced win, not just a material edge


def test_alphabeta_is_deterministic_at_fixed_depth():
    def game():
        a, b = AlphaBetaAgent(depth=2, seed=4), AlphaBetaAgent(depth=2, seed=9)
        s = GameState.new()
        moves = []
        while not s.is_terminal and len(moves) < 60:
            m = (a, b)[s.turn].act(s)
            moves.append(m)
            s = s.apply(m)
        return moves

    assert game() == game()


def test_time_limited_search_returns_quickly():
    import time

    agent = AlphaBetaAgent(depth=64, time_limit=0.3)
    s = GameState.new()
    s = s.apply(s.legal_actions()[0])
    t0 = time.perf_counter()
    assert s.is_legal(agent.act(s))
    assert time.perf_counter() - t0 < 1.5


def test_states_pickle_for_the_search_process():
    import pickle

    s = GameState.new()
    s = s.apply(s.legal_actions()[0])
    t = pickle.loads(pickle.dumps(s))
    assert t.bb == s.bb and t.turn == s.turn and t.history == s.history
    assert t.legal_actions() == s.legal_actions()


def test_worker_cancellation_watermark():
    from types import SimpleNamespace

    from barahai.game import worker

    worker.init(SimpleNamespace(value=0))
    s = GameState.new()
    a = worker.think(1, ("t", 0), "Khiladi", 0, s)
    assert s.is_legal(a)
    worker.init(SimpleNamespace(value=5))
    assert worker.think(3, ("t", 0), "Ustad", 0, s) is None  # stale job never runs
