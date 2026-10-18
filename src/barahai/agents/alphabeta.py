"""Negamax alpha-beta search with iterative deepening and a transposition table.

Depth is counted in *turns*: continuing a chain capture costs no depth, so a
multi-jump is always searched to its end. At the horizon a capture-only
quiescence search runs (under forced capture the side to move has no choice
there anyway), so the search never stops in the middle of an exchange.

With a fixed ``depth`` and no ``time_limit`` the agent is fully deterministic
for a given seed, which makes it a reproducible benchmark opponent. With a
``time_limit`` it plays as deep as the clock allows.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from ..engine import GameState
from ..engine.board import FULL, N_DIRS, SHIFT, STEP_SRC, STRONG, shift_back
from .base import Agent

WIN = 100_000
INF = 10**9
_EXACT, _LOWER, _UPPER = 0, 1, 2
_TT_LIMIT = 400_000


class SearchCancelled(Exception):
    pass


def _mobility(own: int, empty: int) -> int:
    return sum((own & STEP_SRC[d] & shift_back(empty, SHIFT[d])).bit_count() for d in range(N_DIRS))


def evaluate(s: GameState) -> int:
    """Static score for the side to move, in centi-beads."""
    own, opp = s.bb[s.turn], s.bb[1 - s.turn]
    empty = FULL ^ (own | opp)
    material = own.bit_count() - opp.bit_count()
    mobility = _mobility(own, empty) - _mobility(opp, empty)
    strong = (own & STRONG).bit_count() - (opp & STRONG).bit_count()
    return 100 * material + 3 * mobility + 4 * strong


class AlphaBetaAgent(Agent):
    def __init__(
        self,
        depth: int = 4,
        time_limit: float | None = None,
        seed: int | None = 0,
        name: str | None = None,
    ) -> None:
        self.max_depth = depth
        self.time_limit = time_limit
        self._seed = seed
        self.rng = random.Random(seed)
        self.name = name or (f"alphabeta-t{time_limit:g}" if time_limit else f"alphabeta-d{depth}")
        self.tt: dict[int, tuple[int, int, int, int]] = {}
        self.cancelled = False
        # Optional extra stop condition, polled with the clock (e.g. a flag
        # shared with another process).
        self.stop_check: Callable[[], bool] | None = None
        # Stats from the last search, shown in the UI.
        self.last_score = 0
        self.last_depth = 0
        self.nodes = 0

    def reset(self) -> None:
        self.rng = random.Random(self._seed)
        self.tt.clear()

    def cancel(self) -> None:
        """Abort a running search from another thread."""
        self.cancelled = True

    # --- search --------------------------------------------------------------

    def act(self, state: GameState) -> int:
        actions = state.legal_actions()
        if len(actions) == 1:
            return actions[0]
        self.cancelled = False
        self.nodes = 0
        self._deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        if len(self.tt) > _TT_LIMIT:
            self.tt.clear()
        self.rng.shuffle(actions)  # seeded tie-breaking between equal moves
        best = actions[0]
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self._root(state, actions, depth)
            except SearchCancelled:
                if self._stopped():
                    raise
                break
            best, self.last_score, self.last_depth = move, score, depth
            actions.remove(move)
            actions.insert(0, move)  # principal move first on the next iteration
            if abs(score) > WIN - 1000:
                break  # forced result found; deeper search cannot change it
        return best

    def _root(self, s: GameState, actions: list[int], depth: int) -> tuple[int, int]:
        alpha, best_move = -INF, actions[0]
        for a in actions:
            child = s.apply(a)
            if child.turn == s.turn:
                v = self._negamax(child, depth, alpha, INF, 1)
            else:
                v = -self._negamax(child, depth - 1, -INF, -alpha, 1)
            if v > alpha:
                alpha, best_move = v, a
        return alpha, best_move

    def _stopped(self) -> bool:
        return self.cancelled or (self.stop_check is not None and self.stop_check())

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes & 1023 == 0:
            if self._stopped() or (self._deadline and time.perf_counter() > self._deadline):
                raise SearchCancelled

    @staticmethod
    def _terminal_score(s: GameState, ply: int) -> int:
        winner = s.result.winner
        if winner is None:
            return 0
        return WIN - ply if winner == s.turn else -(WIN - ply)

    def _ordered(self, s: GameState, tt_move: int) -> list[int]:
        actions = s.legal_actions()
        actions.sort(key=lambda a: (a != tt_move, not s.is_capture(a)))
        return actions

    def _negamax(self, s: GameState, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        if s.result is not None:
            return self._terminal_score(s, ply)
        if depth <= 0 and s.chain < 0:
            return self._quiesce(s, alpha, beta, ply)

        key = s.key
        entry = self.tt.get(key)
        tt_move = -1
        if entry is not None:
            e_depth, flag, e_score, tt_move = entry
            if e_depth >= depth:
                if flag == _EXACT:
                    return e_score
                if flag == _LOWER and e_score >= beta:
                    return e_score
                if flag == _UPPER and e_score <= alpha:
                    return e_score

        alpha0 = alpha
        best, best_move = -INF, -1
        for a in self._ordered(s, tt_move):
            child = s.apply(a)
            if child.turn == s.turn:
                v = self._negamax(child, depth, alpha, beta, ply + 1)
            else:
                v = -self._negamax(child, depth - 1, -beta, -alpha, ply + 1)
            if v > best:
                best, best_move = v, a
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = _UPPER if best <= alpha0 else _LOWER if best >= beta else _EXACT
        self.tt[key] = (depth, flag, best, best_move)
        return best

    def _quiesce(self, s: GameState, alpha: int, beta: int, ply: int) -> int:
        self._tick()
        if s.result is not None:
            return self._terminal_score(s, ply)
        forced = s.must_capture()
        if not forced:
            stand = evaluate(s)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            best = stand
        else:
            best = -INF
        for a in s.legal_actions():
            if not s.is_capture(a):
                continue
            child = s.apply(a)
            if child.turn == s.turn:
                v = self._quiesce(child, alpha, beta, ply + 1)
            else:
                v = -self._quiesce(child, -beta, -alpha, ply + 1)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        return best
