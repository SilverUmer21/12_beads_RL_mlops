"""Game state, move generation and rule enforcement.

Actions are *atomic*: one step, or one jump. A multi-jump turn is a sequence of
jump actions by the same player; between them the state is "mid-chain" and only
the jumping bead may move. This keeps the action space fixed and small:

    action = direction * 25 + from_square      (0..199)
    STOP   = 200                               (end a chain early; only legal
                                                when captures are optional)

Step and jump never collide on the same id: the neighbour in that direction is
either empty (step) or an enemy bead (jump), never both.

``GameState`` is immutable. ``apply`` returns a new state, which makes undo,
search and replay trivial and keeps states safe to share across threads.
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import (
    FULL,
    JUMP_SRC,
    JUMP_TO,
    N_DIRS,
    N_SQUARES,
    SHIFT,
    START_P0,
    START_P1,
    STEP_SRC,
    STEP_TO,
    iter_bits,
    shift_back,
    square_name,
)
from .rules import STANDARD, Rules

STOP = N_DIRS * N_SQUARES
N_ACTIONS = STOP + 1

_NO_MOVES = (0,) * N_DIRS


class IllegalMoveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Outcome:
    winner: int | None  # None for a draw
    reason: str  # "captured_all" | "blocked" | "quiet" | "repetition" | "max_turns"

    @property
    def is_draw(self) -> bool:
        return self.winner is None


@dataclass(frozen=True, slots=True)
class Move:
    """What an atomic action does on the board."""

    action: int
    src: int
    dst: int
    captured: int | None  # square of the removed bead, for jumps

    @property
    def is_capture(self) -> bool:
        return self.captured is not None


def encode_action(direction: int, src: int) -> int:
    return direction * N_SQUARES + src


def decode_action(action: int) -> tuple[int, int]:
    """(direction, from_square). Not valid for STOP."""
    return divmod(action, N_SQUARES)


def _generate(rules: Rules, own: int, opp: int, chain: int) -> tuple[tuple[int, ...], bool]:
    """Per-direction bitmasks of legal source squares, plus whether STOP is legal."""
    empty = FULL ^ (own | opp)
    src = (1 << chain) if chain >= 0 else own
    jumps = []
    any_jump = 0
    for d in range(N_DIRS):
        delta = SHIFT[d]
        j = src & JUMP_SRC[d] & shift_back(opp, delta) & shift_back(empty, 2 * delta)
        jumps.append(j)
        any_jump |= j
    if chain >= 0:
        return tuple(jumps), not rules.forced_capture
    if any_jump and rules.forced_capture:
        return tuple(jumps), False
    return (
        tuple(
            jumps[d] | (own & STEP_SRC[d] & shift_back(empty, SHIFT[d])) for d in range(N_DIRS)
        ),
        False,
    )


def _can_jump_from(sq: int, own: int, opp: int) -> bool:
    empty = FULL ^ (own | opp)
    for d in range(N_DIRS):
        mid = STEP_TO[d][sq]
        land = JUMP_TO[d][sq]
        if land >= 0 and opp >> mid & 1 and empty >> land & 1:
            return True
    return False


class GameState:
    __slots__ = ("rules", "bb", "turn", "chain", "quiet", "turn_no", "history", "result", "_moves", "_stop")

    rules: Rules
    bb: tuple[int, int]  # bitboards for player 0 and player 1
    turn: int  # player to act
    chain: int  # square of the bead that must keep jumping, or -1
    quiet: int  # completed turns since the last capture
    turn_no: int  # completed turns since the start
    history: tuple[int, ...]  # position keys at turn starts since the last capture
    result: Outcome | None

    def __init__(self) -> None:  # use GameState.new() / apply()
        raise TypeError("use GameState.new()")

    @classmethod
    def _make(
        cls,
        rules: Rules,
        bb: tuple[int, int],
        turn: int,
        chain: int,
        quiet: int,
        turn_no: int,
        history: tuple[int, ...],
    ) -> GameState:
        s = object.__new__(cls)
        s.rules = rules
        s.bb = bb
        s.turn = turn
        s.chain = chain
        s.quiet = quiet
        s.turn_no = turn_no
        s.history = history
        s.result = None
        own, opp = bb[turn], bb[1 - turn]
        if chain < 0:
            s.result = _adjudicate(s, own, opp)
        if s.result is None:
            s._moves, s._stop = _generate(rules, own, opp, chain)
            if chain < 0 and not any(s._moves):
                s.result = Outcome(1 - turn, "blocked")
        if s.result is not None:
            s._moves, s._stop = _NO_MOVES, False
        return s

    @classmethod
    def new(cls, rules: Rules = STANDARD) -> GameState:
        turn = rules.first_player
        key = _key(START_P0, START_P1, turn)
        return cls._make(rules, (START_P0, START_P1), turn, -1, 0, 0, (key,))

    @classmethod
    def from_position(
        cls, p0: int, p1: int, turn: int = 0, rules: Rules = STANDARD, chain: int = -1
    ) -> GameState:
        """Build an arbitrary position (for tests, puzzles and analysis)."""
        if p0 & p1 or (p0 | p1) & ~FULL:
            raise ValueError("overlapping or off-board bitboards")
        if chain >= 0 and not ((p0, p1)[turn] >> chain & 1):
            raise ValueError("chain square must hold a bead of the side to move")
        return cls._make(rules, (p0, p1), turn, chain, 0, 0, (_key(p0, p1, turn),) if chain < 0 else ())

    # --- queries -------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return self.result is not None

    @property
    def key(self) -> int:
        """Exact identity of the position for transposition tables (ignores counters)."""
        return _key(self.bb[0], self.bb[1], self.turn) | (self.chain + 1) << 51

    @property
    def mid_chain(self) -> bool:
        return self.chain >= 0

    def count(self, player: int) -> int:
        return self.bb[player].bit_count()

    def piece_at(self, sq: int) -> int:
        """0 or 1 for a bead of that player, -1 for an empty point."""
        if self.bb[0] >> sq & 1:
            return 0
        if self.bb[1] >> sq & 1:
            return 1
        return -1

    @property
    def move_masks(self) -> tuple[int, ...]:
        """Per-direction bitmask of squares that can move in that direction."""
        return self._moves

    @property
    def can_stop(self) -> bool:
        return self._stop

    def legal_actions(self) -> list[int]:
        out = [d * N_SQUARES + sq for d in range(N_DIRS) for sq in iter_bits(self._moves[d])]
        if self._stop:
            out.append(STOP)
        return out

    def action_mask(self) -> list[bool]:
        """Length-N_ACTIONS legality mask (the hook an RL agent will use)."""
        mask = [False] * N_ACTIONS
        for a in self.legal_actions():
            mask[a] = True
        return mask

    def is_legal(self, action: int) -> bool:
        if action == STOP:
            return self._stop
        if not 0 <= action < STOP:
            return False
        d, sq = divmod(action, N_SQUARES)
        return bool(self._moves[d] >> sq & 1)

    def has_capture(self) -> bool:
        """True when the side to move has at least one jump available."""
        own, opp = self.bb[self.turn], self.bb[1 - self.turn]
        return any(self._moves[d] & shift_back(opp, SHIFT[d]) & own for d in range(N_DIRS))

    def must_capture(self) -> bool:
        """True when every legal action is a jump (forced capture or mid-chain)."""
        return self.result is None and (self.chain >= 0 or (self.rules.forced_capture and self.has_capture()))

    def describe(self, action: int) -> Move:
        if not self.is_legal(action):
            raise IllegalMoveError(f"illegal action {action}")
        if action == STOP:
            return Move(action, self.chain, self.chain, None)
        d, sq = divmod(action, N_SQUARES)
        mid = STEP_TO[d][sq]
        if self.bb[1 - self.turn] >> mid & 1:
            return Move(action, sq, JUMP_TO[d][sq], mid)
        return Move(action, sq, mid, None)

    def is_capture(self, action: int) -> bool:
        if action == STOP or not self.is_legal(action):
            return False
        d, sq = divmod(action, N_SQUARES)
        return bool(self.bb[1 - self.turn] >> STEP_TO[d][sq] & 1)

    def action_for(self, src: int, dst: int) -> int | None:
        """The legal action moving src -> dst, if any (for click/drag input)."""
        for d in range(N_DIRS):
            if self._moves[d] >> src & 1 and dst in (STEP_TO[d][src], JUMP_TO[d][src]):
                a = d * N_SQUARES + src
                if self.describe(a).dst == dst:
                    return a
        return None

    # --- transitions ---------------------------------------------------------

    def apply(self, action: int) -> GameState:
        if self.result is not None:
            raise IllegalMoveError("the game is over")
        if not self.is_legal(action):
            raise IllegalMoveError(f"illegal action {action} ({action_name(action)})")
        turn = self.turn
        own, opp = self.bb[turn], self.bb[1 - turn]
        if action == STOP:
            return self._end_turn(own, opp, captured=True)
        d, sq = divmod(action, N_SQUARES)
        mid = STEP_TO[d][sq]
        if opp >> mid & 1:
            land = JUMP_TO[d][sq]
            own ^= (1 << sq) | (1 << land)
            opp ^= 1 << mid
            if self.rules.chain_capture and opp and _can_jump_from(land, own, opp):
                bb = (own, opp) if turn == 0 else (opp, own)
                return GameState._make(self.rules, bb, turn, land, 0, self.turn_no, ())
            return self._end_turn(own, opp, captured=True)
        own ^= (1 << sq) | (1 << mid)
        return self._end_turn(own, opp, captured=self.chain >= 0)

    def _end_turn(self, own: int, opp: int, captured: bool) -> GameState:
        turn = self.turn
        bb = (own, opp) if turn == 0 else (opp, own)
        nxt = 1 - turn
        key = _key(bb[0], bb[1], nxt)
        if captured:
            quiet, history = 0, (key,)
        else:
            quiet, history = self.quiet + 1, self.history + (key,)
        return GameState._make(self.rules, bb, nxt, -1, quiet, self.turn_no + 1, history)

    def __repr__(self) -> str:
        status = f"result={self.result}" if self.result else f"turn={self.turn}"
        return f"GameState({self.count(0)} v {self.count(1)}, {status}, chain={self.chain})"

    def __str__(self) -> str:
        return render_text(self)


def _key(p0: int, p1: int, turn: int) -> int:
    return p0 | p1 << 25 | turn << 50


def _adjudicate(s: GameState, own: int, opp: int) -> Outcome | None:
    """Turn-start checks. Being blocked is detected after move generation."""
    rules = s.rules
    if not own:
        return Outcome(1 - s.turn, "captured_all")
    if not opp:  # cannot normally happen, but keeps from_position() honest
        return Outcome(s.turn, "captured_all")
    if rules.repetition_limit and s.history.count(s.history[-1]) >= rules.repetition_limit:
        return Outcome(None, "repetition")
    if rules.quiet_limit and s.quiet >= rules.quiet_limit:
        return Outcome(None, "quiet")
    if rules.max_turns and s.turn_no >= rules.max_turns:
        return Outcome(None, "max_turns")
    return None


def action_name(action: int) -> str:
    if action == STOP:
        return "stop"
    if not 0 <= action < STOP:
        return f"?{action}"
    from .board import DIR_NAMES

    d, sq = divmod(action, N_SQUARES)
    return f"{square_name(sq)}:{DIR_NAMES[d]}"


def render_text(s: GameState) -> str:
    glyph = {-1: ".", 0: "o", 1: "x"}
    rows = []
    for r in range(5):
        cells = " ".join(glyph[s.piece_at(r * 5 + c)] for c in range(5))
        rows.append(f"{5 - r} {cells}")
    rows.append("  a b c d e")
    if s.result:
        rows.append(f"result: {s.result}")
    else:
        rows.append(f"to move: {'o' if s.turn == 0 else 'x'}" + (f" (chain from {square_name(s.chain)})" if s.chain >= 0 else ""))
    return "\n".join(rows)
