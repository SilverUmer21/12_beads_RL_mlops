"""Turn a game state into a fixed-size vector for the neural network.

The network always sees the board from the side of the player to move: when it
is player 1's turn the board is rotated 180 degrees, so one network learns both
colours. Actions are rotated the same way. ROT180 is its own inverse, so the
same table converts to and from this "canonical" frame.

Observation layout (106 float32 values, all in [0, 1]):

    0..24    own beads
    25..49   opponent beads
    50..74   empty points
    75..99   one-hot square of the bead that must keep jumping (mid-chain)
    100      quiet turns / quiet limit          (40-quiet-turn draw rule)
    101      repetitions of this position / limit (threefold repetition rule)
    102      turns played / turn cap
    103      own bead count / 12
    104      opponent bead count / 12
    105      1.0 when mid-chain
"""

from __future__ import annotations

import numpy as np

from ..engine import N_ACTIONS, STOP, GameState
from ..engine.board import N_SQUARES, N_SYMMETRIES, ROT180, SYM_DIR, SYM_SQ, transform_bb

N_PLANES = 4
N_SCALARS = 6
OBS_SIZE = N_PLANES * N_SQUARES + N_SCALARS
START_BEADS = 12

_ROT_ACTION = np.arange(N_ACTIONS, dtype=np.int64)
for _a in range(STOP):
    _d, _sq = divmod(_a, N_SQUARES)
    _ROT_ACTION[_a] = SYM_DIR[ROT180][_d] * N_SQUARES + SYM_SQ[ROT180][_sq]
_ROT_ACTION.setflags(write=False)


def to_canonical(action: int, turn: int) -> int:
    """Engine action -> action in the mover's frame."""
    return action if turn == 0 else int(_ROT_ACTION[action])


def from_canonical(action: int, turn: int) -> int:
    """Action in the mover's frame -> engine action (ROT180 is an involution)."""
    return action if turn == 0 else int(_ROT_ACTION[action])


def _bits(bb: int) -> np.ndarray:
    raw = np.frombuffer(bb.to_bytes(4, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:N_SQUARES]


def repetition_count(s: GameState) -> int:
    """How often the current position has occurred since the last capture.

    Mid-chain states carry an empty history (a capture has just happened), so
    they count as 0.
    """
    if not s.history:
        return 0
    return s.history.count(s.history[-1])


def encode(s: GameState) -> np.ndarray:
    own, opp, chain = s.bb[s.turn], s.bb[1 - s.turn], s.chain
    if s.turn == 1:
        own, opp = transform_bb(own, ROT180), transform_bb(opp, ROT180)
        if chain >= 0:
            chain = SYM_SQ[ROT180][chain]
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[0:25] = _bits(own)
    obs[25:50] = _bits(opp)
    obs[50:75] = 1.0 - obs[0:25] - obs[25:50]
    if chain >= 0:
        obs[75 + chain] = 1.0
    r = s.rules
    obs[100] = min(s.quiet / r.quiet_limit, 1.0) if r.quiet_limit else 0.0
    obs[101] = min(repetition_count(s) / r.repetition_limit, 1.0) if r.repetition_limit else 0.0
    obs[102] = min(s.turn_no / r.max_turns, 1.0) if r.max_turns else 0.0
    obs[103] = own.bit_count() / START_BEADS
    obs[104] = opp.bit_count() / START_BEADS
    obs[105] = 1.0 if chain >= 0 else 0.0
    return obs


def symmetry_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index tables for the 8 board symmetries, for data augmentation.

    The rules don't depend on orientation, so rotating or mirroring a position
    (and its moves) gives another valid training example. Returns gather
    indices ``obs_src[k]`` and ``act_src[k]`` (new[j] = old[src[j]]) and the
    forward map ``act_map[k]`` (old action -> new action) for labels.
    """
    obs_src = np.tile(np.arange(OBS_SIZE), (N_SYMMETRIES, 1))
    act_map = np.tile(np.arange(N_ACTIONS), (N_SYMMETRIES, 1))
    for k in range(N_SYMMETRIES):
        for plane in range(N_PLANES):
            for sq in range(N_SQUARES):
                obs_src[k, plane * N_SQUARES + SYM_SQ[k][sq]] = plane * N_SQUARES + sq
        for a in range(STOP):
            d, sq = divmod(a, N_SQUARES)
            act_map[k, a] = SYM_DIR[k][d] * N_SQUARES + SYM_SQ[k][sq]
    act_src = np.argsort(act_map, axis=1)
    return obs_src, act_src, act_map


def canonical_mask(s: GameState) -> np.ndarray:
    """Legal-action mask in the mover's frame (length N_ACTIONS)."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    legal = s.legal_actions()
    if legal:
        idx = np.asarray(legal, dtype=np.int64)
        mask[idx if s.turn == 0 else _ROT_ACTION[idx]] = True
    return mask
