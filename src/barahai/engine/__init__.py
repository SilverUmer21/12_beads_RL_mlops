"""Pure-Python rules engine for Barah Goti (12 Beads). No third-party dependencies."""

from .board import CENTER, EDGES, N_SQUARES, NEIGHBORS, is_strong, parse_square, rc, square, square_name
from .record import GameRecord, Turn, group_turns
from .rules import STANDARD, Rules
from .state import (
    N_ACTIONS,
    STOP,
    GameState,
    IllegalMoveError,
    Move,
    Outcome,
    action_name,
    decode_action,
    encode_action,
)

__all__ = [
    "CENTER",
    "EDGES",
    "N_ACTIONS",
    "N_SQUARES",
    "NEIGHBORS",
    "STANDARD",
    "STOP",
    "GameRecord",
    "GameState",
    "IllegalMoveError",
    "Move",
    "Outcome",
    "Rules",
    "Turn",
    "action_name",
    "decode_action",
    "encode_action",
    "group_turns",
    "is_strong",
    "parse_square",
    "rc",
    "square",
    "square_name",
]
