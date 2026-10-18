"""Board geometry for Barah Goti (12 Beads).

The board is the 5x5 Alquerque grid. Points are numbered row-major from the
top-left corner, ``sq = 5 * row + col``. Player 0 starts on the bottom two rows,
player 1 on the top two.

Every point joins its orthogonal neighbours. Diagonal lines only pass through
the *strong* points, where ``row + col`` is even. Diagonal neighbours of a
strong point are strong too, so every diagonal step lies on a drawn line and
two steps in one direction are always collinear (a legal jump line).

Positions are bitboards: plain Python ints where bit ``sq`` is set when that
point is occupied. Move generation shifts whole bitboards per direction, so it
costs a handful of integer ops per direction instead of a loop over pieces.
"""

from __future__ import annotations

from collections.abc import Iterator

SIZE = 5
N_SQUARES = SIZE * SIZE
FULL = (1 << N_SQUARES) - 1
CENTER = 12

# Clockwise from north. The index arithmetic is load-bearing: d + 4 is the
# opposite direction, d + 2 a quarter turn clockwise, (8 - d) % 8 a mirror.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)
DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
N_DIRS = len(DIRECTIONS)

FILES = "abcde"


def rc(sq: int) -> tuple[int, int]:
    return divmod(sq, SIZE)


def square(row: int, col: int) -> int:
    return row * SIZE + col


def on_board(row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE


def is_strong(sq: int) -> bool:
    """Strong points carry diagonal lines (and so have up to 8 neighbours)."""
    row, col = rc(sq)
    return (row + col) % 2 == 0


def square_name(sq: int) -> str:
    """Algebraic name from player 0's side: files a-e left to right, ranks 1-5 bottom to top."""
    row, col = rc(sq)
    return f"{FILES[col]}{SIZE - row}"


def parse_square(name: str) -> int:
    name = name.strip().lower()
    if len(name) != 2 or name[0] not in FILES or not name[1].isdigit():
        raise ValueError(f"bad square name {name!r}")
    row = SIZE - int(name[1])
    col = FILES.index(name[0])
    if not on_board(row, col):
        raise ValueError(f"square {name!r} is off the board")
    return square(row, col)


def _target(sq: int, d: int, k: int) -> int:
    row, col = rc(sq)
    dr, dc = DIRECTIONS[d]
    if dr and dc and not is_strong(sq):
        return -1
    r2, c2 = row + k * dr, col + k * dc
    return square(r2, c2) if on_board(r2, c2) else -1


# STEP_TO[d][sq]: neighbour of sq in direction d, or -1. JUMP_TO: two points away.
STEP_TO: tuple[tuple[int, ...], ...] = tuple(
    tuple(_target(sq, d, 1) for sq in range(N_SQUARES)) for d in range(N_DIRS)
)
JUMP_TO: tuple[tuple[int, ...], ...] = tuple(
    tuple(_target(sq, d, 2) for sq in range(N_SQUARES)) for d in range(N_DIRS)
)
# Bit offset of one step in each direction.
SHIFT: tuple[int, ...] = tuple(dr * SIZE + dc for dr, dc in DIRECTIONS)
# Squares that can step / jump in direction d without leaving the lines.
STEP_SRC: tuple[int, ...] = tuple(
    sum(1 << sq for sq in range(N_SQUARES) if STEP_TO[d][sq] >= 0) for d in range(N_DIRS)
)
JUMP_SRC: tuple[int, ...] = tuple(
    sum(1 << sq for sq in range(N_SQUARES) if JUMP_TO[d][sq] >= 0) for d in range(N_DIRS)
)
STRONG = sum(1 << sq for sq in range(N_SQUARES) if is_strong(sq))
NEIGHBORS: tuple[tuple[int, ...], ...] = tuple(
    tuple(STEP_TO[d][sq] for d in range(N_DIRS) if STEP_TO[d][sq] >= 0) for sq in range(N_SQUARES)
)
# Undirected line segments (a < b), for drawing the board.
EDGES: tuple[tuple[int, int], ...] = tuple(
    sorted({(min(a, b), max(a, b)) for a in range(N_SQUARES) for b in NEIGHBORS[a]})
)

# Player 0 (bottom) owns rows 3-4 plus the two right-hand middle points; player 1
# is the 180-degree rotation of that. Only the centre starts empty.
START_P0 = sum(1 << sq for sq in range(N_SQUARES) if sq > CENTER)
START_P1 = sum(1 << sq for sq in range(N_SQUARES) if sq < CENTER)


def shift_back(bb: int, delta: int) -> int:
    """Move the bit at ``sq + delta`` onto ``sq`` (callers mask out wrap-around)."""
    return bb >> delta if delta > 0 else (bb << -delta) & FULL


def iter_bits(bb: int) -> Iterator[int]:
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


# --- D4 symmetries -----------------------------------------------------------
# Pieces move in every direction and the strong-point pattern survives every
# rotation and reflection of the square, so the *rules* are invariant under all
# 8 symmetries of the board. (The start position is only invariant under the
# 180-degree turn combined with a colour swap.) Index k: k % 4 quarter turns
# clockwise, then a left-right mirror when k >= 4.

N_SYMMETRIES = 8
ROT180 = 2


def _sym_rc(k: int, row: int, col: int) -> tuple[int, int]:
    for _ in range(k % 4):
        row, col = col, SIZE - 1 - row
    if k >= 4:
        col = SIZE - 1 - col
    return row, col


def _sym_dir(k: int, d: int) -> int:
    d = (d + 2 * (k % 4)) % N_DIRS
    return (N_DIRS - d) % N_DIRS if k >= 4 else d


SYM_SQ: tuple[tuple[int, ...], ...] = tuple(
    tuple(square(*_sym_rc(k, *rc(sq))) for sq in range(N_SQUARES)) for k in range(N_SYMMETRIES)
)
SYM_DIR: tuple[tuple[int, ...], ...] = tuple(
    tuple(_sym_dir(k, d) for d in range(N_DIRS)) for k in range(N_SYMMETRIES)
)
# INVERSE_SYM[k] undoes symmetry k.
INVERSE_SYM: tuple[int, ...] = tuple(
    next(j for j in range(N_SYMMETRIES) if all(SYM_SQ[j][SYM_SQ[k][s]] == s for s in range(N_SQUARES)))
    for k in range(N_SYMMETRIES)
)

# Bitboard transforms via five 5-bit row lookups instead of 25 single-bit moves.
_SYM_ROW_TABLE: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
    tuple(
        tuple(
            sum(1 << SYM_SQ[k][row * SIZE + c] for c in range(SIZE) if value >> c & 1)
            for value in range(1 << SIZE)
        )
        for row in range(SIZE)
    )
    for k in range(N_SYMMETRIES)
)


def transform_bb(bb: int, k: int) -> int:
    table = _SYM_ROW_TABLE[k]
    return (
        table[0][bb & 31]
        | table[1][bb >> 5 & 31]
        | table[2][bb >> 10 & 31]
        | table[3][bb >> 15 & 31]
        | table[4][bb >> 20 & 31]
    )
