"""A deliberately naive, obviously-correct move generator on a 2D grid.

It shares no code with the bitboard engine, so the two can be diffed against
each other. Moves are (src, dst, captured_or_None) tuples on (row, col) points.
"""

from __future__ import annotations

DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def grid_from(p0: int, p1: int) -> list[list[int]]:
    g = [[-1] * 5 for _ in range(5)]
    for sq in range(25):
        if p0 >> sq & 1:
            g[sq // 5][sq % 5] = 0
        elif p1 >> sq & 1:
            g[sq // 5][sq % 5] = 1
    return g


def _lines_from(r: int, c: int):
    for dr, dc in DIRS:
        if dr and dc and (r + c) % 2:
            continue  # weak point: no diagonal lines
        yield dr, dc


def _inside(r: int, c: int) -> bool:
    return 0 <= r < 5 and 0 <= c < 5


def jumps_from(g, player: int, r: int, c: int):
    for dr, dc in _lines_from(r, c):
        mr, mc, lr, lc = r + dr, c + dc, r + 2 * dr, c + 2 * dc
        if _inside(lr, lc) and g[mr][mc] == 1 - player and g[lr][lc] == -1:
            yield (r * 5 + c, lr * 5 + lc, mr * 5 + mc)


def legal_moves(g, player: int, forced: bool, chain: int = -1) -> set:
    if chain >= 0:
        return set(jumps_from(g, player, chain // 5, chain % 5))
    jumps, steps = set(), set()
    for r in range(5):
        for c in range(5):
            if g[r][c] != player:
                continue
            jumps |= set(jumps_from(g, player, r, c))
            for dr, dc in _lines_from(r, c):
                nr, nc = r + dr, c + dc
                if _inside(nr, nc) and g[nr][nc] == -1:
                    steps.add((r * 5 + c, nr * 5 + nc, None))
    if forced and jumps:
        return jumps
    return jumps | steps
