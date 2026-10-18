"""Procedural art for the Chiniot theme: no image files anywhere.

Chiniot (Punjab) is famous for sheesham furniture with inlaid brass. The board
is a lacquered sheesham slab with brass-wire lines; the beads are carnelian
(aqeeq) and Persian turquoise (feroza), both shaded per pixel as real domes
with numpy: Lambert diffuse, Blinn-Phong specular, Fresnel rim, and for
carnelian, light transmitted through the stone.

Everything is baked once per window size and cached; a frame is just blits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pygame

# --- palette -----------------------------------------------------------------

INK = (24, 15, 11)
IVORY = (238, 224, 198)
IVORY_DIM = (176, 160, 136)
BRASS = (212, 168, 86)
BRASS_LIGHT = (250, 224, 150)
BRASS_DARK = (122, 86, 36)
DANGER = (235, 86, 52)
PANEL = (34, 22, 16)

PLAYER_NAMES = ("Carnelian", "Turquoise")
PLAYER_ACCENT = ((238, 112, 58), (80, 206, 196))

LIGHT_DIR = np.array([-0.45, -0.6, 0.66])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


# --- noise -------------------------------------------------------------------


def _upsample(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize with smoothstep weights (soft value noise)."""
    ys = np.linspace(0, a.shape[0] - 1, h)
    xs = np.linspace(0, a.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, a.shape[0] - 1)
    x1 = np.minimum(x0 + 1, a.shape[1] - 1)
    fy = ys - y0
    fx = xs - x0
    fy = (fy * fy * (3 - 2 * fy))[:, None]
    fx = (fx * fx * (3 - 2 * fx))[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def fbm(h: int, w: int, seed: int, octaves: int = 5, cells: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Fractal value noise in [0, 1]. ``cells`` is the coarsest grid (rows, cols)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    cy, cx = cells
    for _ in range(octaves):
        out += amp * _upsample(rng.random((cy + 1, cx + 1)), h, w)
        total += amp
        amp *= 0.5
        cy, cx = cy * 2, cx * 2
    return out / total


def _to_surface(rgb: np.ndarray, alpha: np.ndarray | None = None) -> pygame.Surface:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    if alpha is None:
        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
        return surf
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
    a = pygame.surfarray.pixels_alpha(surf)
    a[:] = np.clip(alpha * 255, 0, 255).astype(np.uint8).T
    del a
    return surf


def _mix(c0, c1, t):
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    return c0 + (c1 - c0) * t[..., None]


# --- wood --------------------------------------------------------------------


def sheesham(w: int, h: int, seed: int = 3) -> np.ndarray:
    """Sheesham (Indian rosewood): warm brown with darker streaks and a lacquer sheen."""
    warp = fbm(h, w, seed, 5, (3, 2))
    streak = fbm(h, w, seed + 1, 4, (24, 2))  # long along x, thin along y
    fine = fbm(h, w, seed + 2, 3, (90, 8))
    y = np.linspace(0, 1, h)[:, None]
    rings = 0.5 + 0.5 * np.sin(2 * math.pi * (y * 15 + warp * 2.6 + streak * 1.1))
    t = np.clip(rings * 0.45 + fine * 0.55, 0, 1)
    rgb = _mix((50, 25, 14), (104, 55, 31), t)
    dark = np.clip((streak - 0.6) * 5, 0, 1)
    rgb = rgb * (1 - 0.4 * dark[..., None])
    # Lacquer: a broad soft highlight from the top-left.
    yy, xx = np.mgrid[0:h, 0:w]
    sheen = np.exp(-(((xx / w - 0.3) ** 2) / 0.12 + ((yy / h - 0.25) ** 2) / 0.06))
    rgb = rgb + sheen[..., None] * np.array([30, 20, 12])
    return rgb


# --- shaded domes ------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    kind: str  # "carnelian" | "turquoise" | "brass"
    seed: int = 0


CARNELIAN = Material("carnelian", 11)
TURQUOISE = Material("turquoise", 23)
BRASS_STUD = Material("brass", 5)
MATERIALS = (CARNELIAN, TURQUOISE)


def _dome(n: int, flatten: float = 0.8):
    """Unit normals of a dome rendered into an n x n grid, plus coverage alpha."""
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - c) / (n / 2)
    y = (yy - c) / (n / 2)
    r = np.sqrt(x * x + y * y)
    inside = np.clip(r, 0, 1)
    z = np.sqrt(np.clip(1 - inside**2, 0, 1)) * flatten
    norm = np.sqrt(x * x + y * y + z * z) + 1e-9
    nx, ny, nz = x / norm, y / norm, z / norm
    alpha = np.clip((1 - r) * n / 2.2, 0, 1)  # ~1px anti-aliased edge
    return x, y, nx, ny, nz, r, alpha


def _shade(n: int, mat: Material) -> tuple[np.ndarray, np.ndarray]:
    x, y, nx, ny, nz, r, alpha = _dome(n)
    lx, ly, lz = LIGHT_DIR
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    hx, hy, hz = lx, ly, lz + 1
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    ndoth = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1)
    fresnel = (1 - np.clip(nz / 0.8, 0, 1)) ** 3

    if mat.kind == "carnelian":
        noise = fbm(n, n, mat.seed, 4, (3, 3))
        bands = 0.5 + 0.5 * np.sin((x * 0.6 + y * 0.9) * 9 + noise * 5)
        albedo = _mix((120, 18, 6), (226, 92, 30), bands * 0.55 + noise * 0.45)
        # Light enters at the top-left and glows out of the lower-right interior.
        through = np.clip(-(x * lx + y * ly) / math.hypot(lx, ly), 0, 1) * np.clip(1 - r**2, 0, 1)
        rgb = albedo * (0.30 + 0.62 * diffuse[..., None])
        rgb += np.array([255, 138, 48]) * (through**1.4 * 0.55)[..., None]
        rgb += 255 * (ndoth**90 * 1.1)[..., None]
        rgb += np.array([255, 190, 140]) * (ndoth**12 * 0.12)[..., None]
        rgb += np.array([255, 120, 70]) * (fresnel * 0.25)[..., None]
    elif mat.kind == "turquoise":
        tone = fbm(n, n, mat.seed + 7, 3, (2, 2))
        albedo = _mix((34, 140, 136), (110, 212, 198), tone)
        # Spiderweb matrix: ridges of two noise layers, faded in patches.
        web = np.zeros((n, n))
        for k, cells in enumerate(((5, 5), (9, 9))):
            noise = fbm(n, n, mat.seed + k, 3, cells)
            web = np.maximum(web, np.clip(1 - np.abs(noise - 0.5) * (60 + 40 * k), 0, 1))
        patch = np.clip((fbm(n, n, mat.seed + 31, 2, (2, 2)) - 0.42) * 4, 0, 1)
        vein = web * patch * 0.85
        albedo = albedo * (1 - vein[..., None]) + np.array([70, 54, 38]) * vein[..., None]
        rgb = albedo * (0.28 + 0.72 * diffuse[..., None])
        rgb += 255 * (ndoth**40 * 0.55)[..., None]  # satin, not glassy
        rgb += np.array([180, 255, 240]) * (fresnel * 0.18)[..., None]
    else:  # brass stud
        albedo = np.array([200, 152, 72], float)
        rgb = albedo * (0.25 + 0.8 * diffuse[..., None])
        rgb += np.array([255, 236, 170]) * (ndoth**30 * 0.9)[..., None]
        rgb += np.array([120, 80, 30]) * (fresnel * 0.4)[..., None]
    return rgb, alpha


@lru_cache(maxsize=32)
def bead(radius: int, mat: Material) -> pygame.Surface:
    """A bead sprite of the given radius, supersampled 2x for clean edges."""
    n = max(4, radius * 4)
    rgb, alpha = _shade(n, mat)
    big = _to_surface(rgb, alpha)
    return pygame.transform.smoothscale(big, (radius * 2, radius * 2))


@lru_cache(maxsize=32)
def soft_disc(radius: int, color: tuple[int, int, int], power: float = 2.0) -> pygame.Surface:
    """Gaussian-ish radial falloff: used for shadows, glows and light spill."""
    n = radius * 2
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / radius
    alpha = np.clip(1 - d, 0, 1) ** power
    rgb = np.broadcast_to(np.array(color, float), (n, n, 3))
    return _to_surface(rgb, alpha)


@lru_cache(maxsize=256)
def glow(radius: int, color: tuple[int, int, int], intensity: float = 1.0) -> pygame.Surface:
    """Premultiplied light for BLEND_RGB_ADD: black is 'no light', so no alpha needed."""
    n = radius * 2
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / radius
    falloff = np.clip(1 - d, 0, 1) ** 2 * intensity
    return _to_surface(np.array(color, float) * falloff[..., None])


@lru_cache(maxsize=16)
def ring(radius: int, width: float, color: tuple[int, int, int]) -> pygame.Surface:
    n = radius * 2 + 4
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    alpha = np.clip(width / 2 + 0.75 - np.abs(d - radius), 0, 1)
    rgb = np.broadcast_to(np.array(color, float), (n, n, 3))
    return _to_surface(rgb, alpha)


# --- geometry helpers ---------------------------------------------------------


def thick_line(surf: pygame.Surface, color, p1, p2, width: float) -> None:
    """A line as a polygon: exact width at any angle (pygame's wide lines are not)."""
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ox, oy = -dy / length * width / 2, dx / length * width / 2
    pygame.draw.polygon(surf, color, [(x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (x2 - ox, y2 - oy), (x1 - ox, y1 - oy)])


def star8(cx: float, cy: float, r: float, inner: float = 0.42, rot: float = 0.0) -> list[tuple[float, float]]:
    """Eight-pointed star (khatim), the commonest motif in Mughal inlay."""
    pts = []
    for i in range(16):
        rr = r if i % 2 == 0 else r * inner
        a = rot + i * math.pi / 8
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


# --- baked layers --------------------------------------------------------------


@dataclass
class BoardGeometry:
    """Where things go for a board of ``size`` pixels with its top-left at ``origin``."""

    origin: tuple[int, int]
    size: int

    @property
    def frame(self) -> float:
        return self.size * 0.135

    @property
    def cell(self) -> float:
        return (self.size - 2 * self.frame) / 4

    def point(self, sq: int) -> tuple[float, float]:
        r, c = divmod(sq, 5)
        ox, oy = self.origin
        return (ox + self.frame + c * self.cell, oy + self.frame + r * self.cell)

    def local_point(self, sq: int, scale: float = 1.0) -> tuple[float, float]:
        r, c = divmod(sq, 5)
        return ((self.frame + c * self.cell) * scale, (self.frame + r * self.cell) * scale)

    @property
    def bead_radius(self) -> int:
        return max(8, int(self.cell * 0.27))

    def hit(self, pos: tuple[int, int]) -> int | None:
        best, best_d = None, self.cell * 0.45
        for sq in range(25):
            px, py = self.point(sq)
            d = math.hypot(pos[0] - px, pos[1] - py)
            if d < best_d:
                best, best_d = sq, d
        return best


def bake_board(geo: BoardGeometry, edges, strong_points, font: pygame.font.Font) -> pygame.Surface:
    """Wood slab, inlay border, brass lines and studs, all in one surface."""
    size = geo.size
    pad = int(size * 0.04)  # room for the slab's drop shadow
    out = pygame.Surface((size + 2 * pad, size + 2 * pad), pygame.SRCALPHA)

    shadow = soft_disc(int(size * 0.62), (0, 0, 0), 1.2)
    shadow = pygame.transform.smoothscale(shadow, (int(size * 1.08), int(size * 1.08)))
    shadow.set_alpha(200)
    out.blit(shadow, shadow.get_rect(center=(pad + size / 2 + size * 0.015, pad + size / 2 + size * 0.03)))

    # The slab itself, with rounded corners.
    wood = _to_surface(sheesham(size, size))
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=int(size * 0.035))
    slab = pygame.Surface((size, size), pygame.SRCALPHA)
    slab.blit(wood, (0, 0))
    slab.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    # Inlay is drawn at 2x and scaled down: cheap, good anti-aliasing.
    ss = 2
    inlay = pygame.Surface((size * ss, size * ss), pygame.SRCALPHA)
    f = geo.frame * ss
    S = size * ss
    band_out, band_in = f * 0.16, f * 0.44
    # Darken the frame band so the inlay reads as set into the wood.
    pygame.draw.rect(inlay, (0, 0, 0, 70), (band_out, band_out, S - 2 * band_out, S - 2 * band_out), border_radius=int(f * 0.2))
    # Brass border wires and a band of diamonds between them.
    for inset, w in ((band_out, 0.022), (band_in, 0.022)):
        pygame.draw.rect(inlay, BRASS, (inset, inset, S - 2 * inset, S - 2 * inset), max(2, int(f * w * 2)), border_radius=int(f * 0.18))
    mid = (band_out + band_in) / 2
    span = S - 2 * mid
    count = 14
    dsz = (band_in - band_out) * 0.30
    for side in range(4):
        for i in range(count + 1):
            t = i / count
            if side == 0:
                cx, cy = mid + span * t, mid
            elif side == 1:
                cx, cy = S - mid, mid + span * t
            elif side == 2:
                cx, cy = mid + span * t, S - mid
            else:
                cx, cy = mid, mid + span * t
            if i in (0, count):
                continue
            big = i % 2 == 0
            d = dsz * (1.0 if big else 0.55)
            pygame.draw.polygon(inlay, BRASS if big else BRASS_DARK, [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)])
    for cx, cy in ((mid, mid), (S - mid, mid), (mid, S - mid), (S - mid, S - mid)):
        pygame.draw.polygon(inlay, BRASS, star8(cx, cy, (band_in - band_out) * 0.62, 0.5, math.pi / 8))
        pygame.draw.circle(inlay, BRASS_DARK, (cx, cy), (band_in - band_out) * 0.14)

    # Lines: a dark routed groove, the brass wire, and a thin specular edge.
    lw = max(3.0, geo.cell * ss * 0.05)
    for a, b in edges:
        p1, p2 = geo.local_point(a, ss), geo.local_point(b, ss)
        thick_line(inlay, (20, 10, 4, 150), (p1[0] + lw * 0.35, p1[1] + lw * 0.5), (p2[0] + lw * 0.35, p2[1] + lw * 0.5), lw * 1.9)
    for a, b in edges:
        thick_line(inlay, BRASS_DARK, geo.local_point(a, ss), geo.local_point(b, ss), lw)
    for a, b in edges:
        p1, p2 = geo.local_point(a, ss), geo.local_point(b, ss)
        thick_line(inlay, BRASS, (p1[0] - lw * 0.12, p1[1] - lw * 0.12), (p2[0] - lw * 0.12, p2[1] - lw * 0.12), lw * 0.55)

    inlay = pygame.transform.smoothscale(inlay, (size, size))
    slab.blit(inlay, (0, 0))

    # Brass studs at every point; strong points (with diagonals) are larger.
    for sq in range(25):
        px, py = geo.local_point(sq)
        r = max(3, int(geo.cell * (0.085 if sq in strong_points else 0.06)))
        slab.blit(soft_disc(r + 3, (0, 0, 0)), (px - r - 3 + 1, py - r - 3 + 2))
        stud = bead(r, BRASS_STUD)
        slab.blit(stud, (px - r, py - r))

    # Engraved coordinates in the frame.
    for i in range(5):
        x, _ = geo.local_point(i)
        _, y = geo.local_point(i * 5)
        for text, pos in ((chr(97 + i), (x, size - geo.frame * 0.54)), (str(5 - i), (geo.frame * 0.54, y))):
            img = font.render(text, True, BRASS_LIGHT)
            img.set_alpha(150)
            slab.blit(img, img.get_rect(center=pos))

    out.blit(slab, (pad, pad))
    return out


def bake_background(w: int, h: int) -> pygame.Surface:
    """Dark walnut wall with a faint brass jali (lattice) of eight-pointed stars."""
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w * 0.42) / w) ** 2 + ((yy - h * 0.45) / h) ** 2)
    glow = np.clip(1 - d * 1.5, 0, 1) ** 1.8
    grain = fbm(h, w, 41, 3, (3, 3))
    rgb = _mix((16, 10, 8), (66, 40, 26), glow * 0.85 + grain * 0.15)
    surf = _to_surface(rgb)

    jali = pygame.Surface((w, h), pygame.SRCALPHA)
    step = 64
    for gy in range(-step, h + step, step):
        for gx in range(-step, w + step, step):
            ox = step // 2 if (gy // step) % 2 else 0
            pygame.draw.polygon(jali, (*BRASS, 14), star8(gx + ox, gy, step * 0.34, 0.55), 1)
            pygame.draw.circle(jali, (*BRASS, 10), (gx + ox + step // 2, gy + step // 2), 2)
    surf.blit(jali, (0, 0))
    return surf


@lru_cache(maxsize=64)
def font(size: int, serif: bool = False, bold: bool = False) -> pygame.font.Font:
    names = "georgia,palatinolinotype,bookantiqua,timesnewroman" if serif else "segoeui,calibri,helvetica,arial"
    return pygame.font.SysFont(names, size, bold=bold)
