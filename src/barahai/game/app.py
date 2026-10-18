"""Barah Goti desktop client (pygame-ce).

Controls
    click a bead, then a glowing point      move (or drag the bead there)
    N new game   U / Ctrl+Z undo   H hint   M sound   S save   L load   Esc deselect

The window never blocks: computer players search in a worker process (see
worker.py for why not a thread), and game states are immutable, so the UI keeps
drawing the current position while a copy of it is searched.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pygame

from ..engine import EDGES, STOP, GameRecord, GameState, group_turns, is_strong
from . import art, worker
from .sound import Sounds

SAVE_PATH = Path.home() / ".barahai" / "saved_game.json"

MODES = ("vs Computer", "Two players", "Watch computers")
LEVELS = ("Naya", "Khiladi", "Ustad")
LEVEL_INFO = {
    "Naya": "beginner: greedy, sometimes careless",
    "Khiladi": "player: searches 3 turns ahead",
    "Ustad": "master: deep search, ~1 s a move",
}
SIDES = ("Carnelian", "Turquoise")
REASONS = {
    "captured_all": "every bead captured",
    "blocked": "no legal move left",
    "quiet": "40 turns without a capture",
    "repetition": "the same position three times",
    "max_turns": "turn limit reached",
}


def ease_out(t: float) -> float:
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# --- transient visuals ---------------------------------------------------------


@dataclass
class Motion:
    player: int
    start: tuple[float, float]
    end: tuple[float, float]
    dst: int
    victim: int | None
    t0: float
    dur: float
    burst_done: bool = False

    def progress(self, now: float) -> float:
        return min(1.0, (now - self.t0) / self.dur)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    size: int


@dataclass
class Flyer:
    """A captured bead flying from the board to its captor's tray."""

    captor: int
    start: tuple[float, float]
    t0: float
    slot: int
    dur: float = 0.7


@dataclass
class Widget:
    rect: pygame.Rect
    kind: str  # "button" | "option"
    key: str
    label: str
    values: tuple[str, ...] = ()
    enabled: bool = True


@dataclass
class Settings:
    mode: str = "vs Computer"
    level: str = "Khiladi"
    side: str = "Carnelian"
    seed: int = field(default_factory=lambda: random.randrange(1 << 30))


# --- the app -------------------------------------------------------------------


class App:
    PANEL_W = 360

    def __init__(self) -> None:
        _dpi_aware()
        pygame.init()
        pygame.display.set_caption("Barah Goti · twelve beads")
        desk_w, desk_h = pygame.display.get_desktop_sizes()[0]
        w, h = min(1320, int(desk_w * 0.88)), min(900, int(desk_h * 0.86))
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_icon(art.bead(16, art.CARNELIAN))
        try:
            pygame.Window.from_display_module().minimum_size = (980, 680)
        except Exception:
            pass
        self.sounds = Sounds()
        ctx = mp.get_context("spawn")
        self.watermark = ctx.Value("q", 0, lock=False)
        self.job = 0
        self.pool = ProcessPoolExecutor(1, mp_context=ctx, initializer=worker.init, initargs=(self.watermark,))
        self.pool.submit(worker.warm_up)  # start the process while the board bakes
        self.clock = pygame.time.Clock()
        self.settings = Settings()

        self.ai_future: Future | None = None
        self.ai_started = 0.0
        self.hint_future: Future | None = None
        self.hint: tuple[int, int, float] | None = None
        self.gen = 0

        self.motion: Motion | None = None
        self.particles: list[Particle] = []
        self.flyers: list[Flyer] = []
        self.selected: int | None = None
        self.hover: int | None = None
        self.drag: tuple[int, tuple[int, int]] | None = None  # (square, press position)
        self.drag_pos: tuple[int, int] | None = None
        self.toast: tuple[str, float] | None = None
        self.scroll = 0
        self.game_over_at: float | None = None
        self.widgets: list[Widget] = []
        self._turns_cache: tuple[int, list] = (-1, [])
        self._pending_resize: float | None = None

        self.layout()
        self.new_game()

    # --- layout & baking -------------------------------------------------------

    def layout(self) -> None:
        w, h = self.screen.get_size()
        self.ui = max(0.8, min(1.5, h / 880))
        u = self.u
        self.panel = pygame.Rect(w - u(self.PANEL_W) - u(24), u(24), u(self.PANEL_W), h - u(48))
        margin = u(30)
        avail_w = self.panel.left - 2 * margin
        size = max(300, min(h - 2 * margin, avail_w))
        ox = margin + (avail_w - size) // 2
        oy = (h - size) // 2
        self.geo = art.BoardGeometry((ox, oy), size)
        strong = frozenset(s for s in range(25) if is_strong(s))
        self.board_img = art.bake_board(self.geo, EDGES, strong, art.font(max(11, int(size * 0.024)), serif=True))
        self.board_pad = int(size * 0.04)
        self.bg = art.bake_background(w, h)
        self.r = self.geo.bead_radius
        self.f_title = art.font(u(38), serif=True, bold=True)
        self.f_sub = art.font(u(15), serif=True)
        self.f_name = art.font(u(21), serif=True, bold=True)
        self.f_ui = art.font(u(15))
        self.f_small = art.font(u(13))
        self.f_count = art.font(u(30), serif=True, bold=True)
        self.f_big = art.font(u(46), serif=True, bold=True)
        self.f_moves = art.font(u(15))
        self._build_widgets()

    def u(self, x: float) -> int:
        return int(round(x * self.ui))

    def _build_widgets(self) -> None:
        u, p = self.u, self.panel
        x, w = p.left + u(18), p.width - u(36)
        y = p.top + u(372)
        self.widgets = []
        for key, label, values in (
            ("mode", "Mode", MODES),
            ("level", "Computer", LEVELS),
            ("side", "You play", SIDES),
        ):
            self.widgets.append(Widget(pygame.Rect(x, y, w, u(30)), "option", key, label, values))
            y += u(34)
        y += u(26)  # room for the level description line
        bw = (w - u(16)) // 3
        for i, (key, label) in enumerate((("new", "New game"), ("undo", "Undo"), ("hint", "Hint"))):
            self.widgets.append(Widget(pygame.Rect(x + i * (bw + u(8)), y, bw, u(36)), "button", key, label))
        self.moves_rect = pygame.Rect(x, y + u(56), w, p.bottom - (y + u(56)) - u(40))

    # --- game control ------------------------------------------------------------

    @property
    def state(self) -> GameState:
        return self.states[-1]

    def new_game(self, record: GameRecord | None = None) -> None:
        self._cancel_search()
        s = self.settings
        # Each seat holds the computer level playing it, or None for a human.
        self.agents: list[str | None] = [None, None]
        self.game_id = self.gen
        self.seed = s.seed + self.gen
        if s.mode == "vs Computer":
            self.agents[1 if s.side == "Carnelian" else 0] = s.level
        elif s.mode == "Watch computers":
            self.agents = [s.level, s.level]
        self.states = record.replay() if record else [GameState.new()]
        self.actions = list(record.actions) if record else []
        self.motion = None
        self.particles.clear()
        self.flyers.clear()
        self.selected = self.drag = self.hint = None
        self.game_over_at = time.perf_counter() if self.state.is_terminal else None
        self.scroll = 0
        self._sync_chain_selection()

    def _cancel_search(self) -> None:
        self.gen += 1
        self.watermark.value = self.job  # every job submitted so far is now stale
        self.ai_future = self.hint_future = None

    def _submit(self, seat: tuple, level: str, state: GameState) -> Future:
        self.job += 1
        key = (self.game_id, *seat)
        seed = self.seed + (seat[0] if isinstance(seat[0], int) else 7)
        fut = self.pool.submit(worker.think, self.job, key, level, seed, state)
        fut.gen = self.gen  # type: ignore[attr-defined]
        return fut

    def is_human(self, player: int) -> bool:
        return self.agents[player] is None

    def human_can_act(self) -> bool:
        return (
            not self.state.is_terminal
            and self.motion is None
            and self.is_human(self.state.turn)
            and self.ai_future is None
        )

    def undo(self) -> None:
        if self.settings.mode == "Watch computers" or len(self.states) == 1:
            return
        self._cancel_search()
        self.motion = None
        self.flyers.clear()
        self._pop()
        while len(self.states) > 1 and (self.state.chain >= 0 or not self.is_human(self.state.turn)):
            self._pop()
        self.game_over_at = None
        self.selected = self.hint = None
        self._sync_chain_selection()
        self.sounds.play("select")

    def _pop(self) -> None:
        self.states.pop()
        self.actions.pop()

    def apply(self, action: int, from_xy: tuple[float, float] | None = None) -> None:
        s = self.state
        mv = s.describe(action)
        self.states.append(s.apply(action))
        self.actions.append(action)
        start = from_xy or self.geo.point(mv.src)
        end = self.geo.point(mv.dst)
        dur = 0.34 if mv.is_capture else 0.2
        if from_xy:
            dur = 0.12 + 0.12 * mv.is_capture
        self.motion = Motion(s.turn, start, end, mv.dst, mv.captured, time.perf_counter(), dur)
        if mv.is_capture:
            self.sounds.play("select")
        self.selected = None
        self.hint = None

    def _sync_chain_selection(self) -> None:
        s = self.state
        if s.chain >= 0 and self.is_human(s.turn):
            self.selected = s.chain

    def targets(self, sq: int) -> dict[int, int]:
        """Legal destinations for the bead on sq: {dst: action}."""
        s = self.state
        out = {}
        for a in s.legal_actions():
            mv = s.describe(a)
            if mv.src == sq and a != STOP:
                out[mv.dst] = a
        return out

    def movable(self) -> set[int]:
        s = self.state
        return {s.describe(a).src for a in s.legal_actions() if a != STOP}

    def flash(self, text: str) -> None:
        self.toast = (text, time.perf_counter())

    def save(self) -> None:
        GameRecord(self.state.rules, list(self.actions)).save(SAVE_PATH)
        self.flash(f"Saved to {SAVE_PATH}")

    def load(self) -> None:
        try:
            record = GameRecord.load(SAVE_PATH)
        except (OSError, ValueError) as e:
            self.flash(f"Could not load: {e.__class__.__name__}")
            return
        self.new_game(record)
        self.flash("Saved game restored")

    # --- input -------------------------------------------------------------------

    def on_board_press(self, pos: tuple[int, int]) -> None:
        if not self.human_can_act():
            return
        sq = self.geo.hit(pos)
        s = self.state
        if sq is None:
            if s.chain < 0:
                self.selected = None
            return
        if self.selected is not None and sq in self.targets(self.selected):
            self.apply(self.targets(self.selected)[sq])
            return
        if s.chain >= 0:
            if sq != s.chain:
                self.sounds.play("invalid")
                self.flash("Keep jumping: the same bead must capture again")
            self.drag = (sq, pos) if sq == s.chain else None
            return
        if s.piece_at(sq) == s.turn:
            if sq in self.movable():
                self.selected = sq
                self.drag = (sq, pos)
                self.sounds.play("select")
            else:
                self.sounds.play("invalid")
                if s.must_capture():
                    self.flash("Capture is mandatory: a glowing bead must jump")
                else:
                    self.flash("That bead has nowhere to go")
                self.selected = None
        else:
            self.selected = None

    def on_board_release(self, pos: tuple[int, int]) -> None:
        if not self.drag:
            return
        sq, press = self.drag
        self.drag = None
        moved = math.hypot(pos[0] - press[0], pos[1] - press[1]) > 8
        self.drag_pos = None
        if not moved or not self.human_can_act():
            return
        dst = self.geo.hit(pos)
        targets = self.targets(sq)
        if dst in targets:
            self.apply(targets[dst], from_xy=pos)
        elif dst != sq:
            self.sounds.play("invalid")

    def on_widget(self, w: Widget, pos: tuple[int, int]) -> None:
        if not w.enabled:
            return
        self.sounds.play("select")
        if w.kind == "button":
            if w.key == "new":
                self.new_game()
            elif w.key == "undo":
                self.undo()
            elif w.key == "hint":
                self.request_hint()
            return
        step = -1 if pos[0] < w.rect.centerx else 1
        current = getattr(self.settings, w.key)
        setattr(self.settings, w.key, w.values[(w.values.index(current) + step) % len(w.values)])
        if not self.actions or self.state.is_terminal:
            self.new_game()
        else:
            self.flash("Press New game to apply")

    def request_hint(self) -> None:
        if not self.human_can_act() or self.hint_future:
            return
        self.hint_future = self._submit(("hint",), "hint", self.state)
        self.flash("Thinking about a hint…")

    def handle(self, e: pygame.event.Event) -> bool:
        if e.type == pygame.QUIT:
            return False
        if e.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
            self._pending_resize = time.perf_counter()
        elif e.type == pygame.MOUSEMOTION:
            self.hover = self.geo.hit(e.pos)
            if self.drag:
                self.drag_pos = e.pos
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for w in self.widgets:
                if w.rect.collidepoint(e.pos):
                    self.on_widget(w, e.pos)
                    return True
            self.on_board_press(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.on_board_release(e.pos)
        elif e.type == pygame.MOUSEWHEEL:
            if self.moves_rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll = max(0, self.scroll - e.y)
        elif e.type == pygame.KEYDOWN:
            ctrl = e.mod & pygame.KMOD_CTRL
            if e.key == pygame.K_n:
                self.new_game()
            elif e.key == pygame.K_u or (ctrl and e.key == pygame.K_z):
                self.undo()
            elif e.key == pygame.K_h:
                self.request_hint()
            elif e.key == pygame.K_m:
                self.flash("Sound on" if self.sounds.toggle() else "Sound off")
            elif e.key == pygame.K_s:
                self.save()
            elif e.key == pygame.K_l:
                self.load()
            elif e.key == pygame.K_ESCAPE:
                if self.state.chain < 0:
                    self.selected = None
                self.hint = None
        return True

    # --- update --------------------------------------------------------------------

    def update(self, dt: float) -> None:
        now = time.perf_counter()
        if self._pending_resize and now - self._pending_resize > 0.2:
            self._pending_resize = None
            self.layout()
            self.motion = None

        m = self.motion
        if m:
            p = m.progress(now)
            if m.victim is not None and not m.burst_done and p >= 0.5:
                m.burst_done = True
                self._burst(m.victim, 1 - m.player)
            if p >= 1:
                self.motion = None
                self.sounds.play("capture" if m.victim is not None else "step")
                self._sync_chain_selection()
                if self.state.is_terminal:
                    self._game_over()

        s = self.state
        if not s.is_terminal and self.motion is None and not self.is_human(s.turn) and self.ai_future is None:
            self.ai_started = now
            self.ai_future = self._submit((s.turn,), self.agents[s.turn], s)
        fut = self.ai_future
        if fut and fut.done():
            min_wait = 0.18 if s.chain >= 0 else 0.45
            if now - self.ai_started >= min_wait:
                self.ai_future = None
                if fut.exception() is not None:
                    raise fut.exception()
                action = fut.result()
                if fut.gen == self.gen and action is not None and self.state.is_legal(action):  # type: ignore[attr-defined]
                    self.apply(action)
        hf = self.hint_future
        if hf and hf.done():
            self.hint_future = None
            ok = hf.exception() is None and hf.result() is not None
            if ok and hf.gen == self.gen and self.state.is_legal(hf.result()):  # type: ignore[attr-defined]
                mv = self.state.describe(hf.result())
                self.hint = (mv.src, mv.dst, now)
                self.toast = None

        for pt in self.particles:
            pt.life -= dt
            pt.x += pt.vx * dt
            pt.y += pt.vy * dt
            pt.vx *= 0.92
            pt.vy *= 0.92
        self.particles = [pt for pt in self.particles if pt.life > 0]
        self.flyers = [f for f in self.flyers if now - f.t0 < f.dur]

    def _burst(self, sq: int, victim_player: int) -> None:
        x, y = self.geo.point(sq)
        color = art.PLAYER_ACCENT[victim_player]
        rng = random.Random()
        for _ in range(26):
            a = rng.uniform(0, 2 * math.pi)
            v = rng.uniform(60, 260) * self.ui
            life = rng.uniform(0.35, 0.8)
            self.particles.append(Particle(x, y, math.cos(a) * v, math.sin(a) * v, life, life, color, rng.choice((2, 3, 4))))
        captor = 1 - victim_player
        slot = self.captured_by(captor) - 1
        self.flyers.append(Flyer(captor, (x, y), time.perf_counter(), slot))

    def _game_over(self) -> None:
        self.game_over_at = time.perf_counter()
        res = self.state.result
        if res.winner is None:
            self.sounds.play("draw")
        elif self.settings.mode == "vs Computer" and not self.is_human(res.winner):
            self.sounds.play("lose")
        else:
            self.sounds.play("win")

    def captured_by(self, player: int) -> int:
        return 12 - self.state.count(1 - player)

    # --- drawing -------------------------------------------------------------------

    def draw(self) -> None:
        now = time.perf_counter()
        scr = self.screen
        scr.blit(self.bg, (0, 0))
        ox, oy = self.geo.origin
        scr.blit(self.board_img, (ox - self.board_pad, oy - self.board_pad))
        self._draw_trail(now)
        self._draw_hint(now)
        self._draw_markers(now)
        self._draw_beads(now)
        self._draw_particles()
        self._draw_panel(now)
        self._draw_flyers(now)
        self._draw_game_over(now)
        self._draw_toast(now)
        pygame.display.flip()

    def _bead_at(self, x: float, y: float, player: int, lift: float = 0.0, scale: float = 1.0) -> None:
        r = self.r
        rr = max(4, int(r * scale))
        shadow = art.soft_disc(int(r * 1.2), (0, 0, 0), 1.6)
        shadow.set_alpha(int(175 - 60 * lift))
        off = r * (0.16 + 0.5 * lift)
        self.screen.blit(shadow, shadow.get_rect(center=(x + off * 0.7, y + off)))
        if player == 0:
            spill = art.glow(int(r * 0.85), (255, 110, 36), 0.3)
            self.screen.blit(spill, spill.get_rect(center=(x + r * 0.42, y + r * 0.52)), special_flags=pygame.BLEND_RGB_ADD)
        img = art.bead(r, art.MATERIALS[player])
        if rr != r:
            img = pygame.transform.smoothscale(img, (rr * 2, rr * 2))
        self.screen.blit(img, img.get_rect(center=(x, y - r * 0.25 * lift)))

    def _draw_beads(self, now: float) -> None:
        s, m = self.state, self.motion
        hidden = set()
        if m:
            hidden.add(m.dst)
        dragging = self.drag and self.drag_pos
        if dragging:
            hidden.add(self.drag[0])
        movable = self.movable() if self.human_can_act() else set()
        pulse = 0.5 + 0.5 * math.sin(now * 5)
        victims = set()
        if self.selected is not None and self.human_can_act():
            for dst, a in self.targets(self.selected).items():
                mv = s.describe(a)
                if mv.captured is not None:
                    victims.add(mv.captured)
        for sq in range(25):
            p = s.piece_at(sq)
            if p < 0 or sq in hidden:
                continue
            x, y = self.geo.point(sq)
            if sq in victims:
                g = art.glow(int(self.r * 1.7), art.DANGER, 0.35 + 0.35 * pulse)
                self.screen.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)
            if self.game_over_at and s.result and s.result.winner == p:
                r_, c_ = divmod(sq, 5)
                wave = 0.5 + 0.5 * math.sin(now * 4 - (r_ + c_) * 0.7)
                g = art.glow(int(self.r * 1.9), art.PLAYER_ACCENT[p], 0.2 + 0.5 * wave)
                self.screen.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)
            lift = 0.0
            if sq == self.selected:
                lift = 0.55
                g = art.glow(int(self.r * 1.8), art.BRASS_LIGHT, 0.45)
                self.screen.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)
            elif sq == self.hover and sq in movable:
                lift = 0.25
            self._bead_at(x, y, p, lift)
            if sq in victims:
                ring = art.ring(int(self.r * 1.12), 3 * self.ui, art.DANGER)
                ring.set_alpha(int(120 + 135 * pulse))
                self.screen.blit(ring, ring.get_rect(center=(x, y)))
            if sq in movable and s.must_capture() and sq != self.selected:
                ring = art.ring(int(self.r * 1.18), 2.2 * self.ui, art.BRASS_LIGHT)
                ring.set_alpha(int(90 + 140 * pulse))
                self.screen.blit(ring, ring.get_rect(center=(x, y)))

        if m:
            p = m.progress(now)
            if m.victim is not None and p < 0.55:
                vx, vy = self.geo.point(m.victim)
                fade = 1 - max(0.0, (p - 0.35) / 0.2)
                img = art.bead(self.r, art.MATERIALS[1 - m.player]).copy()
                img.set_alpha(int(255 * fade))
                self.screen.blit(img, img.get_rect(center=(vx, vy)))
            if m.victim is not None:
                t = ease_in_out(p)
                hop = math.sin(math.pi * p)
                x, y = lerp(m.start, m.end, t)
                self._bead_at(x, y, m.player, lift=hop, scale=1 + 0.18 * hop)
            else:
                x, y = lerp(m.start, m.end, ease_out(p))
                self._bead_at(x, y, m.player, lift=0.35 * (1 - p))
        if dragging:
            x, y = self.drag_pos
            self._bead_at(x, y, s.piece_at(self.drag[0]), lift=1.0, scale=1.08)

    def _draw_markers(self, now: float) -> None:
        if self.selected is None or not self.human_can_act():
            return
        pulse = 0.5 + 0.5 * math.sin(now * 6)
        s = self.state
        for dst, a in self.targets(self.selected).items():
            x, y = self.geo.point(dst)
            if s.is_capture(a):
                g = art.glow(int(self.r * 1.1), art.DANGER, 0.5)
                self.screen.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)
                ring = art.ring(int(self.r * 0.62), 3 * self.ui, (255, 150, 110))
                ring.set_alpha(int(150 + 100 * pulse))
                self.screen.blit(ring, ring.get_rect(center=(x, y)))
            else:
                g = art.glow(int(self.r * 0.95), art.BRASS_LIGHT, 0.35 + 0.25 * pulse)
                self.screen.blit(g, g.get_rect(center=(x, y)), special_flags=pygame.BLEND_RGB_ADD)
                pygame.draw.circle(self.screen, art.BRASS_LIGHT, (x, y), max(3, self.r * 0.2))

    def _draw_trail(self, now: float) -> None:
        """The last turn's path, as a board-sized overlay rebuilt only when it changes."""
        if not self.actions:
            return
        key = (len(self.actions), self.geo.size, self.geo.origin)
        if getattr(self, "_trail_key", None) != key:
            self._trail_key = key
            turn = self._turns()[-1]
            moves = [mv for mv in turn.moves if mv.action != STOP]
            ox, oy = self.geo.origin
            pts = [self.geo.point(moves[0].src)] + [self.geo.point(mv.dst) for mv in moves]
            pts = [(x - ox, y - oy) for x, y in pts]
            layer = pygame.Surface((self.geo.size, self.geo.size), pygame.SRCALPHA)
            color = art.PLAYER_ACCENT[turn.player]
            width = max(3, int(self.r * 0.16))
            for a, b in zip(pts, pts[1:]):
                art.thick_line(layer, (*color, 120), a, b, width)
            pygame.draw.circle(layer, (*color, 150), pts[0], self.r * 0.42, max(2, width // 2))
            self._trail_layer = layer
        self.screen.blit(self._trail_layer, self.geo.origin)

    def _draw_hint(self, now: float) -> None:
        if not self.hint:
            return
        src, dst, t0 = self.hint
        if now - t0 > 6 or not self.human_can_act():
            self.hint = None
            return
        a, b = self.geo.point(src), self.geo.point(dst)
        layer = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        alpha = int(140 + 90 * math.sin(now * 5))
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        tip = (b[0] - math.cos(ang) * self.r * 0.5, b[1] - math.sin(ang) * self.r * 0.5)
        g = art.glow(int(self.r * 1.3), art.BRASS_LIGHT, 0.4 + 0.25 * math.sin(now * 5))
        self.screen.blit(g, g.get_rect(center=b), special_flags=pygame.BLEND_RGB_ADD)
        art.thick_line(layer, (*art.BRASS_LIGHT, alpha), a, tip, max(5, self.r * 0.28))
        head = self.r * 0.55
        pygame.draw.polygon(
            layer,
            (*art.BRASS_LIGHT, alpha),
            [
                (b[0] - math.cos(ang) * self.r * 0.2, b[1] - math.sin(ang) * self.r * 0.2),
                (tip[0] - math.cos(ang - 0.5) * head, tip[1] - math.sin(ang - 0.5) * head),
                (tip[0] - math.cos(ang + 0.5) * head, tip[1] - math.sin(ang + 0.5) * head),
            ],
        )
        self.screen.blit(layer, (0, 0))

    def _draw_particles(self) -> None:
        for pt in self.particles:
            k = pt.life / pt.max_life
            g = art.glow(pt.size * 3, pt.color, round(k * 8) / 8)
            self.screen.blit(g, g.get_rect(center=(pt.x, pt.y)), special_flags=pygame.BLEND_RGB_ADD)

    def _tray_slot(self, player: int, slot: int) -> tuple[float, float]:
        card = self._card_rect(player)
        return (card.left + self.u(74) + slot * self.u(16.5), card.top + self.u(70))

    def _draw_flyers(self, now: float) -> None:
        for f in self.flyers:
            t = ease_in_out(min(1.0, (now - f.t0) / f.dur))
            end = self._tray_slot(f.captor, f.slot)
            x, y = lerp(f.start, end, t)
            y -= math.sin(math.pi * t) * self.u(60)
            rr = max(4, int(self.r + (self.u(7) - self.r) * t))
            img = pygame.transform.smoothscale(art.bead(self.r, art.MATERIALS[1 - f.captor]), (rr * 2, rr * 2))
            self.screen.blit(img, img.get_rect(center=(x, y)))

    # --- panel ---------------------------------------------------------------------

    def _card_rect(self, player: int) -> pygame.Rect:
        u, p = self.u, self.panel
        top = p.top + u(110) + (0 if player == 1 else u(98))
        return pygame.Rect(p.left + u(14), top, p.width - u(28), u(88))

    def _turns(self):
        n = len(self.actions)
        if self._turns_cache[0] != n:
            self._turns_cache = (n, group_turns(self.states, self.actions))
        return self._turns_cache[1]

    def _role(self, player: int) -> str:
        level = self.agents[player]
        if level is None:
            return "You" if self.settings.mode == "vs Computer" else f"Player {player + 1}"
        return f"{level} · computer"

    def _text(self, text: str, f: pygame.font.Font, color, pos, anchor: str = "topleft", alpha: int = 255) -> pygame.Rect:
        img = f.render(text, True, color)
        if alpha < 255:
            img.set_alpha(alpha)
        rect = img.get_rect(**{anchor: pos})
        self.screen.blit(img, rect)
        return rect

    def _draw_panel(self, now: float) -> None:
        u, p, scr = self.u, self.panel, self.screen
        card = pygame.Surface(p.size, pygame.SRCALPHA)
        pygame.draw.rect(card, (*art.PANEL, 225), card.get_rect(), border_radius=u(14))
        pygame.draw.rect(card, (*art.BRASS_DARK, 255), card.get_rect(), 1, border_radius=u(14))
        scr.blit(card, p.topleft)

        # Title, letter-spaced by hand.
        x = p.left + u(22)
        for ch in "BARAH GOTI":
            r = self._text(ch, self.f_title, art.BRASS_LIGHT, (x, p.top + u(18)))
            x = r.right + u(3)
        self._text("twelve beads · the Chiniot board", self.f_sub, art.IVORY_DIM, (p.left + u(24), p.top + u(66)))

        s = self.state
        thinking = self.ai_future is not None
        for player in (1, 0):
            rect = self._card_rect(player)
            active = not s.is_terminal and s.turn == player
            box = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(box, (255, 240, 210, 18 if active else 8), box.get_rect(), border_radius=u(10))
            edge = art.PLAYER_ACCENT[player] if active else art.BRASS_DARK
            pygame.draw.rect(box, (*edge, 255 if active else 120), box.get_rect(), 2 if active else 1, border_radius=u(10))
            scr.blit(box, rect.topleft)
            icon_r = u(19)
            cx, cy = rect.left + u(36), rect.top + u(34)
            if active:
                g = art.glow(int(icon_r * 2), art.PLAYER_ACCENT[player], 0.45 + 0.2 * math.sin(now * 3))
                scr.blit(g, g.get_rect(center=(cx, cy)), special_flags=pygame.BLEND_RGB_ADD)
            img = art.bead(icon_r, art.MATERIALS[player])
            scr.blit(img, img.get_rect(center=(cx, cy)))
            if active and thinking and not self.is_human(player):
                for i in range(8):
                    a = now * 4 + i * math.pi / 4
                    k = (i + 1) / 8
                    pygame.draw.circle(scr, art.PLAYER_ACCENT[player], (cx + math.cos(a) * icon_r * 1.45, cy + math.sin(a) * icon_r * 1.45), max(1, u(2.6) * k))
            self._text(art.PLAYER_NAMES[player], self.f_name, art.IVORY, (rect.left + u(68), rect.top + u(10)))
            self._text(self._role(player), self.f_small, art.IVORY_DIM, (rect.left + u(68), rect.top + u(38)))
            self._text(str(s.count(player)), self.f_count, art.IVORY, (rect.right - u(16), rect.top + u(10)), "topright")
            self._text("beads", self.f_small, art.IVORY_DIM, (rect.right - u(16), rect.top + u(44)), "topright")
            in_flight = sum(1 for f in self.flyers if f.captor == player)
            for slot in range(self.captured_by(player) - in_flight):
                tx, ty = self._tray_slot(player, slot)
                small = art.bead(u(7), art.MATERIALS[1 - player])
                scr.blit(small, small.get_rect(center=(tx, ty)))

        # Status line.
        status, color = self._status(now)
        self._text(status, self.f_ui, color, (p.left + u(22), p.top + u(312)))
        sub = self._substatus()
        if sub:
            self._text(sub, self.f_small, art.IVORY_DIM, (p.left + u(22), p.top + u(336)))

        # Options and buttons.
        mouse = pygame.mouse.get_pos()
        for w in self.widgets:
            hover = w.rect.collidepoint(mouse)
            if w.kind == "option":
                w.enabled = not (w.key == "side" and self.settings.mode != "vs Computer") and not (
                    w.key == "level" and self.settings.mode == "Two players"
                )
                alpha = 255 if w.enabled else 90
                if hover and w.enabled:
                    pygame.draw.rect(scr, (60, 40, 28), w.rect, border_radius=u(6))
                self._text(w.label, self.f_ui, art.IVORY_DIM, (w.rect.left + u(8), w.rect.centery), "midleft", alpha)
                value = getattr(self.settings, w.key)
                vx = w.rect.right - u(8)
                r = self._text("›", self.f_name, art.BRASS, (vx, w.rect.centery - u(2)), "midright", alpha)
                r = self._text(value, self.f_ui, art.IVORY, (r.left - u(10), w.rect.centery), "midright", alpha)
                self._text("‹", self.f_name, art.BRASS, (r.left - u(10), w.rect.centery - u(2)), "midright", alpha)
            else:
                w.enabled = {
                    "undo": self.settings.mode != "Watch computers" and len(self.states) > 1,
                    "hint": self.human_can_act() and self.hint_future is None,
                }.get(w.key, True)
                fill = (88, 60, 30) if hover and w.enabled else (54, 36, 24)
                pygame.draw.rect(scr, fill, w.rect, border_radius=u(8))
                pygame.draw.rect(scr, art.BRASS if w.enabled else art.BRASS_DARK, w.rect, 1, border_radius=u(8))
                self._text(w.label, self.f_ui, art.IVORY if w.enabled else art.IVORY_DIM, w.rect.center, "center", 255 if w.enabled else 110)
        info = LEVEL_INFO[self.settings.level] if self.settings.mode != "Two players" else "two humans, one board"
        last = next(w for w in self.widgets if w.key == "side")
        self._text(info, self.f_small, art.IVORY_DIM, (last.rect.left + u(8), last.rect.bottom + u(4)), alpha=200)

        self._draw_moves()
        self._text("N new · U undo · H hint · M sound · S/L save/load", self.f_small, art.IVORY_DIM, (p.centerx, p.bottom - u(22)), "center", 170)

    def _status(self, now: float) -> tuple[str, tuple[int, int, int]]:
        s = self.state
        if s.result and (self.motion is None):
            if s.result.winner is None:
                return "Draw", art.IVORY
            return f"{art.PLAYER_NAMES[s.result.winner]} wins", art.PLAYER_ACCENT[s.result.winner]
        name = art.PLAYER_NAMES[s.turn]
        if not self.is_human(s.turn):
            dots = "." * (1 + int(now * 3) % 3)
            return f"{name} is thinking{dots}", art.PLAYER_ACCENT[s.turn]
        if s.chain >= 0:
            return "Keep jumping!", art.BRASS_LIGHT
        who = "Your move" if self.settings.mode == "vs Computer" else f"{name} to move"
        return who, art.PLAYER_ACCENT[s.turn]

    def _substatus(self) -> str:
        s = self.state
        if s.result and self.motion is None:
            return REASONS.get(s.result.reason, s.result.reason)
        if s.chain >= 0:
            return "the same bead must capture again"
        if s.must_capture():
            return "a capture is available, so it is mandatory"
        if s.quiet >= 30:
            return f"{s.rules.quiet_limit - s.quiet} quiet turns left before a draw"
        return ""

    def _draw_moves(self) -> None:
        u, r, scr = self.u, self.moves_rect, self.screen
        pygame.draw.line(scr, art.BRASS_DARK, (r.left, r.top - u(10)), (r.right, r.top - u(10)))
        self._text("Moves", self.f_small, art.IVORY_DIM, (r.left + u(4), r.top))
        turns = self._turns()
        rows = [turns[i : i + 2] for i in range(0, len(turns), 2)]
        line_h = u(21)
        visible = max(1, (r.height - u(24)) // line_h)
        max_scroll = max(0, len(rows) - visible)
        self.scroll = min(self.scroll, max_scroll)
        start = max(0, len(rows) - visible - self.scroll)
        y = r.top + u(24)
        for i, row in enumerate(rows[start : start + visible], start=start):
            self._text(f"{i + 1}.", self.f_moves, art.IVORY_DIM, (r.left + u(4), y))
            for j, turn in enumerate(row):
                color = art.PLAYER_ACCENT[turn.player]
                text = turn.notation()
                self._text(text, self.f_moves, color, (r.left + u(40) + j * u(140), y))
            y += line_h

    def _draw_game_over(self, now: float) -> None:
        s = self.state
        if not (s.result and self.game_over_at and self.motion is None):
            return
        t = min(1.0, (now - self.game_over_at - 0.5) / 0.5)
        if t <= 0:
            return
        u = self.u
        ox, oy = self.geo.origin
        size = self.geo.size
        rect = pygame.Rect(0, 0, int(size * 0.68), u(150))
        rect.center = (ox + size // 2, oy + size // 2 + int((1 - ease_out(t)) * u(30)))
        box = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(box, (20, 12, 8, int(225 * t)), box.get_rect(), border_radius=u(16))
        accent = art.PLAYER_ACCENT[s.result.winner] if s.result.winner is not None else art.BRASS
        pygame.draw.rect(box, (*accent, int(255 * t)), box.get_rect(), 2, border_radius=u(16))
        self.screen.blit(box, rect.topleft)
        if s.result.winner is None:
            title = "A draw"
        elif self.settings.mode == "vs Computer":
            title = "You win!" if self.is_human(s.result.winner) else f"{art.PLAYER_NAMES[s.result.winner]} wins"
        else:
            title = f"{art.PLAYER_NAMES[s.result.winner]} wins"
        alpha = int(255 * t)
        self._text(title, self.f_big, accent, (rect.centerx, rect.top + u(52)), "center", alpha)
        self._text(REASONS.get(s.result.reason, ""), self.f_ui, art.IVORY, (rect.centerx, rect.top + u(98)), "center", alpha)
        self._text("N for a new game · U to take back", self.f_small, art.IVORY_DIM, (rect.centerx, rect.top + u(124)), "center", alpha)

    def _draw_toast(self, now: float) -> None:
        if not self.toast:
            return
        text, t0 = self.toast
        age = now - t0
        if age > 2.6:
            self.toast = None
            return
        alpha = int(255 * min(1, (2.6 - age) / 0.4))
        img = self.f_ui.render(text, True, art.IVORY)
        ox, oy = self.geo.origin
        rect = img.get_rect(midtop=(ox + self.geo.size // 2, oy + self.u(6)))
        bg = pygame.Surface((rect.width + self.u(28), rect.height + self.u(14)), pygame.SRCALPHA)
        pygame.draw.rect(bg, (16, 10, 7, int(alpha * 0.85)), bg.get_rect(), border_radius=self.u(10))
        pygame.draw.rect(bg, (*art.BRASS, alpha), bg.get_rect(), 1, border_radius=self.u(10))
        brect = bg.get_rect(center=rect.center)
        self.screen.blit(bg, brect)
        img.set_alpha(alpha)
        self.screen.blit(img, rect)

    # --- main loop -----------------------------------------------------------------

    def run(self) -> None:
        running = True
        try:
            while running:
                dt = self.clock.tick(60) / 1000
                for e in pygame.event.get():
                    if not self.handle(e):
                        running = False
                self.update(dt)
                self.draw()
        finally:
            self._cancel_search()
            self.pool.shutdown(wait=False, cancel_futures=True)
            pygame.quit()


def _dpi_aware() -> None:
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
