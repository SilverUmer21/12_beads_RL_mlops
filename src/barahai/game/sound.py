"""Synthesised sound effects. No audio files: everything is built with numpy at startup.

* step: a bead set down on lacquered wood (a few short inharmonic partials)
* capture: stone-on-stone clack plus a faint glassy ring
* win/draw: plucked-string phrases (Karplus-Strong) in Raag Yaman
"""

from __future__ import annotations

import numpy as np
import pygame

SR = 44100


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(SR * seconds)) / SR


def _partials(freqs, decays, amps, seconds: float) -> np.ndarray:
    t = _t(seconds)
    out = np.zeros_like(t)
    for f, d, a in zip(freqs, decays, amps):
        out += a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
    return out


def _click(seconds: float, seed: int, decay: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = _t(seconds)
    return rng.standard_normal(t.size) * np.exp(-t / decay)


def _pluck(freq: float, seconds: float, seed: int, brightness: float = 0.5) -> np.ndarray:
    """Karplus-Strong, vectorised one delay period at a time."""
    n = int(SR / freq)
    total = int(SR * seconds)
    rng = np.random.default_rng(seed)
    y = np.zeros(total + n + 1)
    burst = rng.uniform(-1, 1, n)
    for _ in range(int((1 - brightness) * 4)):  # soften the attack
        burst = 0.5 * (burst + np.roll(burst, 1))
    y[:n] = burst
    for start in range(n, total, n):
        prev = y[start - n : start]
        before = np.concatenate(([y[start - n - 1]] if start - n - 1 >= 0 else [0.0], prev[:-1]))
        y[start : start + n] = 0.4985 * (prev + before)
    return y[:total]


def _phrase(notes, step: float, hold: float, seed: int) -> np.ndarray:
    total = int(SR * (step * len(notes) + hold))
    out = np.zeros(total)
    for i, f in enumerate(notes):
        start = int(SR * step * i)
        p = _pluck(f, min(hold + step, (total - start) / SR), seed + i)
        out[start : start + p.size] += p
    return out


def _sound(wave: np.ndarray, volume: float) -> pygame.mixer.Sound:
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * volume
    fade = min(200, wave.size)
    wave[-fade:] *= np.linspace(1, 0, fade)
    pcm = (wave * 32767).astype(np.int16)
    channels = pygame.mixer.get_init()[2]
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))


# Raag Yaman on Sa = C4: Ni Re Ga Ma# Pa Dha Ni Sa'
SA, RE, GA, MA_T, PA, DHA, NI = 261.63, 293.66, 329.63, 369.99, 392.0, 440.0, 493.88
LOW_NI = NI / 2


class Sounds:
    def __init__(self) -> None:
        self.enabled = True
        self._bank: dict[str, pygame.mixer.Sound] = {}
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 2, 512)
        except pygame.error:
            self.enabled = False
            return
        wood = _partials((1650, 2480, 3710, 240), (0.018, 0.011, 0.007, 0.05), (1, 0.6, 0.35, 0.5), 0.12)
        wood += 0.25 * _click(0.12, 1)
        self._bank["step"] = _sound(wood, 0.55)
        lift = _partials((2900, 4100), (0.008, 0.005), (1, 0.5), 0.05) + 0.2 * _click(0.05, 2, 0.001)
        self._bank["select"] = _sound(lift, 0.22)
        clack = _partials((3150, 4620, 6080, 900), (0.012, 0.008, 0.005, 0.03), (1, 0.7, 0.4, 0.4), 0.45)
        clack += 0.35 * _click(0.45, 3) + _partials((2093, 3136), (0.22, 0.12), (0.22, 0.1), 0.45)
        self._bank["capture"] = _sound(clack, 0.6)
        thud = _partials((140, 210), (0.04, 0.025), (1, 0.4), 0.12) + 0.15 * _click(0.12, 4, 0.004)
        self._bank["invalid"] = _sound(thud, 0.4)
        self._bank["win"] = _sound(_phrase((LOW_NI, RE, GA, MA_T, DHA, NI, SA * 2), 0.11, 1.1, 10), 0.5)
        self._bank["lose"] = _sound(_phrase((SA * 2, NI, DHA, PA, MA_T, GA, RE), 0.13, 1.0, 20), 0.42)
        self._bank["draw"] = _sound(_phrase((SA, PA, SA * 2), 0.18, 1.2, 30), 0.42)

    def play(self, name: str) -> None:
        if self.enabled and name in self._bank:
            self._bank[name].play()

    def toggle(self) -> bool:
        if self._bank:
            self.enabled = not self.enabled
        return self.enabled
