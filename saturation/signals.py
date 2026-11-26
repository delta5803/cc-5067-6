"""Deterministic test signals (no external files required)."""
from __future__ import annotations

import numpy as np


def sine(fs: float, freq: float, dur: float, amp: float = 0.8,
         phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def log_sweep(fs: float, f0: float = 20.0, f1: float = 20000.0,
              dur: float = 2.0, amp: float = 0.7) -> np.ndarray:
    n = int(fs * dur)
    t = np.arange(n) / fs
    k = np.log(f1 / f0) / dur
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1.0) / k
    return amp * np.sin(phase)


def multitone(fs: float, dur: float = 1.0, amp: float = 0.5,
              freqs: tuple[float, ...] = (200.0, 800.0, 2000.0, 5000.0, 9000.0)
              ) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * dur)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for f in freqs:
        sig += amp / len(freqs) * np.sin(2 * np.pi * f * t)
    return sig, np.asarray(freqs)


def asymmetric_wave(fs: float, freq: float = 300.0, dur: float = 1.0,
                    amp: float = 0.8) -> np.ndarray:
    """Duty-skewed wave: naturally asymmetric, exercises DC production."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    phase = (freq * t) % 1.0
    # 30% duty positive lobe, 70% negative lobe -> non-zero mean input too
    return amp * np.where(phase < 0.3, 1.0, -0.45)


def pluck(fs: float, f0: float = 220.0, dur: float = 3.0,
          amp: float = 0.9, decay: float = 3.5) -> np.ndarray:
    """Karplus-Strong style plucked string with harmonics and noise burst."""
    n = int(fs * dur)
    period = max(2, int(fs / f0))
    rng = np.random.default_rng(7)
    buf = rng.uniform(-1.0, 1.0, period)
    out = np.zeros(n)
    damping = 0.996
    idx = 0
    for i in range(n):
        out[i] = buf[idx]
        nxt = buf[(idx + 1) % period]
        buf[idx] = damping * 0.5 * (buf[idx] + nxt)
        idx = (idx + 1) % period
    env = np.exp(-decay * np.arange(n) / fs)
    # add a few inharmonic partials for spectral interest
    t = np.arange(n) / fs
    partial = sum(0.25 / h * np.sin(2 * np.pi * f0 * h * t)
                  for h in (2, 3, 5))
    return amp * env * (out + partial)


def silence_then(fs: float, sig: np.ndarray, silence_s: float = 1.0
                 ) -> np.ndarray:
    return np.concatenate([np.zeros(int(fs * silence_s)), sig])


def overrange(fs: float, freq: float = 200.0, dur: float = 0.5,
              amp: float = 20.0) -> np.ndarray:
    """Far beyond full-scale: stability / boundedness torture test."""
    return sine(fs, freq, dur, amp=amp)
