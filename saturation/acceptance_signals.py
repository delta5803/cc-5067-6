"""Deterministic inputs for the independent acceptance campaign.

These generators are deliberately separate from ``saturation.signals`` so the
acceptance suite does not share stimulus code with the measurements it
audits. Every signal is fully described by its parameters (no files, no
hand-edited data): two log sweeps covering different frequency spans, a
multitone, a sharp transient and polarity-inverted companions.
"""
from __future__ import annotations

import numpy as np


def _n(fs: float, dur: float) -> int:
    return int(round(fs * dur))


def log_sweep(fs: float, f0: float, f1: float, dur: float,
              amp: float = 0.7, phase0: float = 0.0) -> np.ndarray:
    """Exponential chirp sin(2*pi*f0*(e^{k t}-1)/k)."""
    n = _n(fs, dur)
    t = np.arange(n) / fs
    k = np.log(f1 / f0) / dur
    phase = 2.0 * np.pi * f0 * (np.exp(k * t) - 1.0) / k + phase0
    return amp * np.sin(phase)


def sweep_low(fs: float, dur: float = 1.0, amp: float = 0.7) -> np.ndarray:
    """Sweep A: stays inside the band where harmonics are mostly in-band."""
    return log_sweep(fs, 60.0, 3500.0, dur, amp)


def sweep_high(fs: float, dur: float = 1.0, amp: float = 0.7) -> np.ndarray:
    """Sweep B: crosses the frequencies whose harmonics fold back audibly."""
    return log_sweep(fs, 1200.0, 16000.0, dur, amp)


def multitone(fs: float, dur: float = 1.0, amp: float = 0.5,
              freqs: tuple[float, ...] = (330.0, 2700.0, 6900.0,
                                          11300.0, 15700.0)
              ) -> tuple[np.ndarray, np.ndarray]:
    n = _n(fs, dur)
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for f in freqs:
        sig += (amp / len(freqs)) * np.sin(2.0 * np.pi * f * t)
    return sig, np.asarray(freqs, dtype=np.float64)


def transient(fs: float, dur: float = 1.0, amp: float = 0.9,
              lead_silence: float = 0.1, decay: float = 18.0) -> np.ndarray:
    """One broadband attack followed by an exponential decay (timing probe)."""
    n = _n(fs, dur)
    n0 = _n(fs, lead_silence)
    t = np.arange(n - n0) / fs
    rng = np.random.default_rng(20240517)
    burst = rng.standard_normal(n - n0) * np.exp(-decay * t)
    # a deterministic click anchors the exact attack sample (above the noise)
    burst[0] = 4.0
    out = np.zeros(n)
    out[n0:] = amp * burst
    return out


def inverted(x: np.ndarray) -> np.ndarray:
    """Reverse-polarity companion: y = -x, sample for sample."""
    return -np.asarray(x, dtype=np.float64)


def beat_grid(fs: float, bpm: float, dur: float) -> np.ndarray:
    """Sample indices of beat positions (used to schedule same-beat events)."""
    step = int(round(fs * 60.0 / bpm))
    return np.arange(0, _n(fs, dur), step, dtype=np.int64)
