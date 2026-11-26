"""Spectral and time-domain measurements for saturation evaluation."""
from __future__ import annotations

import numpy as np


def spectrum(x: np.ndarray, fs: float, window: bool = True
             ) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if window:
        w = np.hanning(n)
        xf = np.fft.rfft(x * w)
        scale = np.sum(w)
    else:
        xf = np.fft.rfft(x)
        scale = n
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(xf) / scale
    if window:
        mag *= 2.0
    return freqs, mag


def peak_at(freqs: np.ndarray, mag: np.ndarray, f: float,
            tol_hz: float = 8.0) -> float:
    sel = np.abs(freqs - f) <= tol_hz
    return float(np.max(mag[sel])) if np.any(sel) else 0.0


def harmonic_peaks(freqs: np.ndarray, mag: np.ndarray, f0: float,
                   n_harm: int = 10) -> np.ndarray:
    return np.array([peak_at(freqs, mag, f0 * h) for h in range(1, n_harm + 1)])


def thd_db(peaks: np.ndarray) -> float:
    fund = peaks[0]
    if fund <= 1e-12:
        return -np.inf
    return 20.0 * np.log10(np.sqrt(np.sum(peaks[1:] ** 2)) / fund)


def dc(x: np.ndarray) -> float:
    return float(np.mean(x))


def band_energy(freqs: np.ndarray, mag: np.ndarray, f_lo: float,
                f_hi: float) -> float:
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(mag[sel] ** 2))


def passband_retention(in_freqs: np.ndarray, in_mag: np.ndarray,
                       out_freqs: np.ndarray, out_mag: np.ndarray,
                       f_hi: float) -> float:
    """Retained small-signal band *shape*: correlation of the input/output
    band magnitudes (1 = identical spectral shape). Saturation adds level and
    harmonics, so a raw energy ratio would measure gain rather than bandwidth.
    """
    grid = np.linspace(20.0, f_hi, 400)
    a = np.interp(grid, in_freqs, in_mag)
    b = np.interp(grid, out_freqs, out_mag)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))


def cross_correlation_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    """Sample lag of y relative to x at peak normalized cross-correlation."""
    best_lag, best_val = 0, -np.inf
    x0 = x - x.mean()
    y0 = y - y.mean()
    nx = np.linalg.norm(x0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = x0[-lag:], y0[:lag]
        elif lag > 0:
            a, b = x0[:-lag], y0[lag:]
        else:
            a, b = x0, y0
        if len(a) < 16:
            continue
        val = float(np.dot(a, b) / (nx * np.linalg.norm(b) + 1e-20))
        if val > best_val:
            best_val, best_lag = val, lag
    return best_lag


def alias_bins(f0: float, fs: float, n_harm: int = 12) -> list[float]:
    """Frequencies where harmonics of f0 land after folding around fs/2.

    Returns only the genuinely folded (alias) locations below Nyquist.
    """
    nyq = fs / 2.0
    out = []
    for h in range(2, n_harm + 1):
        f = f0 * h
        if f >= nyq:
            # triangular folding into [0, nyq]
            folded = abs(((f + nyq) % (2 * nyq)) - nyq)
            out.append(float(folded))
    return out


def alias_energy(freqs: np.ndarray, mag: np.ndarray, f0: float, fs: float,
                 n_harm: int = 12, tol_hz: float = 10.0) -> float:
    total = 0.0
    for fa in alias_bins(f0, fs, n_harm):
        total += peak_at(freqs, mag, fa, tol_hz) ** 2
    return float(np.sqrt(total))


def transient_position(x: np.ndarray, threshold_db: float = -45.0) -> int:
    """First sample whose moving RMS crosses threshold (for timing checks)."""
    win = 64
    if len(x) < win:
        return 0
    kernel = np.ones(win) / win
    env = np.sqrt(np.convolve(x * x, kernel, mode="same") + 1e-20)
    thr = 10.0 ** (threshold_db / 20.0) * np.max(env)
    idx = np.flatnonzero(env > thr)
    return int(idx[0]) if len(idx) else len(x)
