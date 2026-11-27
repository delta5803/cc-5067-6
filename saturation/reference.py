"""Independent reference derivations for acceptance expectations.

Nothing here imports the production multirate chain. The numbers the
acceptance suite compares against are derived from textbook formulas and a
self-contained, deliberately different implementation (explicit windowed-sinc
filter instead of the production Kaiser builder; static-curve harmonic series
instead of the stateful saturator), so an implementation bug cannot silently
fix both sides of an assertion.
"""
from __future__ import annotations

import math

import numpy as np

from .transfer import apply_transfer, Transfer


# --------------------------------------------------------------------- #
# Folding geometry (triangular fold around Nyquist, coded from scratch)
# --------------------------------------------------------------------- #
def folded_frequency(f_harm: float, fs: float) -> float:
    """Where a component at f_harm lands after sampling at fs.

    Repeated reflection inside [0, fs/2]: walk the frequency down by whole
    Nyquist spans and mirror on every other span.
    """
    nyq = fs / 2.0
    if f_harm <= nyq:
        return float(f_harm)
    span = f_harm / nyq
    k = int(math.floor(span))
    if k % 2 == 0:
        return float(f_harm - k * nyq)
    return float((k + 1) * nyq - f_harm)


def expected_alias_bins(f0: float, fs: float, n_harm: int = 12
                        ) -> list[float]:
    """Folded locations of harmonics that start above Nyquist."""
    nyq = fs / 2.0
    out = []
    for h in range(2, n_harm + 1):
        fh = f0 * h
        if fh >= nyq:
            fa = folded_frequency(fh, fs)
            if fa > 0.0 and all(abs(fa - g) > 1e-9 for g in out):
                out.append(fa)
    return out


# --------------------------------------------------------------------- #
# Static-curve harmonic series (zero memory: the honest expected value)
# --------------------------------------------------------------------- #
def static_harmonic_series(f0: float, drive: float, kind: str | Transfer,
                           amp: float = 0.9, n_harm: int = 12,
                           periods: int = 257) -> np.ndarray:
    """Magnitude of harmonics 1..n_harm produced by the bare transfer curve.

    The static nonlinearity on a pure sine is the ground truth the stateful
    chain can only approximate; no filters, no one-poles are involved.
    """
    n = periods * 4096
    t = np.arange(n) / n * periods * 2.0 * np.pi
    y = apply_transfer(amp * np.sin(t), drive, kind)
    spec = np.abs(np.fft.rfft(y)) * 2.0 / n
    bins = [int(round(h * periods)) for h in range(1, n_harm + 1)]
    return spec[bins]


# --------------------------------------------------------------------- #
# Self-contained factor-2 offline chain (independent of production code)
# --------------------------------------------------------------------- #
def _windowed_sinc(cutoff_hz: float, sample_rate: float,
                   half_taps: int = 64) -> np.ndarray:
    """Hann-windowed sinc with explicit gain normalisation."""
    n = np.arange(-half_taps, half_taps + 1)
    wc = 2.0 * math.pi * cutoff_hz / sample_rate
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(n == 0, wc / math.pi,
                     np.sin(wc * n) / (math.pi * n))
    h *= 0.5 * (1.0 + np.cos(math.pi * n / (half_taps + 1)))
    return h / np.sum(h)


def offline_factor2(x: np.ndarray, drive: float, kind: str | Transfer,
                    base_rate: float, cutoff_hz: float) -> np.ndarray:
    """Minimal independent x2 oversampler: zero-stuff -> windowed sinc ->
    static curve -> windowed sinc -> decimate. No shared code with the
    production MultirateEngine; delay lines are handled by full convolution.
    """
    up = np.zeros(len(x) * 2)
    up[::2] = x
    h_up = _windowed_sinc(cutoff_hz, base_rate * 2.0) * 2.0
    h_down = _windowed_sinc(cutoff_hz, base_rate * 2.0)
    filt = np.convolve(up, h_up, mode="same")
    sat = apply_transfer(filt, drive, kind)
    lp = np.convolve(sat, h_down, mode="same")
    return lp[::2]


def alias_lower_bound_x2(f0: float, drive: float, kind: str | Transfer,
                         base_rate: float, cutoff_hz: float,
                         amp: float = 0.9, n_harm: int = 12,
                         dur: float = 1.0) -> dict:
    """Alias level of an independent x2 chain.

    For a tone whose harmonics fold at base rate, the unfiltered x1 path must
    show *more* alias than this filtered x2 reference, while a production x4/x8
    chain must show *less*. The value is therefore a discriminating bound that
    a healthy chain straddles, derived entirely outside the production code.
    """
    n = int(base_rate * dur)
    t = np.arange(n) / base_rate
    x = amp * np.sin(2.0 * np.pi * f0 * t)
    y = offline_factor2(x, drive, kind, base_rate, cutoff_hz)
    freqs = np.fft.rfftfreq(n, 1.0 / base_rate)
    mag = 2.0 * np.abs(np.fft.rfft(y * np.hanning(n))) / np.sum(np.hanning(n))
    bins = expected_alias_bins(f0, base_rate, n_harm)
    parts = [_peak(freqs, mag, f) ** 2 for f in bins]
    return {"bound_energy": float(np.sqrt(np.sum(parts))),
            "bins_hz": bins,
            "method": "offline_factor2_windowed_sinc"}


# --------------------------------------------------------------------- #
# Band-difference estimator (energy added above base Nyquist by the curve)
# --------------------------------------------------------------------- #
def band_difference_alias(f0: float, drive: float, kind: str | Transfer,
                          base_rate: float, amp: float = 0.9,
                          n_harm: int = 12, dur: float = 1.0) -> dict:
    """Run the static curve at 4x rate and compare linear vs nonlinear output
    in the mirror/fold bands above the base Nyquist. Energy that only exists
    after the nonlinearity and sits where decimation would fold it is the
    alias potential, measured without any production filter.
    """
    hi_rate = base_rate * 4
    n = int(hi_rate * dur)
    t = np.arange(n) / hi_rate
    x = amp * np.sin(2.0 * np.pi * f0 * t)
    lin = x  # unity-gain linear reference
    nl = apply_transfer(x, drive, kind)
    mag_lin = 2.0 * np.abs(np.fft.rfft(lin * np.hanning(n))) / np.sum(
        np.hanning(n))
    mag_nl = 2.0 * np.abs(np.fft.rfft(nl * np.hanning(n))) / np.sum(
        np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / hi_rate)
    bins_hi = [f0 * h for h in range(2, n_harm + 1) if f0 * h >= base_rate / 2]
    added = 0.0
    for fh in bins_hi:
        added += max(_peak(freqs, mag_nl, fh) - _peak(freqs, mag_lin, fh),
                     0.0) ** 2
    # scale onto a base-rate magnitude axis (same tone, longer FFT -> same
    # component magnitude; the sum is already in comparable amplitude units)
    return {"bound_energy": float(np.sqrt(added)),
            "high_rate_bins_hz": bins_hi,
            "method": "band_difference_static_curve_4x"}


def _peak(freqs: np.ndarray, mag: np.ndarray, f: float,
          tol_hz: float = 12.0) -> float:
    sel = np.abs(freqs - f) <= tol_hz
    return float(np.max(mag[sel])) if np.any(sel) else 0.0


# --------------------------------------------------------------------- #
# Independent latency derivation (Kaiser textbook formula, coded directly)
# --------------------------------------------------------------------- #
def expected_latency_base(factor: int, base_rate: float, cutoff_hz: float,
                          ripple_db: float = 60.0,
                          width_hz: float | None = None) -> int:
    """Expected base-rate delay of up+down FIRs, from the Kaiser estimate.

    N ~= (A - 7.95) / (2.285 * dw), rounded to a type-1 length whose group
    delay is a multiple of the factor, then (up+down) converted to base-rate
    samples. This is the closed form behind filter-design libraries, coded
    here independently so latency is asserted against a derivation, not the
    engine's own bookkeeping.
    """
    if factor == 1:
        return 0
    hi = base_rate * factor
    if width_hz is None:
        width_hz = max(cutoff_hz * 0.1, min(500.0, (hi / 2 - cutoff_hz) * 0.4))
    dw = 2.0 * math.pi * width_hz / hi
    n_taps = int(math.ceil((ripple_db - 7.95) / (2.285 * dw)))
    if n_taps % 2 == 0:
        n_taps += 1
    rem = (n_taps - 1) % (2 * factor)
    if rem:
        n_taps += 2 * factor - rem
    delay_hi = (n_taps - 1) // 2
    return 2 * (delay_hi // factor)


# --------------------------------------------------------------------- #
# Polarity decomposition: even-order vs odd-order content
# --------------------------------------------------------------------- #
def polarity_split(y: np.ndarray, y_inv: np.ndarray) -> tuple[np.ndarray,
                                                              np.ndarray]:
    """Separate even- and odd-symmetric response components.

    Given outputs for x and -x: odd part = (y - y_inv)/2 (fundamental and odd
    harmonics of an odd-symmetric curve), even part = (y + y_inv)/2 (DC and
    even harmonics; ideally only leakage for tanh/hard_clip/sine).
    """
    odd = 0.5 * (y - y_inv)
    even = 0.5 * (y + y_inv)
    return even, odd
