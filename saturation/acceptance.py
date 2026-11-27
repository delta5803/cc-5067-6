"""Acceptance verdicts for alias control, path alignment and latency.

Every threshold is either a physical invariant (folding geometry, reported
versus independently-derived latency) or a margin chosen on measured healthy
and deliberately-broken recordings. The module returns structured evidence so
the test suite asserts on recorded numbers rather than a single boolean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import spectrum, peak_at, band_energy, passband_retention
from .reference import (
    expected_alias_bins, expected_latency_base, alias_lower_bound_x2,
    band_difference_alias,
)
from . import acceptance_signals


@dataclass
class AliasEvidence:
    factor: int
    measured_alias_energy: float
    bound_energy: float
    bound_method: str
    fold_bins_hz: list
    fold_levels: dict
    image_energy: float
    latency_samples: int
    expected_latency_samples: int
    polarity_even_energy: float
    polarity_odd_energy: float
    checks: dict = field(default_factory=dict)


def comb_notch_depth(x_dry: np.ndarray, y_sum: np.ndarray, fs: float,
                     f_lo: float = 200.0, f_hi: float = 10000.0) -> dict:
    """Comb-filter signature relative to a healthy parallel-sum baseline.

    A correctly latency-compensated equal-gain dry/wet sum is coherent; a
    dry-branch offset turns the sum into |1 + e^{jw k}| with periodic deep
    notches. Because the saturation itself colours the spectrum, the healthy
    shape is divided out: render an aligned reference bus on the *same*
    stimulus and compare magnitude bin by bin. A healthy bus matches its own
    baseline (ratio ~1, no deep bins); a misaligned bus develops bins well
    below unity at the comb notch frequencies.
    """
    f, mag = spectrum(y_sum, fs)
    _, mag_ref = spectrum(x_dry, fs)  # caller passes the healthy-baseline sum
    n = min(len(f), len(mag), len(mag_ref))
    f, mag, mag_ref = f[:n], mag[:n], mag_ref[:n]
    band = (f >= f_lo) & (f <= f_hi)
    ratio = (mag[band] + 1e-12) / (mag_ref[band] + 1e-12)
    deepest = float(np.min(ratio))
    p5 = float(np.percentile(ratio, 5))
    frac_deep = float(np.mean(ratio < 0.5))
    # time-domain differential: aligned branches differ only by saturation,
    # misaligned branches carry an additional delay error visible as extra RMS
    return {"deepest_ratio": deepest,
            "p5_ratio": p5,
            "fraction_below_minus_6db": frac_deep,
            "notch_db": float(-20.0 * np.log10(deepest + 1e-12))}


def comb_evidence(bus, stimulus: np.ndarray, fs: float,
                  healthy_sum: np.ndarray) -> dict:
    """Render a (possibly faulty) bus and score its comb response against a
    pre-rendered healthy-bus sum of the same stimulus."""
    y = bus.process_offline(stimulus, 512)[: len(healthy_sum)]
    return comb_notch_depth(healthy_sum, y, fs)


def measure_passband(engine, fs: float, cutoff_hz: float,
                     drive: float = 1.0, reference_engine=None) -> dict:
    """Passband response from a low-level sweep (nonlinearity near-transparent).

    Two independent sweeps cover different spans; shape correlation and
    high-band energy loss are recorded so a brick-wall "fake anti-alias"
    filter cannot pass acceptance. The multirate chain is judged against the
    factor-1 engine on the *same* sweep, which divides out the saturator's
    own one-pole colouration; the energy-loss band stays below that pole's
    corner so only the resampling filters are measured.
    """
    rows = {}
    for name, sweep in (("low", acceptance_signals.sweep_low),
                        ("high", acceptance_signals.sweep_high)):
        x = sweep(fs, dur=1.0, amp=0.05)

        def render(eng):
            y = np.concatenate([eng.process(x), eng.flush()])
            d = eng.delay_base
            return y[d:d + len(x)] if len(y) >= d + len(x) else np.pad(
                y[d:], (0, len(x) - len(y[d:])))

        fo, mo = spectrum(render(engine), fs)
        if reference_engine is None:
            fi, mi = spectrum(x, fs)
        else:
            fi, mi = spectrum(render(reference_engine), fs)
        f_hi = min(cutoff_hz * 0.55, 10000.0)
        shape = passband_retention(fi, mi, fo, mo, f_hi)
        e_ref = band_energy(fi, mi, 4000.0, f_hi)
        e_wet = band_energy(fo, mo, 4000.0, f_hi)
        rows[name] = {
            "shape_corr": float(shape),
            "highband_loss_db": float(10.0 * np.log10(
                (e_wet + 1e-20) / (e_ref + 1e-20))),
        }
    return rows


def _render_aligned(engine, x: np.ndarray) -> np.ndarray:
    y = np.concatenate([engine.process(x), engine.flush()])
    d = engine.delay_base
    return y[d:d + len(x)] if len(y) >= d + len(x) else np.pad(
        y[d:], (0, len(x) - len(y[d:])))


def polarity_evidence(engine_pos, engine_neg, x: np.ndarray, fs: float
                      ) -> dict:
    """Even/odd decomposition from a matched engine pair on x and -x.

    Both engines are configured identically and rendered fresh, so the
    positive and negative outputs share the same delay alignment.
    """
    from .reference import polarity_split
    y_pos = _render_aligned(engine_pos, x)
    y_neg = _render_aligned(engine_neg, -x)
    even, odd = polarity_split(y_pos, y_neg)
    _, me = spectrum(even, fs)
    _, mo = spectrum(odd, fs)
    return {"even_energy": float(np.sqrt(np.sum(me ** 2))),
            "odd_energy": float(np.sqrt(np.sum(mo ** 2)))}


def measure_alias(engine, x: np.ndarray, fs: float, f0: float,
                  drive: float, kind: str, factor: int,
                  cutoff_hz: float, y_polarity: np.ndarray | None = None
                  ) -> AliasEvidence:
    """Render one factor and record fold energy, images and latency on a tone."""
    y = _render_aligned(engine, x)
    d = engine.delay_base
    freqs, mag = spectrum(y, fs)
    bins = expected_alias_bins(f0, fs, n_harm=12)
    levels = {round(f, 1): peak_at(freqs, mag, f, 10.0) for f in bins}
    alias_e = float(np.sqrt(sum(v * v for v in levels.values())))
    # image energy: spectral content above the design cutoff (mirror region)
    image_e = float(np.sqrt(band_energy(freqs, mag, cutoff_hz + 400.0,
                                        fs / 2.0 - 100.0)))
    exp_delay = expected_latency_base(factor, fs, cutoff_hz)
    even_e = odd_e = 0.0
    if y_polarity is not None:
        from .reference import polarity_split
        n2 = min(len(y), len(y_polarity))
        even, odd = polarity_split(y[:n2], y_polarity[:n2])
        _, me = spectrum(even, fs)
        _, mo = spectrum(odd, fs)
        even_e = float(np.sqrt(np.sum(me ** 2)))
        odd_e = float(np.sqrt(np.sum(mo ** 2)))
    return AliasEvidence(
        factor=factor, measured_alias_energy=alias_e,
        bound_energy=float("nan"), bound_method="",
        fold_bins_hz=bins, fold_levels=levels,
        image_energy=image_e, latency_samples=int(d),
        expected_latency_samples=int(exp_delay),
        polarity_even_energy=even_e, polarity_odd_energy=odd_e)


def select_alias_estimator(fs: float, f0: float, drive: float, kind: str,
                           cutoff_hz: float) -> dict:
    """Choose the discriminating alias bound from measurement, not assumption.

    Both independent estimators are evaluated against the production chain's
    actual x1 and x8 recordings. The usable bound is the one that sits
    *between* them with the wider margin (so x1 fails the bound and x8 passes
    it); the decision and margins are returned as evidence.
    """
    from .resample import MultirateEngine
    n = int(fs)
    t = np.arange(n) / fs
    x = 0.9 * np.sin(2.0 * np.pi * f0 * t)
    measured = {}
    for factor in (1, 8):
        eng = MultirateEngine(factor, fs, drive=drive, kind=kind,
                              cutoff_hz=cutoff_hz)
        ev = measure_alias(eng, x, fs, f0, drive, kind, factor, cutoff_hz)
        measured[factor] = ev.measured_alias_energy
    est_a = alias_lower_bound_x2(f0, drive, kind, fs, cutoff_hz)
    est_b = band_difference_alias(f0, drive, kind, fs)
    candidates = {
        "offline_factor2": (est_a["bound_energy"], est_a["method"]),
        "band_difference": (est_b["bound_energy"], est_b["method"]),
    }
    verdict = {}
    for name, (bound, method) in candidates.items():
        # discriminating power: how far the bound sits from both recordings
        margin_low = 20 * np.log10(measured[1] / max(bound, 1e-15))
        margin_high = 20 * np.log10(max(bound, 1e-15)
                                    / max(measured[8], 1e-15))
        verdict[name] = {"bound_energy": bound, "method": method,
                         "db_above_x1": margin_low,
                         "db_below_x8": margin_high,
                         "discriminates": bool(
                             measured[1] > bound > measured[8])}
    usable = [n for n, v in verdict.items() if v["discriminates"]]
    chosen = max(usable, key=lambda n: min(verdict[n]["db_above_x1"],
                                           verdict[n]["db_below_x8"]),
                 default=None)
    return {"chosen": chosen, "candidates": verdict,
            "measured_x1": measured[1], "measured_x8": measured[8],
            "f0_hz": f0}


def latency_verdict(ev: AliasEvidence, tol: int = 0) -> dict:
    """Reported latency must match the independent textbook derivation."""
    ok = abs(ev.latency_samples - ev.expected_latency_samples) <= tol
    return {"ok": ok, "reported": ev.latency_samples,
            "expected": ev.expected_latency_samples,
            "delta": ev.latency_samples - ev.expected_latency_samples}


def impulse_timing(engine, fs: float, impulse_at: int = 64,
                   amp: float = 1e-4, half_window: int = 6) -> dict:
    """Locate the response peak and its sub-sample centroid.

    A decimation-grid fault keeps the integer delay but drags the centroid by
    a fraction of a base-rate sample, which the integer peak index alone would
    miss. Both numbers are returned as independent timing evidence.
    """
    x = np.zeros(impulse_at + 256, dtype=np.float64)
    x[impulse_at] = amp
    y = np.concatenate([engine.process(x), engine.flush()])
    peak = int(np.argmax(np.abs(y)))
    lo, hi = max(0, peak - half_window), min(len(y), peak + half_window + 1)
    idx = np.arange(lo, hi, dtype=np.float64)
    w = np.abs(y[lo:hi])
    centroid = float(np.dot(idx, w) / (np.sum(w) + 1e-20))
    expected_peak = impulse_at + engine.delay_base
    return {"peak_index": peak,
            "expected_peak_index": expected_peak,
            "peak_delta": peak - expected_peak,
            "centroid": centroid,
            "expected_centroid": float(expected_peak),
            "centroid_delta": centroid - float(expected_peak)}
