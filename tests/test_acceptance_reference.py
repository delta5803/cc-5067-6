"""The independent reference itself must be correct before anything trusts it.

These checks validate the folding geometry against hand-computed values, the
self-contained x2 chain against basic multirate invariants, and the latency
formula against the production filter's actual tap count.
"""
import numpy as np
import pytest

from saturation import reference as ref
from saturation.resample import MultirateEngine

FS = 48000
NYQ = FS / 2.0


def test_folded_frequency_hand_computed():
    # 39 kHz at fs=48k (Nyquist 24k): 39k lies in span [24k,48k] and mirrors
    # to 48k - 39k = 9k. 57k lies in [48k,72k] mirroring to 57k-48k = 9k.
    assert ref.folded_frequency(39000.0, FS) == pytest.approx(9000.0, abs=1e-9)
    assert ref.folded_frequency(57000.0, FS) == pytest.approx(9000.0, abs=1e-9)
    # 26k in the first mirror span -> 48k-26k = 22k
    assert ref.folded_frequency(26000.0, FS) == pytest.approx(22000.0, abs=1e-9)
    # anything below Nyquist is returned unchanged
    assert ref.folded_frequency(13000.0, FS) == pytest.approx(13000.0)


def test_expected_alias_bins_only_contains_folded_locations():
    bins = ref.expected_alias_bins(3400.0, FS, n_harm=12)
    # every bin is below Nyquist and corresponds to a harmonic >= Nyquist
    for f in bins:
        assert 0.0 < f < NYQ
        harmonic = next(3400.0 * h for h in range(2, 13)
                        if abs(ref.folded_frequency(3400.0 * h, FS) - f) < 1e-9)
        assert harmonic >= NYQ
    # distinct, sorted locations
    assert len(bins) == len(set(round(b, 6) for b in bins))


def test_static_harmonic_series_odd_only_for_symmetric_curve():
    peaks = ref.static_harmonic_series(1000.0, 6.0, "tanh", amp=0.9,
                                       n_harm=8)
    # tanh of a sine has no even harmonics and no DC
    assert peaks[0] > peaks[1]  # fundamental dominates
    assert peaks[1] == pytest.approx(0.0, abs=1e-9)   # 2nd (even) ~ 0
    assert peaks[3] == pytest.approx(0.0, abs=1e-9)   # 4th (even) ~ 0
    assert peaks[2] > 1e-4                            # 3rd (odd) present


def test_static_harmonic_series_asymmetric_has_even_content():
    odd = ref.static_harmonic_series(1000.0, 6.0, "tanh", n_harm=6)
    asym = ref.static_harmonic_series(1000.0, 6.0, "asym", n_harm=6)
    assert asym[1] > 1e-4                 # 2nd harmonic appears
    assert odd[1] == pytest.approx(0.0, abs=1e-9)


def test_offline_factor2_preserves_base_rate_length_and_bandlimits():
    rng = np.random.default_rng(0)
    x = 0.5 * rng.standard_normal(FS)
    y = ref.offline_factor2(x, 4.0, "tanh", FS, cutoff_hz=18000.0)
    assert len(y) == len(x)
    # the independent chain attenuates content above its cutoff (real LPF,
    # not a passthrough): compare out-of-band vs in-band energy
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / FS)
    in_band = mag[(freqs > 1000) & (freqs < 12000)].mean()
    out_band = mag[(freqs > 20000) & (freqs < 23500)].mean()
    assert 20 * np.log10(out_band / in_band + 1e-20) < -40.0


def test_expected_latency_matches_production_tap_group_delay():
    cut = 18000.0
    for factor in (2, 4, 8):
        eng = MultirateEngine(factor, FS, drive=2.0, cutoff_hz=cut)
        # independent derivation must equal the engine's reported delay
        assert ref.expected_latency_base(factor, FS, cut) == eng.delay_base
        # and the actual FIR taps really have that group delay
        up_delay = (len(eng.up.taps) - 1) // 2 // factor
        down_delay = (len(eng.down.taps) - 1) // 2 // factor
        assert up_delay + down_delay == eng.delay_base


def test_expected_latency_zero_for_factor_one():
    assert ref.expected_latency_base(1, FS, 18000.0) == 0


def test_polarity_split_separates_even_and_odd():
    n = 1024
    t = np.arange(n)
    y = np.sin(t) + 0.3 * np.cos(2 * t) + 0.1
    y_inv = -np.sin(t) + 0.3 * np.cos(2 * t) + 0.1  # even part unchanged
    even, odd = ref.polarity_split(y, y_inv)
    assert np.allclose(odd, np.sin(t), atol=1e-12)
    assert np.allclose(even, 0.3 * np.cos(2 * t) + 0.1, atol=1e-12)
