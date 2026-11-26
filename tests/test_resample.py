import numpy as np
import pytest

from saturation.resample import MultirateEngine, design_kaiser_lpf, _FIR
from saturation.transfer import Transfer
from saturation import signals

FS = 48000


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_block_size_invariance(factor):
    rng = np.random.default_rng(1)
    x = signals.pluck(FS, f0=196.0, dur=1.0) + 0.05 * rng.standard_normal(FS)
    ref = MultirateEngine(factor, FS, drive=5.0)
    y_full = np.concatenate([ref.process(x), ref.flush()])

    chunked = MultirateEngine(factor, FS, drive=5.0)
    parts = []
    for bs in (37, 128, 257, 1024):
        pass
    for start in range(0, len(x), 257):
        parts.append(chunked.process(x[start:start + 257]))
    parts.append(chunked.flush())
    y_chunk = np.concatenate(parts)
    assert len(y_full) == len(y_chunk)
    assert np.allclose(y_full, y_chunk, atol=1e-10)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_impulse_delay_matches_design_within_memory_lag(factor):
    eng = MultirateEngine(factor, FS, drive=1.0, kind=Transfer.TANH)
    # very small impulse keeps the saturator in its linear region
    x = np.zeros(512)
    x[0] = 1e-4
    y = np.concatenate([eng.process(x), eng.flush()])
    peak = int(np.argmax(np.abs(y)))
    # FIR group delay plus at most one sample of one-pole memory lag
    assert abs(peak - eng.delay_base) <= 1
    # tail fully drains: output length equals input plus reported delay
    assert len(y) == len(x) + eng.delay_base
    # nothing left behind after a second flush
    second = eng.flush()
    assert np.allclose(second, np.zeros_like(second), atol=1e-12)


def test_real_upsampling_not_sample_repeat():
    fs_hi = FS * 4
    taps = design_kaiser_lpf(FS * 0.45, fs_hi) * 4
    fir = _FIR(taps)
    up = np.zeros(400)
    up[::4] = np.sign(np.sin(np.arange(100) * 1.7))  # zero-stuffed wide-band
    y = fir.process(up)
    # a sample-and-hold would repeat identical values; a real band-limited
    # interpolation produces distinct inter-sample values
    between = y[1::4]
    assert np.std(between) > 0.01
    # and the interpolator attenuates the spectral images above cutoff
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1.0 / fs_hi)
    image = spec[(freqs > FS * 0.6) & (freqs < FS * 1.4)]
    base = spec[(freqs > 100) & (freqs < FS * 0.4)]
    assert 20 * np.log10(image.max() / base.max() + 1e-20) < -40.0


def test_oversampling_reduces_alias_energy():
    from saturation.analysis import spectrum, alias_energy
    f0 = 3400.0
    x = signals.sine(FS, f0, 1.0, amp=0.9)
    alias = {}
    for factor in (1, 4):
        eng = MultirateEngine(factor, FS, drive=8.0, kind=Transfer.HARD_CLIP)
        y = np.concatenate([eng.process(x), eng.flush()])
        y = y[eng.delay_base:]
        freqs, mag = spectrum(y[: FS], FS)
        alias[factor] = alias_energy(freqs, mag, f0, FS)
    assert alias[4] < alias[1] * 0.5


def test_overrange_input_stays_finite_and_tightly_bounded():
    x = signals.overrange(FS, amp=50.0, dur=0.2)
    for factor in (1, 8):
        eng = MultirateEngine(factor, FS, drive=20.0, kind=Transfer.TANH)
        y = np.concatenate([eng.process(x), eng.flush()])
        assert np.isfinite(y).all()
        # the static curve clips at 1.0; only the anti-imaging FIR's gain
        # compensation may overshoot, and by a small documented margin
        margin = 1.10 if factor > 1 else 1.0 + 1e-9
        assert np.max(np.abs(y)) <= margin


def test_static_curve_bounds_hold_without_filtering():
    from saturation.transfer import apply_transfer, Transfer
    x = signals.overrange(FS, amp=100.0, dur=0.1)
    for kind in Transfer:
        y = apply_transfer(x, 50.0, kind)
        assert np.isfinite(y).all()
        assert np.max(np.abs(y)) <= kind.output_bound + 1e-12


def test_filter_states_persist_and_then_reset():
    eng = MultirateEngine(4, FS, drive=3.0)
    x = signals.sine(FS, 440.0, 0.2)
    eng.process(x[:4800])
    state_before = eng.up.state.copy()
    assert np.any(state_before != 0.0)
    eng.reset()
    assert np.all(eng.up.state == 0.0)
    assert eng.saturator._z_in == 0.0
