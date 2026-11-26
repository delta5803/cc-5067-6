"""The anti-alias chain must not 'cheat' by cutting all high frequencies:
bandwidth retention and added distortion have to be demonstrated together."""
import numpy as np

from saturation.analysis import spectrum, passband_retention, band_energy
from saturation.resample import MultirateEngine
from saturation import signals

FS = 48000


def _run(x, factor, drive=1.0):
    eng = MultirateEngine(factor, FS, drive=drive)
    y = np.concatenate([eng.process(x), eng.flush()])
    return y[eng.delay_base:eng.delay_base + len(x)]


def test_small_signal_bandwidth_retained_through_chain():
    # low level, low drive: the nonlinearity is nearly transparent, so the
    # multirate chain itself must preserve bandwidth up to its design cutoff
    x = signals.log_sweep(FS, f0=50.0, f1=20000.0, dur=2.0, amp=0.05)
    for factor in (1, 4, 8):
        y = _run(x, factor, drive=1.0)
        fi, mi = spectrum(x, FS)
        fo, mo = spectrum(y, FS)
        shape = passband_retention(fi, mi, fo, mo, 18000.0)
        assert shape > 0.95, factor
        # absolute high-band energy kept within 3 dB of the dry input
        e_dry = band_energy(fi, mi, 5000.0, 18000.0)
        e_wet = band_energy(fo, mo, 5000.0, 18000.0)
        assert 10 * np.log10(e_wet / e_dry) > -3.0


def test_alias_reduction_does_not_collapse_high_frequency_content():
    # High-band alias probe: a 19 kHz tone's 3rd harmonic (57 kHz) folds to
    # 9 kHz inside the passband; the 2nd harmonic (38 kHz) folds to 10 kHz.
    # Those folded lines exist without OS and are suppressed with it, while a
    # legitimate 15 kHz tone (below cutoff) must survive the same chain.
    t = np.arange(FS) / FS
    probe = 0.5 * np.sin(2 * np.pi * 15000 * t)
    aliaser = 0.9 * np.sin(2 * np.pi * 19000 * t)
    x = probe + aliaser
    from saturation.analysis import peak_at
    rows = {}
    for factor in (1, 8):
        eng = MultirateEngine(factor, FS, drive=12.0, kind="hard_clip")
        y = np.concatenate([eng.process(x), eng.flush()])
        y = y[eng.delay_base:eng.delay_base + len(x)]
        f, mag = spectrum(y, FS)
        # 3*19k=57k -> folds to 9k; 2*19k=38k -> folds to 10k (both in-band)
        rows[factor] = {
            "legit_15k": peak_at(f, mag, 15000.0, 8.0),
            "alias_9k": peak_at(f, mag, 9000.0, 8.0),
            "alias_10k": peak_at(f, mag, 10000.0, 8.0),
        }
    # without OS the folded harmonics sit well above the noise floor
    assert rows[1]["alias_9k"] > 0.01
    assert rows[8]["alias_9k"] < rows[1]["alias_9k"] * 0.1
    # genuine 15 kHz content retained within 6 dB: bandwidth is not brick-walled
    assert rows[8]["legit_15k"] > rows[1]["legit_15k"] * 0.5
