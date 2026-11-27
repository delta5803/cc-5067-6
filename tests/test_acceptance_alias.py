"""Acceptance for alias control: spectra, folding, images, passband, latency.

Thresholds come with a positive example (healthy high-level drive) and a
negative example (extremely low level), and every injected fault has its own
switch and is required to fail the matching check while leaving unrelated
checks intact — so the suite can distinguish harmonic fold-back from a
decimation/path error rather than reporting a generic "aliasing" failure.
"""
import numpy as np
import pytest

from saturation import acceptance as ac
from saturation import reference as ref
from saturation.faults import FaultyMultirateEngine
from saturation.resample import MultirateEngine
from saturation.transfer import Transfer

FS = 48000
CUTOFF = 18000.0
F0 = 3400.0
DRIVE = 8.0
KIND = Transfer.HARD_CLIP


def _tone(amp=0.9):
    t = np.arange(FS) / FS
    return amp * np.sin(2 * np.pi * F0 * t)


def _engine(factor, **kw):
    return MultirateEngine(factor, FS, drive=DRIVE, kind=KIND,
                           cutoff_hz=CUTOFF, **kw)


# --------------------------------------------------------------------- #
# Estimator is chosen from measurement, not assumed
# --------------------------------------------------------------------- #
def test_alias_estimator_chosen_from_measured_separation():
    sel = ac.select_alias_estimator(FS, F0, DRIVE, KIND, CUTOFF)
    # the measured x1 recording must alias hard, x8 must be far below it
    assert sel["measured_x1"] > 50 * sel["measured_x8"]
    # exactly the method that straddles both recordings is selected
    assert sel["chosen"] == "offline_factor2"
    chosen = sel["candidates"][sel["chosen"]]
    assert chosen["discriminates"] is True
    # the rejected band-difference method is recorded with the reason
    rejected = sel["candidates"]["band_difference"]
    assert rejected["discriminates"] is False


# --------------------------------------------------------------------- #
# Healthy chain: fold energy, images, passband, reported latency
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_oversampling_suppresses_fold_energy_below_independent_bound(factor):
    bound = ref.alias_lower_bound_x2(F0, DRIVE, KIND, FS, CUTOFF)["bound_energy"]
    ev = ac.measure_alias(_engine(factor), _tone(), FS, F0, DRIVE, KIND,
                          factor, CUTOFF)
    # production high-factor chain must beat the independent x2 lower bound,
    # while factor 1 (measured separately) sits far above it
    assert ev.measured_alias_energy < bound
    assert ev.measured_alias_energy < 1e-3


def test_factor1_alias_exceeds_bound_and_factor8_passes():
    bound = ref.alias_lower_bound_x2(F0, DRIVE, KIND, FS, CUTOFF)["bound_energy"]
    x1 = ac.measure_alias(_engine(1), _tone(), FS, F0, DRIVE, KIND, 1, CUTOFF)
    x8 = ac.measure_alias(_engine(8), _tone(), FS, F0, DRIVE, KIND, 8, CUTOFF)
    assert x1.measured_alias_energy > 10 * bound
    assert x8.measured_alias_energy < bound


def test_reported_latency_matches_independent_derivation_all_factors():
    for factor in (1, 2, 4, 8):
        ev = ac.measure_alias(_engine(factor), _tone(), FS, F0, DRIVE, KIND,
                              factor, CUTOFF)
        verdict = ac.latency_verdict(ev)
        assert verdict["ok"], verdict
        assert verdict["delta"] == 0


def test_impulse_timing_integer_and_subsample_healthy():
    for factor in (1, 2, 4, 8):
        timing = ac.impulse_timing(_engine(factor), FS)
        assert timing["peak_delta"] in (0, 1)
        # healthy centroid stays within ~1 base-rate sample of design
        assert abs(timing["centroid_delta"]) < 1.0


def test_passband_retained_on_two_sweeps_relative_to_factor1():
    # near-linear drive so the nonlinearity itself does not colour the sweep
    ref1 = MultirateEngine(1, FS, drive=1.0, cutoff_hz=CUTOFF)
    for factor in (2, 4, 8):
        eng = MultirateEngine(factor, FS, drive=1.0, cutoff_hz=CUTOFF)
        pb = ac.measure_passband(eng, FS, CUTOFF, reference_engine=ref1)
        for span in ("low", "high"):
            assert pb[span]["shape_corr"] > 0.999
            assert pb[span]["highband_loss_db"] > -1.0


def test_mirror_and_fold_components_independently_measured():
    ev1 = ac.measure_alias(_engine(1), _tone(), FS, F0, DRIVE, KIND, 1, CUTOFF)
    ev8 = ac.measure_alias(_engine(8), _tone(), FS, F0, DRIVE, KIND, 8, CUTOFF)
    # folded bins predicted independently are real, sizeable lines at factor 1
    assert sum(ev1.fold_levels.values()) > 0.05
    # both fold energy and mirror-region energy collapse with oversampling
    assert ev8.measured_alias_energy < ev1.measured_alias_energy * 0.01
    assert ev8.image_energy < ev1.image_energy * 0.05


# --------------------------------------------------------------------- #
# Positive vs negative level examples
# --------------------------------------------------------------------- #
def test_high_level_drives_alias_low_level_is_clean():
    hi = ac.measure_alias(_engine(8), _tone(0.9), FS, F0, DRIVE, KIND, 8, CUTOFF)
    quiet_eng = MultirateEngine(8, FS, drive=1.0, kind="tanh", cutoff_hz=CUTOFF)
    lo = ac.measure_alias(quiet_eng, _tone(1e-4), FS, F0, 1.0, "tanh", 8, CUTOFF)
    assert hi.measured_alias_energy > 1e-5       # positive example: detectable
    assert lo.measured_alias_energy < 1e-12      # negative example: noise floor
    assert lo.measured_alias_energy < hi.measured_alias_energy * 1e-6


# --------------------------------------------------------------------- #
# Reverse polarity separates passband distortion from folding
# --------------------------------------------------------------------- #
def test_reverse_polarity_odd_even_separation():
    x = _tone()
    for kind, expect_even in (("tanh", False), ("asym", True)):
        eng_pos = MultirateEngine(4, FS, drive=6.0, kind=kind,
                                  cutoff_hz=CUTOFF)
        eng_neg = MultirateEngine(4, FS, drive=6.0, kind=kind,
                                  cutoff_hz=CUTOFF)
        pol = ac.polarity_evidence(eng_pos, eng_neg, x, FS)
        if expect_even:
            assert pol["even_energy"] > 0.05
        else:
            assert pol["even_energy"] < 1e-9
        assert pol["odd_energy"] > 1.0


# --------------------------------------------------------------------- #
# Independent fault switch 1: removing the anti-alias filter MUST fail
# --------------------------------------------------------------------- #
def test_removing_antialias_filter_fails_alias_but_keeps_latency():
    healthy = ac.measure_alias(_engine(8), _tone(), FS, F0, DRIVE, KIND,
                               8, CUTOFF)
    faulty_eng = FaultyMultirateEngine(8, FS, drive=DRIVE, kind=KIND,
                                       cutoff_hz=CUTOFF,
                                       remove_antialias=True)
    faulty = ac.measure_alias(faulty_eng, _tone(), FS, F0, DRIVE, KIND,
                              8, CUTOFF)
    # the acceptance assertion a healthy chain passes must reject this one
    bound = ref.alias_lower_bound_x2(F0, DRIVE, KIND, FS, CUTOFF)["bound_energy"]
    assert faulty.measured_alias_energy > 10 * bound
    assert faulty.measured_alias_energy > healthy.measured_alias_energy * 50
    # timing report is unchanged: this is a spectral failure, not a delay bug
    assert ac.latency_verdict(faulty)["ok"] is True


def test_antialias_fault_is_off_by_default_and_bit_identical():
    x = _tone()
    a = FaultyMultirateEngine(4, FS, drive=DRIVE, kind=KIND, cutoff_hz=CUTOFF)
    b = MultirateEngine(4, FS, drive=DRIVE, kind=KIND, cutoff_hz=CUTOFF)
    ya = np.concatenate([a.process(x), a.flush()])
    yb = np.concatenate([b.process(x), b.flush()])
    assert np.allclose(ya, yb, atol=1e-12)


# --------------------------------------------------------------------- #
# Independent fault switch 2: decimation grid error MUST be identified
# --------------------------------------------------------------------- #
def test_decimation_grid_error_detected_as_timing_not_alias():
    grid_eng = FaultyMultirateEngine(8, FS, drive=DRIVE, kind=KIND,
                                     cutoff_hz=CUTOFF, decimation_phase=4)
    # alias stays low (filter still present) so an alias-only test would miss it
    ev = ac.measure_alias(grid_eng, _tone(), FS, F0, DRIVE, KIND, 8, CUTOFF)
    assert ev.measured_alias_energy < 1e-3
    # but the sub-sample timing moves while the reported latency is unchanged
    timing = ac.impulse_timing(grid_eng, FS)
    healthy_timing = ac.impulse_timing(_engine(8), FS)
    assert timing["centroid_delta"] > healthy_timing["centroid_delta"] + 0.3
    assert ac.latency_verdict(ev)["ok"] is True
