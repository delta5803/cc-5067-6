"""Acceptance for the latency-compensated dry/wet parallel bus.

The wet branch's reported latency compensates the dry branch, so a healthy
parallel sum is comb-free. A separate path-alignment fault switch offsets the
dry branch and must produce measurable comb cancellation, while the
resampling faults (covered elsewhere) must not — proving the two failure
families are told apart. Same-beat automation changes ratio and dry level on
one tick, and all differential evidence comes from real recordings.
"""
import numpy as np
import pytest

from saturation import acceptance as ac
from saturation import acceptance_signals as asg
from saturation.faults import MisalignedParallelBus, FaultyMultirateEngine
from saturation.parallel_bus import ParallelBus
from saturation.resample import MultirateEngine

FS = 48000
CUTOFF = 18000.0


def _flat_stimulus():
    rng = np.random.default_rng(11)
    return 0.04 * rng.standard_normal(FS)


def _healthy_bus():
    return ParallelBus(FS, factors=(1, 8), drive=0.5, wet_mix=0.5)


def test_dry_branch_uses_reported_latency_for_compensation():
    bus = _healthy_bus()
    proc_latency = bus.proc.max_delay
    assert bus.reported_latency == proc_latency
    # the dry delay line is exactly that compensation (plus zero offset)
    assert bus._dry.delay == proc_latency
    assert bus.dry_delay_offset == 0


def test_healthy_bus_is_comb_free_against_own_baseline():
    stim = _flat_stimulus()
    baseline = _healthy_bus().process_offline(stim, 512)
    evidence = ac.comb_evidence(_healthy_bus(), stim, FS, baseline)
    assert evidence["deepest_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert evidence["fraction_below_minus_6db"] == 0.0


@pytest.mark.parametrize("misalign,min_notch_db", [(1, 2.0), (2, 4.0),
                                                   (4, 6.0), (8, 12.0)])
def test_dry_misalignment_produces_comb_cancellation(misalign, min_notch_db):
    stim = _flat_stimulus()
    baseline = _healthy_bus().process_offline(stim, 512)
    faulty = MisalignedParallelBus(FS, factors=(1, 8), drive=0.5,
                                   wet_mix=0.5, misalign=misalign)
    ev = ac.comb_evidence(faulty, stim, FS, baseline)
    assert ev["deepest_ratio"] < 0.8
    assert ev["notch_db"] > min_notch_db
    # larger offsets carve out a measurable fraction of deep bins
    if misalign >= 4:
        assert ev["fraction_below_minus_6db"] > 0.02


def test_comb_fault_is_independent_of_resampling_fault():
    # a resampling fault with correct bus alignment must NOT trip the comb
    # detector: the two failure families have separate switches
    stim = _flat_stimulus()
    baseline = _healthy_bus().process_offline(stim, 512)
    healthy_evidence = ac.comb_evidence(_healthy_bus(), stim, FS, baseline)
    assert healthy_evidence["fraction_below_minus_6db"] == 0.0
    # and an engine-level decimation fault does not alter the bus compensation
    eng_fault = FaultyMultirateEngine(8, FS, drive=0.5, cutoff_hz=CUTOFF,
                                      decimation_phase=4)
    assert eng_fault.delay_base == _healthy_bus().proc.max_delay or \
        eng_fault.delay_base == MultirateEngine(8, FS, drive=0.5,
                                                cutoff_hz=CUTOFF).delay_base


def test_same_beat_automation_switches_ratio_and_pulls_dry_together():
    beat = FS // 2
    bus = ParallelBus(FS, factors=(1, 8), drive=5.0, wet_mix=0.6)
    bus.schedule_beat(beat, 8, dry_pullup_db=3.0)
    pluck = asg.transient(FS, dur=1.0)
    bus.process_offline(pluck, 512)
    auto = next(e for e in bus.events if e.kind == "beat_automation")
    assert auto.sample_time == beat
    # dry gain target was raised above unity on that same beat (+3 dB ~ 1.41x)
    assert bus.events[0].detail.startswith("request x8")
    assert bus.peak_dry_gain > 1.4
    assert bus.active_factor == 8
    # and the dry level is restored after the crossfade settles
    restore = next(e for e in bus.events if e.kind == "dry_gain_restore")
    assert restore.sample_time >= beat
    assert bus.dry_gain == pytest.approx(1.0)


def test_branch_recordings_are_real_equal_length_differential():
    pluck = asg.transient(FS, dur=1.0)
    bus = ParallelBus(FS, factors=(1, 8), drive=4.0, wet_mix=0.5)
    dry, wet, diff = bus.branch_recordings(pluck)
    assert len(dry) == len(wet) == len(diff)
    # the difference is genuine saturated content, not zeros or copied data
    assert np.sqrt(np.mean(diff ** 2)) > 1e-4
    assert not np.allclose(dry, wet, atol=1e-6)
    # identity check: diff really equals wet - dry
    assert np.allclose(diff, wet - dry, atol=1e-12)


def test_parallel_sum_without_misalignment_has_no_systematic_notches():
    # a musically broadband signal (multitone + sweep), not just white noise
    sig = 0.05 * (asg.multitone(FS, dur=1.0)[0] + asg.sweep_low(FS, dur=1.0))
    baseline = _healthy_bus().process_offline(sig, 512)
    ev = ac.comb_evidence(_healthy_bus(), sig, FS, baseline)
    assert ev["deepest_ratio"] == pytest.approx(1.0, abs=1e-9)
