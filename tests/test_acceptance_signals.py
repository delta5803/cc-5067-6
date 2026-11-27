"""Acceptance stimulus generators and the clean-directory campaign runner."""
import json
import os

import numpy as np

from saturation import acceptance_signals as asg
from saturation import reference as ref

FS = 48000


def test_two_sweeps_cover_different_frequency_spans():
    from scipy.signal import hilbert
    lo = asg.sweep_low(FS, dur=1.0)
    hi = asg.sweep_high(FS, dur=1.0)
    assert len(lo) == len(hi) == FS

    # instantaneous frequency recovered from the analytic-signal phase;
    # trim the edges where the Hilbert transform is unreliable
    def instant_freq(x):
        phase = np.unwrap(np.angle(hilbert(x)))
        d = np.diff(phase) / (2.0 * np.pi) * FS
        return d[FS // 10:-FS // 10]

    flo, fhi = instant_freq(lo), instant_freq(hi)
    assert np.median(fhi) > np.median(flo)
    assert fhi.max() > 12000.0          # high sweep reaches folding territory
    assert flo.max() < 3800.0           # low sweep stays mostly in-band
    assert fhi.min() > 1000.0           # high sweep starts well above the low one


def test_multitone_returns_declared_frequencies():
    sig, freqs = asg.multitone(FS, dur=1.0)
    assert len(sig) == FS
    assert len(freqs) >= 4
    spec = np.abs(np.fft.rfft(sig * np.hanning(FS)))
    grid = np.fft.rfftfreq(FS, 1.0 / FS)
    for f in freqs:
        assert spec[np.argmin(np.abs(grid - f))] > 0.0


def test_transient_has_anchored_attack_and_decay():
    tr = asg.transient(FS, dur=1.0, lead_silence=0.1)
    n0 = int(0.1 * FS)
    assert np.all(tr[:n0] == 0.0)
    assert np.argmax(np.abs(tr)) == n0          # deterministic click anchor
    # energy decays after the attack
    early = np.sqrt(np.mean(tr[n0:n0 + 100] ** 2))
    late = np.sqrt(np.mean(tr[-2000:] ** 2))
    assert early > 10 * late + 1e-12


def test_inverted_signal_is_exact_reverse_polarity():
    x = asg.sweep_low(FS)
    xi = asg.inverted(x)
    assert np.allclose(xi, -x, atol=1e-15)
    assert np.allclose(x + xi, 0.0, atol=1e-15)


def test_beat_grid_is_evenly_spaced():
    beats = asg.beat_grid(FS, bpm=120.0, dur=2.0)
    # 120 BPM = 2 beats/s -> 0.5 s spacing
    assert np.all(np.diff(beats) == FS // 2)
    assert beats[0] == 0


def test_acceptance_runner_runs_in_clean_directory(tmp_path):
    # exercise the campaign's section builders against a freshly created
    # directory; the runner's main() itself wipes and recreates the path, so
    # nothing pre-existing or hand-edited is ever read
    from experiments import run_acceptance as runner
    out_dir = str(tmp_path / "output_acceptance")
    os.makedirs(out_dir, exist_ok=False)
    report = {"sample_rate": FS, "cutoff_hz": runner.CUTOFF,
              "f0_hz": runner.F0, "drive": runner.DRIVE,
              "transfer": runner.KIND.value,
              "independent_reference": "saturation.reference",
              "alias_bins_hz": ref.expected_alias_bins(runner.F0, FS)}
    runner.section_alias_control(report, out_dir)
    runner.section_parallel_bus(report, out_dir)
    with open(os.path.join(out_dir, "acceptance_report.json"), "w") as fh:
        json.dump(report, fh)

    # evidence files were freshly written, nothing pre-existing was read
    assert os.path.isfile(os.path.join(out_dir, "acceptance_report.json"))
    wavs = [f for f in os.listdir(out_dir) if f.endswith(".wav")]
    assert {"tone_dry.wav", "fault_no_antialias.wav",
            "bus_healthy_sum.wav", "bus_automation.wav"} <= set(wavs)

    # the report carries the independently derived expectations and verdicts
    assert report["estimator_selection"]["chosen"] is not None
    assert report["alias_control"]["factors"][8]["latency"]["ok"] is True
    # anti-alias removal must show a large alias gain and unchanged latency
    noaa = report["alias_faults"]["remove_antialias"]
    assert noaa["alias_gain_db_vs_healthy"] > 30.0
    assert noaa["latency"]["ok"] is True
    # comb: aligned self is clean, misalignment is not
    assert report["bus_comb"]["aligned_self"][
        "fraction_below_minus_6db"] == 0.0
    assert report["bus_comb"]["misalign_8"]["deepest_ratio"] < 0.2
    # same-beat automation evidence
    assert report["bus_automation"][
        "dry_pullup_and_switch_same_sample"] is True
