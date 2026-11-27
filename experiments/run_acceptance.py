"""Independent acceptance campaign for the saturation multirate chain.

Run from a *clean* output directory (created fresh each run, never read back):

    PYTHONPATH=/workspace python3 -P experiments/run_acceptance.py

The campaign exercises two independent failure families with their own fault
switches and records real per-run differential evidence:

  A. resampling / alias control — high-level drive vs extremely low level,
     fold/mirror/passband/latency measurements, anti-alias removal and a
     decimation-grid fault, two sweeps and reverse polarity;
  B. path alignment — the latency-compensated dry/wet parallel bus, same-beat
     ratio change with a dry pull-up, and a dry-branch misalignment that must
     produce comb cancellation.

All expected values come from ``saturation.reference`` (independent
derivations) and from healthy recordings made in this same run; nothing is
hand-edited or loaded from a previous run. A machine-readable report and WAV
evidence are written into a freshly created directory.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from saturation.analysis import spectrum, passband_retention, band_energy
from saturation import acceptance_signals as asg
from saturation import acceptance as ac
from saturation import reference as ref
from saturation.faults import FaultyMultirateEngine, MisalignedParallelBus
from saturation.parallel_bus import ParallelBus
from saturation.resample import MultirateEngine
from saturation.transfer import Transfer
from saturation.wavio import write_wav

FS = 48000
CUTOFF = min(FS * 0.45, 18000.0)
F0 = 3400.0
DRIVE = 8.0
KIND = Transfer.HARD_CLIP


def _tone(amp: float = 0.9) -> np.ndarray:
    t = np.arange(FS) / FS
    return amp * np.sin(2.0 * np.pi * F0 * t)


def _engine(factor: int, **kw):
    return MultirateEngine(factor, FS, drive=DRIVE, kind=KIND,
                           cutoff_hz=CUTOFF, **kw)


def section_alias_control(report: dict, out_dir: str) -> None:
    x = _tone()
    write_wav(f"{out_dir}/tone_dry.wav", FS, x)

    # ---- estimator selection decided from measurement ---- #
    selection = ac.select_alias_estimator(FS, F0, DRIVE, KIND, CUTOFF)
    report["estimator_selection"] = selection
    chosen = selection["chosen"]
    bound = selection["candidates"][chosen]["bound_energy"]

    rows = {}
    recordings = {}
    for factor in (1, 2, 4, 8):
        eng = _engine(factor)
        ev = ac.measure_alias(eng, x, FS, F0, DRIVE, KIND, factor, CUTOFF)
        timing = ac.impulse_timing(_engine(factor), FS)
        passband = ac.measure_passband(
            _engine(factor), FS, CUTOFF,
            reference_engine=MultirateEngine(1, FS, drive=DRIVE, kind=KIND,
                                             cutoff_hz=CUTOFF))
        pol_eng_pos = _engine(factor)
        pol_eng_neg = _engine(factor)
        pol = ac.polarity_evidence(pol_eng_pos, pol_eng_neg, x, FS)
        rows[factor] = {
            "alias_energy": ev.measured_alias_energy,
            "image_energy": ev.image_energy,
            "fold_levels": ev.fold_levels,
            "latency": ac.latency_verdict(ev),
            "impulse_timing": timing,
            "passband": passband,
            "polarity_even_energy": pol["even_energy"],
            "polarity_odd_energy": pol["odd_energy"],
            "alias_vs_bound_db": float(20.0 * np.log10(
                (ev.measured_alias_energy + 1e-20) / (bound + 1e-20))),
        }
        # keep an actual wet recording for differential evidence
        full = np.concatenate([_engine(factor).process(x),
                               _engine(factor).flush()])
        recordings[factor] = full
        write_wav(f"{out_dir}/tone_wet_x{factor}.wav", FS,
                  full[_engine(factor).delay_base:])
    report["alias_control"] = {"bound": bound, "bound_estimator": chosen,
                               "factors": rows}

    # ---- positive / negative level examples ---- #
    hi = _engine(8)
    lo = MultirateEngine(8, FS, drive=1.0, kind="tanh", cutoff_hz=CUTOFF)
    ev_hi = ac.measure_alias(hi, _tone(0.9), FS, F0, DRIVE, KIND, 8, CUTOFF)
    x_quiet = _tone(1e-4)
    ev_lo = ac.measure_alias(lo, x_quiet, FS, F0, 1.0, "tanh", 8, CUTOFF)
    report["level_examples"] = {
        "high_level_drive": {"alias_energy": ev_hi.measured_alias_energy,
                             "image_energy": ev_hi.image_energy},
        "extremely_low_level": {"alias_energy": ev_lo.measured_alias_energy,
                                "image_energy": ev_lo.image_energy},
    }

    # ---- independent fault switches, each with real differential ---- #
    faults = {}
    healthy_x8 = recordings[8]
    # switch 1: anti-alias removal -> alias/image must rise, timing unchanged
    eng_noaa = FaultyMultirateEngine(8, FS, drive=DRIVE, kind=KIND,
                                     cutoff_hz=CUTOFF, remove_antialias=True)
    ev_noaa = ac.measure_alias(eng_noaa, x, FS, F0, DRIVE, KIND, 8, CUTOFF)
    rec_noaa = np.concatenate([eng_noaa.process(x), eng_noaa.flush()])
    write_wav(f"{out_dir}/fault_no_antialias.wav", FS, rec_noaa)
    faults["remove_antialias"] = {
        "alias_energy": ev_noaa.measured_alias_energy,
        "image_energy": ev_noaa.image_energy,
        "latency": ac.latency_verdict(ev_noaa),
        "alias_gain_db_vs_healthy": float(20.0 * np.log10(
            (ev_noaa.measured_alias_energy + 1e-20)
            / (rows[8]["alias_energy"] + 1e-20))),
        "recording_diff_rms": float(np.sqrt(np.mean(
            (rec_noaa[:len(healthy_x8)] - healthy_x8) ** 2))),
    }
    # switch 2: decimation grid -> sub-sample timing shift, alias stays low
    eng_grid = FaultyMultirateEngine(8, FS, drive=DRIVE, kind=KIND,
                                     cutoff_hz=CUTOFF, decimation_phase=4)
    ev_grid = ac.measure_alias(eng_grid, x, FS, F0, DRIVE, KIND, 8, CUTOFF)
    timing_grid = ac.impulse_timing(eng_grid, FS)
    faults["decimation_grid"] = {
        "alias_energy": ev_grid.measured_alias_energy,
        "image_energy": ev_grid.image_energy,
        "impulse_timing": timing_grid,
        "latency_report_unchanged": ac.latency_verdict(ev_grid)["ok"],
    }
    report["alias_faults"] = faults

    # ---- two sweeps: passband distortion kept apart from folding ---- #
    sweep_rows = {}
    for name, gen in (("sweep_low", asg.sweep_low),
                      ("sweep_high", asg.sweep_high)):
        sw = gen(FS, dur=1.0, amp=0.7)
        per = {}
        for factor in (1, 8):
            eng = _engine(factor)
            y = np.concatenate([eng.process(sw), eng.flush()])
            d = eng.delay_base
            y = y[d:d + len(sw)]
            fi, mi = spectrum(sw, FS)
            fo, mo = spectrum(y, FS)
            per[factor] = {
                "shape_corr": passband_retention(fi, mi, fo, mo,
                                                 min(CUTOFF, 16000.0)),
                "fold_band_energy": band_energy(fo, mo, CUTOFF + 400.0,
                                                FS / 2.0 - 100.0),
            }
        sweep_rows[name] = per
    report["sweeps"] = sweep_rows


def section_parallel_bus(report: dict, out_dir: str) -> None:
    rng = np.random.default_rng(11)
    stimulus = 0.04 * rng.standard_normal(FS)  # flat, near-linear for comb
    write_wav(f"{out_dir}/bus_stimulus.wav", FS, stimulus)

    # healthy baseline rendered once, reused as the differential reference
    healthy = ParallelBus(FS, factors=(1, 8), drive=0.5, wet_mix=0.5)
    healthy_sum = healthy.process_offline(stimulus, 512)
    write_wav(f"{out_dir}/bus_healthy_sum.wav", FS, healthy_sum)
    report["bus_reported_latency"] = healthy.reported_latency

    comb = {}
    for misalign in (1, 2, 4, 8):
        faulty = MisalignedParallelBus(
            FS, factors=(1, 8), drive=0.5, wet_mix=0.5, misalign=misalign)
        evidence = ac.comb_evidence(faulty, stimulus, FS, healthy_sum)
        comb[f"misalign_{misalign}"] = evidence
    # the healthy bus against its own baseline must be notch-free
    comb["aligned_self"] = ac.comb_evidence(
        ParallelBus(FS, factors=(1, 8), drive=0.5, wet_mix=0.5),
        stimulus, FS, healthy_sum)
    report["bus_comb"] = comb

    # same-beat automation: ratio change and dry pull-up fire on one beat
    beat = int(FS * 0.5)
    bus = ParallelBus(FS, factors=(1, 8), drive=5.0, wet_mix=0.6)
    bus.schedule_beat(beat, 8, dry_pullup_db=3.0)
    pluck = asg.transient(FS, dur=1.0)
    out = bus.process_offline(pluck, 512)
    write_wav(f"{out_dir}/bus_automation.wav", FS, out)
    report["bus_automation"] = {
        "events": [{"sample_time": e.sample_time, "kind": e.kind,
                    "detail": e.detail} for e in bus.events],
        "dry_pullup_and_switch_same_sample": any(
            e.sample_time == beat and e.kind == "beat_automation"
            for e in bus.events),
        "peak_dry_gain": bus.peak_dry_gain,
        "final_dry_gain": bus.dry_gain,
        "final_factor": bus.active_factor,
    }

    # real per-branch differential recording (wet vs latency-compensated dry)
    dry, wet, diff = ParallelBus(
        FS, factors=(1, 8), drive=4.0, wet_mix=0.5).branch_recordings(pluck)
    report["bus_branch_differential"] = {
        "dry_rms": float(np.sqrt(np.mean(dry ** 2))),
        "wet_rms": float(np.sqrt(np.mean(wet ** 2))),
        "wet_minus_dry_rms": float(np.sqrt(np.mean(diff ** 2))),
        "same_length": bool(len(dry) == len(wet)),
    }


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output_acceptance")
    # clean directory: never reuse hand-edited or stale evidence
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    report = {"sample_rate": FS, "cutoff_hz": CUTOFF, "f0_hz": F0,
              "drive": DRIVE, "transfer": KIND.value,
              "independent_reference": "saturation.reference",
              "alias_bins_hz": ref.expected_alias_bins(F0, FS)}
    section_alias_control(report, out_dir)
    section_parallel_bus(report, out_dir)
    with open(os.path.join(out_dir, "acceptance_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n[acceptance] clean evidence written to {out_dir}")


if __name__ == "__main__":
    main()
