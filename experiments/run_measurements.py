"""Offline measurement campaign.

Runs the real multirate chain on synthetic test material and writes WAVs,
spectra and a JSON metrics report into output/. Also builds the two dynamic
switching scenarios requested:

  A. long silence, then high factor enabled (warm-up path)
  B. sustained pluck, same switch while sounding (crossfade / defer path)

Time alignment is checked against a dry-side beat grid and a full-length
high-factor reference recording.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saturation import signals
from saturation.analysis import (
    spectrum, harmonic_peaks, thd_db, dc, alias_energy, alias_bins,
    passband_retention, transient_position, band_energy, peak_at,
    cross_correlation_lag,
)
from saturation.processor import SaturationProcessor, SwitchDecision
from saturation.resample import MultirateEngine
from saturation.transfer import Transfer, transfer_names
from saturation.wavio import write_wav

FS = 48000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "output")
DRIVE = 6.0


def _engine(factor: int, drive: float = DRIVE,
            kind: Transfer = Transfer.TANH) -> MultirateEngine:
    return MultirateEngine(factor, FS, drive=drive, kind=kind)


def run_offline(x: np.ndarray, factor: int, drive: float = DRIVE,
                kind: Transfer = Transfer.TANH) -> np.ndarray:
    eng = _engine(factor, drive, kind)
    return np.concatenate([eng.process(x), eng.flush()])


def trim_delay(y: np.ndarray, factor: int) -> np.ndarray:
    d = _engine(factor).delay_base
    return y[d:]


# --------------------------------------------------------------------- #
def experiment_single_tone(report: dict) -> None:
    f0 = 3400.0  # harmonics fold back inside the audible band at FS=48k
    x = signals.sine(FS, f0, 1.0, amp=0.9)
    rows = {}
    spectra = {}
    for factor in (1, 2, 4, 8):
        y = trim_delay(run_offline(x, factor), factor)[: len(x)]
        write_wav(f"{OUT}/tone_f{factor}.wav", FS, y)
        freqs, mag = spectrum(y, FS)
        spectra[factor] = (freqs, mag)
        peaks = harmonic_peaks(freqs, mag, f0, 8)
        rows[factor] = {
            "thd_db": thd_db(peaks),
            "alias_energy": alias_energy(freqs, mag, f0, FS),
            "dc": dc(y),
            "peak": float(np.max(np.abs(y))),
            "fold_frequencies_hz": alias_bins(f0, FS, 8),
        }
    report["single_tone"] = {"f0_hz": f0, "factors": rows}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for factor, (freqs, mag) in ((1, spectra[1]), (8, spectra[8])):
        m = 20 * np.log10(mag + 1e-12)
        ax.plot(freqs / 1000, m, label=f"factor {factor}", lw=0.9, alpha=0.85)
    for fa in alias_bins(f0, FS, 8):
        ax.axvline(fa / 1000, color="r", ls=":", lw=0.7)
    ax.set_xlabel("frequency [kHz]"); ax.set_ylabel("magnitude [dB]")
    ax.set_title(f"Single-tone {f0:.0f} Hz, drive {DRIVE}: aliases (dotted) "
                 "factor 1 vs 8")
    ax.set_xlim(0, FS / 2000); ax.set_ylim(-120, 5)
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/single_tone.png", dpi=130)
    plt.close(fig)


def experiment_sweep(report: dict) -> None:
    x = signals.log_sweep(FS, dur=2.0, amp=0.7)
    result = {}
    for factor in (1, 8):
        y = trim_delay(run_offline(x, factor), factor)[: len(x)]
        write_wav(f"{OUT}/sweep_f{factor}.wav", FS, y)
        f_in, m_in = spectrum(x, FS)
        f_out, m_out = spectrum(y, FS)
        result[factor] = {
            "passband_retention_18k": passband_retention(
                f_in, m_in, f_out, m_out, 18000.0),
            "highband_energy_10k_20k": band_energy(f_out, m_out, 10000, 20000),
        }
    report["sweep"] = result

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for factor, ax in ((1, axes[0]), (8, axes[1])):
        y = trim_delay(run_offline(x, factor), factor)[: len(x)]
        f, t, Sxx = __import__("scipy.signal", fromlist=["spectrogram"]).spectrogram(
            y, FS, nperseg=2048)
        ax.pcolormesh(t, f / 1000, 10 * np.log10(Sxx + 1e-12),
                      shading="auto", vmin=-100, vmax=-20, cmap="magma")
        ax.set_ylabel("kHz"); ax.set_title(f"log-sweep spectrogram, factor {factor}")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/sweep.png", dpi=130); plt.close(fig)


def experiment_multitone(report: dict) -> None:
    # 7 kHz + 11 kHz sum to 18 kHz (in-band); their 2nd harmonics at 14/22 kHz
    # fold (22 kHz -> 2 kHz at 48 k Nyquist), exercising real alias paths.
    x, freqs_in = signals.multitone(
        FS, dur=1.0,
        freqs=(700.0, 3000.0, 7000.0, 11000.0, 15000.0))
    rows = {}
    for factor in (1, 8):
        y = trim_delay(run_offline(x, factor), factor)[: len(x)]
        write_wav(f"{OUT}/multitone_f{factor}.wav", FS, y)
        f, mag = spectrum(y, FS)
        intermod = 0.0
        # intermodulation products around sums/differences of input tones
        for i, fa in enumerate(freqs_in):
            for fb in freqs_in[i + 1:]:
                for fc in (abs(fa - fb), fa + fb):
                    if not any(abs(fc - t) < 8.0 for t in freqs_in):
                        intermod += peak_at(f, mag, fc, 6.0) ** 2
        # explicit folded locations of harmonics above Nyquist, excluding any
        # frequency already occupied by an input tone (those are not aliases)
        folds = []
        for f0 in freqs_in:
            for fa in alias_bins(f0, FS, n_harm=4):
                if all(abs(fa - t) > 25.0 for t in freqs_in):
                    folds.append(fa)
        fold_energy = float(np.sqrt(sum(
            peak_at(f, mag, fa, 8.0) ** 2 for fa in set(folds))))
        rows[factor] = {
            "intermod_energy": float(np.sqrt(intermod)),
            "fold_energy_new_lines_only": fold_energy,
            "fold_locations_hz": sorted(set(round(v, 1) for v in folds)),
            "dc": dc(y),
        }
    report["multitone"] = rows
    # dedicated two-tone intermodulation / folding test: high-level tones whose
    # 3rd-order products and folded harmonics sit well above the noise floor
    t = np.arange(FS) / FS
    # 13k tone: 2nd harmonic 26k folds to 22k, 3rd 39k folds to 9k at FS=48k;
    # neither location holds an input tone, so both are unambiguous aliases.
    # At factor 8 (384 kHz) those harmonics stay in-band and are filtered out.
    two = 0.9 * np.sin(2 * np.pi * 13000 * t)
    two_rows = {}
    for factor in (1, 8):
        y = trim_delay(run_offline(two, factor, drive=12.0,
                                   kind=Transfer.HARD_CLIP), factor)[: len(two)]
        write_wav(f"{OUT}/twotone_f{factor}.wav", FS, y)
        f, mag = spectrum(y, FS)
        probes = {
            "fold_26k_to_22k": 22000.0,
            "fold_39k_to_9k": 9000.0,
            "inband_fundamental_13k": 13000.0,
        }
        two_rows[factor] = {k: peak_at(f, mag, v, 8.0) for k, v in probes.items()}
    report["two_tone"] = two_rows


def experiment_asymmetric(report: dict) -> None:
    x = signals.asymmetric_wave(FS, dur=1.0)
    rows = {}
    for kind in (Transfer.TANH, Transfer.ASYM):
        y = trim_delay(run_offline(x, factor=4, kind=kind), 4)[: len(x)]
        write_wav(f"{OUT}/asym_{kind.value}.wav", FS, y)
        rows[kind.value] = {"dc_out": dc(y), "input_dc": dc(x),
                            "peak": float(np.max(np.abs(y)))}
    report["asymmetric"] = rows


def experiment_dynamic_switch(report: dict) -> None:
    block = 512
    # scenario A: one second of silence before the pluck, request high factor
    pluck = signals.pluck(FS, f0=220.0, dur=2.0)
    sig_a = signals.silence_then(FS, pluck, silence_s=1.0)
    # scenario B: sustained plucks with no deliberate silence gap; the second
    # phrase decays into a quiet tail where the deferred switch can warm up
    sig_b = np.concatenate([
        signals.pluck(FS, f0=196.0, dur=1.5, decay=2.2),
        signals.pluck(FS, f0=261.6, dur=2.2, decay=4.5) * 0.9,
        np.zeros(int(0.5 * FS)),
    ])

    def scenario(sig: np.ndarray, name: str, budget: int):
        proc = SaturationProcessor(FS, factors=(1, 8), drive=DRIVE,
                                   double_run_budget=budget)
        ref = SaturationProcessor(FS, factors=(8,), drive=DRIVE)
        out, ref_out = [], []
        requested = False
        for i, s in enumerate(range(0, len(sig), block)):
            chunk = sig[s:s + block]
            if not requested and i == 2:
                proc.request_factor(8)
                requested = True
            out.append(proc.process_block(chunk))
            ref_out.append(ref.process_block(chunk))
        out.append(proc.flush()); ref_out.append(ref.flush())
        y = np.concatenate(out); y_ref = np.concatenate(ref_out)
        write_wav(f"{OUT}/{name}_switch.wav", FS, y)
        write_wav(f"{OUT}/{name}_ref_f8.wav", FS, y_ref)
        # timing check 1: envelope transients (dry-side beat grid proxy)
        env_dt = transient_position(y) - transient_position(y_ref)
        # timing check 2: after the switch has fully completed both branches
        # run the same factor-8 chain; cross-correlation there must show zero lag
        last_event = proc.events[-1]
        settle = last_event.sample_time + proc.crossfade_n
        n = min(len(y), len(y_ref))
        # use the longest fully-settled window available after the switch
        win = min(FS, n - settle)
        if win >= 4096:
            xcorr_dt = cross_correlation_lag(
                y_ref[settle:settle + win], y[settle:settle + win],
                max_lag=64)
        else:
            xcorr_dt = cross_correlation_lag(y_ref[:n], y[:n], max_lag=128)
        events = [
            dict(block=e.block_index, t_s=round(e.sample_time / FS, 4),
                 old=e.old_factor, new=e.new_factor,
                 decision=e.decision.value, rms_db=round(e.rms_db, 1),
                 double_run=e.double_run_samples)
            for e in proc.events
        ]
        return {"envelope_offset_samples": int(env_dt),
                "xcorr_offset_samples_vs_ref": int(xcorr_dt),
                "events": events,
                "final_factor": proc.active_factor}, y, y_ref

    ra, ya, yref_a = scenario(sig_a, "silence_then", budget=block)
    rb, yb, yref_b = scenario(sig_b, "sustained", budget=block)
    report["dynamic_switch"] = {"silence_then_high_factor": ra,
                                "sustained_pluck_switch": rb}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = np.arange(len(yb)) / FS
    ax.plot(t, yb, lw=0.7, label="streaming output (switch 1->8)")
    ax.plot(t, yref_b, lw=0.5, alpha=0.6,
            label="reference factor 8 entire time")
    labelled = set()
    for e in rb["events"]:
        if e["decision"] == "defer":
            ax.axvline(e["t_s"], color="#888", ls=":", lw=0.4)
            continue
        label = (f"switch ({e['decision']}) @ {e['t_s']:.2f}s"
                 if e["decision"] not in labelled else None)
        labelled.add(e["decision"])
        ax.axvline(e["t_s"], color="r", ls="--", lw=0.9, label=label)
    ax.set_xlabel("time [s]"); ax.set_ylabel("amplitude")
    ax.set_title("Sustained pluck: dynamic factor switch vs full high-rate "
                 "reference (grey dots = deferred attempts)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/dynamic_switch.png", dpi=130)
    plt.close(fig)


def experiment_drive_sweep(report: dict) -> None:
    x = signals.sine(FS, 1000.0, 0.5, amp=0.8)
    rows = []
    for drive in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        y = trim_delay(run_offline(x, factor=4, drive=drive), 4)[: len(x)]
        f, mag = spectrum(y, FS)
        rows.append({"drive": drive, "thd_db": thd_db(
            harmonic_peaks(f, mag, 1000.0, 8)), "peak": float(np.max(np.abs(y)))})
    report["drive_sweep"] = rows


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    report = {"sample_rate": FS, "drive_default": DRIVE,
              "transfers": transfer_names()}
    experiment_single_tone(report)
    experiment_sweep(report)
    experiment_multitone(report)
    experiment_asymmetric(report)
    experiment_dynamic_switch(report)
    experiment_drive_sweep(report)
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
