"""Rendering backend for the mastering audition panel.

The panel never invents audio: every result is produced by the same
SaturationProcessor / MultirateEngine used by the test suite. Failures are
returned as structured errors so the UI can surface them on the saturation
controls instead of failing silently.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saturation.analysis import spectrum, harmonic_peaks, thd_db, dc
from saturation.processor import SaturationProcessor
from saturation.resample import MultirateEngine
from saturation.transfer import Transfer, transfer_names
from saturation import signals
from saturation import acceptance as ac
from saturation.faults import FaultyMultirateEngine
from saturation import reference as ref

FS = 48000
MAX_SECONDS = 6.0
SOURCES = ("pluck", "sweep", "multitone", "asymmetric", "overrange")


def build_source(name: str) -> np.ndarray:
    if name == "pluck":
        return signals.pluck(FS, f0=220.0, dur=3.0)
    if name == "sweep":
        return signals.log_sweep(FS, dur=3.0, amp=0.7)
    if name == "multitone":
        x, _ = signals.multitone(FS, dur=2.0)
        return x
    if name == "asymmetric":
        return signals.asymmetric_wave(FS, dur=2.0)
    if name == "overrange":
        return signals.overrange(FS, freq=180.0, dur=2.0, amp=8.0)
    raise ValueError(f"unknown source {name!r}")


def _wav_bytes(x: np.ndarray, fs: int = FS) -> bytes:
    """16-bit PCM WAV in memory (no soundcard / PortAudio needed)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 0.95:
        x = x * (0.95 / m)
    pcm = np.clip(x, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    data = pcm16.tobytes()
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(data)
    return buf.getvalue()


def _harmonic_plot(x: np.ndarray, y: np.ndarray, factor: int,
                   drive: float, kind: str) -> str:
    n = min(len(x), len(y), FS * 4)
    fi, mi = spectrum(x[:n], FS)
    fo, mo = spectrum(y[:n], FS)
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.plot(fi / 1000, 20 * np.log10(mi + 1e-12), color="#9aa5b1",
            lw=0.8, label="dry input")
    ax.plot(fo / 1000, 20 * np.log10(mo + 1e-12), color="#d64545",
            lw=0.9, label=f"saturated (x{factor})")
    ax.set_xlim(0, FS / 2000)
    ax.set_ylim(-110, 5)
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("dB")
    ax.set_title(f"Harmonic distribution — drive {drive:g}, {kind}, "
                 f"oversampling x{factor}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class RenderResult:
    ok: bool
    error: str | None
    wav_b64: str
    dry_b64: str
    plot_b64: str
    metrics: dict


def render(source: str, drive: float, factor: int, kind: str,
           block_size: int = 1024) -> RenderResult:
    try:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if not (0.05 <= float(drive) <= 32.0) or not math.isfinite(drive):
            raise ValueError("drive out of range [0.05, 32]")
        if int(factor) not in (1, 2, 4, 8):
            raise ValueError("factor must be 1, 2, 4 or 8")
        if kind not in transfer_names():
            raise ValueError(f"kind must be one of {transfer_names()}")

        x = build_source(source)[: int(FS * MAX_SECONDS)]
        proc = SaturationProcessor(FS, factors=(int(factor),), drive=float(drive),
                                   kind=Transfer(kind))
        outs = []
        for s in range(0, len(x), int(block_size)):
            outs.append(proc.process_block(x[s:s + block_size]))
        outs.append(proc.flush())
        y = np.concatenate(outs)
        # align wet to dry using the common constant latency
        d = proc.max_delay
        y_aligned = y[d:d + len(x)]
        if len(y_aligned) < len(x):
            y_aligned = np.pad(y_aligned, (0, len(x) - len(y_aligned)))

        fi, mi = spectrum(x[: min(FS, len(x))], FS)
        band = (fi >= 80.0) & (fi <= 4000.0)
        f0_est = float(fi[band][np.argmax(mi[band])])
        freqs, mag = spectrum(y_aligned[: min(FS, len(y_aligned))], FS)
        peaks = harmonic_peaks(freqs, mag, f0_est, 8)
        metrics = {
            "f0_hz": round(f0_est, 1),
            "thd_db": round(float(thd_db(peaks)), 2),
            "dc_out": round(float(dc(y_aligned)), 6),
            "peak_out": round(float(np.max(np.abs(y_aligned))), 4),
            "rms_db": round(float(20 * math.log10(
                np.sqrt(np.mean(y_aligned ** 2)) + 1e-12)), 2),
            "latency_samples": int(d),
            "factor": int(factor),
            "drive": float(drive),
            "transfer": kind,
            "harmonics": [round(float(p), 5) for p in peaks[:6]],
        }
        return RenderResult(
            ok=True, error=None,
            wav_b64=base64.b64encode(_wav_bytes(y_aligned)).decode("ascii"),
            dry_b64=base64.b64encode(_wav_bytes(x)).decode("ascii"),
            plot_b64=_harmonic_plot(x, y_aligned, int(factor),
                                    float(drive), kind),
            metrics=metrics,
        )
    except Exception as exc:  # structured failure fed back to the controls
        return RenderResult(
            ok=False, error=f"{type(exc).__name__}: {exc}",
            wav_b64="", dry_b64="", plot_b64="", metrics={})


# --------------------------------------------------------------------- #
# Regression sample: every number is produced live by the DSP core. There
# is no fixed/hard-coded harmonic data; the panel button calls this and the
# returned audio/spectrum come from fresh MultirateEngine renders.
# --------------------------------------------------------------------- #
REGRESSION_F0 = 3400.0
REGRESSION_CUTOFF = min(FS * 0.45, 18000.0)


def regression_sample(drive: float = 8.0, factor: int = 8,
                      kind: str = "hard_clip") -> dict:
    """Render the alias-control regression live and return measured evidence.

    Compares a healthy engine against the two independent fault switches
    (anti-alias removal, decimation-grid error) using the same independent
    reference the acceptance suite uses. Intended for the panel button.
    """
    if not (0.05 <= float(drive) <= 32.0):
        raise ValueError("drive out of range [0.05, 32]")
    if int(factor) not in (2, 4, 8):
        raise ValueError("regression factor must be 2, 4 or 8")
    if kind not in transfer_names():
        raise ValueError(f"kind must be one of {transfer_names()}")

    t = np.arange(FS) / FS
    x = 0.9 * np.sin(2.0 * np.pi * REGRESSION_F0 * t)

    def render(engine):
        full = np.concatenate([engine.process(x), engine.flush()])
        return full[engine.delay_base:engine.delay_base + FS]

    healthy_eng = MultirateEngine(factor, FS, drive=drive, kind=kind,
                                  cutoff_hz=REGRESSION_CUTOFF)
    noaa_eng = FaultyMultirateEngine(factor, FS, drive=drive, kind=kind,
                                     cutoff_hz=REGRESSION_CUTOFF,
                                     remove_antialias=True)
    grid_eng = FaultyMultirateEngine(factor, FS, drive=drive, kind=kind,
                                     cutoff_hz=REGRESSION_CUTOFF,
                                     decimation_phase=4)
    y_healthy = render(healthy_eng)
    y_noaa = render(noaa_eng)
    y_grid = render(grid_eng)

    ev_healthy = ac.measure_alias(healthy_eng, x, FS, REGRESSION_F0, drive,
                                  kind, factor, REGRESSION_CUTOFF)
    ev_noaa = ac.measure_alias(noaa_eng, x, FS, REGRESSION_F0, drive, kind,
                               factor, REGRESSION_CUTOFF)
    ev_grid = ac.measure_alias(grid_eng, x, FS, REGRESSION_F0, drive, kind,
                               factor, REGRESSION_CUTOFF)
    bound = ref.alias_lower_bound_x2(REGRESSION_F0, drive, kind, FS,
                                     REGRESSION_CUTOFF)["bound_energy"]

    # live harmonic distribution of the healthy render (no canned data)
    freqs, mag = spectrum(y_healthy, FS)
    harmonics = harmonic_peaks(freqs, mag, REGRESSION_F0, 8)

    return {
        "ok": True,
        "params": {"drive": float(drive), "factor": int(factor),
                   "kind": kind, "f0_hz": REGRESSION_F0},
        "independent_bound_energy": bound,
        "healthy": {
            "alias_energy": ev_healthy.measured_alias_energy,
            "image_energy": ev_healthy.image_energy,
            "latency_samples": ev_healthy.latency_samples,
            "expected_latency": ev_healthy.expected_latency_samples,
            "harmonics": [float(p) for p in harmonics],
            "wav_b64": base64.b64encode(_wav_bytes(y_healthy)).decode(),
        },
        "fault_remove_antialias": {
            "alias_energy": ev_noaa.measured_alias_energy,
            "image_energy": ev_noaa.image_energy,
            "latency_samples": ev_noaa.latency_samples,
            "alias_gain_db": float(20 * np.log10(
                (ev_noaa.measured_alias_energy + 1e-20)
                / (ev_healthy.measured_alias_energy + 1e-20))),
            "wav_b64": base64.b64encode(_wav_bytes(y_noaa)).decode(),
        },
        "fault_decimation_grid": {
            "alias_energy": ev_grid.measured_alias_energy,
            "impulse_timing": ac.impulse_timing(grid_eng, FS),
            "healthy_impulse_timing": ac.impulse_timing(
                MultirateEngine(factor, FS, drive=drive, kind=kind,
                                cutoff_hz=REGRESSION_CUTOFF), FS),
            "latency_samples": ev_grid.latency_samples,
        },
        "dry_b64": base64.b64encode(_wav_bytes(x)).decode(),
    }
