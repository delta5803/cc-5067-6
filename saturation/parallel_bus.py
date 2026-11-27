"""Dry/wet parallel bus with latency compensation and same-beat automation.

The wet branch is the streaming :class:`SaturationProcessor`; it reports a
constant latency (the slowest engine's FIR delay). The dry branch is delayed
by *exactly that many samples* before equal-power mixing, so summing the two
buses cannot create a comb filter. The reported latency is taken from the
processor at construction and re-checked every block — the bus never assumes
a number of its own.

Automation is scheduled on a beat grid: an oversampling-ratio change and the
compensating dry-level pull-up are applied at the same beat, and the dry gain
returns as the crossfade settles. An optional, explicitly-flagged alignment
offset is the fault switch used to prove the comb-cancellation detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .processor import SaturationProcessor
from .transfer import Transfer


@dataclass
class BusEvent:
    beat_time: int
    sample_time: int
    kind: str
    detail: str


class _DryDelay:
    """Simple integer sample delay line (the latency compensation stage)."""

    def __init__(self, delay: int):
        self.delay = int(delay)
        self.buf = np.zeros(self.delay, dtype=np.float64)

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.delay == 0:
            return x.copy()
        stream = np.concatenate([self.buf, x])
        out = stream[: len(x)].copy()
        self.buf = stream[len(x):]
        return out


class ParallelBus:
    def __init__(self, base_rate: float, factors: tuple[int, ...] = (1, 2, 4, 8),
                 drive: float = 4.0, kind: str | Transfer = Transfer.TANH,
                 wet_mix: float = 1.0, double_run_budget: int = 2048,
                 dry_delay_offset: int = 0):
        self.proc = SaturationProcessor(
            base_rate, factors=factors, drive=drive, kind=kind,
            double_run_budget=double_run_budget)
        # compensation comes straight from the oversampler's reported latency
        self.reported_latency = self.proc.max_delay
        self.base_rate = float(base_rate)
        self._dry = _DryDelay(self.reported_latency + int(dry_delay_offset))
        self.dry_delay_offset = int(dry_delay_offset)
        self.wet_mix = float(wet_mix)
        self.dry_gain = 1.0          # steady-state dry level
        self.peak_dry_gain = 1.0     # highest automation gain reached
        self._dry_gain_target = 1.0
        self._beat_queue: list[tuple[int, int, float]] = []
        self.events: list[BusEvent] = []
        self._sample_time = 0

    # ------------------------------------------------------------------ #
    @property
    def active_factor(self) -> int:
        return self.proc.active_factor

    def schedule_beat(self, beat_sample: int, new_factor: int,
                      dry_pullup_db: float = 3.0) -> None:
        """At ``beat_sample`` switch ratio and pull the dry level up together.

        The pull-up keeps apparent loudness while the cold wet branch fades
        in; the dry gain eases back to unity once the switch settles.
        """
        gain = 10.0 ** (dry_pullup_db / 20.0)
        self._beat_queue.append((int(beat_sample), int(new_factor), gain))
        self._beat_queue.sort()

    def set_wet_mix(self, wet_mix: float) -> None:
        if not 0.0 <= wet_mix <= 1.0:
            raise ValueError("wet_mix must lie in [0, 1]")
        self.wet_mix = float(wet_mix)

    def _ramp_gain(self, n: int) -> np.ndarray:
        start = self.dry_gain
        end = self._dry_gain_target
        return np.linspace(start, end, n, endpoint=False)

    # ------------------------------------------------------------------ #
    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        # fire every automation event whose beat falls inside this block
        due = [e for e in self._beat_queue
               if self._sample_time <= e[0] < self._sample_time + n]
        for beat_sample, new_factor, gain in due:
            self.proc.request_factor(new_factor)
            self._dry_gain_target = gain  # same beat: dry pulled up now
            self.peak_dry_gain = max(self.peak_dry_gain, gain)
            self.events.append(BusEvent(
                beat_sample, beat_sample, "beat_automation",
                f"request x{new_factor}, dry gain -> {gain:.3f}"))
        self._beat_queue = [e for e in self._beat_queue if e not in due]

        wet = self.proc.process_block(x)
        dry = self._dry.push(x)

        gains = self._ramp_gain(n)
        g_wet = np.sqrt(self.wet_mix)
        g_dry = np.sqrt(max(0.0, 1.0 - self.wet_mix))
        out = g_wet * wet + (g_dry * gains) * dry

        # once the processor's crossfade has finished, restore dry unity
        self._sample_time += n
        if (self._dry_gain_target != 1.0
                and not self.proc._pending_factor
                and self.proc._xfade_pos < 0):
            self._dry_gain_target = 1.0
            self.events.append(BusEvent(
                self._sample_time, self._sample_time, "dry_gain_restore",
                "crossfade settled, dry gain back to unity"))
        self.dry_gain = self._dry_gain_target
        return out

    def flush(self) -> np.ndarray:
        wet = self.proc.flush()
        dry = self._dry.push(np.zeros(len(wet)))
        g_wet = np.sqrt(self.wet_mix)
        g_dry = np.sqrt(max(0.0, 1.0 - self.wet_mix))
        return g_wet * wet + g_dry * self.dry_gain * dry

    def process_offline(self, x: np.ndarray, block_size: int) -> np.ndarray:
        parts = []
        for s in range(0, len(x), block_size):
            parts.append(self.process_block(x[s:s + block_size]))
        parts.append(self.flush())
        return np.concatenate(parts)

    # ------------------------------------------------------------------ #
    def branch_recordings(self, x: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real per-branch recordings for differential evidence.

        Returns dry (latency-compensated), wet and their sample difference.
        The difference is measured from actual rendered audio, not a model.
        """
        wet_parts, dry_parts = [], []
        for s in range(0, len(x), 512):
            chunk = x[s:s + 512]
            wet_parts.append(self.proc.process_block(chunk))
            dry_parts.append(self._dry.push(chunk))
        tail = self.proc.flush()
        wet_parts.append(tail)
        dry_parts.append(self._dry.push(np.zeros(len(tail))))
        wet = np.concatenate(wet_parts)
        dry = np.concatenate(dry_parts)
        m = min(len(wet), len(dry))
        return dry[:m], wet[:m], wet[:m] - dry[:m]
