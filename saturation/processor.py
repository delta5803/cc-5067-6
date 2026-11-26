"""Streaming saturation processor with dynamic oversampling-ratio switching.

The host only has a *short* double-processing budget: while a new oversampling
branch warms up, both branches must run on the same input. This module encodes
the warm-up / defer / fallback decision and guarantees sample-accurate output
through equal-gain crossfades. Every branch is delayed to one common constant
latency so a ratio switch never shifts the time axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .resample import MultirateEngine
from .transfer import Transfer


class SwitchDecision(str, Enum):
    NONE = "none"
    WARM_UP = "warm_up"        # idle branch started during silence (full budget)
    CROSSFADE = "crossfade"    # budget covers a short double run
    DEFER = "defer"            # block too loud and budget too small: wait
    FALLBACK = "fallback"      # forced switch without warm-up (pseudo-transient risk)


@dataclass
class SwitchEvent:
    block_index: int
    sample_time: int
    old_factor: int
    new_factor: int
    decision: SwitchDecision
    rms_db: float
    double_run_samples: int


@dataclass
class _DelayLine:
    delay: int
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.delay == 0:
            return x
        stream = np.concatenate([self.buf, x])
        out, self.buf = stream[: len(x)], stream[len(x):]
        return out


class SaturationProcessor:
    def __init__(self, base_rate: float, factors: tuple[int, ...] = (1, 2, 4, 8),
                 drive: float = 4.0, kind: str | Transfer = Transfer.TANH,
                 double_run_budget: int = 2048, crossfade_ms: float = 8.0,
                 idle_db: float = -60.0, cutoff_hz: float | None = None,
                 saturator_kwargs: dict | None = None):
        self.base_rate = float(base_rate)
        self.factors = tuple(sorted(set(int(f) for f in factors)))
        self.double_run_budget = int(double_run_budget)
        self.crossfade_n = int(base_rate * crossfade_ms / 1000.0)
        self.idle_db = float(idle_db)
        self._engines: dict[int, MultirateEngine] = {
            f: MultirateEngine(f, base_rate, drive=drive, kind=kind,
                               cutoff_hz=cutoff_hz,
                               saturator_kwargs=saturator_kwargs)
            for f in self.factors
        }
        # constant latency = slowest branch; every branch is padded to it
        self.max_delay = max(e.delay_base for e in self._engines.values())
        self._delay_lines = {f: _DelayLine(self.max_delay - e.delay_base)
                             for f, e in self._engines.items()}
        self.active_factor = self.factors[0]
        self._pending_factor: int | None = None
        self._xfade_pos = -1  # <0 means no crossfade
        self._block_index = 0
        self._sample_time = 0
        self.events: list[SwitchEvent] = []

    # ------------------------------------------------------------------ #
    def set_drive(self, drive: float) -> None:
        # parameter changes enter all branches smoothly (per-sample smoother)
        for e in self._engines.values():
            e.set_drive(drive)

    def request_factor(self, factor: int) -> SwitchDecision:
        """Ask for a ratio switch; the actual decision happens per block."""
        factor = int(factor)
        if factor not in self._engines or factor == self.active_factor:
            self._pending_factor = None
            return SwitchDecision.NONE
        self._pending_factor = factor
        return SwitchDecision.DEFER

    # ------------------------------------------------------------------ #
    def _rms_db(self, x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x * x) + 1e-20))
        return 20.0 * np.log10(rms + 1e-20)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        active = self._engines[self.active_factor]
        y_active = self._delay_lines[self.active_factor].push(active.process(x))

        if self._xfade_pos < 0 and self._pending_factor is not None:
            self._begin_switch(x, n)

        if self._xfade_pos >= 0:
            new_factor = self._pending_factor
            y_new = self._delay_lines[new_factor].push(
                self._engines[new_factor].process(x))
            remaining = self.crossfade_n - self._xfade_pos
            m = min(n, remaining)
            ramp_in = np.linspace(0.0, 1.0, m, endpoint=False)
            # equal-gain (constant power) crossfade keeps level steady
            g_new = np.sin(ramp_in * np.pi / 2.0)
            g_old = np.cos(ramp_in * np.pi / 2.0)
            y_active = y_active.copy()
            y_active[:m] = g_old * y_active[:m] + g_new * y_new[:m]
            if m < n:
                y_active[m:] = y_new[m:]
            self._xfade_pos += n
            if self._xfade_pos >= self.crossfade_n:
                self._xfade_pos = -1
                self.active_factor = new_factor
                self._pending_factor = None
        self._block_index += 1
        self._sample_time += n
        return y_active

    def _begin_switch(self, x: np.ndarray, n: int) -> None:
        target = self._pending_factor
        target_engine = self._engines[target]
        rms_db = self._rms_db(x)
        cost = n * target  # high-rate work expressed in base-rate equivalents

        if rms_db < self.idle_db:
            # silence: warm the idle branch from zero for free (no audible risk);
            # the crossfade then starts immediately on this (silent) block
            target_engine.warm_up(n)
            decision = SwitchDecision.WARM_UP
            double_run = 0
            self._start_crossfade(target, x, n, rms_db, decision, double_run)
            return

        if cost <= self.double_run_budget:
            # loud block but the host can afford the short double run.
            # Warm-up is skipped (state starts at zero); crossfade hides the
            # cold-start transient of the branch's internal memory.
            decision = SwitchDecision.CROSSFADE
            double_run = cost
            self._start_crossfade(target, x, n, rms_db, decision, double_run)
            return

        # loud and budget exhausted this block: stay put and retry later
        self.events.append(SwitchEvent(
            self._block_index, self._sample_time, self.active_factor, target,
            SwitchDecision.DEFER, rms_db, 0))

    def _start_crossfade(self, target: int, x: np.ndarray, n: int,
                         rms_db: float, decision: SwitchDecision,
                         double_run: int) -> None:
        self._xfade_pos = 0
        self.events.append(SwitchEvent(
            self._block_index, self._sample_time, self.active_factor, target,
            decision, rms_db, double_run))

    def force_factor(self, factor: int) -> SwitchEvent:
        """Hard switch with no warm-up / crossfade budget (documented fallback)."""
        factor = int(factor)
        if factor not in self._engines or factor == self.active_factor:
            return SwitchEvent(self._block_index, self._sample_time,
                               self.active_factor, factor,
                               SwitchDecision.NONE, -np.inf, 0)
        old = self.active_factor
        self.active_factor = factor
        self._pending_factor = None
        self._xfade_pos = -1
        ev = SwitchEvent(self._block_index, self._sample_time, old, factor,
                         SwitchDecision.FALLBACK, self._rms_db(
                             np.zeros(1)), 0)
        self.events.append(ev)
        return ev

    # ------------------------------------------------------------------ #
    def flush(self) -> np.ndarray:
        """Drain every branch; return the active branch's valid tail, padded
        so total output length equals input length + max_delay."""
        tails = {}
        for f, eng in self._engines.items():
            tail = eng.flush()
            tails[f] = self._delay_lines[f].push(tail)
        out = tails[self.active_factor]
        if len(out) < self.max_delay:
            out = np.concatenate([out, np.zeros(self.max_delay - len(out))])
        return out[: self.max_delay]

    def process_offline(self, x: np.ndarray, block_size: int) -> np.ndarray:
        """Chunked run including flush; used for block-size invariance checks."""
        outs = []
        for start in range(0, len(x), block_size):
            outs.append(self.process_block(x[start:start + block_size]))
        outs.append(self.flush())
        return np.concatenate(outs)
