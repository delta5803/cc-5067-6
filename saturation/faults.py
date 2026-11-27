"""Controlled fault injection for the acceptance campaign.

Two *independent* switches exercise the two failure modes the acceptance
suite must tell apart:

1. ``FaultyMultirateEngine`` — defects inside the resampling chain:
   * ``remove_antialias`` replaces the downsampling filter with a pure delay
     (same reported latency, no attenuation): fold/mirror energy floods the
     band while the timing report is unchanged;
   * ``decimation_phase`` rolls the filtered high-rate stream by k ticks
     before decimation: the retained sample grid slips by a sub-sample, so
     the impulse centroid moves without changing the reported latency.

2. ``MisalignedParallelBus`` — a path-alignment defect: the dry branch is
   offset by ``misalign`` samples while the wet branch keeps the reported
   latency. The spectrum still looks like a healthy saturation, but the
   parallel sum develops comb-filter notches.

Faults default OFF so every wrapper behaves bit-for-bit like the production
class unless a switch is explicitly set.
"""
from __future__ import annotations

import numpy as np

from .resample import MultirateEngine
from .parallel_bus import ParallelBus


class FaultyMultirateEngine(MultirateEngine):
    def __init__(self, *args, remove_antialias: bool = False,
                 decimation_phase: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_antialias = bool(remove_antialias)
        self.decimation_phase = int(decimation_phase)
        if self.remove_antialias and self.down is not None:
            # pure delay of identical length: delay report unchanged, no LPF
            n = len(self.down.taps)
            identity = np.zeros(n)
            identity[(n - 1) // 2] = 1.0
            self.down.taps = identity
            self.down.reset()

    @property
    def effective_decimation_phase(self) -> int:
        return self.decimation_phase

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.factor == 1 or self.decimation_phase == 0:
            return super().process(x)
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        L = self.factor
        up_in = np.zeros(n * L, dtype=np.float64)
        up_in[::L] = x
        filtered = self.up.process(up_in)
        saturated = self.saturator.process(filtered)
        lp = self.down.process(saturated)
        # Decimation-grid fault: the downsampler retains every Lth sample from
        # a rolled high-rate stream. With the anti-alias filter in place the
        # base-rate alias level is physically unchanged, but the retained grid
        # slips by a sub-sample, moving the measured impulse centroid while the
        # reported latency stays at its design value.
        shift = int(self.decimation_phase)
        if shift > 0:
            lp = np.concatenate([np.zeros(shift), lp[:-shift]])
        elif shift < 0:
            lp = np.concatenate([lp[-shift:], np.zeros(-shift)])
        dec = lp[::L]
        if len(dec) < n:
            dec = np.concatenate([dec, np.zeros(n - len(dec))])
        return dec[:n]


class MisalignedParallelBus(ParallelBus):
    """Parallel bus whose dry compensation is deliberately off by k samples."""

    def __init__(self, *args, misalign: int = 0, **kwargs):
        super().__init__(*args, dry_delay_offset=int(misalign), **kwargs)
        self.misalign = int(misalign)
