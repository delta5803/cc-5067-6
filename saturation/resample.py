"""Real multirate up/downsampling chain.

Upsampling is zero-stuffing followed by a Kaiser-windowed low-pass (never
sample-and-hold / repeated samples). Downsampling is the same low-pass
class of filter before decimation. All FIR filters keep their delay-line
state across process blocks, so chunked processing matches one offline pass.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import kaiserord, firwin

from .saturator import MemorySaturator
from .transfer import Transfer


def design_kaiser_lpf(cutoff_hz: float, sample_rate: float,
                      ripple_db: float = 60.0, multiple_of: int = 1) -> np.ndarray:
    """Kaiser-windowed FIR low-pass.

    With ``multiple_of=L`` the number of taps is forced so that the group
    delay (N-1)/2 is an integer multiple of L: a hard requirement for
    sample-accurate L-fold (de)interleaving without fractional phases.
    """
    nyq = sample_rate / 2.0
    if not 0.0 < cutoff_hz < nyq:
        raise ValueError("cutoff must lie inside (0, Nyquist)")
    width = max(cutoff_hz * 0.1, min(500.0, (nyq - cutoff_hz) * 0.4))
    n_taps, beta = kaiserord(ripple_db, width / (sample_rate / 2.0))
    if n_taps % 2 == 0:
        n_taps += 1  # type-1 FIR (integer group delay)
    if multiple_of > 1:
        # (N-1)/2 must be a multiple of multiple_of -> N-1 multiple of 2L
        rem = (n_taps - 1) % (2 * multiple_of)
        if rem:
            n_taps += 2 * multiple_of - rem
    return firwin(n_taps, cutoff_hz, window=("kaiser", beta), fs=sample_rate)


class _FIR:
    """Streaming FIR with persistent state and exact delay() reporting."""

    def __init__(self, taps: np.ndarray):
        self.taps = np.asarray(taps, dtype=np.float64)
        self.state = np.zeros(len(self.taps) - 1, dtype=np.float64)

    def reset(self) -> None:
        self.state.fill(0.0)

    @property
    def delay(self) -> int:
        return (len(self.taps) - 1) // 2

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if len(self.state) == 0:
            return np.convolve(x, self.taps, mode="full")[: len(x)]
        buf = np.concatenate([self.state, x])
        y = np.convolve(buf, self.taps, mode="valid")
        self.state = buf[-(len(self.taps) - 1):].copy()
        return y

    def flush(self) -> np.ndarray:
        """Drain the delay line with zero input."""
        if len(self.state) == 0:
            return np.empty(0)
        tail = np.zeros(len(self.taps) - 1)
        out = self.process(tail)
        return out


class MultirateEngine:
    """factor-1 (no oversampling) or factor-L oversampling saturation chain.

    Base rate fs -> upsample by L (zero-stuff + LPF) -> saturate at L*fs ->
    low-pass -> decimate by L. All filter/saturator states persist per block.
    """

    def __init__(self, factor: int, base_rate: float, drive: float,
                 kind: str | Transfer = Transfer.TANH,
                 cutoff_hz: float | None = None, ripple_db: float = 60.0,
                 saturator_kwargs: dict | None = None):
        if factor < 1:
            raise ValueError("oversampling factor must be >= 1")
        self.factor = int(factor)
        self.base_rate = float(base_rate)
        self.high_rate = base_rate * factor
        self.cutoff_hz = float(cutoff_hz) if cutoff_hz else min(base_rate * 0.45, 18000.0)
        self.kind = kind.value if isinstance(kind, Transfer) else str(kind)
        kw = dict(saturator_kwargs or {})
        # saturator time constants are frequency parameters, rescale with rate
        self.saturator = MemorySaturator(
            drive=drive, kind=self.kind, sample_rate=self.high_rate, **kw)
        if factor == 1:
            self.up = None
            self.down = None
            self._up_delay = 0
            self._down_delay = 0
        else:
            # anti-imaging filter at high rate, gain L to compensate zero stuffing
            up_taps = (design_kaiser_lpf(self.cutoff_hz, self.high_rate,
                                         ripple_db, multiple_of=factor)
                       * factor)
            down_taps = design_kaiser_lpf(self.cutoff_hz, self.high_rate,
                                          ripple_db, multiple_of=factor)
            self.up = _FIR(up_taps)
            self.down = _FIR(down_taps)
            # delays expressed in base-rate samples (exact multiples of L)
            self._up_delay = self.up.delay // factor
            self._down_delay = self.down.delay // factor

    @property
    def drive(self) -> float:
        return self.saturator.drive

    def set_drive(self, drive: float) -> None:
        self.saturator.set_drive(drive)

    @property
    def delay_base(self) -> int:
        """Total base-rate sample delay through the chain."""
        return self._up_delay + self._down_delay

    def reset(self) -> None:
        if self.up is not None:
            self.up.reset()
            self.down.reset()
        self.saturator.reset()

    def warm_up(self, base_samples: int) -> None:
        """Idle the whole chain on zeros for base_samples base-rate ticks."""
        if base_samples > 0:
            self.process(np.zeros(int(base_samples), dtype=np.float64))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.factor == 1:
            return self.saturator.process(x)
        n = len(x)
        up_in = np.zeros(n * self.factor, dtype=np.float64)
        up_in[:: self.factor] = x
        filtered = self.up.process(up_in)
        saturated = self.saturator.process(filtered)
        lp = self.down.process(saturated)
        # taps are designed with group delay aligned to L, so phase is zero
        return lp[:: self.factor][:n]

    def flush(self) -> np.ndarray:
        """Drain all FIR states; returns the valid tail in base rate,
        trimmed to the engine's base-rate delay."""
        if self.factor == 1:
            # one-poles only: no finite FIR tail to drain
            return np.empty(0)
        n_hi = (len(self.up.taps) - 1) + (len(self.down.taps) - 1)
        f_up = self.up.process(np.zeros(n_hi, dtype=np.float64))
        sat = self.saturator.process(f_up)
        lp = self.down.process(sat)
        dec = lp[:: self.factor]
        return dec[: self.delay_base]
