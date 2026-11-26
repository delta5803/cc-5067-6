"""Stateful saturation stage with internal memory.

The memory comes from input/output one-pole filters and a per-sample drive
smoother. Two independent MemorySaturator instances (e.g. a freshly created
oversampled branch versus one that has been processing signal) therefore
*do* produce different first-block behaviour, which the processor's warm-up
machinery has to account for.
"""
from __future__ import annotations

import numpy as np

from .transfer import Transfer, apply_scalar


class MemorySaturator:
    def __init__(self, drive: float = 1.0, kind: str | Transfer = Transfer.TANH,
                 sample_rate: float = 48000.0, in_hz: float = 12000.0,
                 out_hz: float = 12000.0, drive_tau_ms: float = 5.0):
        self.sample_rate = float(sample_rate)
        self.kind = kind.value if isinstance(kind, Transfer) else str(kind)
        self._drive = float(drive)
        self._drive_target = float(drive)
        self.in_hz = float(in_hz)
        self.out_hz = float(out_hz)
        self.drive_tau_ms = float(drive_tau_ms)
        self._z_in = 0.0
        self._z_out = 0.0
        self._a_in = self._coef(self.in_hz)
        self._a_out = self._coef(self.out_hz)
        self._a_drive = 1.0 - np.exp(-1.0 /
                                     (sample_rate * drive_tau_ms / 1000.0))

    def _coef(self, cutoff_hz: float) -> float:
        # one-pole low-pass coefficient
        return float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz / self.sample_rate))

    @property
    def drive(self) -> float:
        return self._drive_target

    def set_drive(self, drive: float) -> None:
        if drive <= 0.0 or not np.isfinite(drive):
            raise ValueError("drive must be a positive finite number")
        self._drive_target = float(drive)

    def reset(self) -> None:
        self._z_in = 0.0
        self._z_out = 0.0
        self._drive = self._drive_target

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        z_in, z_out = self._z_in, self._z_out
        drive, drive_target = self._drive, self._drive_target
        a_in, a_out, a_d = self._a_in, self._a_out, self._a_drive
        kind = self.kind
        for i, xi in enumerate(x):
            drive += (drive_target - drive) * a_d
            z_in += a_in * (xi - z_in)
            wet = apply_scalar(z_in, drive, kind)
            z_out += a_out * (wet - z_out)
            y[i] = z_out
        self._z_in, self._z_out, self._drive = z_in, z_out, drive
        return y

    def warm_up(self, n_samples: int) -> None:
        """Run zero input so the state matches a branch idle on silence."""
        if n_samples > 0:
            self.process(np.zeros(int(n_samples), dtype=np.float64))
