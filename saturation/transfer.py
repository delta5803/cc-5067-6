"""Explicit, static nonlinear transfer functions.

Every transfer is a pure function y = f(x * drive) (or the equivalent closed
form) with a documented bound so the output can never become NaN/Inf even for
extreme input levels.
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np


class Transfer(str, Enum):
    TANH = "tanh"
    HARD_CLIP = "hard_clip"
    SINE = "sine"
    ASYM = "asym"

    @property
    def symmetric(self) -> bool:
        return self in (Transfer.TANH, Transfer.HARD_CLIP, Transfer.SINE)

    @property
    def output_bound(self) -> float:
        if self is Transfer.SINE:
            return 1.0
        if self is Transfer.ASYM:
            # negative lobe saturates at -0.5, positive at +1
            return 1.0
        return 1.0


_TRANSFERS = {t.value: t for t in Transfer}


def transfer_names() -> list[str]:
    return [t.value for t in Transfer]


def apply_scalar(x: float, drive: float, kind: str | Transfer) -> float:
    """Scalar version of the transfer curve (used in stateful sample loops)."""
    kind = kind.value if isinstance(kind, Transfer) else _TRANSFERS[kind]
    z = x * drive
    if kind == Transfer.TANH:
        return math.tanh(z)
    if kind == Transfer.HARD_CLIP:
        if z > 1.0:
            return 1.0
        if z < -1.0:
            return -1.0
        return z
    if kind == Transfer.SINE:
        if z > math.pi / 2.0:
            z = math.pi / 2.0
        elif z < -math.pi / 2.0:
            z = -math.pi / 2.0
        return math.sin(z)
    if z >= 0.0:
        return math.tanh(z)
    return 0.5 * math.tanh(z)


def apply_transfer(x: np.ndarray, drive: float, kind: str | Transfer) -> np.ndarray:
    """Apply the static transfer curve element-wise.

    Parameters
    ----------
    x : input signal (any finite amplitude is accepted)
    drive : positive gain applied before the nonlinearity
    """
    kind = kind.value if isinstance(kind, Transfer) else _TRANSFERS[kind]
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite samples")
    if drive <= 0 or not math.isfinite(drive):
        raise ValueError("drive must be a positive finite number")

    z = np.asarray(x, dtype=np.float64) * drive
    if kind == Transfer.TANH:
        return np.tanh(z)
    if kind == Transfer.HARD_CLIP:
        return np.clip(z, -1.0, 1.0)
    if kind == Transfer.SINE:
        return np.sin(np.clip(z, -math.pi / 2.0, math.pi / 2.0))
    # Asymmetric soft clip: positive lobe -> tanh (bounded +1),
    # negative lobe -> 0.5 * tanh (bounded -0.5). Produces a DC component.
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = np.tanh(z[pos])
    out[~pos] = 0.5 * np.tanh(z[~pos])
    return out


def dc_offset(drive: float, kind: str | Transfer, amp: float = 10.0,
              n: int = 1_000_001) -> float:
    """Mean output for a full-scale symmetric sine test (DC production)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return float(np.mean(apply_transfer(amp * np.sin(t), drive, kind)))
