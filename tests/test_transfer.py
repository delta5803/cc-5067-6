import numpy as np
import pytest

from saturation.transfer import (
    Transfer, apply_transfer, apply_scalar, dc_offset,
)


@pytest.mark.parametrize("kind", Transfer)
def test_output_bounded_and_finite(kind):
    x = np.linspace(-50.0, 50.0, 10001)
    y = apply_transfer(x, drive=10.0, kind=kind)
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= kind.output_bound + 1e-12


@pytest.mark.parametrize("kind", [Transfer.TANH, Transfer.HARD_CLIP, Transfer.SINE])
def test_symmetric_curve(kind):
    x = np.linspace(-10.0, 10.0, 513)
    y = apply_transfer(x, 2.5, kind)
    y_neg = apply_transfer(-x, 2.5, kind)
    assert np.allclose(y_neg, -y, atol=1e-12)
    assert abs(dc_offset(4.0, kind)) < 1e-9


def test_asymmetric_produces_dc():
    dc = dc_offset(4.0, Transfer.ASYM)
    assert dc > 0.0  # positive lobe saturates harder -> positive mean
    x = np.linspace(-20.0, 20.0, 1001)
    y = apply_transfer(x, 4.0, Transfer.ASYM)
    assert np.min(y) >= -0.5 - 1e-12 and np.max(y) <= 1.0 + 1e-12


def test_scalar_matches_vectorized():
    x = np.linspace(-3.0, 3.0, 37)
    for kind in Transfer:
        vec = apply_transfer(x, 2.0, kind)
        scal = np.array([apply_scalar(float(v), 2.0, kind) for v in x])
        assert np.allclose(vec, scal, atol=1e-12)


def test_zero_input_zero_output_all_curves():
    for kind in Transfer:
        assert apply_scalar(0.0, 3.0, kind) == 0.0


def test_invalid_drive_rejected():
    with pytest.raises(ValueError):
        apply_transfer(np.zeros(4), drive=0.0, kind=Transfer.TANH)
    with pytest.raises(ValueError):
        apply_transfer(np.zeros(4), drive=float("nan"), kind=Transfer.TANH)
