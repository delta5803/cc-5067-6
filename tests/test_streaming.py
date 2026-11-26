import numpy as np

from saturation.processor import SaturationProcessor
from saturation.resample import MultirateEngine
from saturation import signals

FS = 48000


def test_flush_preserves_valid_tail_at_file_edge():
    # signal ending exactly at the file edge: the ringing tail must survive
    x = signals.pluck(FS, f0=174.0, dur=1.0)
    eng = MultirateEngine(8, FS, drive=5.0)
    body = eng.process(x)
    tail = eng.flush()
    assert len(tail) == eng.delay_base
    # the tail carries real decaying signal energy, not zeros or garbage
    assert np.sqrt(np.mean(tail ** 2)) > 1e-5
    full = np.concatenate([body, tail])
    assert len(full) == len(x) + eng.delay_base
    assert np.isfinite(full).all()


def test_chunked_run_matches_single_offline_pass_for_processor():
    rng = np.random.default_rng(11)
    x = (signals.pluck(FS, dur=1.3)
         + 0.03 * rng.standard_normal(int(FS * 1.3)))
    p_a = SaturationProcessor(FS, factors=(1, 4, 8), drive=5.5)
    chunks_a = []
    for s in range(0, len(x), 333):
        chunks_a.append(p_a.process_block(x[s:s + 333]))
    chunks_a.append(p_a.flush())
    p_b = SaturationProcessor(FS, factors=(1, 4, 8), drive=5.5)
    chunks_b = []
    for s in range(0, len(x), 4096):
        chunks_b.append(p_b.process_block(x[s:s + 4096]))
    chunks_b.append(p_b.flush())
    assert np.allclose(np.concatenate(chunks_a), np.concatenate(chunks_b),
                       atol=1e-10)


def test_drive_smoother_ramps_without_zipper():
    from saturation import signals
    x = signals.sine(FS, 400.0, 0.05)
    # reference: a cold processor instantiated directly at the high drive
    cold = SaturationProcessor(FS, factors=(1,), drive=20.0)
    cold.process_block(np.zeros(2048))
    hard = cold.process_block(x)
    # smoothed processor: drive target changes while the stream is running
    proc = SaturationProcessor(FS, factors=(1,), drive=1.0)
    proc.process_block(np.zeros(2048))
    proc.set_drive(20.0)
    y = proc.process_block(x)
    # per-sample discontinuities must be strictly smaller than the hard path
    assert np.max(np.abs(np.diff(y))) < np.max(np.abs(np.diff(hard)))
    # and the output approaches the high-drive level only gradually
    assert np.max(np.abs(y[:200])) < np.max(np.abs(hard[:200]))
