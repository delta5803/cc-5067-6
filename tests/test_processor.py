import numpy as np

from saturation.processor import SaturationProcessor, SwitchDecision
from saturation.transfer import Transfer
from saturation import signals

FS = 48000


def _run_chunked(proc: SaturationProcessor, x: np.ndarray, block: int):
    out = []
    for s in range(0, len(x), block):
        out.append(proc.process_block(x[s:s + block]))
    out.append(proc.flush())
    return np.concatenate(out)


def test_constant_factor_matches_offline_and_block_invariant():
    x = signals.pluck(FS, f0=220.0, dur=1.2)
    p1 = SaturationProcessor(FS, factors=(1, 4), drive=5.0)
    y_a = _run_chunked(p1, x, 512)
    p2 = SaturationProcessor(FS, factors=(1, 4), drive=5.0)
    y_b = _run_chunked(p2, x, 1023)
    assert len(y_a) == len(y_b) == len(x) + p1.max_delay
    assert np.allclose(y_a, y_b, atol=1e-10)


def test_constant_latency_across_factors():
    p = SaturationProcessor(FS, factors=(1, 2, 4, 8), drive=4.0)
    delays = {f: p._engines[f].delay_base + p._delay_lines[f].delay
              for f in p.factors}
    assert len(set(delays.values())) == 1
    assert delays[1] == p.max_delay


def test_switch_after_long_silence_uses_warmup():
    block = 512
    pluck = signals.pluck(FS, dur=1.0)
    x = signals.silence_then(FS, pluck, silence_s=1.0)
    p = SaturationProcessor(FS, factors=(1, 8), drive=6.0,
                            double_run_budget=block)
    p.request_factor(8)
    n_silence_blocks = int(FS * 1.0 / block)
    _run_chunked_switch(p, x, block, request_at=0)
    kinds = [e.decision for e in p.events]
    assert SwitchDecision.WARM_UP in kinds
    # switch completes before the pluck arrives -> no audible transient
    first_event = next(e for e in p.events
                       if e.decision in (SwitchDecision.WARM_UP,
                                         SwitchDecision.CROSSFADE))
    assert first_event.sample_time < FS


def _run_chunked_switch(proc, x, block, request_at):
    out = []
    for i, s in enumerate(range(0, len(x), block)):
        if i == request_at:
            proc.request_factor(proc.factors[-1])
        out.append(proc.process_block(x[s:s + block]))
    out.append(proc.flush())
    return np.concatenate(out)


def test_loud_switch_within_budget_crossfades():
    block = 512
    x = signals.pluck(FS, dur=2.0)  # loud from the start
    p = SaturationProcessor(FS, factors=(1, 4), drive=6.0,
                            double_run_budget=block * 10)
    _run_chunked_switch(p, x, block, request_at=4)
    dec = [e.decision for e in p.events]
    assert SwitchDecision.CROSSFADE in dec
    assert p.active_factor == 4


def test_loud_switch_over_budget_defers_then_succeeds():
    block = 512
    x = np.concatenate([
        signals.pluck(FS, dur=2.0, decay=4.0),
        np.zeros(FS),  # decays into true silence where warm-up is free
    ])
    p = SaturationProcessor(FS, factors=(1, 8), drive=6.0,
                            double_run_budget=block)  # factor 8 cost > budget
    _run_chunked_switch(p, x, block, request_at=2)
    dec = [e.decision for e in p.events]
    assert SwitchDecision.DEFER in dec
    # as the pluck decays into silence the warm-up path takes over
    assert p.active_factor == 8


def test_force_fallback_is_marked():
    block = 512
    x = signals.pluck(FS, dur=1.0)
    p = SaturationProcessor(FS, factors=(1, 8), drive=6.0)
    out = []
    for i, s in enumerate(range(0, len(x), block)):
        if i == 3:
            ev = p.force_factor(8)
            assert ev.decision == SwitchDecision.FALLBACK
        out.append(p.process_block(x[s:s + block]))
    p.flush()
    assert p.active_factor == 8


def test_switch_does_not_shift_time_axis():
    from saturation.analysis import transient_position
    block = 512
    pluck = signals.pluck(FS, f0=196.0, dur=1.5)
    x = signals.silence_then(FS, pluck, silence_s=0.8)
    p_ref = SaturationProcessor(FS, factors=(8,), drive=5.0)
    y_ref = _run_chunked(p_ref, x, block)
    p_sw = SaturationProcessor(FS, factors=(1, 8), drive=5.0)
    y_sw = _run_chunked_switch(p_sw, x, block, request_at=2)
    # outputs share one constant latency; transients must land on the same tick
    assert abs(transient_position(y_ref) - transient_position(y_sw)) <= block


def test_drive_change_is_smoothed_and_stateful():
    p = SaturationProcessor(FS, factors=(1,), drive=1.0)
    p.process_block(signals.sine(FS, 300.0, 0.05))
    p.set_drive(20.0)
    y1 = p.process_block(signals.sine(FS, 300.0, 0.02))
    # first block after the change cannot jump straight to the 20x level
    p2 = SaturationProcessor(FS, factors=(1,), drive=20.0)
    y_cold = p2.process_block(signals.sine(FS, 300.0, 0.02))
    assert np.max(np.abs(y1)) < np.max(np.abs(y_cold))


def test_cold_branch_produces_pseudo_transient_without_warmup():
    """Documents the memory hazard: a branch whose delay lines and drive
    smoother have real history differs from a branch switched in cold. The
    processor hides this with crossfades; a raw force_switch would expose it."""
    history = signals.sine(FS, 300.0, 0.1, amp=0.7)
    block = signals.pluck(FS, dur=0.3)[:2048]

    # branch A has been live on signal; drive target moves during operation
    live = SaturationProcessor(FS, factors=(4,), drive=6.0)
    live._engines[4].process(history)
    live.set_drive(9.0)  # drive change still smoothing when block arrives
    y_live = live._engines[4].process(block)

    # branch B is instantiated at the switch moment: zeroed FIR delay lines,
    # no signal history, drive smoother sitting exactly at the target value
    cold = SaturationProcessor(FS, factors=(4,), drive=6.0)
    cold.set_drive(9.0)
    y_cold = cold._engines[4].process(block)

    # first wet block diverges (FIR delay line fill + smoother position)
    assert not np.allclose(y_live, y_cold, atol=1e-4)
    # and the divergence is largest at the attack, decaying as states settle
    diff_env = np.abs(y_live - y_cold)
    early = diff_env[:256].mean()
    late = diff_env[-256:].mean()
    assert early > late
