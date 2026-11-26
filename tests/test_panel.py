from saturation.panel_backend import render, SOURCES
from saturation.transfer import transfer_names


def test_render_all_sources_and_factors():
    for source in SOURCES:
        for factor in (1, 8):
            r = render(source, drive=4.0, factor=factor, kind="tanh")
            assert r.ok, r.error
            assert len(r.wav_b64) > 1000
            assert len(r.dry_b64) > 1000
            assert len(r.plot_b64) > 1000
            assert r.metrics["factor"] == factor
            assert isinstance(r.metrics["thd_db"], float)
            assert r.metrics["latency_samples"] >= 0


def test_render_all_transfers():
    for kind in transfer_names():
        r = render("asymmetric", drive=5.0, factor=4, kind=kind)
        assert r.ok, r.error


def test_render_rejects_bad_params_back_to_controls():
    assert not render("pluck", drive=999.0, factor=8, kind="tanh").ok
    assert not render("pluck", drive=2.0, factor=3, kind="tanh").ok
    assert not render("nope", drive=2.0, factor=1, kind="tanh").ok
    assert not render("pluck", drive=2.0, factor=1, kind="bogus").ok


def test_oversampled_render_keeps_wet_length_aligned_to_dry():
    import base64, io, wave
    for factor in (1, 8):
        r = render("overrange", drive=12.0, factor=factor, kind="hard_clip")
        assert r.ok
        with wave.open(io.BytesIO(base64.b64decode(r.wav_b64))) as wf:
            wet_n = wf.getnframes()
        with wave.open(io.BytesIO(base64.b64decode(r.dry_b64))) as wf:
            dry_n = wf.getnframes()
        # latency compensation makes wet and dry exactly the same length
        assert wet_n == dry_n
