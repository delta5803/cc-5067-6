"""The mastering-panel regression sample is driven by the real DSP core.

These checks guard the requirement that panel buttons never serve fixed or
canned harmonic data: every run re-renders through MultirateEngine / fault
wrappers, changing the drive measurably changes the harmonics, and the
injected faults produce the same bad results the acceptance suite records.
"""
import base64
import io
import json
import wave

import pytest

from saturation.panel_backend import regression_sample


def _wav_len(b64):
    with wave.open(io.BytesIO(base64.b64decode(b64))) as wf:
        return wf.getnframes()


def test_regression_renders_real_audio_and_measured_evidence():
    r = regression_sample(drive=8.0, factor=8, kind="hard_clip")
    assert r["ok"] is True
    h, n, g = r["healthy"], r["fault_remove_antialias"], r["fault_decimation_grid"]
    # real WAV payloads, all one second at 48k
    assert _wav_len(h["wav_b64"]) == 48000
    assert _wav_len(r["dry_b64"]) == 48000
    # healthy chain beats the independently derived alias bound
    assert h["alias_energy"] < r["independent_bound_energy"]
    # latency matches the independent derivation
    assert h["latency_samples"] == h["expected_latency"]
    # measured harmonics are non-trivial (real saturation, not zeros/canned)
    assert h["harmonics"][0] > 0.1
    assert len(h["harmonics"]) == 8


def test_regression_faults_are_exposed_not_hidden():
    r = regression_sample(drive=8.0, factor=8, kind="hard_clip")
    n = r["fault_remove_antialias"]
    g = r["fault_decimation_grid"]
    # anti-alias removal: alias rises by tens of dB, latency report unchanged
    assert n["alias_energy"] > r["healthy"]["alias_energy"] * 50
    assert n["alias_gain_db"] > 30.0
    assert n["latency_samples"] == r["healthy"]["latency_samples"]
    # decimation grid: alias stays low, but sub-sample timing shifts
    assert g["alias_energy"] < 1e-3
    assert (g["impulse_timing"]["centroid_delta"]
            > g["healthy_impulse_timing"]["centroid_delta"] + 0.3)


def test_regression_changing_drive_changes_measured_harmonics():
    # the button must call the implementation, not return fixed harmonic data
    low = regression_sample(drive=2.0, factor=8, kind="hard_clip")
    high = regression_sample(drive=16.0, factor=8, kind="hard_clip")
    assert low["healthy"]["harmonics"] != high["healthy"]["harmonics"]
    # harder drive pushes more into the higher harmonics
    tail_low = sum(low["healthy"]["harmonics"][2:])
    tail_high = sum(high["healthy"]["harmonics"][2:])
    assert tail_high > tail_low


def test_regression_rejects_bad_parameters():
    with pytest.raises(ValueError):
        regression_sample(drive=999.0, factor=8)
    with pytest.raises(ValueError):
        regression_sample(drive=8.0, factor=3)
    with pytest.raises(ValueError):
        regression_sample(drive=8.0, factor=8, kind="bogus")


def test_regression_http_endpoint_serves_live_result():
    # exercise the actual HTTP handler path used by the panel button
    from saturation.panel import Handler
    import urllib.parse

    class FakeReq:
        def __init__(self, body):
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = io.BytesIO(body.encode())

    captured = {}

    class FakeHandler(Handler):
        def __init__(self):
            pass

        def _send(self, code, ctype, body):
            captured["code"] = code
            captured["json"] = json.loads(body.decode())

    body = urllib.parse.urlencode({"drive": 8, "factor": 8,
                                   "kind": "hard_clip"})
    h = FakeHandler()
    h.path = "/regression"
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body.encode())
    h.do_POST()
    assert captured["code"] == 200
    assert captured["json"]["ok"] is True
    assert captured["json"]["healthy"]["latency_samples"] >= 0
