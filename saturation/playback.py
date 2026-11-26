"""Optional live listening fallback when PortAudio is available.

The browser panel is the primary UI; this CLI player only works in
environments where sounddevice + PortAudio are installed.
"""
from __future__ import annotations

import numpy as np

from .panel_backend import render


def play(source: str, drive: float, factor: int, kind: str) -> bool:
    try:
        import sounddevice as sd  # noqa
    except Exception as exc:
        print(f"[playback] 实时声卡不可用 ({exc})；请改用浏览器面板试听，"
              "或使用 output/ 下的 WAV 文件。")
        return False
    result = render(source, drive, factor, kind)
    if not result.ok:
        print(f"[playback] 渲染失败: {result.error}")
        return False
    import base64, io, wave
    raw = base64.b64decode(result.wav_b64)
    with wave.open(io.BytesIO(raw)) as wf:
        n = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float64) / 32768
        fs = wf.getframerate()
    print(f"[playback] {source} drive={drive} x{factor} {kind} "
          f"THD={result.metrics['thd_db']}dB")
    sd.play(pcm, fs, blocking=True)
    return True
