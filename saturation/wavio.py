"""16-bit WAV IO kept separate from the DSP chain."""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile


def read_wav(path: str) -> tuple[float, np.ndarray]:
    fs, data = wavfile.read(path)
    data = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return float(fs), data


def write_wav(path: str, fs: float, x: np.ndarray,
              peak: float = 0.95) -> None:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > peak:
        x = x * (peak / m)
    pcm = np.clip(x, -1.0, 1.0)
    wavfile.write(path, int(fs), (pcm * 32767).astype(np.int16))
