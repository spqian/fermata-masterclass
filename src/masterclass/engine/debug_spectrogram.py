"""Server-side spectrogram renderer for the Technical Viewer.

Renders a mel-spectrogram of a time window of a lesson's audio to a PNG,
cached on disk so repeat requests for the same window are free.

Returns the raw PNG bytes; the API layer is responsible for setting
content-type and serving them.
"""
from __future__ import annotations

import hashlib
import io
import logging

import librosa
import matplotlib

matplotlib.use("Agg")  # headless rendering, no GUI backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from masterclass.storage.base import ObjectStorage

_LOG = logging.getLogger(__name__)

DEFAULT_WIDTH = 1100
DEFAULT_HEIGHT = 280
N_MELS = 96
FMAX_HZ = 8000.0
MAX_WINDOW_SEC = 60.0    # cap a single rendered slice to keep latency sane
MIN_WINDOW_SEC = 0.25


def _cache_key(audio_key: str, start: float, end: float, width: int, height: int) -> str:
    digest = hashlib.sha1(f"{audio_key}|{start:.3f}|{end:.3f}|{width}|{height}|v1".encode("utf-8")).hexdigest()
    # Co-locate the cache under the lesson's analysis/ prefix so it travels
    # with the rest of the per-lesson artifacts.
    prefix = audio_key.rsplit("/", 2)[0]  # drop ".../artifacts/audio.wav"
    return f"{prefix}/analysis/_debug_specgrams/{digest}.png"


def render_window(
    *,
    storage: ObjectStorage,
    audio_key: str,
    start_sec: float,
    end_sec: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> bytes:
    """Render a mel-spectrogram PNG for [start_sec, end_sec) of an audio asset.

    Caches the result on the same storage backend so reloads are instant.
    """
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")
    window = end_sec - start_sec
    if window < MIN_WINDOW_SEC:
        raise ValueError(f"window must be at least {MIN_WINDOW_SEC}s")
    if window > MAX_WINDOW_SEC:
        raise ValueError(f"window must be at most {MAX_WINDOW_SEC}s")
    width = int(min(max(width, 200), 2400))
    height = int(min(max(height, 120), 800))

    cache = _cache_key(audio_key, start_sec, end_sec, width, height)
    if storage.exists(cache):
        return storage.read_bytes(cache)

    # librosa.load with offset+duration reads only the requested slice so we
    # do not have to pull the whole multi-MB WAV into memory once libsndfile
    # is seeking the file handle. Wrap bytes in BytesIO so libsndfile sees a
    # file-like object regardless of storage backend.
    audio_bytes = storage.read_bytes(audio_key)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, offset=start_sec, duration=window)
    if y.size == 0:
        raise ValueError("requested window is outside the audio length")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX_HZ, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#0b0d12")
    ax = fig.add_subplot(1, 1, 1, facecolor="#0b0d12")
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX_HZ,
        cmap="magma",
        ax=ax,
    )
    # Shift the x-axis so the displayed times match the lesson's wall-clock,
    # not the slice-local offset starting at zero. FuncFormatter avoids the
    # set_xticklabels warning matplotlib emits when ticks aren't pinned.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _pos: f"{t + start_sec:.2f}"))
    ax.set_xlabel("time (s)", color="#c5c0b3")
    ax.set_ylabel("mel freq (Hz)", color="#c5c0b3")
    ax.tick_params(colors="#7a7669")
    for spine in ax.spines.values():
        spine.set_color("#3a3530")
    ax.set_title(f"mel spectrogram · {start_sec:.2f}s → {end_sec:.2f}s", color="#c9a96a", fontsize=10)
    fig.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    png = buf.getvalue()

    try:
        storage.write_bytes(cache, png, content_type="image/png")
    except Exception:
        _LOG.warning("failed to cache spectrogram at %s", cache, exc_info=True)
    return png
