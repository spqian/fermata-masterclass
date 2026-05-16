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
DEFAULT_HEIGHT = 520           # taller so 88 piano keys are legible
# Piano range A0 (27.5 Hz) -> C8 (4186 Hz) is 88 semitones; we go from C1 up so
# the very lowest piano bass (rarely useful) is skipped and the visualization
# starts at a frequency where short windows still resolve cleanly.
BINS_PER_OCTAVE = 24           # quarter-tone resolution -> see pitch deviation (sharp/flat) within a semitone
N_OCTAVES = 7                  # C1 to C8 covers the entire usable piano range
N_BINS = BINS_PER_OCTAVE * N_OCTAVES
FMIN_NOTE = "C1"               # ~32.7 Hz, MIDI 24
FMIN_MIDI = 24
FMAX_MIDI = FMIN_MIDI + N_BINS // (BINS_PER_OCTAVE // 12)  # MIDI at top of plot (C8 = 108)
MAX_WINDOW_SEC = 60.0          # cap a single rendered slice to keep latency sane
MIN_WINDOW_SEC = 0.25

# Pinned plot-area margins (fractions of figure). Frontend uses these to
# translate mouse pixel coordinates -> (time, frequency) for the hover tooltip.
# If you change these, bump the cache-key version below and the matching
# constants in static/technical_viewer.html.
PLOT_LEFT_FRAC = 0.060
PLOT_RIGHT_FRAC = 0.992
PLOT_BOTTOM_FRAC = 0.115
PLOT_TOP_FRAC = 0.935


def _cache_key(audio_key: str, start: float, end: float, width: int, height: int) -> str:
    digest = hashlib.sha1(
        f"{audio_key}|{start:.3f}|{end:.3f}|{width}|{height}|cqt-v3-measured".encode("utf-8")
    ).hexdigest()
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
) -> tuple[bytes, dict]:
    """Render a CQT spectrogram PNG for [start_sec, end_sec) of an audio asset.

    Returns ``(png_bytes, metadata)`` where ``metadata`` contains the true
    pixel-coordinate bbox of the plot area inside the image plus the data
    ranges, so the frontend can translate mouse / overlay coordinates exactly.
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

    cache_png = _cache_key(audio_key, start_sec, end_sec, width, height)
    cache_meta = cache_png + ".json"
    if storage.exists(cache_png) and storage.exists(cache_meta):
        try:
            meta = storage.read_json(cache_meta)
            return storage.read_bytes(cache_png), meta
        except (FileNotFoundError, ValueError, TypeError):
            pass  # corrupt cache, fall through and regenerate

    audio_bytes = storage.read_bytes(audio_key)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, offset=start_sec, duration=window)
    if y.size == 0:
        raise ValueError("requested window is outside the audio length")

    fmin = librosa.note_to_hz(FMIN_NOTE)
    hop_length = 2 ** (N_OCTAVES - 1) * 2
    cqt = librosa.cqt(
        y=y, sr=sr,
        fmin=fmin,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=hop_length,
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#0b0d12")
    ax = fig.add_subplot(1, 1, 1, facecolor="#0b0d12")
    librosa.display.specshow(
        cqt_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="cqt_note",
        fmin=fmin,
        bins_per_octave=BINS_PER_OCTAVE,
        cmap="magma",
        ax=ax,
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _pos: f"{t + start_sec:.2f}"))
    ax.set_xlabel("time (s)", color="#c5c0b3")
    ax.set_ylabel("pitch (note name)", color="#c5c0b3")
    ax.tick_params(colors="#7a7669")
    for octave in range(1, N_OCTAVES + 1):
        c_hz = librosa.note_to_hz(f"C{octave}")
        if fmin <= c_hz <= librosa.note_to_hz(f"C{N_OCTAVES + 1}"):
            ax.axhline(c_hz, color="#3a3530", linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#3a3530")
    ax.set_title(
        f"constant-Q spectrogram (24 bins/octave) · {start_sec:.2f}s → {end_sec:.2f}s",
        color="#c9a96a",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=PLOT_LEFT_FRAC,
        right=PLOT_RIGHT_FRAC,
        bottom=PLOT_BOTTOM_FRAC,
        top=PLOT_TOP_FRAC,
    )

    # IMPORTANT: force a draw before measuring axes bbox; otherwise the bbox
    # may reflect requested rather than actual layout (esp. for tick labels).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox_display = ax.get_window_extent(renderer=renderer)
    # Display coords have origin at bottom-left; convert to top-left origin
    # (the natural coord system for an HTML img).
    plot_left = float(ax_bbox_display.x0)
    plot_right = float(ax_bbox_display.x1)
    plot_bottom = float(height - ax_bbox_display.y0)
    plot_top = float(height - ax_bbox_display.y1)

    # Pitch range: the CQT y-axis runs from fmin (bin 0) at the bottom to
    # fmin * 2**(N_BINS / BINS_PER_OCTAVE) at the top of the topmost bin.
    midi_min = float(librosa.hz_to_midi(fmin))
    midi_max = midi_min + float(N_BINS) / float(BINS_PER_OCTAVE // 12)  # 12 semitones/octave conversion

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    png = buf.getvalue()

    meta = {
        "image_width": width,
        "image_height": height,
        "plot_bbox_px": [plot_left, plot_top, plot_right, plot_bottom],  # x0,y0,x1,y1 top-left origin
        "time_range_sec": [float(start_sec), float(end_sec)],
        "midi_range": [midi_min, midi_max],
        "bins_per_octave": BINS_PER_OCTAVE,
        "schema_version": 1,
    }

    try:
        storage.write_bytes(cache_png, png, content_type="image/png")
        storage.write_json(cache_meta, meta)
    except Exception:
        _LOG.warning("failed to cache spectrogram at %s", cache_png, exc_info=True)
    return png, meta
