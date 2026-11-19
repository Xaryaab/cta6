"""Find where a downloaded clip source really starts in the stream.

HLS sources (Twitch) are cut on segment boundaries, so a section asked for at 1:00:00 can start several
seconds earlier. Captions use stream time, so we match the source's audio against the analysis audio
(the one the transcript came from) and use the measured start instead of the requested one."""
import subprocess
from pathlib import Path

import numpy as np

from .utils import log

SR = 4000          # plenty for waveform matching, keeps it fast
SEARCH = 20.0      # +- seconds around the requested start
SAMPLE = 25.0      # seconds of the source used as the fingerprint


def _pcm(path: Path, ss: float, dur: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, ss):.3f}", "-i", str(path), "-t", f"{dur:.3f}",
                        "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    return a - a.mean() if a.size else a


def source_start(src: Path, audio: Path, audio_start: float, expected: float) -> float:
    """Stream time of t=0 in `src`. `audio` covers the stream from `audio_start`. Falls back to `expected`."""
    for t0 in (3.0, 12.0):
        a = _pcm(src, t0, SAMPLE)
        ref_from = expected + t0 - SEARCH
        b = _pcm(audio, ref_from - audio_start, SAMPLE + 2 * SEARCH)
        if a.size < SR * 5 or b.size <= a.size or not a.any():
            continue
        c = np.correlate(b, a, "valid")
        i = int(np.argmax(c))
        seg = b[i:i + a.size]
        ncc = float(c[i] / (np.linalg.norm(a) * np.linalg.norm(seg) + 1e-9))
        if ncc >= 0.3:
            actual = ref_from + i / SR - t0
            if abs(actual - expected) > 0.15:
                log(f"   source starts {actual - expected:+.2f}s from requested - captions re-synced")
            return actual
    log("   couldn't verify source timing - using requested start")
    return expected
