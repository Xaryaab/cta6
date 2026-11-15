"""Find candidate 'something happened' moments from audio loudness, chat activity and hype words."""
import re
import subprocess
from pathlib import Path

import numpy as np

from . import config

STRONG_WORDS = {
    "wtf", "omg", "holy", "insane", "crazy", "bruh", "lmao", "lmfao", "haha", "hahaha", "ahh", "ahhh",
    "yooo", "yoo", "noo", "nooo", "dead", "clip", "screaming", "help", "run", "chase", "cops", "crash",
}
WEAK_WORDS = {"bro", "dude", "what", "wait", "no", "oh", "yo", "why", "go", "stop", "yes", "god", "shit", "fuck"}
CHAT_HYPE = re.compile(
    r"\b(kekw|lul|lulw|omegalul|lmao|lmfao|pog|poggers|pogchamp|w+|l+|clip|clip it|monkas|kappa|"
    r"wtf|omg|xd+|hah+a+|lol+|\?{2,}|!{2,}|aware|icant|sadge)\b",
    re.I,
)


def loudness_per_second(audio: Path) -> np.ndarray:
    """RMS level (dB) for each second of the audio file."""
    sr = 8000
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    levels = []
    block = sr * 2
    while True:
        buf = proc.stdout.read(block)
        if not buf:
            break
        x = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        levels.append(20 * np.log10(rms + 1e-5))
    proc.wait()
    return np.array(levels, dtype=np.float32)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _z(x: np.ndarray) -> np.ndarray:
    if x.size == 0 or float(x.std()) < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - x.mean()) / x.std(), -3, 6)


def find_candidates(start: float, end: float, loud: np.ndarray, words: list, chat: list) -> list:
    n = max(1, int(end - start) + 1)

    # Audio: spikes relative to the local (2-minute) baseline, so loud streamers aren't always "hype".
    a = np.full(n, loud.min() if loud.size else -60, dtype=np.float32)
    a[: min(n, loud.size)] = loud[:n]
    audio_score = _z(_smooth(a - _smooth(a, 120), 3))

    # Transcript hype words
    kw = np.zeros(n, dtype=np.float32)
    for w in words:
        i = int(w["s"] - start)
        if 0 <= i < n:
            tok = re.sub(r"[^a-z]", "", w["w"].lower())
            if tok in STRONG_WORDS:
                kw[i] += 1.0
            elif tok in WEAK_WORDS:
                kw[i] += 0.3
            if w["w"].endswith("!") or (len(w["w"]) > 2 and w["w"].isupper()):
                kw[i] += 0.3
    kw_score = _z(_smooth(kw, 8))

    # Chat activity: volume + hype emotes. Chat lags the moment by a few seconds -> shift back.
    has_chat = len(chat) > 50
    chat_score = np.zeros(n, dtype=np.float32)
    if has_chat:
        c = np.zeros(n, dtype=np.float32)
        for t, msg in chat:
            i = int(t - start) - 6
            if 0 <= i < n:
                c[i] += 1.0 + (1.0 if CHAT_HYPE.search(msg or "") else 0.0)
        chat_score = _z(_smooth(c, 10) - _smooth(c, 300))

    score = _smooth(1.0 * audio_score + (1.6 * chat_score if has_chat else 0) + 0.8 * kw_score, 5)

    # Peak picking with minimum spacing (non-max suppression)
    order = np.argsort(-score)
    picked: list[int] = []
    edge = 10
    for i in order:
        if len(picked) >= config.CANDIDATES:
            break
        if i < edge and n > 60:
            continue
        if all(abs(int(i) - p) >= config.CANDIDATE_SPACING for p in picked):
            picked.append(int(i))

    return [
        {
            "id": k + 1,
            "peak": round(start + i, 1),
            "score": round(float(score[i]), 2),
            "audio": round(float(audio_score[i]), 2),
            "chat": round(float(chat_score[i]), 2),
            "words": round(float(kw_score[i]), 2),
        }
        for k, i in enumerate(sorted(picked, key=lambda i: -score[i]))
    ]
