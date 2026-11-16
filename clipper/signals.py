"""Find candidate 'something happened' moments.

Signals (each normalised within this video, so quiet and loud streamers both get real peaks):
  - audio spikes vs. the local 2-minute baseline
  - "payoff shape": quiet -> sudden loud within a few seconds (reactions, screams, crashes)
  - hype words in the transcript
  - chat: UNIQUE chatters per 5s (spam-proof) + hype emotes, shifted back for chat delay
  - YouTube "most replayed" heatmap (small, capped bonus)
  - viewer-made Twitch clips on this VOD (strong bonus)
"""
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
CHAT_DELAY = 6  # chat reacts a few seconds after the moment


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


def _payoff(a: np.ndarray) -> np.ndarray:
    """Loudest of the next 3s minus the average of the previous 6s: rewards sudden jumps out of calm."""
    n = a.size
    out = np.zeros(n, dtype=np.float32)
    if n < 12:
        return out
    pad = np.pad(a, (6, 3), mode="edge")
    for i in range(n):
        out[i] = pad[i + 6:i + 9].max() - pad[i:i + 6].mean()
    return np.maximum(out, 0)


def find_candidates(start: float, end: float, loud: np.ndarray, words: list, chat: list,
                    heatmap: list | None = None, twitch_clips: list | None = None) -> list:
    n = max(1, int(end - start) + 1)

    # Audio
    a = np.full(n, loud.min() if loud.size else -60, dtype=np.float32)
    a[: min(n, loud.size)] = loud[:n]
    audio_score = _z(_smooth(a - _smooth(a, 120), 3))
    payoff_score = _z(_smooth(_payoff(a), 3))

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

    # Chat: unique chatters per 5s block (one person spamming counts once) + hype emotes
    has_chat = len(chat) > 50
    chat_score = np.zeros(n, dtype=np.float32)
    if has_chat:
        c = np.zeros(n, dtype=np.float32)
        seen = set()
        for row in chat:
            t, msg = row[0], row[1]
            author = row[2] if len(row) > 2 and row[2] else None
            i = int(t - start) - CHAT_DELAY
            if not 0 <= i < n:
                continue
            if author:
                key = (author, i // 5)
                if key in seen:
                    continue
                seen.add(key)
            c[i] += 1.0 + (0.8 if CHAT_HYPE.search(msg or "") else 0.0)
        chat_score = _z(_smooth(c, 10) - _smooth(c, 300))

    # YouTube most-replayed heatmap
    heat_score = np.zeros(n, dtype=np.float32)
    if heatmap:
        h = np.zeros(n, dtype=np.float32)
        for seg in heatmap:
            i0, i1 = int(seg.get("start_time", 0) - start), int(seg.get("end_time", 0) - start)
            h[max(0, i0):max(0, min(n, i1))] = float(seg.get("value") or 0)
        heat_score = np.clip(_z(h), 0, 3)

    # Viewer-made Twitch clips (the moment is usually near the end of a 30s clip)
    tw = np.zeros(n, dtype=np.float32)
    for clip in twitch_clips or []:
        center = clip["offset"] + clip["duration"] * 0.65 - start
        i0, i1 = int(center - 8), int(center + 8)
        if i1 > 0 and i0 < n:
            tw[max(0, i0):min(n, i1)] += 1.0
    twitch_score = np.minimum(tw, 4) * 1.2

    score = _smooth(
        1.0 * audio_score + 0.6 * payoff_score + 0.8 * kw_score
        + (1.6 * chat_score if has_chat else 0) + 0.5 * heat_score + twitch_score,
        5,
    )

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
            "twitch_clips": int(tw[i]),
        }
        for k, i in enumerate(sorted(picked, key=lambda i: -score[i]))
    ]
