"""Speech-to-text with word timestamps. Groq Whisper (free tier) first, local faster-whisper fallback.

Output format (times are absolute seconds in the source video):
  {"words": [{"w": str, "s": float, "e": float}], "segments": [{"text": str, "s": float, "e": float}]}
"""
import math
import time
from pathlib import Path

from . import config
from .utils import load_json, log, run, save_json


def transcribe(audio: Path, offset: float, length: float, work_dir: Path) -> dict:
    if config.GROQ_API_KEY:
        return _groq(audio, offset, length, work_dir)
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        raise SystemExit(
            "No GROQ_API_KEY in .env and faster-whisper isn't installed.\n"
            "Get a free key at https://console.groq.com/keys (recommended), or run:\n"
            "  uv pip install -r requirements-local.txt"
        )
    return _local(audio, offset)


# ---------------------------------------------------------------- Groq
def _groq(audio: Path, offset: float, length: float, work_dir: Path) -> dict:
    import groq

    client = groq.Groq(api_key=config.GROQ_API_KEY, max_retries=0)
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    n = max(1, math.ceil(length / config.CHUNK_SECONDS))
    words, segments = [], []

    for i in range(n):
        cs = i * config.CHUNK_SECONDS
        cache = chunks_dir / f"c{i:04d}.json"
        if cache.exists():
            data = load_json(cache)
        else:
            mp3 = chunks_dir / f"c{i:04d}.mp3"
            run(["ffmpeg", "-y", "-ss", cs, "-t", config.CHUNK_SECONDS, "-i", audio,
                 "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", mp3])
            log(f"Transcribing chunk {i + 1}/{n} (Groq)...")
            data = _groq_call(client, mp3)
            save_json(cache, data)
            mp3.unlink(missing_ok=True)

        base = offset + cs
        for w in data.get("words") or []:
            words.append({"w": w["word"].strip(), "s": round(base + w["start"], 3), "e": round(base + w["end"], 3)})
        for s in data.get("segments") or []:
            segments.append({"text": s["text"].strip(), "s": round(base + s["start"], 3), "e": round(base + s["end"], 3)})

    return {"words": [w for w in words if w["w"]], "segments": segments}


def _groq_call(client, mp3: Path) -> dict:
    import groq

    payload = mp3.read_bytes()
    for attempt in range(8):
        try:
            resp = client.audio.transcriptions.create(
                file=(mp3.name, payload),
                model=config.GROQ_WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
            data = resp.to_dict() if hasattr(resp, "to_dict") else resp.model_dump()
            return {"words": data.get("words") or [], "segments": data.get("segments") or []}
        except groq.RateLimitError as e:
            wait = _retry_after(e) or 60
            log(f"Groq rate limit hit - waiting {wait:.0f}s (free tier)...")
            time.sleep(wait)
        except (groq.APIConnectionError, groq.InternalServerError) as e:
            wait = 5 * (attempt + 1)
            log(f"Groq error ({type(e).__name__}) - retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Groq transcription kept failing; try again later.")


def _retry_after(err) -> float | None:
    try:
        val = err.response.headers.get("retry-after")
        return float(val) + 1 if val else None
    except Exception:
        return None


# ---------------------------------------------------------------- local
def _decode_16k(audio: Path):
    """Decode with ffmpeg to 16 kHz mono float32 (avoids faster-whisper's PyAV dependency issues)."""
    import subprocess

    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _local(audio: Path, offset: float) -> dict:
    from faster_whisper import WhisperModel

    log(f"Transcribing locally with faster-whisper '{config.LOCAL_WHISPER_MODEL}' on CPU (slow)...")
    model = WhisperModel(config.LOCAL_WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(_decode_16k(audio), word_timestamps=True, vad_filter=False)
    words, segments = [], []
    for seg in segs:
        segments.append({"text": seg.text.strip(), "s": round(offset + seg.start, 3), "e": round(offset + seg.end, 3)})
        for w in seg.words or []:
            if w.word.strip():
                words.append({"w": w.word.strip(), "s": round(offset + w.start, 3), "e": round(offset + w.end, 3)})
        log(f"  ...{seg.end:.0f}s transcribed")
    return {"words": words, "segments": segments}
