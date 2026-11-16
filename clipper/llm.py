"""Shared Gemini helper: JSON responses, retries, free-tier rate-limit handling."""
import json
import re
import time

from . import config
from .utils import log

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _retry_delay(msg: str) -> float:
    m = re.search(r"retry(?:Delay)?\W{0,4}(?:in\s*)?(\d+(?:\.\d+)?)\s*s", msg, re.I)
    return float(m.group(1)) + 2 if m else 40.0


def ask_json(contents, model: str | None = None, temperature: float = 0.4, low_media: bool = False, label: str = "Gemini"):
    """Send a prompt (text or [parts]) and return parsed JSON, or None if it keeps failing."""
    if not available():
        return None
    from google.genai import types

    model = model or config.GEMINI_MODEL
    for attempt in range(6):
        cfg = {"response_mime_type": "application/json", "temperature": temperature}
        if low_media:
            cfg["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
        try:
            resp = _get_client().models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
            )
            data = parse_json(resp.text or "")
            if data is not None:
                return data
            log(f"{label}: unreadable JSON, retrying...")
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                log(f"{label}: daily free-tier quota used up - skipping this step.")
                return None
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = _retry_delay(msg)
                log(f"{label}: rate limited (free tier) - waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            if low_media and ("400" in msg or "INVALID_ARGUMENT" in msg):
                low_media = False  # model may not support media_resolution
                continue
            if any(c in msg for c in ("400", "403", "404")):
                log(f"{label}: request rejected ({msg[:200]})")
                return None
            wait = 5 * (attempt + 1)
            log(f"{label}: error ({msg[:120]}) - retrying in {wait}s...")
            time.sleep(wait)
    return None
