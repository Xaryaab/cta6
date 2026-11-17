"""LLM helpers: JSON responses, retries, free-tier limit handling.

Gemini   - several keys (one per Google account) are rotated; each call starts on the next key to spread the
           per-minute limits, and a key whose daily quota is gone is skipped for the rest of the run.
           Main model first, fallback model when the main one is overloaded (503) or out of quota on every key.
OpenRouter - optional; ask_text_json() sends text-only prompts there (rotating keys and free models) so the
           Gemini quota is kept for the steps that need video/images. Falls back to Gemini if it fails.
"""
import json
import re
import time

from . import config
from .utils import log

_clients: dict = {}
_exhausted: set = set()   # (key, model) pairs whose daily quota ran out this run
_cooldown: dict = {}      # (key, model) -> time.time() when it may be used again (per-minute limits)
_cursor = 0

_or_exhausted: set = set()  # OpenRouter keys out of daily quota
_or_dead: set = set()       # OpenRouter models that are unavailable


def _client(key: str):
    if key not in _clients:
        from google import genai

        _clients[key] = genai.Client(api_key=key)
    return _clients[key]


def available() -> bool:
    return bool(config.GEMINI_API_KEYS)


def _models(model: str | None) -> list[str]:
    main = model or config.GEMINI_MODEL
    fb = config.GEMINI_FALLBACK_MODEL
    return [main] + ([fb] if fb and fb != main else [])


def quota_out(model: str | None = None) -> bool:
    """True once every key x model we can use has hit its daily free-tier limit this run."""
    return all((k, m) in _exhausted for k in config.GEMINI_API_KEYS for m in _models(model))


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
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


def _tag(key: str) -> str:
    return f"key {config.GEMINI_API_KEYS.index(key) + 1}"


def ask_json(contents, model: str | None = None, temperature: float = 0.4, low_media: bool = False,
             label: str = "Gemini"):
    """Send a prompt (text or [parts]) to Gemini and return parsed JSON, or None if it keeps failing."""
    global _cursor
    keys = config.GEMINI_API_KEYS
    if not keys:
        return None
    from google.genai import types

    models = _models(model)
    start = _cursor % len(keys)
    _cursor += 1
    order = keys[start:] + keys[:start]
    # main model on every key first, then the fallback model on every key
    slots = [(k, m) for m in models for k in order]
    overloaded: set = set()  # models that returned 503 during this call
    errors = limited = 0

    while errors < 6:
        usable = [s for s in slots if s not in _exhausted]
        if not usable:
            log(f"{label}: daily free-tier quota used up on all {len(keys)} key(s) - skipping this step.")
            return None
        fresh = [s for s in usable if s[1] not in overloaded] or usable
        now = time.time()
        ready = [s for s in fresh if _cooldown.get(s, 0) <= now]
        if not ready:
            wait = min(_cooldown[s] for s in fresh) - now
            log(f"{label}: all keys rate limited - waiting {wait:.0f}s...")
            time.sleep(max(1.0, wait))
            continue
        key, mdl = ready[0]

        cfg = {"response_mime_type": "application/json", "temperature": temperature}
        if low_media:
            cfg["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
        try:
            resp = _client(key).models.generate_content(
                model=mdl, contents=contents, config=types.GenerateContentConfig(**cfg))
            data = parse_json(resp.text or "")
            if data is not None:
                return data
            errors += 1
            log(f"{label}: unreadable JSON, retrying...")
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                _exhausted.add((key, mdl))
                log(f"{label}: daily quota for {mdl} used up on {_tag(key)} - trying the next one")
                continue
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                _cooldown[(key, mdl)] = time.time() + _retry_delay(msg)
                limited += 1
                if limited > 20:
                    log(f"{label}: still rate limited after many tries - skipping this step.")
                    return None
                if len(keys) == 1:
                    log(f"{label}: rate limited (free tier) - waiting...")
                continue
            if low_media and ("400" in msg or "INVALID_ARGUMENT" in msg):
                low_media = False  # model may not support media_resolution
                continue
            if any(c in msg for c in ("400", "403", "404")):
                log(f"{label}: request rejected ({msg[:200]})")
                return None
            errors += 1
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                if mdl not in overloaded and len(models) > 1:
                    overloaded.add(mdl)
                    log(f"{label}: {mdl} is overloaded - switching to {[m for m in models if m != mdl][0]}")
                    continue
            wait = 5 * errors
            log(f"{label}: error ({msg[:120]}) - retrying in {wait}s...")
            time.sleep(wait)
    return None


def _openrouter(prompt: str, temperature: float, label: str):
    """Try OpenRouter keys x free models. Returns parsed JSON or None."""
    import httpx

    keys = [k for k in config.OPENROUTER_API_KEYS if k not in _or_exhausted]
    for mdl in config.OPENROUTER_MODELS:
        if mdl in _or_dead:
            continue
        for key in keys:
            if key in _or_exhausted:
                continue
            for attempt in range(3):
                try:
                    r = httpx.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "X-Title": "Clipper"},
                        json={"model": mdl, "temperature": temperature,
                              "messages": [{"role": "user", "content": prompt}]},
                        timeout=180,
                    )
                except httpx.HTTPError as e:
                    log(f"{label}: OpenRouter network error ({e}) - retrying...")
                    time.sleep(5)
                    continue
                body = r.text
                if r.status_code == 200:
                    try:
                        data = r.json()
                        if data.get("error"):
                            raise ValueError(str(data["error"])[:150])
                        content = data["choices"][0]["message"].get("content") or ""
                    except (ValueError, KeyError, IndexError) as e:
                        log(f"{label}: OpenRouter {mdl} bad response ({e}) - retrying...")
                        time.sleep(3)
                        continue
                    parsed = parse_json(content)
                    if parsed is not None:
                        log(f"{label}: answered by {mdl}")
                        return parsed
                    log(f"{label}: {mdl} returned unreadable JSON - retrying...")
                    continue
                if r.status_code == 429:
                    if "per-day" in body or "per day" in body.lower() or "free-models-per-day" in body:
                        _or_exhausted.add(key)
                        log(f"{label}: OpenRouter daily limit hit on a key - trying the next one")
                        break
                    time.sleep(15 * (attempt + 1))  # upstream provider busy / per-minute limit
                    continue
                if r.status_code in (401, 402, 403):
                    _or_exhausted.add(key)
                    log(f"{label}: OpenRouter key rejected ({r.status_code}) - skipping it")
                    break
                if r.status_code in (400, 404):
                    _or_dead.add(mdl)
                    log(f"{label}: OpenRouter model {mdl} unavailable - trying the next model")
                    break
                time.sleep(5 * (attempt + 1))  # 5xx
            if mdl in _or_dead:
                break
    return None


def ask_text_json(prompt: str, temperature: float = 0.4, label: str = "LLM"):
    """Text-only prompt: OpenRouter first (if keys set) to save Gemini quota, then Gemini."""
    if config.OPENROUTER_API_KEYS:
        res = _openrouter(prompt, temperature, label)
        if res is not None:
            return res
        log(f"{label}: OpenRouter failed - using Gemini instead")
    return ask_json(prompt, temperature=temperature, label=label)


def text_available() -> bool:
    return available() or bool(config.OPENROUTER_API_KEYS)
