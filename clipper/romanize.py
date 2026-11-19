"""Roman script for Hindi / Urdu streams: Whisper writes them in Devanagari or Urdu script, but the audience
reads Roman Urdu/Hindi ("bhai yeh kya ho gaya"). Only the words inside a clip are converted, at caption time,
so a whole stream costs a handful of small LLM calls."""
import re

from . import llm
from .utils import load_json, log, save_json

_NATIVE = re.compile(r"[ऀ-ॿ؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
BATCH = 150

PROMPT = """Transliterate these Hindi/Urdu words into ROMAN script (Latin letters) the way Pakistani and Indian
viewers type on social media, e.g. "भाई" -> "bhai", "کیا" -> "kya", "क्या हो गया" -> "kya ho gaya".
Rules:
- transliterate, do NOT translate; keep English words as they are
- casual spelling: "hai" not "haiṅ", no accents/diacritics
- return EXACTLY {n} items, one per input word, same order (a word may become two Roman words - keep them in one item)

Return ONLY JSON: {{"roman": ["...", ...]}}

WORDS (JSON array):
{words}
"""


def is_native(text: str) -> bool:
    return bool(_NATIVE.search(text or ""))


def needs_roman(words: list) -> bool:
    """True if a noticeable share of the words are in Devanagari / Urdu script."""
    sample = words[:3000]
    return bool(sample) and sum(1 for w in sample if is_native(w["w"])) / len(sample) > 0.15


def _convert(items: list[str]) -> list[str] | None:
    import json

    res = llm.ask_task("translit", PROMPT.format(n=len(items), words=json.dumps(items, ensure_ascii=False)),
                       temperature=0.1, label="Roman script")
    out = res.get("roman") if isinstance(res, dict) else res if isinstance(res, list) else None
    if not isinstance(out, list) or len(out) != len(items):
        return None
    return [str(x).strip() or orig for x, orig in zip(out, items)]


def romanize_clip(words: list, s: float, e: float, cache_path) -> list:
    """Return the full word list with the words inside [s, e] in Roman script (others untouched).
    Results are cached per clip; words that fail to convert keep their original script."""
    idx = [i for i, w in enumerate(words) if w["e"] > s - 0.5 and w["s"] < e + 0.5 and is_native(w["w"])]
    if not idx:
        return words
    cache = load_json(cache_path) if cache_path.exists() else {}
    todo = [i for i in idx if str(i) not in cache]
    for b in range(0, len(todo), BATCH):
        chunk = todo[b:b + BATCH]
        conv = _convert([words[i]["w"] for i in chunk])
        if conv is None:  # one retry in halves - long batches are where models miscount
            half = len(chunk) // 2 or 1
            parts = [chunk[:half], chunk[half:]] if len(chunk) > 1 else [chunk]
            conv_parts = [(_convert([words[i]["w"] for i in p]) if p else []) for p in parts]
            if any(c is None for c in conv_parts):
                log("   Roman script: conversion failed for part of this clip - keeping original script there")
                conv_parts = [c if c is not None else [words[i]["w"] for i in p] for c, p in zip(conv_parts, parts)]
            conv = [x for c in conv_parts for x in c]
        cache.update({str(i): r for i, r in zip(chunk, conv)})
    save_json(cache_path, cache)
    out = list(words)
    for i in idx:
        if str(i) in cache:
            out[i] = {**words[i], "w": cache[str(i)]}
    return out


TITLE_PROMPT = """These are titles for short clips from a Hindi/Urdu GTA livestream. The audience reads ROMAN
Urdu/Hindi (Latin letters), like "Franklin Ne Banda Gaadi Par Fenk Diya".
Rewrite every title in casual Roman Urdu/Hindi, max 8 words, keep the same meaning and hook. Keep names
(Franklin, Lamar, Michael...) and common English gaming words (police, bike, yacht) as they are.
If a title is already Roman Urdu/Hindi, return it unchanged. Never use Devanagari or Urdu script.

Return ONLY JSON: {{"titles": [<exactly {n} strings, same order>]}}

TITLES:
{titles}
"""


def romanize_titles(titles: list[str], cache_path) -> list[str]:
    """English / native-script titles -> Roman Urdu/Hindi in one call. Cached; unchanged on failure."""
    import json

    cache = load_json(cache_path) if cache_path.exists() else {}
    todo = [t for t in dict.fromkeys(titles) if t and t not in cache]
    if todo:
        res = llm.ask_task("translit", TITLE_PROMPT.format(n=len(todo), titles=json.dumps(todo, ensure_ascii=False)),
                           temperature=0.3, label="Roman titles")
        out = res.get("titles") if isinstance(res, dict) else res if isinstance(res, list) else None
        if isinstance(out, list) and len(out) == len(todo):
            cache.update({t: (str(r).strip() or t) for t, r in zip(todo, out)})
            save_json(cache_path, cache)
        else:
            log("Roman titles: conversion failed - keeping original titles")
    return [cache.get(t, t) for t in titles]


def romanize_text(text: str) -> str:
    """Single short string (a title) -> Roman script. Returns the input unchanged on failure."""
    if not is_native(text):
        return text
    conv = _convert([text])
    return conv[0] if conv else text
