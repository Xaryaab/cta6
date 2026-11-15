"""Ask Gemini to choose the most viral moments among the signal-based candidates."""
import json
import re
import time

from . import config
from .utils import fmt_time, log

PROMPT = """You are an elite short-form video editor who makes viral TikTok / YouTube Shorts / Reels clips
from long gaming livestreams (especially GTA roleplay). Below are {k} candidate moments from a stream,
found by spikes in audio loudness, chat activity and hype words. Each has the transcript around it
with absolute timestamps in seconds.

Pick the {n} BEST clips. A great clip:
- starts right on the hook (no dead air, no "anyway so..."), ideally mid-action or on a strong line
- has a clear payoff: funny reaction, chaos, plot twist, rage, clutch play, wholesome or shocking moment
- is understandable without watching the rest of the stream
- ends right after the punchline/reaction (don't trail off)
- is between {min_len} and {max_len} seconds long

Skip candidates that are just loud gameplay with nothing interesting said, ads, or "be right back" segments.
Clips must not overlap. Use only timestamps that appear inside a candidate's transcript window.

Return ONLY a JSON array, best first, of objects:
{{"candidate": <id>, "start": <seconds>, "end": <seconds>, "title": "<hook caption, max 8 words, no hashtags>",
  "score": <1-100 virality>, "reason": "<one short sentence>"}}

Stream title: {title}

CANDIDATES:
{blocks}
"""


def _context(segments: list, a: float, b: float) -> str:
    lines = [f"[{s['s']:.1f}] {s['text']}" for s in segments if s["e"] >= a and s["s"] <= b]
    return "\n".join(lines) or "(no speech)"


def select_clips(candidates: list, transcript: dict, n: int, min_len: float, max_len: float, title: str) -> list:
    segments = transcript["segments"]
    windows = {}
    blocks = []
    for c in candidates:
        a, b = c["peak"] - config.CONTEXT_BEFORE, c["peak"] + config.CONTEXT_AFTER
        windows[c["id"]] = (a, b)
        blocks.append(
            f"--- Candidate {c['id']} (peak at {c['peak']:.0f}s / {fmt_time(c['peak'])}, "
            f"signal score {c['score']}, chat {c['chat']}, audio {c['audio']}) ---\n{_context(segments, a, b)}"
        )

    picks = None
    if config.GEMINI_API_KEY:
        prompt = PROMPT.format(k=len(candidates), n=n, min_len=int(min_len), max_len=int(max_len),
                               title=title, blocks="\n\n".join(blocks))
        picks = _gemini(prompt)
    else:
        log("No GEMINI_API_KEY - falling back to signal-only picks (add a key for much better clips).")

    if not picks:
        picks = [{"candidate": c["id"], "start": c["peak"] - 20, "end": c["peak"] + 15, "title": "",
                  "score": 0, "reason": "signal-only"} for c in candidates]

    words = transcript["words"]
    final = []
    for p in picks:
        try:
            s, e = float(p["start"]), float(p["end"])
        except (KeyError, TypeError, ValueError):
            continue
        win = windows.get(p.get("candidate"))
        if win:
            s, e = max(s, win[0] - 15), min(e, win[1] + 15)
        s, e = _snap(words, s, e)
        if e - s < min_len:
            e = s + min_len
        if e - s > max_len:
            e = _snap(words, s, s + max_len)[1]
        if any(_overlap((s, e), (f["start"], f["end"])) > 0.3 for f in final):
            continue
        final.append({
            "start": round(s, 2), "end": round(e, 2),
            "title": str(p.get("title") or "").strip(),
            "score": p.get("score", 0), "reason": p.get("reason", ""),
        })
        if len(final) >= n:
            break
    return final


def _gemini(prompt: str) -> list | None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    for attempt in range(5):
        try:
            log(f"Asking Gemini ({config.GEMINI_MODEL}) to pick the best moments...")
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
            )
            data = _parse_json(resp.text or "")
            if isinstance(data, dict):
                data = next((v for v in data.values() if isinstance(v, list)), [])
            if isinstance(data, list) and data:
                return data
            log("Gemini returned no usable clips.")
            return None
        except Exception as e:
            msg = str(e)
            wait = 40 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 5 * (attempt + 1)
            log(f"Gemini error: {msg[:160]} - retrying in {wait}s...")
            time.sleep(wait)
    return None


def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        return json.loads(m.group(0)) if m else None


def _snap(words: list, s: float, e: float) -> tuple[float, float]:
    """Move cut points to word boundaries so no word gets chopped."""
    inside = [w for w in words if w["e"] > s - 0.3 and w["s"] < e + 0.3]
    if inside:
        s = min(s, inside[0]["s"]) if inside[0]["s"] < s else inside[0]["s"]
        e = max(e, inside[-1]["e"]) if inside[-1]["e"] > e else e
    return max(0.0, s - 0.15), e + 0.35


def _overlap(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))
