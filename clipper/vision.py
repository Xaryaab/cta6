"""Visual pass: Gemini WATCHES each shortlisted moment (low-res video + transcript), then a rerank pass
compares all of them head-to-head. Inspired by Opus Clip's ClipAnything and Clips Kitty's rerank."""
from pathlib import Path

from . import config, llm
from .ingest import download_section
from .select_llm import STYLES, context_lines, fit_clip, overlap, user_prompt_text
from .utils import load_json, log, save_json, short_hash

WATCH_PROMPT = """You are an elite short-form editor for gaming livestreams (GTA roleplay). Watch this {dur:.0f}-second
excerpt of a stream (video + audio). Its transcript, with timestamps in seconds FROM THE START OF THIS EXCERPT:
{transcript}

WHAT WE WANT: {style}
{user_prompt}
Decide the single best self-contained short clip inside this excerpt:
- start exactly where the hook begins (action or a strong line), no dead air
- end right after the payoff/reaction
- {min_len}-{max_len} seconds long
- judge what you SEE and HEAR (crashes, chases, reactions, facial expressions, tone), not just the words

Return ONLY JSON:
{{"keep": <true if this is genuinely post-worthy, else false>,
  "start": <seconds from excerpt start>, "end": <seconds from excerpt start>,
  "title": "<scroll-stopping hook caption, max 8 words, no hashtags, no emojis>",
  "what_happens": "<one sentence describing what visually happens>",
  "description": "<1-2 sentence social media caption>",
  "hashtags": ["<5 relevant hashtags without #>"],
  "score": <1-100 how viral this clip would be>}}
"""

RERANK_PROMPT = """You are deciding which clips a GTA streamer clipper should post. Compare these {k} candidate
clips HEAD-TO-HEAD (they were each scored separately, so their scores are not calibrated against each other).

WHAT WE WANT: {style}
{user_prompt}
Rank them by how likely each is to go viral on TikTok/Shorts. Drop clips that are weak, confusing without
context, or near-duplicates of a better one. Return at most {n}.

Return ONLY JSON: {{"order": [<clip ids, best first>]}}

CLIPS:
{blocks}
"""


def watch_and_rank(picks: list, url: str, sources: Path, work: Path, transcript: dict, n: int,
                   min_len: float, max_len: float, style: str, prompt: str) -> list:
    if not llm.available():
        return picks[:n]
    vdir = work / "vision"
    vdir.mkdir(exist_ok=True)
    style_text = STYLES.get(style, STYLES["auto"])
    words, segments = transcript["words"], transcript["segments"]

    watched = []
    for i, p in enumerate(picks, 1):
        a = max(0.0, p["start"] - config.VISION_PAD)
        b = p["end"] + config.VISION_PAD
        key = f"{int(a * 10)}_{int(b * 10)}_{style}_{short_hash(prompt)}"
        cache = vdir / f"{key}.json"
        if cache.exists():
            res = load_json(cache)
        else:
            log(f"Visual pass {i}/{len(picks)}: Gemini watches {a:.0f}s-{b:.0f}s ...")
            try:
                prev = download_section(url, sources, f"preview_{int(a * 10)}_{int(b * 10)}", a, b,
                                        max_height=config.PREVIEW_HEIGHT)
            except Exception as e:
                log(f"   preview download failed ({str(e)[:100]}) - keeping text-pass result")
                watched.append({**p, "id": i})
                continue
            res = _watch(prev, b - a, context_lines(segments, a, b, rel_to=a), style_text, prompt, min_len, max_len)
            if res is None:
                log("   visual pass unavailable - keeping text-pass result")
                watched.append({**p, "id": i})
                continue
            res["_offset"] = a
            save_json(cache, res)

        if not res.get("keep", True):
            log(f"   dropped after watching: {res.get('what_happens', '')[:90]}")
            continue
        try:
            s, e = a + float(res["start"]), a + float(res["end"])
            clip = fit_clip(words, s, e, min_len, max_len) or (p["start"], p["end"])
        except (KeyError, TypeError, ValueError):
            clip = (p["start"], p["end"])
        watched.append({
            "id": i, "start": round(clip[0], 2), "end": round(clip[1], 2),
            "title": (res.get("title") or p["title"]).strip(),
            "score": res.get("score", p.get("score", 0)),
            "reason": res.get("what_happens") or p.get("reason", ""),
            "description": res.get("description", ""),
            "hashtags": [h.lstrip("#") for h in (res.get("hashtags") or [])][:6],
            "watched": True,
        })

    # de-duplicate overlaps (keep higher score)
    watched.sort(key=lambda c: -(c.get("score") or 0))
    unique = []
    for c in watched:
        if all(overlap((c["start"], c["end"]), (u["start"], u["end"])) <= 0.3 for u in unique):
            unique.append(c)
    return _rerank(unique, transcript, n, style_text, prompt)


def _watch(video: Path, dur: float, transcript_text: str, style_text: str, prompt: str, min_len, max_len):
    from google.genai import types

    data = video.read_bytes()
    if len(data) > 18 * 1024 * 1024:
        log("   preview too large for inline upload - skipping visual pass for this clip")
        return None
    text = WATCH_PROMPT.format(dur=dur, transcript=transcript_text, style=style_text,
                               user_prompt=user_prompt_text(prompt), min_len=int(min_len), max_len=int(max_len))
    res = llm.ask_json([types.Part.from_bytes(data=data, mime_type="video/mp4"), text],
                       model=config.GEMINI_VISION_MODEL, temperature=0.3, low_media=True, label="Visual pass")
    if isinstance(res, list):
        res = res[0] if res and isinstance(res[0], dict) else None
    return res if isinstance(res, dict) else None


def _rerank(clips: list, transcript: dict, n: int, style_text: str, prompt: str) -> list:
    if len(clips) <= 1:
        return clips[:n]
    blocks = []
    for c in clips:
        speech = context_lines(transcript["segments"], c["start"], c["end"], rel_to=c["start"])[:500]
        blocks.append(f"--- Clip {c['id']} ({c['end'] - c['start']:.0f}s, individual score {c.get('score')}) ---\n"
                      f"Title: {c['title']}\nWhat happens: {c.get('reason', '')}\nSpeech:\n{speech}")
    log(f"Rerank: comparing {len(clips)} clips head-to-head...")
    res = llm.ask_json(RERANK_PROMPT.format(k=len(clips), n=n, style=style_text, user_prompt=user_prompt_text(prompt),
                                            blocks="\n\n".join(blocks)), temperature=0.2, label="Rerank")
    order = res.get("order") if isinstance(res, dict) else res if isinstance(res, list) else None
    by_id = {c["id"]: c for c in clips}
    ranked = []
    for cid in order or []:
        try:
            c = by_id.pop(int(cid))
        except (KeyError, TypeError, ValueError):
            continue
        ranked.append(c)
    if not ranked:  # rerank failed -> fall back to individual scores
        ranked = sorted(clips, key=lambda c: -(c.get("score") or 0))
    return ranked[:n]
