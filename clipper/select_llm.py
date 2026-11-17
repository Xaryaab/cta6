"""Text pass: Gemini reads the transcript around signal-based candidates and shortlists clips."""
from . import config, llm
from .utils import fmt_time, log

# Clip-type presets (like Opus Clip "genre", tuned for GTA / gaming streams)
STYLES = {
    "auto": "Any kind of highly shareable moment - funny, chaotic, dramatic, impressive or wholesome.",
    "funny": "FUNNY moments: jokes, banter, absurd situations, fails, laughing fits, unexpected one-liners, awkward RP.",
    "chaos": "CHAOTIC ACTION: police chases, shootouts, crashes, explosions, escapes, everything going wrong at once.",
    "rage": "RAGE: the streamer losing it, yelling, tilting, rants, dramatic overreactions, keyboard-slam energy.",
    "drama": "RP DRAMA / STORY: betrayals, confrontations, heated arguments, interrogations, plot twists, emotional scenes.",
    "wholesome": "WHOLESOME moments: kindness, sweet interactions, friendships, heartwarming chat or character moments.",
    "clutch": "SKILL / CLUTCH: insane driving, narrow escapes, clutch wins, heists pulled off, impressive plays.",
}
LENGTHS = {"auto": (15, 60), "short": (10, 30), "medium": (30, 60), "long": (60, 90)}

PROMPT = """You are an elite short-form editor who makes viral TikTok / YouTube Shorts / Reels clips from long
gaming livestreams (especially GTA roleplay). Below are {k} candidate moments found by spikes in audio
loudness, chat activity, viewer-made clips and hype words. Each has the transcript around it with absolute
timestamps in seconds.

WHAT TO LOOK FOR: {style}
{user_prompt}
Pick the {n} BEST clips. A great clip:
- starts right on the hook (no dead air, no "anyway so..."), ideally mid-action or on a strong line
- has a clear payoff: a reaction, punchline, twist or climax
- is understandable without watching the rest of the stream
- ends right after the punchline/reaction (don't trail off)
- is between {min_len} and {max_len} seconds long

Skip loud gameplay where nothing interesting is said, ads, and "be right back" segments.
Clips must not overlap. Use only timestamps inside a candidate's transcript window.

Write titles in the SAME LANGUAGE the streamer speaks.

Return ONLY a JSON array, best first, of objects:
{{"candidate": <id>, "start": <seconds>, "end": <seconds>, "title": "<hook caption, max 8 words, no hashtags>",
  "score": <1-100 virality>, "reason": "<one short sentence>"}}

Stream title: {title}

CANDIDATES:
{blocks}
"""


def user_prompt_text(prompt: str) -> str:
    if not prompt:
        return ""
    return (f"\nThe user specifically asked for: \"{prompt}\". ONLY pick moments that match this request. "
            "If fewer moments match, return fewer clips.\n")


def context_lines(segments: list, a: float, b: float, rel_to: float | None = None) -> str:
    lines = []
    for s in segments:
        if s["e"] >= a and s["s"] <= b:
            t = s["s"] - rel_to if rel_to is not None else s["s"]
            lines.append(f"[{t:.1f}] {s['text']}")
    return "\n".join(lines) or "(no speech)"


def select_clips(candidates: list, transcript: dict, n: int, min_len: float, max_len: float, title: str,
                 style: str = "auto", prompt: str = "") -> list:
    segments = transcript["segments"]
    before = max(config.CONTEXT_BEFORE, max_len * 0.75)
    after = max(config.CONTEXT_AFTER, max_len * 0.75)
    windows, blocks = {}, []
    for c in candidates:
        a, b = c["peak"] - before, c["peak"] + after
        windows[c["id"]] = (a, b)
        extra = f", viewer clips {c['twitch_clips']}" if c.get("twitch_clips") else ""
        blocks.append(
            f"--- Candidate {c['id']} (peak at {c['peak']:.0f}s / {fmt_time(c['peak'])}, signal score {c['score']}, "
            f"chat {c['chat']}, audio {c['audio']}{extra}) ---\n{context_lines(segments, a, b)}"
        )

    picks = None
    if llm.available():
        log(f"Text pass: Gemini reads {len(candidates)} candidates (style: {style}{', prompt' if prompt else ''})...")
        picks = llm.ask_json(PROMPT.format(
            k=len(candidates), n=n, min_len=int(min_len), max_len=int(max_len), title=title,
            style=STYLES.get(style, STYLES["auto"]), user_prompt=user_prompt_text(prompt), blocks="\n\n".join(blocks),
        ), label="Text pass")
        if isinstance(picks, dict):
            picks = next((v for v in picks.values() if isinstance(v, list)), None)
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
        clip = fit_clip(words, s, e, min_len, max_len)
        if not clip or any(overlap(clip, (f["start"], f["end"])) > 0.3 for f in final):
            continue
        final.append({
            "start": round(clip[0], 2), "end": round(clip[1], 2),
            "title": str(p.get("title") or "").strip(),
            "score": p.get("score", 0), "reason": p.get("reason", ""),
        })
        if len(final) >= n:
            break
    return final


def fit_clip(words: list, s: float, e: float, min_len: float, max_len: float) -> tuple | None:
    if e <= s:
        return None
    s, e = snap(words, s, e)
    if e - s < min_len:
        e = s + min_len
    if e - s > max_len:
        e = snap(words, s, s + max_len)[1]
        e = min(e, s + max_len + 1.0)
    return s, e


def snap(words: list, s: float, e: float) -> tuple[float, float]:
    """Move cut points to word boundaries so no word gets chopped."""
    inside = [w for w in words if w["e"] > s - 0.3 and w["s"] < e + 0.3]
    if inside:
        s = min(s, inside[0]["s"]) if inside[0]["s"] < s else inside[0]["s"]
        e = max(e, inside[-1]["e"]) if inside[-1]["e"] > e else e
    return max(0.0, s - 0.15), e + 0.35


def overlap(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))
