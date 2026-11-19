"""Burned-in captions as an ASS subtitle file: several templates + hook title, positioned per layout."""
import re
from pathlib import Path

from . import config

W, H = 1080, 1920

# ASS colours are &HAABBGGRR
YELLOW, GREEN, WHITE, BLACK = "&H0000FFFF", "&H004CFF00", "&H00FFFFFF", "&H00000000"

TEMPLATES = {
    # current style: 2-3 words, active word yellow + slight pop
    "bold": {"size": 88, "max_words": 3, "max_chars": 18, "highlight": YELLOW, "pop": 110, "outline": 7, "upper": True},
    # Hormozi / MrBeast style: 1-2 huge words, green active word, big pop
    "hormozi": {"size": 118, "max_words": 2, "max_chars": 14, "highlight": GREEN, "pop": 122, "outline": 9, "upper": True},
    # clean: white, smaller, no highlight, sentence case
    "minimal": {"size": 72, "max_words": 4, "max_chars": 26, "highlight": None, "pop": 100, "outline": 4, "upper": False},
    # karaoke: whole line shown, colour sweeps across words as they're spoken
    "karaoke": {"size": 84, "max_words": 4, "max_chars": 24, "highlight": YELLOW, "pop": 100, "outline": 6, "upper": True,
                "karaoke": True},
}

# In-game dialogue (when the streamer's speech is tagged separately): a smaller, faded line just under the
# main captions, top-aligned so a 2-row line grows downward, away from them. Alpha 0x60 = ~38% transparent.
GAME = {"size": 50, "max_chars": 70, "colour": "&H60FFFFFF", "outline": 3, "gap": 22, "min_show": 1.2}

# Where things go for each layout. margin = distance from the bottom (captions) / top (title).
POSITIONS = {
    "blur": {"caption_margin": 470, "title_margin": 300, "title_full": True},
    "crop": {"caption_margin": 520, "title_margin": 300, "title_full": False},
    "split": {"caption_margin": 600, "title_margin": 712, "title_full": False},
}


def _t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# Swear words are masked in captions (f***) - the audio is untouched.
_SWEAR = re.compile(r"(motherfuck|fuck|shit|bitch|cunt|nigg|faggot|whore|slut|asshole|pussy|dick|cock)\w*", re.I)
_SAFE = {"cockpit", "cocktail", "dickens", "shitake", "scunthorpe"}


def _censor(w: str) -> str:
    if w.lower().strip(".,!?'\"") in _SAFE:
        return w
    return _SWEAR.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), w)


def _clean(word: str, upper: bool) -> str:
    w = word.replace("{", "(").replace("}", ")").replace("\\", "").strip()
    if config.CENSOR_CAPTIONS:
        w = _censor(w)
    return w.upper() if upper else w


def _groups(words: list, max_words: int, max_chars: int) -> list:
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["s"] - cur[-1]["e"] > 0.6
                    or sum(len(x["w"]) for x in cur) + len(w["w"]) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w["w"].rstrip().endswith((".", "?", "!", ",")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _forward(ws: list) -> None:
    """Whisper sometimes returns overlapping / backwards word times (streamer + game voices talking over each
    other). Force them to move forward so two lines of the same track never sit on screen at once."""
    prev_s = -1e9
    for w in ws:
        w["s"] = max(w["s"], prev_s + 0.04)
        w["e"] = max(w["e"], w["s"] + 0.04)
        prev_s = w["s"]


def build_ass(words: list, clip_start: float, clip_end: float, title: str, layout: str, out: Path,
              template: str = "bold", game: str = "small", game_lines: list | None = None) -> None:
    """words use absolute times; they are shifted so 0 = clip_start.
    game: what to do with words tagged spk="game" (in-game dialogue) -
          "small" = faded line under the main captions, "same" = treat like the streamer, "hide" = drop them.
    game_lines: [{"text", "s", "e"}] in-game dialogue lines (absolute times) drawn in "small" mode - whole
          Whisper segments, whose timing is more reliable than the word times when voices overlap."""
    tp = TEMPLATES.get(template, TEMPLATES["bold"])
    pos = POSITIONS.get(layout, POSITIONS["blur"])
    dur = clip_end - clip_start
    rel = [
        {"raw": w["w"], "w": _clean(w["w"], tp["upper"]), "s": w["s"] - clip_start, "e": w["e"] - clip_start,
         "game": game != "same" and w.get("spk") == "game"}
        for w in words
        if w["e"] > clip_start and w["s"] < clip_end and _clean(w["w"], tp["upper"])
    ]
    rel = [w for w in rel if not w["game"]]
    _forward(rel)
    karaoke = tp.get("karaoke")
    primary, secondary = (tp["highlight"], WHITE) if karaoke else (WHITE, WHITE)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,{tp['size']},{primary},{secondary},{BLACK},&H96000000,-1,0,0,0,100,100,1,0,1,{tp['outline']},4,2,70,70,{pos['caption_margin']},1
Style: Game,Arial,{GAME['size']},{GAME['colour']},{GAME['colour']},&H60000000,&H00000000,-1,0,0,0,100,100,0,0,1,{GAME['outline']},0,8,90,90,{H - pos['caption_margin'] + GAME['gap']},1
Style: Title,Arial Black,62,{BLACK},{BLACK},{WHITE},{WHITE},-1,0,0,0,100,100,0,0,3,22,0,8,90,90,{pos['title_margin']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if title:
        title_end = dur if pos["title_full"] else min(dur, 4.0)
        events.append(f"Dialogue: 1,{_t(0)},{_t(title_end)},Title,,0,0,0,,{_clean(title, True)}")

    groups = _groups(rel, tp["max_words"], tp["max_chars"])
    for gi, g in enumerate(groups):
        g_end = g[-1]["e"]
        if gi + 1 < len(groups) and groups[gi + 1][0]["s"] - g_end < 0.35:
            g_end = groups[gi + 1][0]["s"]  # also cuts a group that would overlap the next one
        g_end = min(g_end, dur)

        if karaoke:  # one event per line, \kf sweeps colour across each word as it's spoken
            parts, cursor = [], g[0]["s"]
            for w in g:
                lead = max(0, int(round((w["s"] - cursor) * 100)))
                length = max(1, int(round((w["e"] - w["s"]) * 100)))
                parts.append((f"{{\\k{lead}}}" if lead else "") + f"{{\\kf{length}}}{w['w']}")
                cursor = w["e"]
            events.append(f"Dialogue: 0,{_t(g[0]['s'])},{_t(g_end)},Caption,,0,0,0,,{' '.join(parts)}")
            continue

        for wi, w in enumerate(g):
            s = g[0]["s"] if wi == 0 else w["s"]
            e = g[wi + 1]["s"] if wi + 1 < len(g) else g_end
            if e <= s:
                e = s + 0.05
            parts = []
            for xi, x in enumerate(g):
                if xi == wi and tp["highlight"]:
                    parts.append(f"{{\\c{tp['highlight']}&\\fscx{tp['pop']}\\fscy{tp['pop']}}}{x['w']}{{\\r}}")
                else:
                    parts.append(x["w"])
            pop = "" if wi else "{\\t(0,80,\\fscx104\\fscy104)\\t(80,160,\\fscx100\\fscy100)}"
            events.append(f"Dialogue: 0,{_t(s)},{_t(min(e, dur))},Caption,,0,0,0,,{pop}{' '.join(parts)}")

    if game == "small" and game_lines:
        gl = []
        for ln in sorted(game_lines, key=lambda x: x["s"]):
            a, b = ln["s"] - clip_start, ln["e"] - clip_start
            text = _clean(ln["text"], False)
            if b <= 0 or a >= dur or not text:
                continue
            # very long lines: split into halves at a space, time shared by length
            chunks = [text]
            while any(len(c) > GAME["max_chars"] for c in chunks):
                c = max(chunks, key=len)
                cut = c.rfind(" ", 0, len(c) // 2 + 10)
                if cut <= 0:
                    break
                k = chunks.index(c)
                chunks[k:k + 1] = [c[:cut], c[cut + 1:]]
            total = sum(len(c) for c in chunks) or 1
            t = a
            for c in chunks:
                span = (b - a) * len(c) / total
                gl.append([max(t, gl[-1][0] + 0.05 if gl else t), t + span, c])
                t += span
        for gi, (a, b, text) in enumerate(gl):
            b = max(b, a + GAME["min_show"])
            if gi + 1 < len(gl):
                b = min(b, gl[gi + 1][0])  # one game line on screen at a time
            a, b = max(0.0, a), min(b, dur)
            if b > a:
                events.append(f"Dialogue: 0,{_t(a)},{_t(b)},Game,,0,0,0,,{text}")

    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
