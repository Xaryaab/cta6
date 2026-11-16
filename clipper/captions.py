"""Burned-in captions as an ASS subtitle file: several templates + hook title, positioned per layout."""
from pathlib import Path

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


def _clean(word: str, upper: bool) -> str:
    w = word.replace("{", "(").replace("}", ")").replace("\\", "").strip()
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


def build_ass(words: list, clip_start: float, clip_end: float, title: str, layout: str, out: Path,
              template: str = "bold") -> None:
    """words use absolute times; they are shifted so 0 = clip_start."""
    tp = TEMPLATES.get(template, TEMPLATES["bold"])
    pos = POSITIONS.get(layout, POSITIONS["blur"])
    dur = clip_end - clip_start
    rel = [
        {"w": _clean(w["w"], tp["upper"]), "s": w["s"] - clip_start, "e": w["e"] - clip_start}
        for w in words
        if w["e"] > clip_start and w["s"] < clip_end and _clean(w["w"], tp["upper"])
    ]
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
            g_end = groups[gi + 1][0]["s"]
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

    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
