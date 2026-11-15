"""TikTok-style burned-in captions as an ASS subtitle file (word-by-word highlight + hook title)."""
from pathlib import Path

W, H = 1080, 1920
HIGHLIGHT = "&H0000FFFF&"  # yellow (ASS colours are &HBBGGRR)


def _t(sec: float) -> str:
    sec = max(0.0, sec)
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _clean(word: str) -> str:
    return word.replace("{", "(").replace("}", ")").replace("\\", "").strip().upper()


def _groups(words: list, max_words: int = 3, max_chars: int = 18) -> list:
    groups, cur = [], []
    for w in words:
        if cur and (
            len(cur) >= max_words
            or w["s"] - cur[-1]["e"] > 0.6
            or sum(len(x["w"]) for x in cur) + len(w["w"]) > max_chars
        ):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w["w"].rstrip().endswith((".", "?", "!", ",")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list, clip_start: float, clip_end: float, title: str, layout: str, out: Path) -> None:
    """words use absolute times; they are shifted so 0 = clip_start."""
    dur = clip_end - clip_start
    rel = [
        {"w": _clean(w["w"]), "s": w["s"] - clip_start, "e": w["e"] - clip_start}
        for w in words
        if w["e"] > clip_start and w["s"] < clip_end and _clean(w["w"])
    ]
    caption_margin = 470 if layout == "blur" else 520
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,4,2,70,70,{caption_margin},1
Style: Title,Arial Black,62,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,22,0,8,90,90,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if title:
        title_end = dur if layout == "blur" else min(dur, 4.0)
        events.append(f"Dialogue: 1,{_t(0)},{_t(title_end)},Title,,0,0,0,,{_clean(title)}")

    groups = _groups(rel)
    for gi, g in enumerate(groups):
        g_end = g[-1]["e"]
        if gi + 1 < len(groups) and groups[gi + 1][0]["s"] - g_end < 0.35:
            g_end = groups[gi + 1][0]["s"]
        for wi, w in enumerate(g):
            s = g[0]["s"] if wi == 0 else w["s"]
            e = g[wi + 1]["s"] if wi + 1 < len(g) else g_end
            if e <= s:
                e = s + 0.05
            parts = []
            for xi, x in enumerate(g):
                if xi == wi:
                    parts.append(f"{{\\c{HIGHLIGHT}\\fscx110\\fscy110}}{x['w']}{{\\r}}")
                else:
                    parts.append(x["w"])
            pop = "{\\fad(0,0)}" if wi else "{\\t(0,80,\\fscx104\\fscy104)\\t(80,160,\\fscx100\\fscy100)}"
            events.append(f"Dialogue: 0,{_t(s)},{_t(min(e, dur))},Caption,,0,0,0,,{pop}{' '.join(parts)}")

    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
