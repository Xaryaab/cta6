"""Who is talking: the streamer or a game character? Whisper hears both but doesn't say which, so each
transcript segment inside a clip is labelled:
  1. segments in Devanagari / Urdu script -> streamer (game characters in GTA speak English)
  2. the rest -> a fast LLM decides whether a line is streamer commentary or scripted in-game dialogue
Words then inherit the label of the segment they fall in. Only clip segments are labelled, so a whole
run costs a few small LLM calls; results are cached."""
import bisect
import json
import re

from . import llm
from .romanize import is_native
from .utils import load_json, log, save_json

PROMPT = """These transcript lines come from a GTA livestream clip, in order. The audio mixes the STREAMER
(live commentary, reactions, talking to chat) with GAME characters (scripted mission dialogue, cutscenes, NPCs:
Lamar, Franklin, Michael, Trevor, Jimmy, Simeon, police, pedestrians).
{lang_rule}
Lines marked "known": "streamer" are certainly the streamer - use them as context. Label every OTHER line.
Return ONLY JSON: {{"labels": {{"<id>": "streamer" | "game", ...}}}} with one entry for each id in {ids}.

LINES:
{lines}
"""

NATIVE_RULE = """The streamer speaks Hindi/Urdu. So a full, fluent English sentence - especially American slang,
swearing in character, or anything that sounds like a written script - is GAME dialogue. The streamer's English
is only short exclamations or a few words mixed into Hindi/Urdu ("bro", "let's go", "what", "oh no", "come on")."""

ENGLISH_RULE = """Scripted lines (character names, mission talk, in-character slang, NPC barks) are GAME dialogue;
reactions to what happens, talking to chat or about the game are the STREAMER."""


def _key(seg: dict) -> str:
    return f"{seg['s']:.2f}"


def label_segments(segments: list, s: float, e: float, cache_path, native_stream: bool) -> dict:
    """Returns {segment_key: "streamer" | "game"} for every segment overlapping [s, e]."""
    segs = [g for g in segments if g["e"] > s and g["s"] < e and g.get("text", "").strip()]
    cache = load_json(cache_path) if cache_path.exists() else {}
    todo = []
    for g in segs:
        k = _key(g)
        if k in cache:
            continue
        if is_native(g["text"]):
            cache[k] = "streamer"
        else:
            todo.append(g)
    if todo:
        todo_ids = {id(g) for g in todo}
        lines, ids = [], []
        for i, g in enumerate(segs):
            row = {"id": i, "text": g["text"]}
            if id(g) not in todo_ids:
                row["known"] = cache[_key(g)]
            else:
                ids.append(i)
            lines.append(row)
        res = llm.ask_task("speaker", PROMPT.format(lang_rule=NATIVE_RULE if native_stream else ENGLISH_RULE,
                                                    ids=ids, lines=json.dumps(lines, ensure_ascii=False)),
                           temperature=0.1, label="Speaker labels")
        labels = res.get("labels") if isinstance(res, dict) else None
        if isinstance(labels, dict) and all(str(i) in labels for i in ids):
            for i in ids:
                g = segs[i]
                cache[_key(g)] = "game" if str(labels[str(i)]).strip().lower().startswith("game") else "streamer"
        else:
            # without the LLM: on a Hindi/Urdu stream English lines are most likely the game; otherwise we
            # can't tell, so treat everything as the streamer (= the old single-track captions)
            log("   speaker labels unavailable - " + ("English lines treated as game dialogue" if native_stream
                                                      else "showing all speech as main captions"))
            for g in todo:
                cache[_key(g)] = "game" if native_stream else "streamer"
    save_json(cache_path, cache)
    return {_key(g): cache[_key(g)] for g in segs}


def _norm(t: str) -> str:
    return re.sub(r"[^\w]", "", (t or "").lower())


_align_cache: dict = {}


def _align(orig_words: list, segments: list) -> list:
    """Segment index for each word, matched by TEXT in order (Whisper's word times are too messy when voices
    overlap, but words and segment text come out in the same order). None where no match was found."""
    key = (id(orig_words), id(segments))
    if key in _align_cache:
        return _align_cache[key]
    out = [None] * len(orig_words)
    starts = [w["s"] for w in orig_words]
    wi = 0
    for si, g in enumerate(segments):
        # re-anchor on time each segment so one bad match can't derail everything after it
        anchor = bisect.bisect_left(starts, g["s"] - 2.0)
        if wi < anchor or (wi < len(starts) and starts[wi] > g["e"] + 2.0):
            wi = anchor
        for tok in (_norm(t) for t in g.get("text", "").split()):
            if not tok:
                continue
            for j in range(wi, min(wi + 6, len(orig_words))):  # small look-ahead for skipped/extra words
                if _norm(orig_words[j]["w"]) == tok:
                    for k in range(wi, j + 1):
                        out[k] = si
                    wi = j + 1
                    break
    _align_cache[key] = out
    return out


def tag_words(words: list, orig_words: list, segments: list, labels: dict, s: float, e: float) -> list:
    """Copy of `words` (possibly romanized, same order as `orig_words`) where words inside [s, e] carry "spk"
    from their segment: matched by text first, by time as a fallback."""
    seg_of = _align(orig_words, segments)
    segs = [g for g in segments if _key(g) in labels]
    out = list(words)
    for i, w in enumerate(words):
        if w["e"] <= s - 0.5 or w["s"] >= e + 0.5:
            continue
        best = segments[seg_of[i]] if seg_of[i] is not None else None
        if best is None or _key(best) not in labels:
            mid = (w["s"] + w["e"]) / 2
            best = next((g for g in segs if g["s"] - 0.05 <= mid <= g["e"] + 0.05), None)
            if best is None and segs:
                best = min(segs, key=lambda g: min(abs(mid - g["s"]), abs(mid - g["e"])))
        if best is not None:
            out[i] = {**w, "spk": labels[_key(best)]}
    return out


def game_lines(segments: list, labels: dict) -> list:
    """In-game dialogue as whole lines with Whisper's segment timing: [{"text", "s", "e"}]."""
    return [{"text": g["text"].strip(), "s": g["s"], "e": g["e"]}
            for g in segments if labels.get(_key(g)) == "game"]
