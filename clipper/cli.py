"""python -m clipper <url> [options]  ->  output/<job>/clips/<settings>/*.mp4"""
import argparse
import time

import numpy as np

from . import config, llm
from .captions import TEMPLATES, build_ass
from .chat import fetch_chat
from .ingest import download_audio, download_section, get_info, job_id, set_cookies
from .layout import detect_facecam, parse_manual
from .render import prepare_fonts, render_clip
from .romanize import needs_roman, romanize_clip, romanize_text, romanize_titles
from .select_llm import LENGTHS, STYLES, select_clips
from .signals import find_candidates, loudness_per_second
from .speakers import game_lines, label_segments, tag_words
from .transcribe import transcribe
from .twitch_clips import fetch_vod_clips
from .utils import ensure_ffmpeg, fmt_time, load_json, log, parse_time, save_json, short_hash, slugify
from .vision import watch_and_rank


def _quota_exit():
    raise SystemExit("Gemini's free daily quota is used up, so clips can't be picked properly right now. "
                     "Everything done so far is saved - rerun the same command after the quota resets "
                     "(midnight Pacific time) and it picks up from here.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="clipper", description="Turn a stream/VOD URL into viral vertical clips.")
    ap.add_argument("url", help="Twitch / YouTube / Kick VOD or video URL")
    ap.add_argument("--clips", type=int, default=8, help="how many clips to make (default 8)")
    ap.add_argument("--style", choices=list(STYLES), default="auto", help="kind of moments to look for")
    ap.add_argument("--prompt", default="", help='custom request, e.g. "every police chase"')
    ap.add_argument("--length", choices=list(LENGTHS), default="auto",
                    help="auto=15-60s, short=10-30s, medium=30-60s, long=60-90s")
    ap.add_argument("--min-len", type=float, default=None, help="override min clip length (s)")
    ap.add_argument("--max-len", type=float, default=None, help="override max clip length (s)")
    ap.add_argument("--layout", choices=["auto", "split", "blur", "crop"], default="auto",
                    help="auto = facecam/gameplay split if a webcam is found, else blur")
    ap.add_argument("--facecam", default="", help="manual webcam box as fractions x,y,w,h (e.g. 0.75,0,0.25,0.3)")
    ap.add_argument("--captions", choices=list(TEMPLATES), default="bold", help="caption template")
    ap.add_argument("--script", choices=["auto", "roman", "native"], default="auto",
                    help="caption script for Hindi/Urdu: auto = Roman when the transcript is Devanagari/Urdu")
    ap.add_argument("--game-dialogue", choices=["small", "same", "hide"], default="small",
                    help="in-game character speech: small = faded line under the streamer's captions (default), "
                         "same = caption it like the streamer, hide = streamer only")
    ap.add_argument("--from", dest="t_from", help="start of window to analyse, e.g. 1:00:00")
    ap.add_argument("--to", dest="t_to", help="end of window to analyse, e.g. 2:00:00")
    ap.add_argument("--no-chat", action="store_true", help="skip chat replay download")
    ap.add_argument("--no-vision", action="store_true", help="skip the visual pass (faster, uses less quota)")
    ap.add_argument("--cookies", default=config.YTDLP_COOKIES_FILE, help="cookies.txt for age-restricted videos")
    ap.add_argument("--cookies-from-browser", default=config.YTDLP_COOKIES_BROWSER, help="firefox / edge / chrome / brave")
    ap.add_argument("--repick", action="store_true", help="ask the LLM again (ignore cached picks)")
    ap.add_argument("--rerender", action="store_true", help="redo captions + render only")
    args = ap.parse_args(argv)

    min_len, max_len = LENGTHS[args.length]
    min_len = args.min_len if args.min_len is not None else min_len
    max_len = args.max_len if args.max_len is not None else max_len

    t0 = time.time()
    ensure_ffmpeg()
    set_cookies(args.cookies, args.cookies_from_browser)

    log("Reading video info...")
    info = get_info(args.url)
    duration = float(info.get("duration") or 0)
    start = parse_time(args.t_from) or 0.0
    end = parse_time(args.t_to) or duration
    if not end:
        raise SystemExit("Couldn't determine video length; pass --to.")
    end = min(end, duration) if duration else end
    if end - start < min_len:
        raise SystemExit("Window is shorter than the minimum clip length.")
    full = start <= 0 and (not duration or end >= duration)
    if info.get("is_live"):
        # VOD of a stream that's still going: lock to a fixed window so the cache doesn't mix lengths
        log(f"Stream is still live - analysing the first {fmt_time(end)} recorded so far.")
        full = False
    tag = "full" if full else f"{int(start)}_{int(end)}"

    job_dir = config.OUTPUT_DIR / job_id(info)
    work = job_dir / f"work_{tag}"
    sources = job_dir / "sources"
    run_tag = f"{args.style}_{args.length}" + (f"_{slugify(args.prompt, 24)}" if args.prompt else "")
    clips_dir = job_dir / "clips" / run_tag
    for d in (job_dir, work, sources, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    save_json(job_dir / "meta.json", {"url": args.url, "title": info.get("title"), "uploader": info.get("uploader"),
                                      "duration": duration})
    log(f"'{info.get('title')}' ({fmt_time(duration)}) -> analysing {fmt_time(start)}-{fmt_time(end)}")
    log(f"Settings: {args.clips} clips | style {args.style} | {int(min_len)}-{int(max_len)}s | layout {args.layout} "
        f"| captions {args.captions}" + (f" | prompt: {args.prompt!r}" if args.prompt else ""))

    # 1. audio
    audio = download_audio(args.url, job_dir, start, end, full, info)

    # 2. transcript
    tr_path = work / "transcript.json"
    if tr_path.exists():
        transcript = load_json(tr_path)
    else:
        transcript = transcribe(audio, start, end - start, work)
        save_json(tr_path, transcript)
    log(f"Transcript: {len(transcript['words'])} words")

    # 3. audience signals: chat, viewer clips, most-replayed heatmap
    chat_path = work / "chat.json"
    if args.no_chat:
        chat = []
    elif chat_path.exists():
        chat = load_json(chat_path)
    else:
        chat = fetch_chat(args.url, start, end, args.cookies)
        save_json(chat_path, chat)

    tw_path = job_dir / "twitch_clips.json"
    if tw_path.exists():
        twitch = load_json(tw_path)
    else:
        twitch = fetch_vod_clips(info)
        if twitch:
            save_json(tw_path, twitch)
    heatmap = info.get("heatmap") or []
    if heatmap:
        log("Using YouTube 'most replayed' heatmap")

    # 4. signals -> candidates
    loud_path = work / "loudness.npy"
    if loud_path.exists():
        loud = np.load(loud_path)
    else:
        log("Analysing audio loudness...")
        loud = loudness_per_second(audio)
        np.save(loud_path, loud)
    candidates = find_candidates(start, end, loud, transcript["words"], chat, heatmap, twitch)
    save_json(work / "candidates.json", candidates)
    log(f"Candidate moments: {len(candidates)}")

    # 5. text pass (shortlist) -> 6. visual pass + rerank
    use_vision = llm.available() and not args.no_vision
    key = f"{args.clips}_{args.style}_{int(min_len)}_{int(max_len)}_{short_hash(args.prompt)}"
    shortlist_n = args.clips * 2 if use_vision else args.clips
    text_path = work / f"text_{key}.json"
    final_path = work / f"final_{key}{'' if use_vision else '_novision'}.json"
    if args.repick:
        text_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)

    if final_path.exists():
        picks = load_json(final_path)
    else:
        if text_path.exists():
            shortlist = load_json(text_path)
        else:
            shortlist = select_clips(candidates, transcript, shortlist_n, min_len, max_len, info.get("title") or "",
                                     args.style, args.prompt)
            if llm.text_available() and all(p.get("reason") == "signal-only" for p in shortlist):
                _quota_exit()
            save_json(text_path, shortlist)
        log(f"Shortlisted {len(shortlist)} clips")
        if use_vision:
            picks = watch_and_rank(shortlist, args.url, sources, work, transcript, args.clips,
                                   min_len, max_len, args.style, args.prompt)
        else:
            picks = shortlist[: args.clips]
        if use_vision and llm.quota_out():
            _quota_exit()
        save_json(final_path, picks)
    if not picks:
        raise SystemExit("No clips selected.")
    log(f"Final clips: {len(picks)}")

    # 7. download full-quality sections
    srcs = []
    for p in picks:
        s, e = p["start"], p["end"]
        pad_s = max(0.0, s - config.SECTION_PAD)
        srcs.append((download_section(args.url, sources, f"src_{int(s * 10)}_{int(e * 10)}", pad_s,
                                      e + config.SECTION_PAD), pad_s))

    # 8. layout
    facecam = parse_manual(args.facecam) if args.facecam else None
    layout = args.layout
    if layout in ("auto", "split") and not facecam:
        facecam = detect_facecam(job_dir, [s for s, _ in srcs])
    if layout == "auto":
        layout = "split" if facecam else "blur"
    if layout == "split" and not facecam:
        log("No facecam found - using blur layout instead (pass --facecam x,y,w,h to set it manually)")
        layout = "blur"
    log(f"Layout: {layout}")

    # 9. captions + render
    prepare_fonts(job_dir)
    words = transcript["words"]
    roman = args.script == "roman" or (args.script == "auto" and needs_roman(words))
    if roman:
        log("Hindi/Urdu script detected - captions and titles will be in Roman script")
        titles = romanize_titles([p["title"] for p in picks], work / "roman_titles.json")
        picks = [{**p, "title": t} for p, t in zip(picks, titles)]
    summary = []
    for i, (p, (src, pad_s)) in enumerate(zip(picks, srcs), 1):
        s, e = p["start"], p["end"]
        if roman:
            words = romanize_clip(words, s, e, work / "roman_words.json")
        if args.game_dialogue != "same":
            labels = label_segments(transcript["segments"], s, e, work / "speakers.json", roman)
            words = tag_words(words, transcript["words"], transcript["segments"], labels, s, e)
            lines = game_lines(transcript["segments"], labels)
        else:
            lines = []
            p = {**p, "title": romanize_text(p["title"])}  # safety net if the title call failed
        name = f"{i:02d}_{slugify(p['title'])}"
        out = clips_dir / f"{name}_{layout}_{args.captions}.mp4"
        log(f"[{i}/{len(picks)}] {fmt_time(s)}-{fmt_time(e)} ({e - s:.0f}s) score {p.get('score')}: {p['title']!r}")
        if out.exists() and not args.rerender:
            log("   already rendered")
        else:
            ass = work / f"{name}.ass"
            build_ass(words, s, e, p["title"], layout, ass, args.captions, args.game_dialogue, lines)
            log("   rendering...")
            render_clip(job_dir, src, s - pad_s, e - s, ass, out, layout, facecam)
        tags = " ".join(f"#{h}" for h in p.get("hashtags") or [])
        (clips_dir / f"{name}.txt").write_text(
            f"{p['title']}\n\n{p.get('description') or p.get('reason', '')}\n\n{tags}\n", encoding="utf-8")
        summary.append({**p, "file": str(out), "timestamp": fmt_time(s), "layout": layout})

    save_json(clips_dir / "clips.json", summary)
    log(f"Done in {fmt_time(time.time() - t0)} -> {clips_dir}")
    for c in summary:
        log(f"  [{c['timestamp']}] score {c.get('score')}: {c['title']}  - {c.get('reason', '')}")
