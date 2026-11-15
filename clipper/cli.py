"""python -m clipper <url> [options]  ->  output/<job>/clips/*.mp4"""
import argparse
import time

import numpy as np

from . import config
from .captions import build_ass
from .chat import fetch_chat
from .ingest import download_audio, download_section, get_info, job_id
from .render import prepare_fonts, render_clip
from .select_llm import select_clips
from .signals import find_candidates, loudness_per_second
from .transcribe import transcribe
from .utils import ensure_ffmpeg, fmt_time, load_json, log, parse_time, save_json, slugify


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="clipper", description="Turn a stream/VOD URL into viral vertical clips.")
    ap.add_argument("url", help="Twitch / YouTube / Kick VOD or video URL")
    ap.add_argument("--clips", type=int, default=8, help="how many clips to make (default 8)")
    ap.add_argument("--layout", choices=["blur", "crop"], default="blur", help="blur = full frame + blurred bg; crop = center crop")
    ap.add_argument("--from", dest="t_from", help="start of window to analyse, e.g. 1:00:00")
    ap.add_argument("--to", dest="t_to", help="end of window to analyse, e.g. 2:00:00")
    ap.add_argument("--min-len", type=float, default=15)
    ap.add_argument("--max-len", type=float, default=60)
    ap.add_argument("--no-chat", action="store_true", help="skip chat replay download")
    ap.add_argument("--repick", action="store_true", help="ask the LLM again (ignore cached picks)")
    ap.add_argument("--rerender", action="store_true", help="redo captions + render only")
    args = ap.parse_args(argv)

    t0 = time.time()
    ensure_ffmpeg()

    log("Reading video info...")
    info = get_info(args.url)
    duration = float(info.get("duration") or 0)
    start = parse_time(args.t_from) or 0.0
    end = parse_time(args.t_to) or duration
    if not end:
        raise SystemExit("Couldn't determine video length; pass --to.")
    end = min(end, duration) if duration else end
    if end - start < args.min_len:
        raise SystemExit("Window is shorter than --min-len.")
    full = start <= 0 and (not duration or end >= duration)
    tag = "full" if full else f"{int(start)}_{int(end)}"

    job_dir = config.OUTPUT_DIR / job_id(info)
    work = job_dir / f"work_{tag}"
    clips_dir = job_dir / "clips"
    for d in (job_dir, work, clips_dir):
        d.mkdir(parents=True, exist_ok=True)
    save_json(job_dir / "meta.json", {"url": args.url, "title": info.get("title"), "uploader": info.get("uploader"),
                                      "duration": duration})
    log(f"'{info.get('title')}' ({fmt_time(duration)}) -> analysing {fmt_time(start)}-{fmt_time(end)}")
    log(f"Job folder: {job_dir}")

    # 1. audio
    audio = download_audio(args.url, job_dir, start, end, full)

    # 2. transcript
    tr_path = work / "transcript.json"
    if tr_path.exists():
        transcript = load_json(tr_path)
    else:
        transcript = transcribe(audio, start, end - start, work)
        save_json(tr_path, transcript)
    log(f"Transcript: {len(transcript['words'])} words")

    # 3. chat
    chat_path = work / "chat.json"
    if args.no_chat:
        chat = []
    elif chat_path.exists():
        chat = load_json(chat_path)
    else:
        chat = fetch_chat(args.url, start, end)
        save_json(chat_path, chat)

    # 4. signals -> candidates
    loud_path = work / "loudness.npy"
    if loud_path.exists():
        loud = np.load(loud_path)
    else:
        log("Analysing audio loudness...")
        loud = loudness_per_second(audio)
        np.save(loud_path, loud)
    candidates = find_candidates(start, end, loud, transcript["words"], chat)
    save_json(work / "candidates.json", candidates)
    log(f"Candidate moments: {len(candidates)}")

    # 5. LLM selection
    picks_path = work / f"picks_{args.clips}.json"
    if picks_path.exists() and not args.repick:
        picks = load_json(picks_path)
    else:
        picks = select_clips(candidates, transcript, args.clips, args.min_len, args.max_len, info.get("title") or "")
        save_json(picks_path, picks)
    if not picks:
        raise SystemExit("No clips selected.")
    log(f"Selected {len(picks)} clips")

    # 6. download sections + captions + render
    prepare_fonts(job_dir)
    sources = job_dir / "sources"
    sources.mkdir(exist_ok=True)
    summary = []
    for i, p in enumerate(picks, 1):
        s, e = p["start"], p["end"]
        name = f"{i:02d}_{slugify(p['title'])}"
        out = clips_dir / f"{name}_{args.layout}.mp4"
        log(f"[{i}/{len(picks)}] {fmt_time(s)}-{fmt_time(e)} ({e - s:.0f}s) {p['title']!r}")
        if out.exists() and not args.rerender:
            log("   already rendered")
        else:
            pad_s = max(0.0, s - config.SECTION_PAD)
            src = download_section(args.url, sources, f"src_{int(s * 10)}_{int(e * 10)}", pad_s, e + config.SECTION_PAD)
            ass = work / f"{name}.ass"
            build_ass(transcript["words"], s, e, p["title"], args.layout, ass)
            log("   rendering...")
            render_clip(job_dir, src, s - pad_s, e - s, ass, out, args.layout)
        summary.append({**p, "file": str(out), "timestamp": fmt_time(s)})

    save_json(clips_dir / "clips.json", summary)
    log(f"Done in {fmt_time(time.time() - t0)} -> {clips_dir}")
    for c in summary:
        log(f"  [{c['timestamp']}] score {c['score']}: {c['title']}  - {c['reason']}")
