"""Downloading via yt-dlp: metadata, audio for analysis, and short video sections for clips."""
import concurrent.futures as cf
import urllib.parse
import urllib.request
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

from .utils import log, run

_BASE = {
    "quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True,
    # YouTube needs a JS runtime to solve its player challenges (required for logged-in / age-restricted
    # playback). Use Node (or Deno if installed) and let yt-dlp fetch its solver script.
    "js_runtimes": {"node": {}, "deno": {}},
    "remote_components": ["ejs:github"],
}


def set_cookies(cookie_file: str = "", browser: str = "") -> None:
    """Needed for age-restricted / sub-only videos. A cookies.txt file is the most reliable on Windows."""
    if cookie_file:
        if not Path(cookie_file).exists():
            raise SystemExit(f"Cookies file not found: {cookie_file}")
        _BASE["cookiefile"] = cookie_file
        log(f"Using cookies file: {cookie_file}")
    elif browser:
        _BASE["cookiesfrombrowser"] = (browser.lower(),)
        log(f"Using cookies from browser: {browser}")


def get_info(url: str) -> dict:
    try:
        with yt_dlp.YoutubeDL({**_BASE, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).lower()
        has_cookies = "cookiefile" in _BASE or "cookiesfrombrowser" in _BASE
        if "confirm your age" in msg or "not a bot" in msg or "sign in" in msg:
            if has_cookies:
                raise SystemExit(
                    "The site still wants a login even with cookies. Your cookies.txt may be expired or from a\n"
                    "non-age-verified account - export a fresh one while logged in on youtube.com.")
            raise SystemExit(
                "This video needs a login (age-restricted or bot check).\n"
                "  1. Install the 'Get cookies.txt LOCALLY' browser extension\n"
                "  2. Open youtube.com while logged in (age-verified account) -> Export -> save as cookies.txt\n"
                "     in the project folder\n"
                "  3. In .env set: YTDLP_COOKIES_FILE=cookies.txt   (or pass --cookies cookies.txt)")
        if "subscriber" in msg or "sub-only" in msg:
            raise SystemExit("This VOD is subscriber-only. Use cookies from an account that's subscribed.")
        raise SystemExit(f"Couldn't open the link: {str(e).replace('ERROR: ', '')[:300]}")
    if info.get("is_live") and "/videos/" not in url:
        raise SystemExit("That's a live channel, not a VOD - pass the stream's /videos/<id> link instead.")
    return info


def job_id(info: dict) -> str:
    return f"{(info.get('extractor_key') or 'src').lower()}_{info['id']}"


def _find(out_dir: Path, stem: str) -> Path | None:
    for p in out_dir.glob(f"{stem}.*"):
        if p.suffix not in (".part", ".ytdl") and not p.name.endswith(".part"):
            return p
    return None


def _hls_audio(info: dict, start: float, end: float, out: Path) -> bool:
    """Fast path for HLS audio (Twitch): fetch the segments covering [start, end] in parallel, then trim to the
    exact window. yt-dlp's ranged download goes through ffmpeg one segment at a time - hours for a long VOD."""
    fmts = [f for f in info.get("formats") or [] if "m3u8" in (f.get("protocol") or "")]
    audio = [f for f in fmts if f.get("vcodec") == "none"] or sorted(fmts, key=lambda f: f.get("tbr") or 1e9)[:1]
    if not audio:
        return False
    m3u8 = audio[0]["url"]
    with urllib.request.urlopen(m3u8, timeout=30) as r:
        lines = r.read().decode("utf-8", "replace").splitlines()
    segs, t, dur = [], 0.0, None
    for ln in lines:
        if ln.startswith("#EXTINF:"):
            dur = float(ln[8:].split(",")[0])
        elif ln and not ln.startswith("#") and dur is not None:
            if t + dur > start and t < end:
                segs.append((t, urllib.parse.urljoin(m3u8, ln)))
            t += dur
            dur = None
    if not segs:
        return False

    def get(u):
        for attempt in range(5):
            try:
                with urllib.request.urlopen(u, timeout=30) as r:
                    return r.read()
            except Exception:
                if attempt == 4:
                    raise

    raw = out.with_suffix(".ts.part")
    log(f"   fetching {len(segs)} audio segments in parallel...")
    with cf.ThreadPoolExecutor(32) as ex, open(raw, "wb") as f:
        for data in ex.map(get, [u for _, u in segs]):
            f.write(data)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start - segs[0][0]:.3f}", "-i", raw, "-t", f"{end - start:.3f}",
         "-vn", "-c:a", "copy", out])
    raw.unlink(missing_ok=True)
    return out.exists()


def download_audio(url: str, out_dir: Path, start: float, end: float, full: bool, info: dict | None = None) -> Path:
    """Audio-only download. When a --from/--to window is used, only that range is fetched."""
    stem = "audio_full" if full else f"audio_{int(start)}_{int(end)}"
    existing = _find(out_dir, stem)
    if existing:
        return existing
    if info and "twitch" in (info.get("extractor_key") or "").lower():
        try:
            log("Downloading audio" + ("" if full else f" for window {int(start)}s-{int(end)}s") + " (fast)...")
            if _hls_audio(info, start, end or float(info.get("duration") or 0), out_dir / f"{stem}.m4a"):
                return out_dir / f"{stem}.m4a"
        except Exception as e:
            log(f"   fast audio download failed ({str(e)[:120]}) - using the normal download")
    log("Downloading audio" + ("" if full else f" for window {int(start)}s-{int(end)}s") + "...")
    opts = {**_BASE, "format": "bestaudio/best", "outtmpl": str(out_dir / f"{stem}.%(ext)s")}
    if not full:
        opts["download_ranges"] = download_range_func(None, [(start, end)])
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    path = _find(out_dir, stem)
    if not path:
        raise RuntimeError("Audio download failed.")
    return path


def download_section(url: str, out_dir: Path, stem: str, start: float, end: float, max_height: int = 1080) -> Path:
    """Download just [start, end] of the video (<= max_height)."""
    existing = _find(out_dir, stem)
    if existing:
        return existing
    opts = {
        **_BASE,
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/wv*+ba/w",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / f"{stem}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    path = _find(out_dir, stem)
    if not path:
        raise RuntimeError(f"Video section download failed ({start:.0f}-{end:.0f}s).")
    return path
