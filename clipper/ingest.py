"""Downloading via yt-dlp: metadata, audio for analysis, and short video sections for clips."""
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

from .utils import log

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
    if info.get("is_live"):
        raise SystemExit("Live streams aren't supported yet - wait for the VOD.")
    return info


def job_id(info: dict) -> str:
    return f"{(info.get('extractor_key') or 'src').lower()}_{info['id']}"


def _find(out_dir: Path, stem: str) -> Path | None:
    for p in out_dir.glob(f"{stem}.*"):
        if p.suffix not in (".part", ".ytdl") and not p.name.endswith(".part"):
            return p
    return None


def download_audio(url: str, out_dir: Path, start: float, end: float, full: bool) -> Path:
    """Audio-only download. When a --from/--to window is used, only that range is fetched."""
    stem = "audio_full" if full else f"audio_{int(start)}_{int(end)}"
    existing = _find(out_dir, stem)
    if existing:
        return existing
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
