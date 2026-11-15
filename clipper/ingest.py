"""Downloading via yt-dlp: metadata, audio for analysis, and short video sections for clips."""
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func

from .utils import log

_BASE = {"quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True}


def get_info(url: str) -> dict:
    with yt_dlp.YoutubeDL(_BASE) as ydl:
        info = ydl.extract_info(url, download=False)
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


def download_section(url: str, out_dir: Path, stem: str, start: float, end: float) -> Path:
    """Download just [start, end] of the video (<=1080p)."""
    existing = _find(out_dir, stem)
    if existing:
        return existing
    opts = {
        **_BASE,
        "format": "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
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
