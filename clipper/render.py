"""Turn a downloaded section into a 1080x1920 clip with burned captions."""
import os
import shutil
from pathlib import Path

from .utils import run

FONT_CANDIDATES = [r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arialbd.ttf"]


def prepare_fonts(job_dir: Path) -> None:
    fonts = job_dir / "fonts"
    fonts.mkdir(exist_ok=True)
    for f in FONT_CANDIDATES:
        if os.path.exists(f) and not (fonts / os.path.basename(f)).exists():
            shutil.copy(f, fonts)


def render_clip(job_dir: Path, src: Path, offset: float, duration: float, ass: Path, out: Path, layout: str) -> None:
    # ffmpeg runs inside job_dir with relative paths: avoids Windows drive-letter escaping in filters.
    rel = lambda p: Path(os.path.relpath(p, job_dir)).as_posix()
    subs = f"ass={rel(ass)}:fontsdir=fonts"
    if layout == "crop":
        graph = f"[0:v]scale=-2:1920,crop=1080:1920,{subs},setsar=1[v]"
    else:  # blur: full frame centered over a blurred, zoomed copy of itself
        graph = (
            "[0:v]split=2[a][b];"
            "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "boxblur=25:3,eq=brightness=-0.08[bg];"
            "[b]scale=1080:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{subs},setsar=1[v]"
        )
    run(
        [
            "ffmpeg", "-y", "-ss", f"{offset:.3f}", "-i", rel(src), "-t", f"{duration:.3f}",
            "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", rel(out),
        ],
        cwd=job_dir,
    )
