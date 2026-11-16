"""Turn a downloaded section into a 1080x1920 clip with burned captions.

Layouts:
  blur  - full frame centred over a blurred copy of itself
  crop  - centre crop
  split - streamer facecam on top (35%), gameplay below (65%)  [Opus "gameplay" / StreamLadder style]
"""
import os
import shutil
from pathlib import Path

from .layout import probe_size
from .utils import run

FONT_CANDIDATES = [r"C:\Windows\Fonts\ariblk.ttf", r"C:\Windows\Fonts\arialbd.ttf"]
TOP_H = 672           # facecam area height in split layout (35% of 1920)
BOT_H = 1920 - TOP_H  # gameplay area


def prepare_fonts(job_dir: Path) -> None:
    fonts = job_dir / "fonts"
    fonts.mkdir(exist_ok=True)
    for f in FONT_CANDIDATES:
        if os.path.exists(f) and not (fonts / os.path.basename(f)).exists():
            shutil.copy(f, fonts)


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _split_graph(src: Path, facecam: dict, subs: str) -> str:
    sw, sh = probe_size(src)
    # facecam: crop the detected box to the top area's aspect ratio (centred), then scale
    fx, fy, fw, fh = facecam["x"] * sw, facecam["y"] * sh, facecam["w"] * sw, facecam["h"] * sh
    target = 1080 / TOP_H
    if fw / fh > target:
        cw, ch = fh * target, fh
    else:
        cw, ch = fw, fw / target
    cx, cy = fx + (fw - cw) / 2, fy + (fh - ch) / 2
    cw, ch = _even(cw), _even(ch)
    cx, cy = max(0, min(sw - cw, int(cx))), max(0, min(sh - ch, int(cy)))
    # gameplay: full-height centre crop to the bottom area's aspect ratio
    gw = _even(min(sw, sh * 1080 / BOT_H))
    gx = (sw - gw) // 2
    return (
        "[0:v]split=2[a][b];"
        f"[a]crop={cw}:{ch}:{cx}:{cy},scale=1080:{TOP_H},setsar=1[top];"
        f"[b]crop={gw}:{_even(sh)}:{gx}:0,scale=1080:{BOT_H},setsar=1[bot];"
        f"[top][bot]vstack=inputs=2,{subs},setsar=1[v]"
    )


def render_clip(job_dir: Path, src: Path, offset: float, duration: float, ass: Path, out: Path, layout: str,
                facecam: dict | None = None) -> None:
    # ffmpeg runs inside job_dir with relative paths: avoids Windows drive-letter escaping in filters.
    rel = lambda p: Path(os.path.relpath(p, job_dir)).as_posix()
    subs = f"ass={rel(ass)}:fontsdir=fonts"
    if layout == "split" and facecam:
        graph = _split_graph(src, facecam, subs)
    elif layout == "crop":
        graph = f"[0:v]scale=-2:1920,crop=1080:1920,{subs},setsar=1[v]"
    else:
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
