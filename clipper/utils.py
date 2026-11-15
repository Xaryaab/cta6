import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[clipper] {msg}", flush=True)


def ensure_ffmpeg() -> None:
    """Use system ffmpeg if present, otherwise download a static build (first run only)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    import static_ffmpeg  # downloads a static build on first use only, then reuses it

    static_ffmpeg.add_paths(weak=True)
    if not shutil.which("ffmpeg"):
        sys.exit("Could not set up ffmpeg. Install it manually and add it to PATH.")


def run(cmd: list, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    res = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        tail = "\n".join((res.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"Command failed: {cmd[0]} ...\n{tail}")
    return res


def parse_time(value: str | None) -> float | None:
    """'1:02:03' / '62:03' / '3723' -> seconds."""
    if value is None or value == "":
        return None
    parts = [float(p) for p in str(value).split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\s-]", "", text or "").strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:max_len].strip("_") or "clip"
