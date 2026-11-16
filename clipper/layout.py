"""Facecam detection for the gaming split layout (facecam on top, gameplay below).
Gemini looks at a frame and returns the webcam overlay's bounding box (box_2d, 0-1000 normalised)."""
import json
import subprocess
from pathlib import Path

from . import llm
from .utils import load_json, log, run, save_json

PROMPT = """This is a frame from a gaming livestream. Is there a WEBCAM / FACECAM overlay showing the real streamer
(a live camera feed of a real person, usually a rectangle in a corner or side of the screen)?
Do NOT count in-game characters, avatars, or people in the game world.

Return ONLY JSON: {"facecam": [ymin, xmin, ymax, xmax]} using coordinates normalised to 0-1000 for the
whole webcam rectangle, or {"facecam": null} if there is no webcam overlay."""


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(video)], capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def parse_manual(value: str) -> dict:
    """'x,y,w,h' as fractions of the frame (e.g. 0.75,0,0.25,0.3)."""
    x, y, w, h = [float(v) for v in value.split(",")]
    return {"x": x, "y": y, "w": w, "h": h}


def detect_facecam(job_dir: Path, sources: list[Path]) -> dict | None:
    """Returns {"x","y","w","h"} as fractions of the frame, or None. Cached per job."""
    cache = job_dir / "facecam.json"
    if cache.exists():
        return load_json(cache).get("box")
    if not llm.available() or not sources:
        return None
    from google.genai import types

    boxes = []
    for src in sources[:2]:  # check two different moments; webcam must appear in both
        frame = job_dir / f"_frame_{src.stem}.jpg"
        try:
            run(["ffmpeg", "-y", "-ss", "3", "-i", src, "-frames:v", "1", "-q:v", "3", frame])
        except RuntimeError:
            continue
        res = llm.ask_json([types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg"), PROMPT],
                           temperature=0.0, label="Facecam detection")
        frame.unlink(missing_ok=True)
        fc = res.get("facecam") if isinstance(res, dict) else None
        if not (isinstance(fc, list) and len(fc) == 4):
            boxes.append(None)
            continue
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) / 1000 for v in fc]
        w, h = xmax - xmin, ymax - ymin
        boxes.append({"x": xmin, "y": ymin, "w": w, "h": h} if w > 0.06 and h > 0.06 and w * h < 0.5 else None)

    box = boxes[0] if boxes and all(boxes) else None
    if box and len(boxes) > 1:  # average the two detections
        box = {k: round((boxes[0][k] + boxes[1][k]) / 2, 4) for k in box}
    save_json(cache, {"box": box})
    log(f"Facecam: {'found at ' + str(box) if box else 'none detected'}")
    return box
