"""Viewer-made Twitch clips for a VOD: the strongest "people thought this was clip-worthy" signal.

Needs a free Twitch developer app (https://dev.twitch.tv/console/apps): TWITCH_CLIENT_ID + TWITCH_CLIENT_SECRET.
"""
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from . import config
from .utils import log

API = "https://api.twitch.tv/helix"


def _get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _token() -> str:
    data = urllib.parse.urlencode({
        "client_id": config.TWITCH_CLIENT_ID, "client_secret": config.TWITCH_CLIENT_SECRET,
        "grant_type": "client_credentials",
    }).encode()
    with urllib.request.urlopen("https://id.twitch.tv/oauth2/token", data=data, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))["access_token"]


def fetch_vod_clips(info: dict) -> list:
    """Returns [{"offset": seconds into VOD, "duration": s, "views": int, "title": str}] for this VOD."""
    if info.get("extractor_key") != "TwitchVod":
        return []
    if not (config.TWITCH_CLIENT_ID and config.TWITCH_CLIENT_SECRET):
        log("Tip: add TWITCH_CLIENT_ID/SECRET to .env to use viewer-made clips as a signal (free).")
        return []
    vod_id = str(info["id"]).lstrip("v")
    try:
        headers = {"Client-Id": config.TWITCH_CLIENT_ID, "Authorization": f"Bearer {_token()}"}
        video = _get(f"{API}/videos?id={vod_id}", headers)["data"][0]
        created = datetime.fromisoformat(video["created_at"].replace("Z", "+00:00"))
        params = {
            "broadcaster_id": video["user_id"], "first": 100,
            "started_at": created.isoformat().replace("+00:00", "Z"),
            "ended_at": (created + timedelta(days=2)).isoformat().replace("+00:00", "Z"),
        }
        clips, cursor = [], None
        for _ in range(20):  # max 2000 clips
            q = dict(params, **({"after": cursor} if cursor else {}))
            page = _get(f"{API}/clips?{urllib.parse.urlencode(q)}", headers)
            for c in page.get("data", []):
                if c.get("video_id") == vod_id and c.get("vod_offset") is not None:
                    clips.append({"offset": float(c["vod_offset"]), "duration": float(c.get("duration") or 30),
                                  "views": int(c.get("view_count") or 0), "title": c.get("title", "")})
            cursor = page.get("pagination", {}).get("cursor")
            if not cursor:
                break
        log(f"Viewer-made Twitch clips for this VOD: {len(clips)}")
        return clips
    except Exception as e:
        log(f"Twitch clips unavailable ({type(e).__name__}: {str(e)[:120]})")
        return []
