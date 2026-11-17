"""Chat replay (Twitch / YouTube) - the strongest 'something happened here' signal for streams."""
import threading
import time

from . import config
from .utils import log


def fetch_chat(url: str, start: float, end: float, cookies: str = "", timeout: float | None = None) -> list:
    """Returns [[time_seconds, message, author], ...]. Empty list if chat isn't available.
    Gives up after `timeout` seconds (Twitch replay pages slowly) and keeps whatever arrived."""
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        return []
    timeout = config.CHAT_TIMEOUT if timeout is None else timeout
    out, stop = [], threading.Event()

    def worker():
        try:
            chat = ChatDownloader(cookies=cookies or None).get_chat(url, start_time=int(start), end_time=int(end) + 1)
            for msg in chat:
                if stop.is_set():
                    return
                t = msg.get("time_in_seconds")
                if t is None:
                    continue
                author = (msg.get("author") or {}).get("name") or ""
                out.append([round(float(t), 1), msg.get("message") or "", author])
                if len(out) % 10000 == 0:
                    log(f"  ...{len(out)} chat messages")
        except Exception as e:  # chat is optional - never fail the job over it
            log(f"Chat unavailable ({type(e).__name__}: {e}). Continuing without chat signal.")

    log(f"Downloading chat replay (max {timeout / 60:.0f} min)...")
    th = threading.Thread(target=worker, daemon=True)
    t0 = time.time()
    th.start()
    th.join(timeout)
    if th.is_alive():
        stop.set()
        msgs = list(out)
        covered = msgs[-1][0] if msgs else start
        if covered >= end - 0.1 * (end - start):
            log(f"Chat download timed out but covered the window - using {len(msgs)} messages.")
            return msgs
        # partial chat would bias picks toward the covered part of the window
        log(f"Chat download timed out after {(time.time() - t0) / 60:.0f} min (only reached "
            f"{covered / 60:.0f} of {end / 60:.0f} min) - continuing without chat signal.")
        return []
    log(f"Chat messages: {len(out)}")
    return out
