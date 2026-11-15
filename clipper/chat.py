"""Chat replay (Twitch / YouTube) - the strongest 'something happened here' signal for streams."""
from .utils import log


def fetch_chat(url: str, start: float, end: float) -> list:
    """Returns [[time_seconds, message], ...]. Empty list if chat isn't available."""
    try:
        from chat_downloader import ChatDownloader
    except ImportError:
        return []
    out = []
    try:
        log("Downloading chat replay (can take a while for long VODs)...")
        chat = ChatDownloader().get_chat(url, start_time=int(start), end_time=int(end) + 1)
        for msg in chat:
            t = msg.get("time_in_seconds")
            if t is None:
                continue
            out.append([round(float(t), 1), msg.get("message") or ""])
            if len(out) % 10000 == 0:
                log(f"  ...{len(out)} chat messages")
    except Exception as e:  # chat is optional - never fail the job over it
        log(f"Chat unavailable ({type(e).__name__}: {e}). Continuing without chat signal.")
    log(f"Chat messages: {len(out)}")
    return out
