import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _key(name: str) -> str:
    """Read an API key; treat placeholders like 'your_..._here' as not set."""
    val = os.getenv(name, "").strip().strip('"').strip("'")
    return "" if (not val or val.lower().startswith("your_")) else val


GEMINI_API_KEY = _key("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Model that WATCHES video in the visual pass (defaults to GEMINI_MODEL)
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "").strip() or GEMINI_MODEL

GROQ_API_KEY = _key("GROQ_API_KEY")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()

LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "small").strip()

# Cookies for age-restricted videos (pick one): path to cookies.txt, or a browser name (firefox, edge, chrome, brave)
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "").strip().strip('"')
if YTDLP_COOKIES_FILE and not Path(YTDLP_COOKIES_FILE).is_absolute():
    YTDLP_COOKIES_FILE = str(ROOT / YTDLP_COOKIES_FILE)  # relative paths = relative to the project folder
YTDLP_COOKIES_BROWSER = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()

# Optional: viewer-made Twitch clips as a signal (free app at https://dev.twitch.tv/console/apps)
TWITCH_CLIENT_ID = _key("TWITCH_CLIENT_ID")
TWITCH_CLIENT_SECRET = _key("TWITCH_CLIENT_SECRET")

OUTPUT_DIR = ROOT / "output"

# Pipeline tuning
CHUNK_SECONDS = 600          # audio chunk size sent to Groq (~5 MB at 48 kbps mono)
CANDIDATES = 30              # candidate moments sent to the LLM
CANDIDATE_SPACING = 40       # min seconds between candidate peaks
CONTEXT_BEFORE = 45          # transcript context around a peak given to the LLM
CONTEXT_AFTER = 40
SECTION_PAD = 2.0            # extra seconds downloaded around each clip
VISION_PAD = 10.0            # extra seconds around a shortlisted clip that Gemini gets to watch
PREVIEW_HEIGHT = 360         # resolution of the clips Gemini watches (small = fast + fewer tokens)
