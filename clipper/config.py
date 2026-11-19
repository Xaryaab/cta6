import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

def _key(name: str) -> str:
    """Read an API key; treat placeholders like 'your_..._here' as not set."""
    val = os.getenv(name, "").strip().strip('"').strip("'")
    return "" if (not val or val.lower().startswith("your_")) else val


def _key_list(*names: str) -> list[str]:
    """Collect keys from single-key vars and comma-separated list vars, deduplicated, placeholders dropped."""
    out = []
    for name in names:
        for v in os.getenv(name, "").split(","):
            v = v.strip().strip('"').strip("'")
            if v and not v.lower().startswith("your_") and v not in out:
                out.append(v)
    return out


# Several free keys (from different Google accounts) are rotated when one hits its limit.
GEMINI_API_KEYS = _key_list("GEMINI_API_KEY", "GEMINI_API_KEYS")
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# OpenRouter (optional) handles the text-only steps (text pass, rerank) so Gemini quota is saved for watching video.
OPENROUTER_API_KEYS = _key_list("OPENROUTER_API_KEY", "OPENROUTER_API_KEYS")
OPENROUTER_MODELS = [m.strip() for m in (os.getenv("OPENROUTER_MODELS") or
                     "google/gemma-4-31b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free,inclusionai/ling-3.2-flash:free"
                     ).split(",") if m.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Model that WATCHES video in the visual pass (defaults to GEMINI_MODEL)
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "").strip() or GEMINI_MODEL
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL") or "gemini-3.7-flash"  # used when the main model returns 503
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT") or 600)  # seconds; chat replay is optional, never let it hang a run

GROQ_API_KEYS = _key_list("GROQ_API_KEY", "GROQ_API_KEYS")
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()
# Groq text models (free, very fast) used by the router for small text steps like rerank. Tried in order.
GROQ_TEXT_MODELS = [m.strip() for m in (os.getenv("GROQ_TEXT_MODELS") or
                    "openai/gpt-oss-120b,llama-3.3-70b-versatile,openai/gpt-oss-20b").split(",") if m.strip()]
# Groq's free tier rejects big prompts (tokens-per-minute cap) - prompts estimated above this skip Groq.
GROQ_MAX_PROMPT_TOKENS = int(os.getenv("GROQ_MAX_PROMPT_TOKENS") or 6000)

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
CENSOR_CAPTIONS = True      # mask swear words in burned-in captions (f***); audio untouched
SECTION_PAD = 2.0            # extra seconds downloaded around each clip
VISION_PAD = 10.0            # extra seconds around a shortlisted clip that Gemini gets to watch
VISION_STOP_EXTRA = 0.5      # visual pass stops once it has clips * (1 + this) good clips (saves Gemini quota)
PREVIEW_HEIGHT = 360         # resolution of the clips Gemini watches (small = fast + fewer tokens)
