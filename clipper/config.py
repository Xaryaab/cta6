import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()

LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "small").strip()

OUTPUT_DIR = ROOT / "output"

# Pipeline tuning
CHUNK_SECONDS = 600          # audio chunk size sent to Groq (~5 MB at 48 kbps mono)
CANDIDATES = 30              # candidate moments sent to the LLM
CANDIDATE_SPACING = 40       # min seconds between candidate peaks
CONTEXT_BEFORE = 45          # transcript context around a peak given to the LLM
CONTEXT_AFTER = 40
SECTION_PAD = 2.0            # extra seconds downloaded around each clip
