# Clipper

Paste a Twitch / YouTube / Kick VOD URL → get vertical (9:16) clips of the most viral moments, with TikTok-style captions burned in.

## How it works
1. Downloads **audio only** (fast, small)
2. Transcribes with word timestamps: **Groq Whisper** (free key), or local faster-whisper on CPU as a fallback
3. Downloads **chat replay** (Twitch/YouTube) and finds chat explosions
4. Scores every second: loudness spikes + chat spikes + hype words → top 30 candidates
5. **Gemini** (free key) picks the best N clips and writes hook titles
6. Downloads only those video sections → 1080x1920 + animated captions → `output/<job>/clips/`

## Setup (Windows)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
# optional offline transcription fallback:
uv pip install --python .venv -r requirements-local.txt
copy .env.example .env   # then paste your free keys
```
- Gemini key: https://aistudio.google.com/apikey
- Groq key: https://console.groq.com/keys

FFmpeg is downloaded automatically on the first run if it isn't installed.

## Usage
```powershell
.venv\Scripts\python -m clipper <url> --clips 8
.venv\Scripts\python -m clipper <url> --from 1:00:00 --to 2:00:00   # test a slice of a long VOD
.venv\Scripts\python -m clipper <url> --layout crop                  # center crop instead of blurred bg
.venv\Scripts\python -m clipper <url> --rerender                     # redo captions/render only
.venv\Scripts\python -m clipper <url> --repick                       # ask Gemini again
```
Every stage is cached in `output/<job>/`, so reruns are fast and free-tier rate limits just resume where they stopped.
