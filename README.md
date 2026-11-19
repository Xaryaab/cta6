# Clipper

Paste a Twitch / YouTube / Kick VOD URL → get vertical (9:16) clips of the most viral moments, with TikTok-style captions burned in.

## How it works
1. Downloads **audio only** (fast, small)
2. Transcribes with word timestamps: **Groq Whisper** (free key), or local faster-whisper on CPU as a fallback
3. Audience signals: **unique chatters** + hype emotes from chat replay, **viewer-made Twitch clips**, YouTube **"most replayed"** heatmap
4. Scores every second: loudness spikes + quiet-to-loud "payoff" jumps + hype words + audience signals → top 30 candidates
5. **Text pass:** Gemini reads the transcript around each candidate and shortlists 2× the clips you asked for
6. **Visual pass:** Gemini *watches* each shortlisted moment (360p video + audio) and sets exact cut points, title, description and hashtags
7. **Rerank:** all watched clips are compared head-to-head and the best N are kept
8. Detects the streamer's **facecam** → facecam/gameplay split layout, captions in your chosen template → `output/<job>/clips/<style>/`

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
- **More free quota:** put extra Gemini keys (other Google accounts) in `GEMINI_API_KEYS=k2,k3,...` - they're rotated
  when one hits its limit. Add free OpenRouter keys in `OPENROUTER_API_KEYS=...` and the text-only steps (text pass,
  rerank) run there, so Gemini quota is kept for watching video.
- **Model routing:** each AI step goes to the provider that suits it, falling through to the next on failure.
  Text pass (long, needs judgment): OpenRouter → Groq → Gemini. Rerank (short): Groq → OpenRouter → Gemini.
  Watching video / facecam: Gemini only. Prompts too big for Groq's free tier skip it automatically, and the
  visual pass stops early once it has enough good clips.

FFmpeg is downloaded automatically on the first run if it isn't installed.

## Usage
```powershell
.venv\Scripts\python -m clipper <url> --clips 8
.venv\Scripts\python -m clipper <url> --style funny            # auto|funny|chaos|rage|drama|wholesome|clutch
.venv\Scripts\python -m clipper <url> --prompt "every police chase"
.venv\Scripts\python -m clipper <url> --length short           # auto 15-60s | short 10-30 | medium 30-60 | long 60-90
.venv\Scripts\python -m clipper <url> --captions hormozi        # bold | hormozi | minimal | karaoke
.venv\Scripts\python -m clipper <url> --layout split           # auto | split | blur | crop
.venv\Scripts\python -m clipper <url> --facecam 0.75,0,0.25,0.3  # set webcam box manually (x,y,w,h fractions)
.venv\Scripts\python -m clipper <url> --from 1:00:00 --to 2:00:00  # only analyse a slice
.venv\Scripts\python -m clipper <url> --no-vision              # skip the visual pass (faster)
.venv\Scripts\python -m clipper <url> --rerender / --repick
```
Each clip also gets a `.txt` with a ready-to-paste title, description and hashtags.

Optional: add `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET` (free app at https://dev.twitch.tv/console/apps) to use viewer-made clips as a signal.

Every stage is cached in `output/<job>/`, so reruns are fast and free-tier rate limits just resume where they stopped.
