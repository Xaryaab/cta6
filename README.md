# Clipper

Paste a Twitch, YouTube or Kick VOD link and get vertical (9:16) clips of its best moments, with captions burned in
and a ready-to-paste title, caption and hashtags for each one. Built for clipping long gaming streams (GTA 6
marathons were the test bed) but works on any stream.

It runs on your machine with your own API keys. Gemini and Groq both have free tiers that are enough to try it.

## How it works

1. Downloads the **audio** only (Twitch audio is fetched in parallel, so an 18-hour VOD takes minutes)
2. Transcribes it with word timestamps: **Groq Whisper**, or local faster-whisper on CPU as a fallback
3. Collects audience signals: unique chatters and hype emotes from chat replay, viewer-made Twitch clips,
   YouTube's "most replayed" heatmap
4. Scores every second: loudness spikes, quiet-to-loud payoffs, hype words and the audience signals, then keeps the
   top 30 candidates
5. **Text pass:** an LLM reads the transcript around each candidate and shortlists twice as many clips as you asked for
6. **Visual pass:** Gemini watches each shortlisted moment (360p video and audio) and sets the cut points, title,
   description and hashtags
7. **Rerank:** the watched clips are compared against each other and the best N are kept
8. **Render:** finds the streamer's facecam for a facecam-over-gameplay layout, re-syncs captions to each downloaded
   section, masks swear words on screen (not in the audio) and writes the clips to `output/<job>/clips/`

Every stage is cached in `output/<job>/`, so reruns are fast and a run stopped by a rate limit resumes where it left off.

## Setup

You need **Python 3.10+**. FFmpeg is downloaded automatically on the first run if it isn't installed.
For YouTube links, [Node.js](https://nodejs.org) is also recommended (yt-dlp uses it to get past YouTube's checks).

```bash
git clone <this repo>
cd <repo folder>
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

macOS / Linux:
```bash
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and add your keys:

| Key | Needed for | Get one |
|---|---|---|
| `GEMINI_API_KEY` | picking and watching moments, facecam detection | https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | fast transcription | https://console.groq.com/keys |
| `OPENROUTER_API_KEYS` | optional: runs text-only steps on free models to save Gemini quota | https://openrouter.ai/keys |
| `TWITCH_CLIENT_ID` / `_SECRET` | optional: viewer-made clips as a signal | https://dev.twitch.tv/console/apps |

Free tiers have daily limits. You can list several keys comma-separated (`GEMINI_API_KEYS`, `GROQ_API_KEYS`) and
they're rotated when one runs out. No Groq key? Install `requirements-local.txt` to transcribe on CPU instead (slow).

## Usage

Windows uses `.venv\Scripts\python`, macOS / Linux `.venv/bin/python`:

```bash
python -m clipper <vod-url>                                  # 8 clips, sensible defaults
python -m clipper <vod-url> --clips 5 --style funny          # auto | funny | chaos | rage | drama | wholesome | clutch
python -m clipper <vod-url> --prompt "every police chase"    # describe what you want
python -m clipper <vod-url> --length short                   # auto 15-60s | short 10-30 | medium 30-60 | long 60-90
python -m clipper <vod-url> --captions hormozi               # bold | hormozi | minimal | karaoke
python -m clipper <vod-url> --layout crop                    # auto | split | blur | crop
python -m clipper <vod-url> --facecam 0,0,0.26,0.25          # set the webcam box yourself (x,y,w,h as fractions)
python -m clipper <vod-url> --from 1:00:00 --to 3:00:00      # only analyse part of the VOD
python -m clipper <vod-url> --no-vision                      # skip the visual pass (faster, less Gemini quota)
python -m clipper <vod-url> --rerender                       # redo captions and render only, no new picks
python -m clipper <vod-url> --repick                         # ask the AI to pick again
```

Run `python -m clipper --help` for every option, including `--script` (Roman-script captions for Hindi/Urdu streams)
and `--game-dialogue` (how in-game character speech is captioned).

Each clip comes with a `.txt` file holding a title, a caption and hashtags.

**Twitch VODs of streams that are still live** work too: the clipper analyses what has been recorded so far.

### Tips

- **Gemini overloaded (503 errors)?** The clipper switches to the fallback model and retries. If both stay down,
  rerun with `--no-vision`. Your transcript and picks are cached, so nothing is redone.
- **Wrong layout?** `--facecam` sets the webcam box by hand. `--layout crop` suits footage without a facecam box.
- **Swear words** are masked in captions by default. Set `CENSOR_CAPTIONS = False` in `clipper/config.py` to turn it off.
- **Age-restricted or sub-only VODs:** export a `cookies.txt` from your browser and set `YTDLP_COOKIES_FILE` in `.env`.

## Responsible use

Only clip content you have the rights or permission to use, such as a streamer's official clipping campaign, and
follow the terms of the platforms you download from and post to. You are responsible for what you publish.

## License

MIT - see [LICENSE](LICENSE).
