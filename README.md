# M&M Conference Assistant

A real-time web app that listens to a Surgical Morbidity & Mortality (M&M) conference, automatically detects questions from the audience, and generates AI-powered answers — all displayed live in your browser.

---

## How It Works

Every 10 seconds of audio is transcribed using OpenAI's Whisper speech-to-text model. The transcript is then analyzed by GPT-4o-mini to detect whether a question was asked. If one is found, the question and answer appear instantly on the webpage.

```
Microphone → Whisper (speech-to-text) → GPT-4o-mini (Q&A detection) → Browser
```

---

## Requirements

- Python 3.9+
- An OpenAI API key
- `ffmpeg` installed on your system
- PortAudio installed on your system (for microphone access)

### Installing system dependencies

**macOS:**
```bash
brew install ffmpeg portaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg libportaudio2
```

---

## Setup

**1. Clone or download this project**

```
your_project/
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."
```

To avoid doing this every time, add the line above to your `~/.zshrc` or `~/.bashrc` file.

**4. Run the app**

```bash
python app.py
```

**5. Open your browser**

Go to `http://127.0.0.1:5001`

---

## Usage

1. Open the app in your browser
2. Click **Start Listening** — the app will begin capturing audio from your microphone
3. Speak naturally; the app processes audio in 10-second chunks
4. When a question is detected, it appears on the page with an AI-generated answer
5. Click **Stop Listening** to end the session
6. All Q&A from the session is preserved — reloading the page will not lose results

---

## Troubleshooting

**"Access to localhost was denied" (HTTP 403)**
This is caused by macOS AirPlay Receiver occupying port 5000. The app runs on port 5001 to avoid this, so make sure you're visiting `http://127.0.0.1:5001` and not port 5000.

**Page loads but no results appear**
Check the terminal where you ran `python app.py`. You should see log lines like `[WHISPER]`, `[GPT]`, and `[QA]` as audio is processed. If you see `[ERROR]`, the error message will tell you what went wrong.

**Whisper produces empty or garbled output**
This can happen if the room is quiet or there's too much background noise. The app automatically skips empty transcripts and tries again with the next chunk.

**Microphone not detected**
Make sure your system has granted microphone permission to the terminal application you're using.

---

## Configuration

You can adjust these variables near the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `chunk_duration` | `10` | Seconds of audio per processing cycle |
| `model` | `"small"` | Whisper model size (`base`, `small`, `medium`, `large`) |

Larger Whisper models are more accurate but slower. `small` is a good balance for most use cases.

---

## Tech Stack

| Component | Tool |
|---|---|
| Web framework | Flask |
| Speech-to-text | OpenAI Whisper (local) |
| Question detection & answering | OpenAI GPT-4o-mini |
| Audio capture | sounddevice |
| Browser updates | Server-Sent Events (SSE) |