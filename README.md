# M&M Conference Assistant

A real-time web app that listens to a Surgical Morbidity & Mortality (M&M) conference, automatically detects questions from the audience, and generates AI-powered answers — all displayed live in your browser. Also supports uploading PDF case documents for instant AI-generated summaries.

---

## How It Works

### Live Audio Q&A

Every 10 seconds of audio is transcribed using OpenAI's Whisper speech-to-text model. The transcript is then analyzed by GPT-4o-mini to detect whether a question was asked. If one is found, the question and answer appear instantly on the webpage.

```
Microphone → Whisper (speech-to-text) → GPT-4o-mini (Q&A detection) → Browser
```

### PDF Document Analysis

Upload PDF case files, conference presentations, or medical reports. The app extracts all text using pdfplumber and sends it to GPT-4o-mini to generate a concise 3-5 paragraph summary covering main topics, key findings, and recommendations.

```
PDF Upload → Text Extraction → GPT-4o-mini (summarization) → Browser
```

### Real-Time Updates

The app uses **Server-Sent Events (SSE)** to push updates to your browser instantly. When you see `[SSE] Client connected. Total: 1` in the terminal, it means your browser has established a live connection and is ready to receive updates without refreshing the page.

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
├── .env
└── templates/
    └── index.html
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your OpenAI API key**

Create a `.env` file in the project root directory with your OpenAI API key:

```bash
# Create .env file
echo 'OPENAI_API_KEY=sk-your-actual-api-key-here' > .env
```

Or manually create `.env` and add:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Replace `sk-your-actual-api-key-here` with your actual OpenAI API key from https://platform.openai.com/api-keys

> **⚠️ Important:** Never commit `.env` to git. The `.env` file is already in `.gitignore` to protect your secrets.

**4. Run the app**

```bash
python app.py
```

You should see:
```
Loading Whisper model...
Whisper model loaded.
 * Running on http://127.0.0.1:5001
```

**5. Open your browser**

Go to `http://127.0.0.1:5001`

When the page loads, you'll see `[SSE] Client connected. Total: 1` in the terminal — this means your browser is connected and ready to receive real-time updates.

---

## Usage

### Live Audio Q&A

1. Click **Start Listening** — the app will begin capturing audio from your microphone
2. The status indicator will turn green and pulse
3. Speak naturally; the app processes audio in 10-second chunks
4. When a question is detected, it appears in the **Questions & Answers** panel (right side) with an AI-generated answer
5. Click **Stop Listening** to end the session

### PDF Document Summaries

1. Click **Choose PDF file** and select a document
2. Click **Analyze** — the app will extract text and generate a summary
3. The summary appears in the **Case Summaries** panel (left side) with the filename and timestamp
4. Upload multiple PDFs; each will appear as a separate card

### Persistence

All Q&A and PDF summaries from the current session are preserved:
- Reloading the page will not lose results
- Opening the page in multiple tabs shows the same data
- Closing all tabs will clear the session history

---

## Troubleshooting

**"Access to localhost was denied" (HTTP 403)**

This is caused by macOS AirPlay Receiver occupying port 5000. The app runs on port 5001 to avoid this, so make sure you're visiting `http://127.0.0.1:5001` and not port 5000.

**Page loads but no results appear**

Check the terminal where you ran `python app.py`. You should see log lines like:
- `[WHISPER]` — audio transcription
- `[GPT]` — question detection
- `[QA]` — question/answer broadcast
- `[PDF]` — PDF processing
- `[SSE]` — browser connections

If you see `[ERROR]`, the error message will tell you what went wrong.

**"[PDF ERROR] No 'pdf' key in request.files"**

The file upload isn't reaching the server. Check:
- You clicked "Choose PDF file" and selected a file
- The filename appears next to the upload button
- The "Analyze" button is enabled (not grayed out)

**Whisper produces empty or garbled output**

This can happen if the room is quiet or there's too much background noise. The app automatically skips empty transcripts and tries again with the next chunk.

**Microphone not detected**

Make sure your system has granted microphone permission to the terminal application you're using.

**What does "[SSE] Client connected. Total: 1" mean?**

This is normal! It means your browser has successfully connected to the real-time update stream. The number shows how many browser tabs are currently connected. For example:
- Open 1 tab → `Total: 1`
- Open 2 tabs → `Total: 2`  
- Close a tab → `Total: 1`

---

## Configuration

You can adjust these variables near the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `chunk_duration` | `10` | Seconds of audio per processing cycle |
| `model` | `"small"` | Whisper model size (`base`, `small`, `medium`, `large`) |
| `MAX_CONTENT_LENGTH` | `50 MB` | Maximum PDF file size |

Larger Whisper models are more accurate but slower. `small` is a good balance for most use cases.

---

## Understanding the Code

### The Big Picture

The app does three things simultaneously:
1. **Listens** to your microphone
2. **Processes** what it hears (transcription → Q&A detection)
3. **Broadcasts** results to your browser in real-time

### Key Components

**Audio Capture (`audio_callback`)**
- Runs automatically every 0.5 seconds
- Collects audio chunks and puts them in a queue

**Processing Thread (`process_audio`)**
- Runs in parallel (separate thread)
- Waits for 10 seconds of audio to accumulate
- Sends to Whisper → GPT-4o-mini → broadcasts results

**PDF Processing (`upload_pdf`)**
- Extracts text using pdfplumber
- Sends to GPT-4o-mini for summarization
- Broadcasts summary to all connected browsers

**Server-Sent Events (`/stream` route)**
- Each browser tab gets its own message queue
- Server pushes updates instantly when they happen
- Replays history when you reload the page

**Why Threading?**
The audio processing happens in a separate thread so the Flask web server can still respond to other requests (like PDF uploads or the `/stream` endpoint) while audio is being processed.

---

## Tech Stack

| Component | Tool |
|---|---|
| Web framework | Flask |
| Speech-to-text | OpenAI Whisper (local) |
| Question detection & answering | OpenAI GPT-4o-mini |
| PDF text extraction | pdfplumber |
| Audio capture | sounddevice |
| Browser updates | Server-Sent Events (SSE) |

---

## File Structure

```
your_project/
├── app.py              # Flask server + audio/PDF processing
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── templates/
    └── index.html     # Frontend UI
```

**app.py** contains all the backend logic:
- Routes (`/`, `/start`, `/stop`, `/upload_pdf`, `/stream`)
- Audio processing thread
- PDF text extraction and summarization
- Whisper and GPT integration

**index.html** is a self-contained single-page app:
- All HTML, CSS, and JavaScript in one file
- Establishes SSE connection on page load
- Handles file uploads and UI updates

---

## Tips

- **Multiple PDFs**: Upload as many as you want; they all stay in the left panel
- **Long Sessions**: The app remembers everything until you close all browser tabs
- **Multiple Users**: Each person can open the page on their own device and see the same results
- **Debug Mode**: All terminal logs are prefixed with `[WHISPER]`, `[GPT]`, `[PDF]`, etc. to make troubleshooting easy