import os
import json
import queue
import threading
import datetime
import tempfile
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

from flask import Flask, render_template, Response, jsonify
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import whisper

app = Flask(__name__)
client = OpenAI()

# ----------------------------
# Configuration
# ----------------------------
samplerate = 16000
chunk_duration = 10
blocksize = int(samplerate * 0.5)

audio_q = queue.Queue()

is_recording = False
recording_lock = threading.Lock()

questions_and_answers = []  # Q&A history, replayed on SSE reconnect

# Per-client SSE queues
_sse_clients = []
_sse_clients_lock = threading.Lock()


def broadcast(event_type: str, data: dict):
    """Push an event into every connected client's queue."""
    payload = {"type": event_type, "data": data}
    with _sse_clients_lock:
        for q in _sse_clients:
            q.put(payload)


initial_prompt = (
    "This is a surgical morbidity and mortality (M&M) conference. "
    "The speakers are residents, attendings, and moderators discussing surgical cases. "
    "Common terms: EKG, troponin, lab, diabetes, GI, ICU, vasopressors, norepinephrine, "
    "intubation, extubation, wound infection, dehiscence, drain output."
)

print("Loading Whisper model...")
model = whisper.load_model("small")
print("Whisper model loaded.")

# ----------------------------
# OpenAI Q&A detection
# ----------------------------
def detect_and_answer_question(text: str):
    prompt = f"""
    You are an assistant for a Surgical Morbidity and Mortality Conference.

    Analyze the following transcript segment:
    "{text}"

    Task:
    1. Determine if this segment contains a question from the audience.
    2. If YES, extract and refine the question for grammar and clarity.
    3. Provide a medical answer if you can based on your knowledge.
    4. If you cannot answer due to insufficient context, state that clearly.

    Respond in this JSON format:
    {{
        "has_question": true/false,
        "question": "the refined question or null",
        "answer": "your answer or explanation of why you cannot answer"
    }}

    Only include "question" and "answer" if has_question is true.
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result = completion.choices[0].message.content.strip()

    try:
        if result.startswith("```json"):
            result = result.split("```json")[1].split("```")[0].strip()
        elif result.startswith("```"):
            result = result.split("```")[1].split("```")[0].strip()
        return json.loads(result)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse JSON: {result}")
        return {"has_question": False}


# ----------------------------
# Audio callback
# ----------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    if is_recording:
        audio_q.put(indata.copy())


# ----------------------------
# Processing thread
# ----------------------------
def process_audio():
    audio_buffer = []
    target_samples = int(chunk_duration * samplerate)
    print("[THREAD] Processing thread started.")

    while is_recording:
        # Collect audio blocks until we have a full chunk
        try:
            audio_block = audio_q.get(timeout=1)
        except queue.Empty:
            continue

        audio_buffer.append(audio_block)
        total_samples = sum(len(b) for b in audio_buffer)

        if total_samples < target_samples:
            continue

        # We have enough audio — process it
        audio = np.concatenate(audio_buffer, axis=0)[:target_samples]
        audio_buffer = []

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio, samplerate)
                tmp_path = tmp.name

            try:
                print(f"[WHISPER] Transcribing chunk...")
                result = model.transcribe(
                    tmp_path,
                    language="en",
                    task="transcribe",
                    initial_prompt=initial_prompt,
                )
                transcript = result["text"].strip()
                print(f"[WHISPER] Got: {transcript[:80]}...")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            # Skip empty or hallucinated outputs
            if (
                not transcript
                or transcript == initial_prompt.strip()
                or "The speakers are residents, attendings" in transcript
            ):
                print("[WHISPER] Skipping empty/hallucinated transcript.")
                continue

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Update log line on frontend
            broadcast("status", {"timestamp": timestamp, "text": f"Analyzing segment at {timestamp}…"})

            print(f"[GPT] Analyzing for questions...")
            analysis = detect_and_answer_question(transcript)
            print(f"[GPT] Result: {analysis}")

            if analysis.get("has_question"):
                question = analysis.get("question", "")
                answer = analysis.get("answer", "")

                qa_entry = {
                    "timestamp": timestamp,
                    "question": question,
                    "answer": answer,
                }
                questions_and_answers.append(qa_entry)
                broadcast("qa", qa_entry)
                print(f"[QA] Question detected and broadcast.")
            else:
                broadcast("no_question", {"timestamp": timestamp})
                print(f"[QA] No question detected.")

        except Exception as e:
            print(f"[ERROR] {e}")
            broadcast("error", {"message": str(e)})

    print("[THREAD] Processing thread exiting.")


stream = None
proc_thread = None


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_recording():
    global is_recording, stream, proc_thread

    with recording_lock:
        if is_recording:
            return jsonify({"status": "already_running"})

        is_recording = True

        # Flush stale audio
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

        stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            blocksize=blocksize,
        )
        stream.start()
        print("[STREAM] Audio stream started.")

        proc_thread = threading.Thread(target=process_audio, daemon=True)
        proc_thread.start()

    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop_recording():
    global is_recording, stream

    with recording_lock:
        if not is_recording:
            return jsonify({"status": "not_running"})

        is_recording = False

        if stream:
            stream.stop()
            stream.close()
            stream = None
            print("[STREAM] Audio stream stopped.")

    return jsonify({"status": "stopped"})


@app.route("/stream")
def sse_stream():
    """Server-Sent Events — each client gets its own queue."""
    client_q = queue.Queue()

    with _sse_clients_lock:
        _sse_clients.append(client_q)
    print(f"[SSE] Client connected. Total: {len(_sse_clients)}")

    def generate():
        try:
            # Replay Q&A history for this client
            for qa in questions_and_answers:
                yield "data: {}\n\n".format(json.dumps({"type": "qa", "data": qa}))

            while True:
                try:
                    event = client_q.get(timeout=30)
                    yield "data: {}\n\n".format(json.dumps(event))
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _sse_clients_lock:
                try:
                    _sse_clients.remove(client_q)
                except ValueError:
                    pass
            print(f"[SSE] Client disconnected. Total: {len(_sse_clients)}")

    return Response(generate(), mimetype="text/event-stream")


@app.route("/status")
def status():
    return jsonify({
        "is_recording": is_recording,
        "qa_count": len(questions_and_answers),
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False, port=5001)