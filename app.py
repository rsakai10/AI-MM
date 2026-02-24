import os
import json
import queue
import threading
import datetime
import tempfile
import warnings
import numpy as np
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

load_dotenv()

from flask import Flask, render_template, Response, jsonify, request
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import whisper
import pdfplumber

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

client = OpenAI()

KB_DIR = os.environ.get("KB_DIR", "./kb_store")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

kb_index = None
kb_meta: List[Dict[str, Any]] = []
kb_ready = False

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
pdf_summaries = []  # PDF summaries, replayed on SSE reconnect

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
# PDF Processing
# ----------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"--- Page {i+1} ---\n{page_text}\n\n"
    return text


def summarize_text(text, filename):
    """Generate a summary using GPT in structured M&M format."""
    # Truncate if text is very long (GPT has token limits)
    max_chars = 30000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Document truncated due to length...]"
    
    prompt = f"""
    You are analyzing a document from a Surgical Morbidity & Mortality (M&M) conference.
    
    Document: {filename}
    
    Content:
    {text}
    
    Provide a comprehensive M&M case analysis structured in the following format. If the document doesn't contain information for a specific section, write "Not specified in document" for that section:
    
    ## Case Information
    
    ## Case Review
    
    ## Potential Contributing Factors
    
    ## Cause Analysis
    
    ## Literature Review (medical literature, guidelines, or best practices)
    
    ## Summary & Action Items

    """
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    return completion.choices[0].message.content.strip()

# RAG functions
def load_kb():
    global kb_index, kb_meta, kb_ready

    kb_ready = False
    kb_meta = []
    kb_index = None

    kb_dir = Path(KB_DIR)
    index_path = kb_dir / "kb.index"
    meta_path = kb_dir / "kb_meta.jsonl"

    if not index_path.exists() or not meta_path.exists():
        print(f"[RAG] KB not found at {KB_DIR}. RAG will be disabled.")
        return

    kb_index = faiss.read_index(str(index_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            kb_meta.append(json.loads(line))

    kb_ready = True
    print(f"[RAG] KB loaded. Chunks={len(kb_meta)}")


def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def retrieve_context(query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    if not kb_ready or kb_index is None or not kb_meta:
        return []

    qvec = embed_query(query)
    scores, indices = kb_index.search(qvec, top_k)

    hits: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(kb_meta):
            continue
        item = kb_meta[idx]
        hits.append(
            {
                "score": float(score),
                "source": item.get("source", ""),
                "chunk_index": item.get("chunk_index", -1),
                "text": item.get("text", ""),
            }
        )
    return hits


def format_rag_context(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "No retrieved knowledge base context."

    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        source = h.get("source", "unknown")
        chunk_index = h.get("chunk_index", -1)
        score = h.get("score", 0.0)
        text = h.get("text", "")
        parts.append(
            f"[Retrieved Chunk {i}] "
            f"source={source}, chunk_index={chunk_index}, score={score:.4f}\n"
            f"{text}"
        )
    return "\n\n".join(parts)


load_kb()

# ----------------------------
# OpenAI Q&A detection
# ----------------------------

def detect_and_answer_question(text: str):
    # Step 1: detect whether there is a question and extract a refined question
    detection_prompt = f"""
You are an assistant for a Surgical Morbidity and Mortality Conference.

Analyze the following transcript segment:
\"\"\"{text}\"\"\"

Task:
1. Determine if this segment contains a question from the audience.
2. If YES, extract and refine the question for grammar and clarity.
3. Do not answer the question yet.

Respond in JSON only with this format:
{{
  "has_question": true/false,
  "question": "refined question string or null"
}}
"""

    detection_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": detection_prompt}],
        temperature=0.0,
    )

    detection_result = detection_completion.choices[0].message.content.strip()

    try:
        if detection_result.startswith("```json"):
            detection_result = detection_result.split("```json", 1)[1].split("```", 1)[0].strip()
        elif detection_result.startswith("```"):
            detection_result = detection_result.split("```", 1)[1].split("```", 1)[0].strip()
        detection = json.loads(detection_result)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse detection JSON: {detection_result}")
        return {"has_question": False}

    if not detection.get("has_question"):
        return {"has_question": False}

    question = (detection.get("question") or "").strip()
    if not question:
        return {"has_question": False}

    # Step 2: retrieve KB context using RAG
    hits = retrieve_context(question, top_k=RAG_TOP_K)
    rag_context = format_rag_context(hits)

    # Step 3: answer with RAG context
    answer_prompt = f"""
    You are an assistant for a Surgical Morbidity and Mortality Conference.
    
    You must answer the question using the retrieved knowledge base context when it is relevant.
    If the retrieved context is insufficient or not directly applicable, say what is missing and then provide a cautious answer based on general medical knowledge.
    Do not fabricate citations or facts.
    Be concise but clinically useful.
    
    Question:
    {question}
    
    Retrieved knowledge base context:
    {rag_context}
    
    Return JSON only in this format:
    {{
      "has_question": true,
      "question": "refined question string",
      "answer": "final answer",
      "rag_used": true/false,
      "retrieved_chunks": [
        {{
          "source": "path",
          "chunk_index": 0,
          "score": 0.0
        }}
      ]
    }}
    """


    answer_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.1,
    )

    answer_result = answer_completion.choices[0].message.content.strip()

    try:
        if answer_result.startswith("```json"):
            answer_result = answer_result.split("```json", 1)[1].split("```", 1)[0].strip()
        elif answer_result.startswith("```"):
            answer_result = answer_result.split("```", 1)[1].split("```", 1)[0].strip()
        parsed = json.loads(answer_result)

        # Fallback if model omitted retrieval metadata
        if "retrieved_chunks" not in parsed:
            parsed["retrieved_chunks"] = [
                {
                    "source": h["source"],
                    "chunk_index": h["chunk_index"],
                    "score": h["score"],
                }
                for h in hits
            ]
        if "rag_used" not in parsed:
            parsed["rag_used"] = len(hits) > 0

        return parsed
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse answer JSON: {answer_result}")
        return {
            "has_question": True,
            "question": question,
            "answer": "I detected a question, but the answer formatter failed. Please retry.",
            "rag_used": len(hits) > 0,
            "retrieved_chunks": [
                {
                    "source": h["source"],
                    "chunk_index": h["chunk_index"],
                    "score": h["score"],
                }
                for h in hits
            ],
        }
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
                    "rag_used": analysis.get("rag_used", False),
                    "retrieved_chunks": analysis.get("retrieved_chunks", []),
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


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Handle PDF upload, extract text, and generate summary."""
    print(f"[PDF] Upload request received")
    print(f"[PDF] request.files keys: {list(request.files.keys())}")
    print(f"[PDF] request.form keys: {list(request.form.keys())}")
    
    if "pdf" not in request.files:
        print("[PDF ERROR] No 'pdf' key in request.files")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["pdf"]
    print(f"[PDF] File object: {file}")
    print(f"[PDF] Filename: {file.filename}")
    
    if file.filename == "":
        print("[PDF ERROR] Empty filename")
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.lower().endswith(".pdf"):
        print(f"[PDF ERROR] Not a PDF: {file.filename}")
        return jsonify({"error": "File must be a PDF"}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"[PDF] Processing {filename}...")
        
        # Extract text
        text = extract_text_from_pdf(filepath)
        
        if not text.strip():
            os.unlink(filepath)
            print("[PDF ERROR] No text extracted")
            return jsonify({"error": "Could not extract text from PDF"}), 400
        
        print(f"[PDF] Extracted {len(text)} characters, generating summary...")
        
        # Generate summary
        summary = summarize_text(text, filename)
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        pdf_entry = {
            "timestamp": timestamp,
            "filename": filename,
            "summary": summary,
            "char_count": len(text)
        }
        
        pdf_summaries.append(pdf_entry)
        broadcast("pdf_summary", pdf_entry)
        
        print(f"[PDF] Summary generated and broadcast.")
        
        # Clean up temp file
        os.unlink(filepath)
        
        return jsonify({"status": "success", "summary": pdf_entry})
    
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        import traceback
        traceback.print_exc()
        if 'filepath' in locals() and os.path.exists(filepath):
            os.unlink(filepath)
        return jsonify({"error": str(e)}), 500


@app.route("/stream")
def sse_stream():
    """Server-Sent Events — each client gets its own queue."""
    client_q = queue.Queue()

    with _sse_clients_lock:
        _sse_clients.append(client_q)
    print(f"[SSE] Client connected. Total: {len(_sse_clients)}")

    def generate():
        try:
            # Replay history for this client
            for qa in questions_and_answers:
                yield "data: {}\n\n".format(json.dumps({"type": "qa", "data": qa}))
            for pdf in pdf_summaries:
                yield "data: {}\n\n".format(json.dumps({"type": "pdf_summary", "data": pdf}))

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
        "pdf_count": len(pdf_summaries),
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False, port=5001)