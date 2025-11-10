from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from datetime import datetime, UTC
from dotenv import load_dotenv
from gtts import gTTS
from cerebras.cloud.sdk import Cerebras
from openai import OpenAI  #only for transcripts
import os, sys, traceback

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_2")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY_2 in .env for transcription")
oa_client = OpenAI(api_key=OPENAI_API_KEY)

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise RuntimeError("Missing CEREBRAS_API_KEY in .env (use your csk-… key)")
cb_client = Cerebras(api_key=CEREBRAS_API_KEY)

CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
#qwen-3-235b-a22b-instruct-2507  for better use maybe


DEBUG = os.environ.get("EDRICO_DEBUG", "1") == "1"
def log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)



CAPTURE_DIR = os.path.expanduser(os.environ.get("EDRICO_CAPTURE_DIR", "/tmp/edrico-captures"))
os.makedirs(CAPTURE_DIR, exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = HERE

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  #50mb


#prompt
SYSTEM_MSG = """You are a careful, context-aware listener that answers short recorded questions. 
Listen to the entire clip and understand what the speaker means, not just what they say. 
Reply only to the main or final question in the clip. 
Answer in one short sentence or number. Never repeat, explain, or add filler. 
You understand conversational intent. 
If someone mentions a topic and then asks about it (for example, “Eldorado Gold is a mining company. Do you know what it specializes in?”), interpret it as a direct question about that topic (“What does Eldorado Gold specialize in?”). 
If they say “Do you know what…” or “Have you heard of…”, treat it as “What is…”. 

Examples: “What’s 14 times 13 again?” → “182.” 
“Who came up with evolution?” → “Darwin.” 
“That’s a good point, Mr. Kiladze… what did Heisenberg do?” → “Uncertainty principle.”
“We really like your work, Mr. Nicholas. How does a DCF evaluation happen?” → “Discounted cash flow analysis using projected future cash and discount rate.” 
“Hello, we really like your opinion, but now about car companies. Do you know what BYD is?” → “Chinese electric vehicle manufacturer.” 
“…” → “Unclear.”
"""

def transcribe_file(path: str) -> str:
    with open(path, "rb") as f:
        resp = oa_client.audio.transcriptions.create(
            model="whisper-tiny.en",
            #whisper-1
            #gpt-4o-mini-transcribe
            #gpt-4o-transcribe
            file=f,
            response_format="text",
            language="en",
        )
    return resp.strip() if isinstance(resp, str) else getattr(resp, "text", "").strip()

def synthesize_tts(text: str):
    try:
        if not text or not text.strip():
            return None
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        fname = f"edrico-tts-{ts}.mp3"
        out_path = os.path.join(CAPTURE_DIR, fname)
        gTTS(text=text, lang="en", slow=False).save(out_path)
        size = os.path.getsize(out_path)
        return {"url": f"/captures/{fname}", "mimetype": "audio/mpeg", "size": size, "filename": fname}
    except Exception as e:
        log("[tts] error:", e)
        return None


def _extract_content_from_completion(completion):
    try:
        log("[cerebras completion type]", type(completion))
        log("[cerebras completion raw]", completion)
    except Exception:
        pass

    if isinstance(completion, dict):
        chs = completion.get("choices") or []
        if chs:
            ch0 = chs[0]
            msg = ch0.get("message") if isinstance(ch0, dict) else None
            if msg:
                cnt = msg.get("content")
                if isinstance(cnt, str) and cnt.strip():
                    return cnt
            if isinstance(ch0, dict) and isinstance(ch0.get("text"), str):
                t = ch0["text"].strip()
                if t:
                    return t
            d = ch0.get("delta") if isinstance(ch0, dict) else None
            if isinstance(d, dict) and isinstance(d.get("content"), str):
                t = d["content"].strip()
                if t:
                    return t
        for k in ("output_text", "text", "content"):
            v = completion.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return ""

    try:
        chs = getattr(completion, "choices", None) or []
        if chs:
            ch0 = chs[0]
            msg = getattr(ch0, "message", None)
            if msg:
                cnt = getattr(msg, "content", None)
                if isinstance(cnt, str) and cnt.strip():
                    return cnt
            t = getattr(ch0, "text", None)
            if isinstance(t, str) and t.strip():
                return t
            d = getattr(ch0, "delta", None)
            if d:
                cnt2 = getattr(d, "content", None)
                if isinstance(cnt2, str) and cnt2.strip():
                    return cnt2
    except Exception as e:
        log("[extract error]", e, traceback.format_exc())

    try:
        return str(completion)
    except Exception:
        return ""

def ask_assistant(prompt: str) -> str:
    try:
        completion = cb_client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.2,
            top_p=1.0,
            max_completion_tokens=64,
        )
        content = _extract_content_from_completion(completion)
        text = content.strip() if isinstance(content, str) else ""
        if not text:
            log("[empty content] falling back to Unclear.")
            return "Unclear."
        return text
    except Exception as e:
        log("[ask_assistant exception]", repr(e))
        log(traceback.format_exc())
        return f"(assistant error: {e.__class__.__name__}: {e})"


#routes

@app.post("/transcribe")
def transcribe_only():
    log("[transcribe] incoming request, content_length=", request.content_length)
    if "audio" not in request.files:
        return jsonify(ok=False, error="no audio file field"), 400

    file = request.files["audio"]
    ext = ".wav"
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    fname = f"edrico-{ts}{ext}"
    save_path = os.path.join(CAPTURE_DIR, fname)
    file.save(save_path)
    size = os.path.getsize(save_path)
    log(f"[transcribe] saved {fname} ({size} bytes) -> {save_path}")

    try:
        transcript = transcribe_file(save_path)
        return jsonify(
            ok=True,
            filename=fname,
            url=f"/captures/{fname}",
            mimetype=(file.mimetype or "").lower(),
            size=size,
            transcript=transcript,
        ), 200
    except Exception as e:
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
        log("[transcribe] error:", err)
        return jsonify(
            ok=False,
            filename=fname,
            url=f"/captures/{fname}",
            mimetype=(file.mimetype or "").lower(),
            transcript="",
            error=err
        ), 502

@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        prompt = (data.get("text") or "").strip()
        if not prompt:
            return jsonify(ok=False, error="missing 'text'"), 400
        reply = ask_assistant(prompt)
        tts_info = synthesize_tts("" if reply.startswith("(assistant error:") else reply)
        return jsonify(ok=True, assistant_reply=reply, tts=tts_info), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 502

@app.get("/health")
def health():
    return jsonify(ok=True)

@app.get("/")
def serve_index():
    return app.send_static_file("index.html")

@app.get("/captures/<path:fname>")
def serve_capture(fname):
    return send_from_directory(CAPTURE_DIR, fname, conditional=True)

@app.post("/capture")
def capture():
    log("[capture] incoming request, content_length=", request.content_length)
    if "audio" not in request.files:
        log("[capture] no audio field, keys:", list(request.files.keys()))
        return jsonify(ok=False, error="no audio file field"), 400

    file = request.files["audio"]
    ext = ".wav"
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    fname = f"edrico-{ts}{ext}"
    save_path = os.path.join(CAPTURE_DIR, fname)
    file.save(save_path)
    size = os.path.getsize(save_path)
    log(f"[capture] saved file {fname} ({size} bytes) -> {save_path}")

    try:
        transcript = transcribe_file(save_path)
        reply = ask_assistant(transcript) if transcript else "Unclear."
        tts_info = synthesize_tts("" if reply.startswith("(assistant error:") else reply)

        log(f"[capture] transcript len={len(transcript)}; reply len={len(reply)}; tts={'ok' if tts_info else 'none'}")
        return jsonify(
            ok=True,
            filename=fname,
            url=f"/captures/{fname}",
            mimetype=(file.mimetype or "").lower(),
            size=size,
            transcript=transcript,
            assistant_reply=reply,
            tts=tts_info
        ), 200
    except Exception as e:
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
        log("[capture] error:", err)
        return jsonify(
            ok=False,
            filename=fname,
            url=f"/captures/{fname}",
            mimetype=(file.mimetype or "").lower(),
            transcript="",
            assistant_reply="",
            tts=None,
            error=err
        ), 502

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)