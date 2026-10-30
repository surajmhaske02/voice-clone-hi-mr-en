"""Local web app: clone a voice and speak English / Hindi / Marathi in it.

Engines: "best" / "balanced" = OmniVoice with a 10 s / 6 s voice prompt (accent + timbre),
         "fast" = Piper + OpenVoice (timbre only, ~0.5 s),
         "trained" = per-voice Piper model distilled from OmniVoice (distill/), Piper speed.
Audio is streamed clause by clause so playback starts before the whole sentence is done.

Run:  .venv-omni/bin/python app.py   ->  http://127.0.0.1:7860
"""
import base64, io, json, os, re, subprocess, tempfile, threading, time
import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from engines import LANGS, ROOT, BestEngine, FastEngine, TrainedEngine, crossfade_concat

VOICE_DIR = os.path.join(ROOT, "voices")
os.makedirs(VOICE_DIR, exist_ok=True)
SR_IN = 22050

fast = FastEngine()
best = BestEngine()
trained = TrainedEngine()
lock = threading.Lock()  # one inference at a time (MPS / torch not safe across request threads)
app = Flask(__name__, static_folder=None)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]", "_", (name or "").strip())[:40] or "voice"


def voice_path(name):
    return os.path.join(VOICE_DIR, safe_name(name))


def load_any_audio(file_storage, sr=SR_IN):
    """Decode browser webm/ogg/mp3/wav via ffmpeg to mono float32; trim edge silence."""
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_storage.filename or "")[1] or ".bin") as f:
        file_storage.save(f.name)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    idx = np.where(np.abs(audio) > 0.01)[0]
    return audio[idx[0]:idx[-1] + 1] if len(idx) else audio


def wav_b64(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "web"), "index.html")


@app.get("/api/voices")
def voices():
    out = []
    for n in sorted(os.listdir(VOICE_DIR)):
        meta = os.path.join(VOICE_DIR, n, "meta.json")
        if os.path.exists(meta):
            out.append({**json.load(open(meta)), "trained": TrainedEngine.langs(os.path.join(VOICE_DIR, n))})
    return jsonify(out)


@app.post("/api/enroll")
def enroll():
    if "audio" not in request.files:
        return jsonify(error="no audio"), 400
    name = safe_name(request.form.get("name"))
    try:
        audio = load_any_audio(request.files["audio"])
    except subprocess.CalledProcessError:
        return jsonify(error="could not decode audio"), 400
    secs = len(audio) / SR_IN
    if secs < 5:
        return jsonify(error=f"recording too short ({secs:.1f}s); speak 10-20 s"), 400
    vd = voice_path(name)
    os.makedirs(vd, exist_ok=True)  # keep voices/<name>/piper/: trained models take hours to rebuild
    for f in os.listdir(vd):
        if os.path.isfile(os.path.join(vd, f)):
            os.remove(os.path.join(vd, f))
    sf.write(os.path.join(vd, "ref.wav"), audio, SR_IN)
    with lock:
        t = time.perf_counter()
        try:
            fast.enroll(audio, vd)
            ref_text = best.enroll(audio, SR_IN, vd, ref_text=(request.form.get("ref_text") or "").strip() or None)
        except Exception as e:
            app.logger.exception("enroll failed")
            return jsonify(error=f"cloning failed: {e}"), 500
        finally:
            best.free_asr()
        ms = (time.perf_counter() - t) * 1000
    meta = {"name": name, "seconds": round(secs, 1), "ref_text": ref_text}
    json.dump(meta, open(os.path.join(vd, "meta.json"), "w"), ensure_ascii=False)
    return jsonify(**meta, ms=round(ms))


@app.get("/api/sample/<name>")
def sample(name):
    return send_from_directory(voice_path(name), "ref.wav")


@app.post("/api/speak")
def speak():
    d = request.get_json(force=True)
    vd, lang, text = voice_path(d.get("voice")), d.get("lang", "en"), (d.get("text") or "").strip()
    engine = d.get("engine", "best")
    if not os.path.exists(os.path.join(vd, "meta.json")) or lang not in LANGS or not text:
        return jsonify(error="pick a voice, language and text"), 400
    if engine == "trained" and lang not in TrainedEngine.langs(vd):
        return jsonify(error=f"no trained model for this voice in this language yet (see distill/README.md)"), 400

    def gen():
        with lock:
            t0 = time.perf_counter()
            if engine == "trained":
                sr, it = trained.SR, trained.speak(text, lang, vd)
            elif engine == "fast":
                sr, it = fast.SR, fast.speak(text, lang, vd, tau=float(d.get("tau", 0.3)), raw=bool(d.get("raw")))
            else:
                mode = engine if engine in best.MODES else "best"
                sr, it = best.SR, best.speak(text, lang, vd, mode=mode, steps=int(d.get("steps", 8)))
            chunks = []
            for i, c in enumerate(it):
                chunks.append(c)
                yield json.dumps({"i": i, "ms": round((time.perf_counter() - t0) * 1000),
                                  "audio": wav_b64(c, sr)}) + "\n"
            full = crossfade_concat(chunks, sr)
            yield json.dumps({"done": True, "ms": round((time.perf_counter() - t0) * 1000),
                              "seconds": round(len(full) / sr, 2), "audio": wav_b64(full, sr)}) + "\n"

    return Response(stream_with_context(gen()), mimetype="application/x-ndjson")


@app.post("/api/convert")
def convert_speech():
    """Speech-to-speech: recorded speech -> cloned voice (OpenVoice converter)."""
    vd = voice_path(request.form.get("voice"))
    if "audio" not in request.files or not os.path.exists(os.path.join(vd, "fast_se.pth")):
        return jsonify(error="pick a voice and record speech"), 400
    audio = load_any_audio(request.files["audio"])
    with lock:
        t = time.perf_counter()
        out = fast.speech_to_speech(audio, vd, tau=float(request.form.get("tau", 0.3)))
        ms = (time.perf_counter() - t) * 1000
    return jsonify(ms=round(ms), seconds=round(len(audio) / SR_IN, 2), audio=wav_b64(out, fast.SR))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, threaded=True)
