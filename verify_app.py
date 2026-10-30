"""End-to-end check of the running app: speak with every engine/language, then transcribe the
result with Whisper to confirm the words are actually there (I can't listen; this is the next best).

Usage: .venv-omni/bin/python verify_app.py <voice-name> [engine ...]
"""
import base64, io, json, sys, urllib.error, urllib.request
import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

URL = "http://127.0.0.1:7860/api/speak"
TEXT = {
    "en": "Hello friend, I am coming at four in the afternoon. Please keep the tea ready.",
    "hi": "नमस्ते दोस्त, मैं आज दोपहर चार बजे आ रहा हूँ। कृपया चाय तैयार रखना।",
    "mr": "नमस्कार मित्रांनो, मी आज दुपारी चार वाजता येतोय. कृपया चहा तयार ठेवा.",
}
WHISPER_LANG = {"en": "english", "hi": "hindi", "mr": "marathi"}

voice = sys.argv[1]
engines = sys.argv[2:] or ["best", "balanced", "fast"]
asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", device="cpu", dtype=torch.float32)

for engine in engines:
    for lang, text in TEXT.items():
        req = urllib.request.Request(URL, json.dumps({"voice": voice, "engine": engine, "lang": lang, "text": text}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            msgs = [json.loads(line) for line in urllib.request.urlopen(req)]
        except urllib.error.HTTPError as e:
            print(f"{engine:8s} [{lang}] skipped: {json.loads(e.read()).get('error')}", flush=True)
            continue
        first, done = msgs[0], msgs[-1]
        audio, sr = sf.read(io.BytesIO(base64.b64decode(done["audio"])))
        heard = asr({"raw": audio.astype(np.float32), "sampling_rate": sr},
                    generate_kwargs={"language": WHISPER_LANG[lang], "task": "transcribe"})["text"].strip()
        print(f"{engine:8s} [{lang}] first {first['ms']:5d} ms  done {done['ms']:5d} ms  "
              f"{done['seconds']:4.1f}s audio | heard: {heard}", flush=True)
