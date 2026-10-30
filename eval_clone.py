"""Objective clone accuracy + latency for every engine (fast / balanced / best) in en, hi, mr.

Accuracy = ECAPA speaker-verification cosine between generated speech and a HELD-OUT part of
the speaker's real recording (higher = sounds more like them; same-person real speech ~0.6-0.8).
Usage: .venv-omni/bin/python eval_clone.py   (stop the web app first: both use the GPU)
"""
import os, tempfile, time
import numpy as np
import librosa
import torch
from speechbrain.inference.speaker import EncoderClassifier

from engines import FastEngine, BestEngine, ROOT

RES = os.path.join(ROOT, "OpenVoice/resources")
SPEAKERS = {  # name: (file, enroll window s, held-out window s)
    "spk0_male_en": ("demo_speaker0.mp3", (0, 10), (10, 19)),
    "spk2_male_en": ("demo_speaker2.mp3", (0, 12), (15, 29)),
    "ref_female_en": ("example_reference.mp3", (0, 12), (30, 58)),
}
TEXT = {
    "en": "Hello friend, I am coming at four in the afternoon. Please keep the tea ready.",
    "hi": "नमस्ते दोस्त, मैं आज दोपहर चार बजे आ रहा हूँ। कृपया चाय तैयार रखना।",
    "mr": "नमस्कार मित्रांनो, मी आज दुपारी चार वाजता येतोय. कृपया चहा तयार ठेवा.",
}
ENGINES = ("fast", "balanced", "best")

ecapa = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=os.path.join(ROOT, "cache/ecapa"), run_opts={"device": "cpu"})


def emb(wav, sr):
    w = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        return ecapa.encode_batch(torch.from_numpy(w)[None]).flatten()


def run(gen):
    """gen yields chunks; returns (audio, first-chunk ms, total ms)."""
    t = time.perf_counter(); chunks = []; first = None
    for c in gen:
        chunks.append(c)
        if first is None:
            first = (time.perf_counter() - t) * 1000
    return np.concatenate(chunks), first, (time.perf_counter() - t) * 1000


fast = FastEngine()
best = BestEngine()
rows = []
for name, (f, (e0, e1), (h0, h1)) in SPEAKERS.items():
    full, _ = librosa.load(os.path.join(RES, f), sr=22050)
    enroll = full[int(e0 * 22050):int(e1 * 22050)]
    held = emb(full[int(h0 * 22050):int(h1 * 22050)], 22050)
    vd = tempfile.mkdtemp()
    fast.enroll(enroll, vd)
    best.enroll(enroll, 22050, vd)
    print(f"\n== {name}")
    for lang, text in TEXT.items():
        for eng in ENGINES:
            if eng == "fast":
                audio, first, total = run(fast.speak(text, lang, vd)); sr = fast.SR
            else:
                audio, first, total = run(best.speak(text, lang, vd, mode=eng)); sr = best.SR
            score = torch.cosine_similarity(emb(audio, sr), held, dim=0).item()
            print(f"  [{lang}] {eng:8s} similarity {score:.3f}   first audio {first:6.0f} ms   total {total:6.0f} ms", flush=True)
            rows.append((lang, eng, score, first))

print("\n== average over speakers")
for lang in TEXT:
    for eng in ENGINES:
        r = [x for x in rows if x[0] == lang and x[1] == eng]
        print(f"  [{lang}] {eng:8s} similarity {np.mean([x[2] for x in r]):.3f}   first audio {np.mean([x[3] for x in r]):6.0f} ms")
