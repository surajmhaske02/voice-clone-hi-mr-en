"""Hack #2, step 1: OmniVoice (teacher) speaks N sentences per language in a cloned voice.

Output per language, in Piper's training format:
  distill/data/<voice>/<lang>/wavs/0001.wav ...  (22.05 kHz mono)
  distill/data/<voice>/<lang>/metadata.csv       ("0001.wav|text")
Resumable: re-run the same command and finished sentences are skipped.

Usage:
  .venv-omni/bin/python distill/make_dataset.py --voice suraj --langs hi mr en --n 400
Stop the web app first (both use the GPU). On the M4 Mac this is ~10 s per sentence.
"""
import argparse, os, re, sys, time
import numpy as np
import librosa
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from engines import BestEngine, ROOT  # noqa: E402

SR_OUT = 22050  # Piper "medium" voices are 22.05 kHz


def plausible(audio, sr, text):
    """Reject obvious teacher failures: too short/long for the text, or near-silent."""
    letters = len(re.sub(r"[\W\d_]", "", text))
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 3 <= letters / max(dur, 1e-3) <= 22 and rms > 0.01


def clean(audio, sr):
    audio, _ = librosa.effects.trim(audio, top_db=35)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR_OUT)
    peak = np.max(np.abs(audio)) or 1.0
    audio = 0.9 * audio / peak  # consistent level across the dataset
    pad = np.zeros(int(0.1 * SR_OUT), dtype=np.float32)
    return np.concatenate([pad, audio.astype(np.float32), pad])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="name of a voice cloned in the web app (voices/<name>)")
    ap.add_argument("--langs", nargs="+", default=["hi", "mr", "en"])
    ap.add_argument("--n", type=int, default=400, help="sentences per language (~400 = ~25-30 min audio)")
    ap.add_argument("--mode", default="best", choices=["best", "balanced"])
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()

    vd = os.path.join(ROOT, "voices", a.voice)
    if not os.path.exists(os.path.join(vd, f"omni_{a.mode}.pt")):
        sys.exit(f"voice '{a.voice}' not found; clone it in the web app first")
    eng = BestEngine()

    for lang in a.langs:
        sents = [s.strip() for s in open(os.path.join(HERE, "sentences", f"{lang}.txt")) if s.strip()][: a.n]
        out = os.path.join(HERE, "data", a.voice, lang)
        os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
        meta_path = os.path.join(out, "metadata.csv")
        done = {}
        if os.path.exists(meta_path):
            for line in open(meta_path):
                f, t = line.rstrip("\n").split("|", 1)
                done[f] = t
        t0, made, rejected = time.time(), 0, 0
        with open(meta_path, "a") as meta:
            for i, text in enumerate(sents, 1):
                name = f"{i:04d}.wav"
                if name in done:
                    continue
                audio = eng.model.generate(text=text, language=lang, voice_clone_prompt=eng._prompt(vd, a.mode),
                                           num_step=a.steps)[0]
                if not plausible(audio, eng.SR, text):
                    rejected += 1
                    continue
                sf.write(os.path.join(out, "wavs", name), clean(audio, eng.SR), SR_OUT)
                meta.write(f"{name}|{text}\n")
                meta.flush()
                made += 1
                if made % 10 == 0:
                    per = (time.time() - t0) / made
                    left = (len(sents) - i) * per / 60
                    print(f"[{lang}] {i}/{len(sents)}  {per:.1f}s/sentence  ~{left:.0f} min left  rejected {rejected}", flush=True)
        total = sum(sf.info(os.path.join(out, "wavs", f)).duration for f in os.listdir(os.path.join(out, "wavs")))
        print(f"[{lang}] done: {len(os.listdir(os.path.join(out, 'wavs')))} clips, {total/60:.1f} min of audio -> {out}", flush=True)


if __name__ == "__main__":
    main()
