"""Benchmark OmniVoice zero-shot cloning (en / hi / mr) on this Mac.

Usage: .venv-omni/bin/python omni_test.py --ref my_voice.wav [--steps 32 16 8]
"""
import argparse, os, time
import soundfile as sf
import torch
from omnivoice import OmniVoice

SENTENCES = {
    "en": "Hello friend, I am coming at four in the afternoon. Please keep the tea ready.",
    "hi": "नमस्ते दोस्त, मैं आज दोपहर चार बजे आ रहा हूँ। कृपया चाय तैयार रखना।",
    "mr": "नमस्कार मित्रांनो, मी आज दुपारी चार वाजता येतोय. कृपया चहा तयार ठेवा.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--ref-text", default=None, help="transcript of ref (skips Whisper)")
ap.add_argument("--steps", type=int, nargs="+", default=[32, 16, 8])
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--out", default="out_omni")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

t = time.perf_counter()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=a.device, dtype=torch.float16)
print(f"load {time.perf_counter()-t:.1f}s on {a.device}")

t = time.perf_counter()
if a.ref_text is None:
    model.load_asr_model()
prompt = model.create_voice_clone_prompt(a.ref, ref_text=a.ref_text)
print(f"enroll {time.perf_counter()-t:.1f}s  ref_text: {prompt.ref_text!r}")

model.generate(text="warm up", language="en", voice_clone_prompt=prompt, num_step=8)
for steps in a.steps:
    for lang, text in SENTENCES.items():
        if a.device == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        wav = model.generate(text=text, language=lang, voice_clone_prompt=prompt, num_step=steps)[0]
        ms = (time.perf_counter() - t) * 1000
        dur = len(wav) / model.sampling_rate
        sf.write(f"{a.out}/{lang}_s{steps}.wav", wav, model.sampling_rate)
        print(f"steps={steps:2d} [{lang}] {dur:.1f}s audio in {ms:.0f} ms (RTF {ms/1000/dur:.2f})")
