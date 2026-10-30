"""OpenVoice V2 clone test: Piper base TTS (en/hi/mr) -> ToneColorConverter -> cloned voice.

Usage:
  python clone_test.py --ref my_voice.wav            # enroll + speak test sentences
  python clone_test.py --ref my_voice.wav --threads 1  # rough single-core (board-like) timing
"""
import argparse, os, re, sys, time, wave, io
import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "OpenVoice"))
from openvoice.api import ToneColorConverter  # noqa: E402
from openvoice.mel_processing import spectrogram_torch  # noqa: E402
from piper import PiperVoice  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "OpenVoice/checkpoints_v2/converter")
VOICES = {
    "en": "piper_voices/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "hi": "piper_voices/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
    "mr": "piper_voices/mr/mr_IN/google/medium/mr_IN-google-medium.onnx",
}
SENTENCES = {
    "en": "Hello friend, I am coming at four in the afternoon. Please keep the tea ready.",
    "hi": "नमस्ते दोस्त, मैं आज दोपहर चार बजे आ रहा हूँ। कृपया चाय तैयार रखना।",
    "mr": "नमस्कार मित्रांनो, मी आज दुपारी चार वाजता येतोय. कृपया चहा तयार ठेवा.",
}
SR = 22050  # converter sample rate


def piper_tts(voice, text):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        voice.synthesize_wav(text, w)
    buf.seek(0)
    audio, sr = sf.read(buf, dtype="float32")
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def se_from_audio(conv, audio):
    y = torch.FloatTensor(audio).unsqueeze(0)
    hps = conv.hps
    spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                             hps.data.hop_length, hps.data.win_length, center=False)
    with torch.no_grad():
        return conv.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)


def convert(conv, audio, src_se, tgt_se, tau=0.3):
    hps = conv.hps
    y = torch.FloatTensor(audio).unsqueeze(0)
    spec = spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate,
                             hps.data.hop_length, hps.data.win_length, center=False)
    lens = torch.LongTensor([spec.size(-1)])
    with torch.no_grad():
        out = conv.model.voice_conversion(spec, lens, sid_src=src_se, sid_tgt=tgt_se, tau=tau)
    return out[0][0, 0].numpy()


def first_chunk(text):
    # first clause (up to comma / danda) else first 4 words
    m = re.match(r"^(.+?[,،।.!?])\s", text)
    if m and len(m.group(1).split()) <= 6:
        return m.group(1)
    return " ".join(text.split()[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="10-30 s clean recording of target voice")
    ap.add_argument("--src", help="optional: real speech to convert (speech-to-speech test)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    # upstream base class rejects enable_watermark kwarg; stub wavmark and drop watermark model
    import types
    sys.modules.setdefault("wavmark", types.SimpleNamespace(load_model=lambda: torch.nn.Identity()))
    conv = ToneColorConverter(os.path.join(CKPT, "config.json"), device="cpu")
    conv.watermark_model = None
    conv.load_ckpt(os.path.join(CKPT, "checkpoint.pth"))

    # ---- enrollment: voice fingerprint ----
    ref, _ = librosa.load(a.ref, sr=SR)
    t = time.perf_counter()
    tgt_se = se_from_audio(conv, ref)
    print(f"enroll: {len(ref)/SR:.1f}s ref -> fingerprint {tuple(tgt_se.shape)} "
          f"({tgt_se.numel()*4} bytes) in {(time.perf_counter()-t)*1000:.0f} ms")
    torch.save(tgt_se, os.path.join(a.out, "tgt_se.pth"))

    for lang, path in VOICES.items():
        voice = PiperVoice.load(os.path.join(ROOT, path))
        # base voice fingerprint (do once per base voice, cache on device)
        src_se = se_from_audio(conv, piper_tts(voice, SENTENCES[lang]))
        text = SENTENCES[lang]

        piper_tts(voice, "warm up")  # warm up
        convert(conv, piper_tts(voice, "warm up"), src_se, tgt_se)

        t0 = time.perf_counter(); base = piper_tts(voice, text); t1 = time.perf_counter()
        cloned = convert(conv, base, src_se, tgt_se); t2 = time.perf_counter()
        dur = len(base) / SR
        sf.write(os.path.join(a.out, f"{lang}_base.wav"), base, SR)
        sf.write(os.path.join(a.out, f"{lang}_cloned.wav"), cloned, SR)

        fc = first_chunk(text)
        c0 = time.perf_counter(); fb = piper_tts(voice, fc); convert(conv, fb, src_se, tgt_se)
        c1 = time.perf_counter()
        print(f"[{lang}] {dur:.1f}s audio | full: tts {(t1-t0)*1000:.0f} ms + clone {(t2-t1)*1000:.0f} ms "
              f"(RTF {(t2-t0)/dur:.2f}) | first chunk '{fc}' -> {(c1-c0)*1000:.0f} ms to first audio")

    if a.src:
        src_audio, _ = librosa.load(a.src, sr=SR)
        s_se = se_from_audio(conv, src_audio)
        t = time.perf_counter(); vc = convert(conv, src_audio, s_se, tgt_se)
        print(f"[speech-to-speech] {len(src_audio)/SR:.1f}s in {(time.perf_counter()-t)*1000:.0f} ms")
        sf.write(os.path.join(a.out, "s2s_cloned.wav"), vc, SR)


if __name__ == "__main__":
    main()
