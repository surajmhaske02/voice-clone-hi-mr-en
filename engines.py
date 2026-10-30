"""Voice cloning engines for English / Hindi / Marathi.

BestEngine  OmniVoice zero-shot TTS (k2-fsa, 0.8B). Copies timbre AND accent/prosody.
            Speaker similarity ~0.70-0.77 (ECAPA, held-out real speech) in all three languages.
FastEngine  Piper base TTS -> OpenVoice V2 tone-color converter. Starts in ~0.5 s on CPU but
            only copies tone colour: similarity ~0.2-0.28. Kept for speed / speech-to-speech.
See eval_clone.py for how the numbers were measured.
"""
import io, os, re, sys, types, wave
import numpy as np
import soundfile as sf
import librosa
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "OpenVoice"))
sys.modules.setdefault("wavmark", types.SimpleNamespace(load_model=lambda: torch.nn.Identity()))
from openvoice.api import ToneColorConverter  # noqa: E402
from openvoice.mel_processing import spectrogram_torch  # noqa: E402
from piper import PiperVoice, SynthesisConfig  # noqa: E402

LANGS = ("en", "hi", "mr")
PV = os.path.join(ROOT, "piper_voices")

# Fixed base voices. Auto-picking the "closest" base speaker was tried and scored worse.
FAST_BASES = {
    "en": (f"{PV}/en/en_US/lessac/medium/en_US-lessac-medium.onnx", None),
    "hi": (f"{PV}/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx", None),
    "mr": (f"{PV}/mr/mr_IN/google/medium/mr_IN-google-medium.onnx", 0),
}
# Lines the web page asks people to read at enrollment. Transcripts that match them are replaced by
# the exact text, because Whisper misspells Hindi/Marathi and the clone prompt needs an exact transcript.
ENROLL_SCRIPT = [
    "Hello, today I am recording my voice so that this device can speak like me.",
    "नमस्ते, आज मौसम बहुत अच्छा है और मैं अपने दोस्तों के साथ बाज़ार जा रहा हूँ।",
    "नमस्कार, मी आज माझा आवाज रेकॉर्ड करत आहे. हा आवाज छान आणि स्पष्ट असावा.",
]
CALIBRATION = {
    "en": "The quick brown fox jumps over the lazy dog. I am testing my new voice today, and it sounds good.",
    "hi": "नमस्ते, मेरा नाम राहुल है। आज मौसम बहुत अच्छा है और मैं बाज़ार जा रहा हूँ।",
    "mr": "नमस्कार, माझे नाव राहुल आहे. आज हवामान खूप छान आहे आणि मी बाजारात जात आहे.",
}


# ---------------------------------------------------------------- text chunking
_SENT = re.compile(r"(?<=[.!?।॥])\s+")


def chunk_text(text, first_words=4):
    """Short first chunk (first clause or ~4 words) for low latency, then whole sentences."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sents = [s for s in _SENT.split(text) if s.strip()]
    first, rest = sents[0], sents[1:]
    m = re.match(r"^(.+?[,;:])\s+(.+)$", first)
    if m and len(m.group(1).split()) <= 7:
        head, tail = m.group(1), m.group(2)
    elif len(first.split()) > first_words + 3:
        w = first.split()
        head, tail = " ".join(w[:first_words]), " ".join(w[first_words:])
    else:
        head, tail = first, ""
    return [head] + ([tail] if tail else []) + rest


def crossfade_concat(chunks, sr, fade_ms=5, gap_ms=60):
    """Join chunks with a short pause and raised-cosine fades (no clicks)."""
    n = int(sr * fade_ms / 1000)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    out = []
    for c in chunks:
        c = c.astype(np.float32).copy()
        if len(c) > 2 * n:
            c[:n] *= ramp
            c[-n:] *= ramp[::-1]
        out += [c, gap]
    return np.concatenate(out[:-1]) if out else np.zeros(0, np.float32)


def snap_to_script(text, script=ENROLL_SCRIPT, min_match=0.5):
    """If `text` is a (misspelled) piece of a known script line, return that piece spelled exactly."""
    from difflib import SequenceMatcher
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()  # noqa: E731
    heard = norm(text)
    best = None
    for line in script:
        words = line.split()
        blocks = [b for b in SequenceMatcher(None, norm(line), heard, autojunk=False).get_matching_blocks() if b.size]
        matched = sum(b.size for b in blocks)
        if not heard or matched / len(heard) < min_match:
            continue
        # span of the line covered by the heard words, widened by unmatched words at either end
        start = max(0, blocks[0].a - blocks[0].b)
        end = min(len(words), blocks[-1].a + blocks[-1].size + (len(heard) - blocks[-1].b - blocks[-1].size))
        if best is None or matched > best[0]:
            best = (matched, " ".join(words[start:end]))
    return best[1] if best else text


def best_window(wav, sr, secs):
    """Loudest-speech window of ~`secs` seconds, edges moved to the quietest nearby frame so no word
    is cut in half (a cut word makes the transcript and audio disagree and hurts cloning)."""
    n = int(secs * sr)
    if len(wav) <= n:
        return wav.astype(np.float32)
    hop = sr // 2
    energy = [np.mean(wav[i:i + n] ** 2) for i in range(0, len(wav) - n + 1, hop)]
    start = int(np.argmax(energy)) * hop
    fr = sr // 50  # 20 ms frames
    frame_e = np.array([np.mean(wav[i:i + fr] ** 2) for i in range(0, len(wav) - fr, fr)])

    def quiet(pos, lo, hi):
        a, b = max(0, (pos + lo) // fr), min(len(frame_e), (pos + hi) // fr)
        return pos if a >= b else (a + int(np.argmin(frame_e[a:b]))) * fr

    s = quiet(start, -int(0.6 * sr), int(0.3 * sr))
    e = quiet(start + n, -int(0.8 * sr), int(0.6 * sr))
    return wav[s:max(e, s + sr)].astype(np.float32)


# ---------------------------------------------------------------- fast engine
class FastEngine:
    SR = 22050

    def __init__(self):
        ck = os.path.join(ROOT, "OpenVoice/checkpoints_v2/converter")
        self.conv = ToneColorConverter(os.path.join(ck, "config.json"), device="cpu")
        self.conv.watermark_model = None
        self.conv.load_ckpt(os.path.join(ck, "checkpoint.pth"))
        self.voices = {l: PiperVoice.load(p) for l, (p, _) in FAST_BASES.items()}
        self.base_ses = {l: self.embed(self.piper(l, CALIBRATION[l])) for l in LANGS}

    def piper(self, lang, text):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self.voices[lang].synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=FAST_BASES[lang][1]))
        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        return audio if sr == self.SR else librosa.resample(audio, orig_sr=sr, target_sr=self.SR)

    def _spec(self, audio):
        h = self.conv.hps.data
        return spectrogram_torch(torch.FloatTensor(audio).unsqueeze(0), h.filter_length, h.sampling_rate,
                                 h.hop_length, h.win_length, center=False)

    def embed(self, audio):
        with torch.no_grad():
            return self.conv.model.ref_enc(self._spec(audio).transpose(1, 2)).unsqueeze(-1)

    def enroll(self, audio, voice_dir):
        torch.save(self.embed(audio), os.path.join(voice_dir, "fast_se.pth"))

    def convert(self, audio, src_se, tgt_se, tau=0.3):
        spec = self._spec(audio)
        with torch.no_grad():
            out = self.conv.model.voice_conversion(spec, torch.LongTensor([spec.size(-1)]),
                                                   sid_src=src_se, sid_tgt=tgt_se, tau=tau)
        return out[0][0, 0].numpy()

    def speak(self, text, lang, voice_dir, tau=0.3, raw=False):
        tgt = torch.load(os.path.join(voice_dir, "fast_se.pth"))
        for chunk in chunk_text(text):
            base = self.piper(lang, chunk)
            yield base if raw else self.convert(base, self.base_ses[lang], tgt, tau)

    def speech_to_speech(self, audio, voice_dir, tau=0.3):
        tgt = torch.load(os.path.join(voice_dir, "fast_se.pth"))
        return self.convert(audio, self.embed(audio), tgt, tau)


# ---------------------------------------------------------------- best engine
class BestEngine:
    SR = 24000

    # Prompt length drives per-step cost. Measured (avg ECAPA similarity / first sound on M4):
    #   best     10 s -> 0.70-0.77 / ~4-6 s      balanced  6 s -> 0.61-0.67 / ~2.5 s
    MODES = {"best": 10.0, "balanced": 6.0}

    def __init__(self, device=None):
        from omnivoice import OmniVoice
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=self.device, dtype=dtype)
        self.prompts = {}

    def enroll(self, audio, sr, voice_dir, ref_text=None):
        """One clone prompt per mode. The prompt's audio and transcript must match exactly: a wrong
        transcript makes the clone garble words and talk too slowly. So the recording is split at
        pauses, each phrase is transcribed on its own, implausible transcripts are dropped, and the
        prompt is built from the most reliable phrases (English first: Whisper is best at it).

        A user-typed transcript matches the whole clip, so then the clip is used untrimmed for every mode.
        """
        wav = librosa.resample(audio, orig_sr=sr, target_sr=self.SR) if sr != self.SR else audio
        phrases = None if ref_text else self._phrases(wav)
        texts = {}
        for mode, secs in self.MODES.items():
            if ref_text:
                w, text = wav, ref_text
            else:
                w, text = self._select(phrases, secs) if phrases else (best_window(wav, self.SR, secs), None)
                text = text or self.transcribe(w)
            prompt = self.model.create_voice_clone_prompt((torch.from_numpy(w)[None], self.SR), ref_text=text)
            prompt.save(os.path.join(voice_dir, f"omni_{mode}.pt"))
            self.prompts[(voice_dir, mode)] = prompt
            texts[mode] = prompt.ref_text
        return texts["best"]

    def _phrases(self, wav):
        """Split at pauses -> [(audio, text, is_latin)] for phrases with a plausible transcript."""
        iv = librosa.effects.split(wav, top_db=30, frame_length=1024, hop_length=256)
        merged = []
        for s, e in iv:  # join pieces separated by short gaps (< 0.35 s) into phrases
            if merged and s - merged[-1][1] < int(0.35 * self.SR):
                merged[-1][1] = e
            else:
                merged.append([s, e])
        out = []
        for s, e in merged:
            s, e = max(0, s - int(0.05 * self.SR)), min(len(wav), e + int(0.05 * self.SR))
            dur = (e - s) / self.SR
            if not 0.8 <= dur <= 12:
                continue
            heard = self.transcribe(wav[s:e])
            text = snap_to_script(heard)
            letters = len(re.sub(r"[\W\d_]", "", text))
            if not text or not 4 <= letters / dur <= 22:  # nothing heard / hallucinated or truncated
                continue
            # reliability rank: 0 = matched a script line, 1 = English (Whisper's best language), 2 = other
            latin = bool(re.search(r"[A-Za-z]", text)) and not re.search(r"[ऀ-ॿ]", text)
            rank = 0 if any(text in line for line in ENROLL_SCRIPT) else 1 if latin else 2
            out.append((wav[s:e], text, rank))
        return out

    def _select(self, phrases, secs):
        """Most reliable phrases (see rank in _phrases, then recording order) up to ~secs, kept in order."""
        order = sorted(range(len(phrases)), key=lambda i: (phrases[i][2], i))
        pick, total = [], 0.0
        for i in order:
            d = len(phrases[i][0]) / self.SR
            if pick and total + d > secs * 1.25:
                continue
            pick.append(i)
            total += d
            if total >= secs * 0.8:
                break
        pick.sort()
        gap = np.zeros(int(0.25 * self.SR), dtype=np.float32)
        audio = np.concatenate([x for i in pick for x in (phrases[i][0], gap)][:-1]).astype(np.float32)
        return audio, " ".join(phrases[i][1] for i in pick)

    MLX_ASR = "mlx-community/whisper-large-v3-turbo"

    def transcribe(self, wav):
        """Whisper often labels spoken Hindi/Marathi as Urdu and writes Arabic script, which no longer
        matches the audio for the clone prompt. Retry forced to Hindi (Devanagari) when that happens.

        On Apple Silicon this uses mlx-whisper: ~2 s per phrase vs ~8-10 s for the PyTorch pipeline
        (Whisper always encodes a 30 s window, so each short phrase costs a full pass)."""
        try:
            import mlx_whisper
        except ImportError:
            mlx_whisper = None
        if mlx_whisper:
            w16 = librosa.resample(wav, orig_sr=self.SR, target_sr=16000).astype(np.float32)
            def run(language=None):  # temperature capped: the full fallback ladder can take 10+ s on a bad phrase
                return mlx_whisper.transcribe(w16, path_or_hf_repo=self.MLX_ASR, language=language, temperature=(0.0, 0.4),
                                              condition_on_previous_text=False)["text"].strip()
        else:
            if self.model._asr_pipe is None:
                self.model.load_asr_model()
            def run(language=None):  # the pipeline consumes its input dict, so build a new one per call
                kw = {"generate_kwargs": {"language": language, "task": "transcribe"}} if language else {}
                return self.model._asr_pipe({"raw": wav, "sampling_rate": self.SR}, **kw)["text"].strip()

        text = run()
        if re.search(r"[؀-ۿ]", text):
            text = run("hi" if mlx_whisper else "hindi")
        return text

    def free_asr(self):
        self.model._asr_pipe = None
        try:
            from mlx_whisper.transcribe import ModelHolder
            ModelHolder.model = None
        except ImportError:
            pass
        if self.device == "mps":
            torch.mps.empty_cache()

    def _prompt(self, voice_dir, mode):
        if (voice_dir, mode) not in self.prompts:
            from omnivoice.models.omnivoice import VoiceClonePrompt
            self.prompts[(voice_dir, mode)] = VoiceClonePrompt.load(os.path.join(voice_dir, f"omni_{mode}.pt"))
        return self.prompts[(voice_dir, mode)]

    def speak(self, text, lang, voice_dir, mode="best", steps=8, speed=None):
        """Short first clause first (fast first sound), then one sentence at a time.

        Not batched: batching chunks of different lengths in one generate() call returned noise for
        the padded items on MPS.
        """
        p = self._prompt(voice_dir, mode)
        for chunk in chunk_text(text):
            yield self.model.generate(text=chunk, language=lang, voice_clone_prompt=p,
                                      num_step=steps, speed=speed)[0]


# ---------------------------------------------------------------- trained (distilled) engine
class TrainedEngine:
    """Per-voice Piper model fine-tuned on OmniVoice output (distill/). Teacher-like voice at Piper speed;
    the same .onnx files run on the device."""
    SR = 22050

    def __init__(self):
        self.voices = {}

    @staticmethod
    def path(voice_dir, lang):
        return os.path.join(voice_dir, "piper", f"{lang}.onnx")

    @classmethod
    def langs(cls, voice_dir):
        return [l for l in LANGS if os.path.exists(cls.path(voice_dir, l))]

    def speak(self, text, lang, voice_dir):
        p = self.path(voice_dir, lang)
        mtime = os.path.getmtime(p)
        if self.voices.get(p, (None, 0))[1] != mtime:  # reload after re-training
            self.voices[p] = (PiperVoice.load(p), mtime)
        voice = self.voices[p][0]
        for chunk in chunk_text(text):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                voice.synthesize_wav(chunk, w)
            buf.seek(0)
            audio, sr = sf.read(buf, dtype="float32")
            yield audio if sr == self.SR else librosa.resample(audio, orig_sr=sr, target_sr=self.SR)
