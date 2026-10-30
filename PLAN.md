# Low-cost voice-cloning handheld: test results + build plan

## 0. Update (2026-10-30): accuracy upgrade with OmniVoice

The web app (`app.py`, run with `.venv-omni/bin/python app.py`) now has three engines.

| Engine | Model | Voice similarity* (en / hi / mr) | First sound on M4 Mac | Copies accent? |
|---|---|---|---|---|
| **Best** | OmniVoice, 10 s voice prompt | **0.77 / 0.70 / 0.70** | ~5-6 s | yes |
| **Balanced** | OmniVoice, 6 s voice prompt | 0.67 / 0.63 / 0.61 | ~3 s | yes |
| **Fast** | Piper + OpenVoice V2 | 0.27 / 0.28 / 0.21 | ~0.4 s | no, tone only |

\*ECAPA speaker-verification cosine against held-out real speech, averaged over 3 speakers (`eval_clone.py`). Real recordings of the same person score about 0.6-0.8, so Best is in "same person" range. Fast is not.

- OmniVoice (k2-fsa, 0.8B, 600+ languages incl. Marathi) gives about **3x higher similarity** than OpenVoice. `verify_app.py` transcribes the generated audio with Whisper, and every engine produces the full sentence in all 3 languages.
- **Latency limit:** about 600 ms per decode step on the M4 GPU, so Best and Balanced are slower than real time on this Mac. An NVIDIA GPU is reported as RTF ~0.025, which would bring this to sub-second. That is not measured here.
- **License:** the OmniVoice weights are **CC-BY-NC (non-commercial)**. For a product, use AI4Bharat IndicF5 (MIT, hi + mr; gated on HuggingFace, needs a login) or Chatterbox (MIT, en + hi).
- **Bugs found and fixed:**
  - Batched generation returned noise on MPS, so chunks are now generated one at a time.
  - Whisper wrote Hindi in Urdu script and mangled Marathi. The voice prompt is now built from pause-split phrases with plausible transcripts, and phrases that match the on-screen reading lines are replaced by those exact lines.
  - Auto-picking the "closest" Piper base speaker made the Fast engine worse, so it was reverted.
- **For the device:** Hack #2 still applies. Use OmniVoice (or IndicF5 for a commercial product) as the *teacher* to generate the synthetic dataset, then fine-tune a small Piper model per voice. The device gets OmniVoice-level accent and Piper-level speed.
- **Hack #2 is now built:** see `distill/README.md`. Run `bash distill/run_all.sh <voice>`, which produces `voices/<voice>/piper/<lang>.onnx`. Those files are the web app's **Trained** engine, and they are what gets copied to the device.

## 1. What was tested (Mac M4, 2026-10-30)

Pipeline = Hack #1: `text -> Piper base TTS (en / hi / mr) -> OpenVoice V2 ToneColorConverter + 1 KB voice fingerprint -> cloned audio`

| Step | Result |
|---|---|
| Enrollment (19 s reference to fingerprint) | 50-80 ms, fingerprint = 256 floats = **1,024 bytes** |
| English, 4.1 s sentence | TTS 157 ms + clone 712 ms; first clause "Hello friend," heard after **222 ms** |
| Hindi, 5.9 s sentence | TTS 212 ms + clone 1045 ms; first clause after **310 ms** |
| Marathi, 5.9 s sentence | TTS 220 ms + clone 1149 ms; first clause after **398 ms** |
| Speech-to-speech (mic voice to cloned voice) | 29.5 s of speech converted in 5.2 s |
| Single thread (closer to board) | first clause 330 / 460 / 581 ms (en / hi / mr) |
| Converter as ONNX | 128 MB FP32, matches PyTorch (max diff 0.004) |
| Converter INT8 | **no speed gain, no size gain**: the HiFi-GAN decoder (convolutions) does most of the work, and ONNX Runtime CPU can't run INT8 convs |

Listen: `out/*_base.wav` (stock voice) vs `out/*_cloned.wav` (cloned) vs `out/s2s_cloned.wav`.

## 2. Reality check on the pasted "hacks"

- **OpenVoice is real and works for Hindi and Marathi.** The converter only copies *timbre*, so the input language doesn't matter. Rhythm and accent still come from the Piper base voice.
- **The "540 ms on a Rs 3,500 RK3566" claim is too optimistic for Hack #1.** An M4 core is roughly 8-10x faster than a Cortex-A55 core. My estimate for the Radxa ZERO 3W (4x A55) is converter RTF ~1.2-1.5 plus Piper ~0.4, so it would **not keep up in real time**, and the first clause would arrive after ~1.5-3 s. This is an estimate; measure on the board.
- **Hack #2 (distill the voice into one small TTS model) is the right fit for a cheap board.** At runtime there is only one VITS model and no converter. Piper medium runs faster than real time on A55/A72-class CPUs.
- **Things I couldn't verify, so I didn't use them:** the `AaryanAKAanthu1/OpenVoice-onnx` repo (unknown author; I exported ONNX myself from the official MyShell weights instead), "Pocket TTS v4.0 Small" numbers, and the RKNN NPU speedups. VITS has dynamic shapes, which are painful to run on the RK3566 NPU. Plan on CPU only.
- The official S3 checkpoint link is dead. Weights now come from HuggingFace `myshell-ai/OpenVoiceV2`.

## 3. Recommended architecture

```
SETUP (laptop / Colab, once per voice, ~1-2 h)
  record 30-60 s of the person reading
    -> teacher makes ~1 h of synthetic speech in their voice (en + hi + mr sentences)
         teacher A (easy, tested here): Piper -> OpenVoice converter
         teacher B (best Indian accent): AI4Bharat IndicF5 on a Colab GPU
    -> fine-tune Piper (VITS) starting from the existing hi / mr / en checkpoints
    -> export one .onnx file per language (~60 MB medium, ~20 MB low)
    -> copy to the device SD card

DEVICE (standalone, no internet)
  button / text in -> clause chunker (first 3-5 words first)
    -> Piper ONNX (that person's voice) -> ring buffer -> I2S amp -> speaker
```

- Optional "guest mode" (Hack #1 on the device): only on a Pi 5 or RK3588 class board, ~Rs 6-9k.
- Mic in, cloned voice out live (voice changer): run the OpenVoice converter on 1-2 s mic chunks. This also needs a Pi 5 class board. Expect ~1 s of lag on a cheap board; it can't be truly live.

## 4. Handheld BOM (approximate prices; check before buying)

| Part | Price |
|---|---|
| Radxa ZERO 3W 2 GB (RK3566) | ~Rs 3,500 |
| INMP441 I2S mic | ~Rs 150 |
| MAX98357A I2S amp + 40 mm 3 W speaker | ~Rs 260 |
| 1S LiPo 2000 mAh + IP5306 charge/boost module | ~Rs 400 |
| 16 GB SD card, 2 buttons, 3D-printed case | ~Rs 400 |
| **Total** | **~Rs 4,700** |

Cheaper option: Sipeed LicheeRV Nano (~Rs 2,250, mic and amp on board). Only Piper *low/x_low* models will fit, and they are untested here.

Honest alternative: a used or cheap Android phone (Rs 5-7k) is 5-10x faster than these SBCs and already has the battery, mic and speaker. It can run Hack #1 live.

## 5. Licensing (matters if you sell this)

- OpenVoice V2: MIT. Commercial use is OK.
- Piper Marathi `mr_IN-google`: CC BY-SA 4.0. Commercial use is OK with attribution and share-alike.
- Piper Hindi `hi_IN-rohan`: trained on IITM IndicTTS data, whose license **restricts commercial use**. Get permission or use another Hindi base voice.
- Cloning someone else's voice needs their consent. Add a consent step in the enrollment flow.

## 6. Next steps

1. Record your own voice and listen to the cloned output:
   `ffmpeg -f avfoundation -i ":0" -t 30 -ac 1 -ar 22050 my_voice.wav`
   then `python clone_test.py --ref my_voice.wav`
2. Build the Hack #2 data generator (teacher A) and fine-tune Piper on Colab.
3. Buy the Radxa ZERO 3W and benchmark Piper and the OpenVoice ONNX on it. That gives real numbers in place of the estimates above.
