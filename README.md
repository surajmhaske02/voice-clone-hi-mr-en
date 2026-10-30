# Voice Clone (English, Hindi, Marathi)

Clone a voice from a short recording, then speak any sentence in that voice in English, Hindi or Marathi.
The goal is a low-cost handheld device (Radxa ZERO 3W, I2S mic and speaker; see `PLAN.md`).

## Engines in the web app

| Engine | How it works | Voice match | First sound (M4 Mac) |
|---|---|---|---|
| Best | OmniVoice, 10 s voice prompt | highest | ~5-6 s |
| Balanced | OmniVoice, 6 s voice prompt | high | ~3 s |
| Fast | Piper + OpenVoice V2 tone converter | low (tone only) | ~0.4 s |
| Trained | small Piper model fine-tuned on your voice (`distill/`) | close to Best | ~50 ms |

## Run

```bash
python3.10 -m venv .venv-omni
.venv-omni/bin/pip install torch==2.8.0 omnivoice piper-tts librosa soundfile speechbrain flask mlx-whisper
git clone https://github.com/myshell-ai/OpenVoice && .venv-omni/bin/pip install -e OpenVoice
# OpenVoice V2 weights: HuggingFace myshell-ai/OpenVoiceV2 -> OpenVoice/checkpoints_v2/converter
# Piper voices (en lessac, hi rohan, mr google): HuggingFace rhasspy/piper-voices -> piper_voices/
.venv-omni/bin/python app.py      # http://localhost:7860
```

Read the 3 lines on the page in a 15-20 s recording, press Clone (~25 s), then type text and press Speak.

## Files

- `app.py`, `web/index.html`: Flask web app
- `engines.py`: Fast, Best and Trained engines
- `distill/`: Hack #2, distill the cloned voice into a small fast Piper model (see `distill/README.md`)
- `eval_clone.py`, `verify_app.py`: voice-similarity and Whisper checks
- `clone_test.py`, `export_onnx.py`, `omni_test.py`: early experiments
- `PLAN.md`: test results, device plan, parts list

Models, voice recordings and training data are not in this repo (`.gitignore`).

## License notes

OmniVoice weights are CC-BY-NC (non-commercial). Piper `hi_IN-rohan` uses IITM data with commercial limits.
Clone only voices whose owners agree.
