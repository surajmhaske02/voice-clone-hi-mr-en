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

## Setup

**You need:** macOS (Apple Silicon is best) or Linux, Python 3.10, git, ffmpeg, about 10 GB of free disk and
16 GB of RAM. An NVIDIA GPU on Linux works too and is faster.

**1. Install the system tools** (skip any you already have)

macOS:
```bash
brew install python@3.10 git ffmpeg
```
Ubuntu / Debian:
```bash
sudo apt install python3.10 python3.10-venv git ffmpeg
```

**2. Get the code**
```bash
git clone https://github.com/surajmhaske02/voice-clone-hi-mr-en.git
cd voice-clone-hi-mr-en
```

**3. Run the setup script.** It creates the Python env (`.venv-omni`), installs `requirements.txt`, and downloads
the OpenVoice code and weights plus the Piper voices for English, Hindi and Marathi. It takes about 10 minutes.
```bash
bash setup.sh
```
If `python3.10` has another name on your machine: `PYTHON=/path/to/python3.10 bash setup.sh`.

**4. Start the app**
```bash
.venv-omni/bin/python app.py
```
Open http://localhost:7860. The first start downloads OmniVoice and Whisper (about 3 GB), so it takes a few minutes once.

## Use

1. **Clone:** type a voice name, press Record and read the 3 lines shown on the page (15-20 s). You can also
   upload an audio file. Press **Clone** and wait about 25 s.
2. **Speak:** pick the voice, a language (English, Hindi, Marathi) and an engine, type a sentence, press **Speak**.
   Start with **Best** for the closest voice, or **Fast** to hear something right away.
3. **Voice to voice:** record yourself speaking and hear it back in the cloned voice (Fast engine).

## Train your own fast voice (optional)

The **Trained** engine needs a small model trained on your cloned voice: a few hours per language on a Mac, faster
on a GPU or free Google Colab.
```bash
bash setup.sh --train                 # one time: training env .venv-train
bash distill/run_all.sh <voice-name>  # Hindi, Marathi, English; resumable
```
Close the web app while training. Details, Colab steps and timings are in `distill/README.md`.

## Troubleshooting

- **`Error processing file '.../piper/phontab'`**: the folder path is too long for espeak (limit about 160
  characters). Move the project to a shorter path, for example `~/voice-clone-hi-mr-en`, and run `bash setup.sh` again.
- **`ffmpeg` not found / "could not decode audio"**: install ffmpeg (step 1).
- **Very slow on a 16 GB Mac**: don't run training and the web app at the same time.

## Files

- `setup.sh`, `requirements.txt`, `requirements-train.txt`: setup and Python packages
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
