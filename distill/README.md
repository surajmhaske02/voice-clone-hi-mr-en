# Hack #2: distill your cloned voice into a small, fast model

The **Best** engine (OmniVoice, 0.8B) sounds most like you but is slow. Here it acts as a *teacher*: it reads
~400 sentences per language in your cloned voice, and a small **Piper** model (≈23M params, 61 MB ONNX) is
fine-tuned on that audio. The result is the web app's **Trained** engine. It starts speaking in about 0.4 s and runs on
a Raspberry Pi-class CPU, so it's the model the handheld device ships with.

```
voices/<name>/omni_best.pt ─► make_dataset.py ─► data/<name>/<lang>/{wavs/,metadata.csv}
   (clone in web app)          (OmniVoice teacher)
                                       │
                                       ▼
                         train_voice.py (Piper fine-tune from rhasspy checkpoint)
                                       │
                                       ▼
                         voices/<name>/piper/<lang>.onnx (+ .onnx.json)  ─►  web app "Trained" / device
```

## Run it

Clone the voice in the web app first, then **stop the web app** (the teacher and the trainer both use the GPU).

**All on this Mac:**
```bash
bash distill/run_all.sh suraj            # hi, then mr, then en; resumable, so re-run after any stop
tail -f distill/run_all.log              # progress
```
Defaults: `N=400` sentences per language (≈25–30 min of audio) and `STEPS=3000` fine-tune steps. Override them like this:
`N=600 STEPS=5000 bash distill/run_all.sh suraj hi`.

Stop it at any time with `pkill -f run_all.sh; pkill -f make_dataset.py; pkill -f piper_fit.py`.

**Faster training on a free Colab GPU:** make the datasets on the Mac
(`.venv-omni/bin/python distill/make_dataset.py --voice suraj`), then run `bash distill/pack_for_colab.sh suraj`. Open
`distill/colab_train.ipynb` in Colab with a T4 GPU, upload the zip, run the cells, and unzip the downloaded
`piper_suraj.zip` into this folder.

## Steps and files

| File | What it does |
|---|---|
| `make_sentences.py` | Builds 1,200 clean sentences per language from Wikipedia plus everyday phrases (no digits or mixed scripts) and writes them to `sentences/` |
| `make_dataset.py` | Teacher: OmniVoice speaks each sentence in the cloned voice, drops implausible outputs, and writes 22.05 kHz wavs plus `metadata.csv`. About 6 s per sentence on the M4 |
| `train_voice.py` | Fine-tunes from the rhasspy checkpoint (en: lessac; hi and mr: rohan, since the Marathi checkpoint is multi-speaker), keeps the best-sounding checkpoint (UTMOS `val_mos`), and exports ONNX into `voices/<name>/piper/` |
| `piper_fit.py` | Piper's trainer, set to keep 2 checkpoints instead of 10 (each is 807 MB) |
| `run_all.sh` | Runs the dataset and training steps for each language |
| `colab_train.ipynb`, `pack_for_colab.sh` | The same training, run on a Colab GPU |

## Measured on this Mac (M4, 16 GB)

- Teacher: about 5.7–7.3 s per sentence. 400 Hindi sentences took 46 min and gave 33 min of audio, with 0 rejected.
- Memory: uncapped, the trainer grew to 20 GB on a 16 GB Mac and swapped, which made it *and* the web app crawl
  (a clone took 10+ min). `run_all.sh` now caps MPS memory. Don't use the web app while training on a 16 GB Mac, and
  keep the Mac awake (`caffeinate -ims -w <run_all pid>`): it slept overnight and training stalled.
- Trainer: about **5 s per step** on the Mac GPU (`mps`, batch 16), so 3,000 steps ≈ **4 h per language**. A Colab
  GPU should be much faster (not measured here). The smoke test covered dataset → 30 steps → ONNX → Whisper check in the app.
- Trained model in the app: **first sound in 413 ms, a whole sentence in 582 ms**.
- Disk: about 1.6 GB per language for checkpoints in `distill/runs/` (you can delete them after export).

## Notes

- **Quality ceiling:** the student can only be as close to your voice as the teacher, and Piper's prosody is a bit
  flatter. Use more sentences (`N=800`) and more steps (`STEPS=5000`) for a closer match.
- **Licenses:** the OmniVoice teacher is **non-commercial**, so this pipeline is fine for prototypes. For a product, swap
  in an MIT-licensed teacher (AI4Bharat IndicF5 for hi/mr; it's gated, so log in to HuggingFace first). The Hindi base checkpoint
  (rohan) comes from IITM IndicTTS data, which has commercial-use limits. lessac (en) has its own license.
- **Consent:** clone only voices whose owners have agreed.
