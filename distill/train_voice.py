"""Hack #2, step 2: fine-tune a small Piper voice per language on the teacher dataset, export ONNX.

Result: voices/<voice>/piper/<lang>.onnx (+ .onnx.json), picked up by the web app as the
"Trained" engine and copyable as-is to the device (Piper runs on a Raspberry Pi class CPU).
Resumable: re-running continues from the last checkpoint.

Usage:
  .venv-train/bin/python distill/train_voice.py --voice suraj --langs hi mr en --steps 3000
"""
import argparse, glob, os, re, shutil, subprocess, sys
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Base checkpoint to fine-tune from (rhasspy/piper-checkpoints) and espeak-ng voice per language.
# Marathi uses the Hindi checkpoint: the Marathi one is multi-speaker and won't load into a
# single-speaker run, and Hindi shares the script and most sounds.
BASE = {
    "en": ("en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt", "en-us"),
    "hi": ("hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt", "hi"),
    "mr": ("hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt", "mr"),
}


def default_accelerator():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def train(voice, lang, steps, batch, accel, val_every):
    data = os.path.join(HERE, "data", voice, lang)
    if not os.path.exists(os.path.join(data, "metadata.csv")):
        sys.exit(f"no dataset at {data}; run distill/make_dataset.py first")
    run = os.path.join(HERE, "runs", voice, lang)
    os.makedirs(run, exist_ok=True)
    ckpt_rel, espeak = BASE[lang]
    base = hf_hub_download("rhasspy/piper-checkpoints", ckpt_rel, repo_type="dataset",
                           local_dir=os.path.join(HERE, "checkpoints"))
    base_step = int(re.search(r"step=(\d+)", ckpt_rel).group(1))
    last = sorted(glob.glob(os.path.join(run, "lightning_logs", "version_*", "checkpoints", "last.ckpt")))
    start = last[-1] if last else base  # resume if a previous run exists
    print(f"[{lang}] training from {os.path.relpath(start, ROOT)} for {steps} steps on {accel}", flush=True)
    cmd = [sys.executable, os.path.join(HERE, "piper_fit.py"), "fit",
           "--data.voice_name", f"{voice}_{lang}",
           "--data.csv_path", os.path.join(data, "metadata.csv"),
           "--data.audio_dir", os.path.join(data, "wavs"),
           "--data.espeak_voice", espeak,
           "--data.cache_dir", os.path.join(run, "cache"),
           "--data.config_path", os.path.join(run, "config.json"),
           "--data.batch_size", str(batch),
           "--model.sample_rate", "22050",
           "--trainer.accelerator", accel, "--trainer.devices", "1",
           "--trainer.max_steps", str(base_step + steps),
           "--trainer.check_val_every_n_epoch", str(val_every),
           "--trainer.default_root_dir", run,
           "--ckpt_path", start]
    subprocess.run(cmd, check=True)
    return run


def export(voice, lang, run):
    ckpts = glob.glob(os.path.join(run, "lightning_logs", "version_*", "checkpoints", "*.ckpt"))
    best = [c for c in ckpts if os.path.basename(c).startswith("best-mos")]
    pick = max(best or [c for c in ckpts if c.endswith("last.ckpt")], key=os.path.getmtime)
    out_dir = os.path.join(ROOT, "voices", voice, "piper")
    os.makedirs(out_dir, exist_ok=True)
    onnx = os.path.join(out_dir, f"{lang}.onnx")
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", pick, "--output-file", onnx],
                   check=True)
    shutil.copy(os.path.join(run, "config.json"), onnx + ".json")
    print(f"[{lang}] exported {os.path.relpath(pick, ROOT)} -> {os.path.relpath(onnx, ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--langs", nargs="+", default=["hi", "mr", "en"])
    ap.add_argument("--steps", type=int, default=3000, help="fine-tune steps per language (2000-5000 typical)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accelerator", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--val-every", type=int, default=5, help="validate every N epochs")
    ap.add_argument("--export-only", action="store_true", help="skip training, export existing checkpoints")
    a = ap.parse_args()
    accel = a.accelerator or default_accelerator()
    for lang in a.langs:
        run = os.path.join(HERE, "runs", a.voice, lang) if a.export_only else \
            train(a.voice, lang, a.steps, a.batch, accel, a.val_every)
        export(a.voice, lang, run)


if __name__ == "__main__":
    main()
