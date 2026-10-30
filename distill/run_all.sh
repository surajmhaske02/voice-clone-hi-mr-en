#!/usr/bin/env bash
# Hack #2 end to end on this Mac: teacher dataset -> fine-tune -> ONNX, one language at a time
# (so the first language is usable early). Resumable: just run it again.
#
#   bash distill/run_all.sh suraj            # hi, mr, en
#   bash distill/run_all.sh suraj hi         # only Hindi
#
# Env overrides: N=400 sentences/lang, STEPS=3000 fine-tune steps/lang, BATCH=16.
# The MPS caps stop the trainer growing past RAM (it reached 20 GB on a 16 GB Mac and swapped).
# Stop the web app first: the teacher and the trainer both need the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

VOICE="${1:?usage: run_all.sh <voice> [langs...]}"
shift
LANGS=("${@:-hi mr en}")
LANGS=(${LANGS[*]})
N="${N:-400}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-16}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.8}"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.6}"

for lang in "${LANGS[@]}"; do
    echo "===== [$lang] 1/2 teacher dataset ($N sentences) $(date +%H:%M)"
    .venv-omni/bin/python distill/make_dataset.py --voice "$VOICE" --langs "$lang" --n "$N"
    echo "===== [$lang] 2/2 fine-tune ($STEPS steps) + export $(date +%H:%M)"
    .venv-train/bin/python distill/train_voice.py --voice "$VOICE" --langs "$lang" --steps "$STEPS" --batch "$BATCH"
    echo "===== [$lang] done: voices/$VOICE/piper/$lang.onnx $(date +%H:%M)"
done
