#!/usr/bin/env bash
# Zip what Colab needs to fine-tune on a GPU: trainer scripts + the teacher dataset of one voice.
#   bash distill/pack_for_colab.sh suraj   ->  distill/colab_suraj.zip  (upload it in colab_train.ipynb)
set -euo pipefail
cd "$(dirname "$0")/.."
VOICE="${1:?usage: pack_for_colab.sh <voice>}"
rm -f "distill/colab_${VOICE}.zip"
zip -qr "distill/colab_${VOICE}.zip" distill/piper_fit.py distill/train_voice.py "distill/data/${VOICE}"
echo "wrote distill/colab_${VOICE}.zip ($(du -h "distill/colab_${VOICE}.zip" | cut -f1))"
