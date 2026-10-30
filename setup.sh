#!/usr/bin/env bash
# One-time setup: Python env, OpenVoice code + weights, Piper base voices.
#
#   bash setup.sh            # web app (.venv-omni)
#   bash setup.sh --train    # also the training env (.venv-train) for distill/
#
# Needs Python 3.10 (set PYTHON=/path/to/python3.10 to pick one), git and ffmpeg. Re-running is safe.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.10}"
OPENVOICE_COMMIT=74a1d14

command -v "$PYTHON" >/dev/null || { echo "Python 3.10 not found. macOS: brew install python@3.10 | Ubuntu: sudo apt install python3.10 python3.10-venv"; exit 1; }
command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found; the app needs it to read recordings. macOS: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg"

echo "== 1/4 Python env .venv-omni"
[ -d .venv-omni ] || "$PYTHON" -m venv .venv-omni
.venv-omni/bin/pip install -q --upgrade pip
.venv-omni/bin/pip install -q -r requirements.txt

echo "== 2/4 OpenVoice code"
if [ ! -d OpenVoice ]; then
    git clone -q https://github.com/myshell-ai/OpenVoice.git
    git -C OpenVoice checkout -q "$OPENVOICE_COMMIT"
fi

echo "== 3/4 Model files (OpenVoice V2 converter, Piper voices for en / hi / mr)"
.venv-omni/bin/python - <<'EOF'
import os
from huggingface_hub import hf_hub_download

for f in ("converter/config.json", "converter/checkpoint.pth"):
    hf_hub_download("myshell-ai/OpenVoiceV2", f, local_dir="OpenVoice/checkpoints_v2")
for v in ("en/en_US/lessac/medium/en_US-lessac-medium", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium",
          "mr/mr_IN/google/medium/mr_IN-google-medium"):
    for ext in (".onnx", ".onnx.json"):
        hf_hub_download("rhasspy/piper-voices", v + ext, local_dir="piper_voices")
print("model files ready")
EOF

if [ "${1:-}" = "--train" ]; then
    echo "== 4/4 Training env .venv-train"
    [ -d .venv-train ] || "$PYTHON" -m venv .venv-train
    .venv-train/bin/pip install -q --upgrade pip
    .venv-train/bin/pip install -q -r requirements-train.txt
    [ -d piper1-gpl ] || git clone -q --depth 1 https://github.com/OHF-Voice/piper1-gpl.git
    # Piper's trainer needs a small C extension (monotonic alignment) that the pip package doesn't ship built
    MA="$(.venv-train/bin/python -c 'import os, piper; print(os.path.join(os.path.dirname(piper.__file__), "train/vits/monotonic_align"))')"
    CYTHONIZE="$PWD/.venv-train/bin/cythonize"
    cp piper1-gpl/src/piper/train/vits/monotonic_align/core.pyx "$MA/"
    (cd "$MA" && mkdir -p monotonic_align && "$CYTHONIZE" -q -i core.pyx && mv core*.so monotonic_align/)
    .venv-train/bin/python -c "from piper.train.vits.monotonic_align import maximum_path; print('trainer ready')"
else
    echo "== 4/4 Training env skipped (run: bash setup.sh --train)"
fi

echo
echo "Done. Start the app:  .venv-omni/bin/python app.py   then open http://localhost:7860"
echo "First start downloads OmniVoice and Whisper (~3 GB), so it takes a few minutes once."
