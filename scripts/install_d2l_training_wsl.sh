#!/usr/bin/env bash
# Run inside Ubuntu WSL after Windows has restarted and WSL is functional.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D2L_REPO="$(cd "${SCRIPT_DIR}/../../doc-to-lora" && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WSL cannot see the NVIDIA GPU; check the Windows driver before installing CUDA packages." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y build-essential git curl libaio-dev python3-dev
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$D2L_REPO"
if ! grep -qxF '.venv-linux/' .git/info/exclude; then
  printf '\n.venv-linux/\n' >> .git/info/exclude
fi
uv python install 3.10
uv venv --python 3.10 .venv-linux
uv pip install --python .venv-linux/bin/python torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124
UV_PROJECT_ENVIRONMENT=.venv-linux uv sync --locked
uv pip install --python .venv-linux/bin/python tokenizers==0.21.0
uv pip install --python .venv-linux/bin/python 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
uv pip install --python .venv-linux/bin/python flashinfer-python==0.2.2 -i https://flashinfer.ai/whl/cu124/torch2.6
.venv-linux/bin/python -c 'import torch, deepspeed, flash_attn, flashinfer, vllm; print("CUDA", torch.cuda.is_available(), "Torch", torch.__version__)'
