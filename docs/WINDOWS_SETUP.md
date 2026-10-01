# Windows inference setup

These steps install the two repositories in separate environments for the D2L/RAG experiment. They were exercised with an RTX 3060, Python 3.12, Torch 2.6 CUDA 12.4, and Qdrant 1.17.1. They do not install D2L's Linux training stack or RAG's Ollama generator.

Run PowerShell from the directory containing the three sibling repositories.

## Python and uv

```powershell
py -m pip install --user uv
py -m uv python install 3.12
```

## RAG environment

```powershell
Set-Location vietnamese-rag-system
py -m uv sync --locked --group dev --python 3.12
```

The lockfile currently installs CPU Torch in the RAG environment. Retrieval works on CPU; the RAG worker exits before D2L loads Gemma. Create an ignored `.env` with `QDRANT_HOST=127.0.0.1` and `QDRANT_PORT=6333`. Redis is optional because the repo falls back to in-memory caching when it is unavailable.

The RAG lockfile has `qdrant-client==1.17.1`. Use a Qdrant 1.17.1 server to avoid the client's version warning. An [official Windows release archive](https://github.com/qdrant/qdrant/releases/tag/v1.17.1) is available. Extract `qdrant.exe` to a local tools directory and run it with its working directory in the ignored `vietnamese-rag-system/qdrant_storage/v1.17.1` directory:

```powershell
$qdrantData = (New-Item -ItemType Directory -Force -Path .\qdrant_storage\v1.17.1).FullName
Start-Process -FilePath "$env:LOCALAPPDATA\qdrant-1.17.1\qdrant.exe" -WorkingDirectory $qdrantData -WindowStyle Hidden
.\.venv\Scripts\python.exe -c "from qdrant_client import QdrantClient; print(QdrantClient(host='localhost',port=6333).get_collections())"
```

If the executable is stored elsewhere, change `-FilePath`. Qdrant must be started again after a reboot. An empty collection list means the server is healthy but no document has been ingested yet.

## D2L inference environment

```powershell
Set-Location ..\doc-to-lora
py -m uv venv --python 3.12 .venv
py -m uv pip install --python .venv\Scripts\python.exe torch==2.6.0 --torch-backend=cu124
py -m uv pip install --python .venv\Scripts\python.exe transformers==4.51.3 peft==0.15.2 accelerate==1.6.0 datasets==3.6.0 einops jaxtyping pyyaml bitsandbytes==0.46.1 opt-einsum rouge-score pypdf pytest hf_xet
$env:PYTHONUTF8 = '1'
py -m uv pip install --python .venv\Scripts\python.exe --no-deps -e .
```

`PYTHONUTF8=1` is needed for editable installation on Windows because this repository's `setup.py` reads its Unicode README using the active locale. The inference-only installation intentionally skips Linux-only packages such as DeepSpeed, vLLM, and FlashAttention.

Accept the Gemma model terms on Hugging Face and authenticate locally. Do not put the token in a repository or chat.

```powershell
.\.venv\Scripts\hf.exe auth login
```

Then download the public D2L checkpoint and gated Gemma model. The following command populates the standard Hugging Face cache for Gemma and a gitignored `trained_d2l` folder for the checkpoint:

```powershell
@'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download('SakanaAI/doc-to-lora', filename='gemma_demo/checkpoint-80000/pytorch_model.bin', local_dir='trained_d2l')
snapshot_download('google/gemma-2-2b-it', allow_patterns=['*.json', '*.model', '*.safetensors'])
'@ | .\.venv\Scripts\python.exe -
```

Verify CUDA and D2L imports:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel; print(torch.__version__, torch.cuda.is_available())"
```

## Experiment package and smoke test

```powershell
Set-Location ..\trustmargin-experiments
py -m uv pip install --python ..\doc-to-lora\.venv\Scripts\python.exe -e .
..\doc-to-lora\.venv\Scripts\python.exe -m pytest -q
```

The experiment defaults to the sibling repo paths, the D2L checkpoint above, and `vietnamese-rag-system/.venv/Scripts/python.exe` for retrieval. The D2L context encoder cannot internalize an entire long PDF in this GPU configuration. The preparation script extracts a short passage from the sample PDF, creates three smoke rows, and ingests the **same passage** through the RAG repo's vector-store implementation. Artifacts live under ignored `outputs/setup_smoke_*`; the collection is `setup_smoke_legal`.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
..\vietnamese-rag-system\.venv\Scripts\python.exe scripts\prepare_smoke_data.py
..\doc-to-lora\.venv\Scripts\python.exe scripts\smoke_test.py --input outputs\setup_smoke_input.jsonl --document outputs\setup_smoke_document.txt --collection setup_smoke_legal --limit 3
```
