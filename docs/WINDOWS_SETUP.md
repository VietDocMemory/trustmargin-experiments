# Windows inference setup

These steps install the two repositories in separate environments for the D2L/RAG experiment. They were exercised with an RTX 3060, Python 3.12, Torch 2.6 CUDA 12.4, and Qdrant 1.17.1. Native Windows serves D2L inference. D2L training dependencies require WSL Ubuntu and a Windows restart after enabling its optional features.

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

The lockfile currently installs CPU Torch in the RAG environment. Retrieval works on CPU; the RAG worker exits before D2L loads Gemma. The ignored `.env` sets Qdrant at `127.0.0.1:6333`, Redis at `127.0.0.1:6379`, and Ollama at `127.0.0.1:11434/api/chat` with `qwen2.5:7b-instruct`.

On Windows, install Ollama and the Redis-compatible Memurai Developer service:

```powershell
winget install --id Ollama.Ollama --exact --silent --accept-package-agreements --accept-source-agreements
winget install --id Memurai.MemuraiDeveloper --exact --silent --accept-package-agreements --accept-source-agreements
Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull qwen2.5:7b-instruct
```

Memurai Developer is intended for development use; its service starts automatically. Run `scripts/start_rag_services.ps1` from the experiment repository after reboot to start Qdrant, Ollama, and the FastAPI app when needed.

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

The experiment defaults to the sibling repo paths, the D2L checkpoint above, and `vietnamese-rag-system/.venv/Scripts/python.exe` for retrieval. The preparation script extracts a short passage from the sample PDF, creates three smoke rows, and ingests the **same passage** through the RAG repo's vector-store implementation. Artifacts live under ignored `outputs/setup_smoke_*`; the collection is `setup_smoke_legal`.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
..\vietnamese-rag-system\.venv\Scripts\python.exe scripts\prepare_smoke_data.py
..\doc-to-lora\.venv\Scripts\python.exe scripts\smoke_test.py --input outputs\setup_smoke_input.jsonl --document outputs\setup_smoke_document.txt --collection setup_smoke_legal --limit 3
```

For the 109-page legal PDF, `scripts/internalize_full_pdf.py` extracts every page and saves one D2L adapter per page with a coverage manifest in `outputs/full_pdf_d2l`. This verifies that all pages can pass through D2L. The full PDF is about 90,000 Gemma tokens and cannot form a single adapter through `internalize`; per-page adapters need a page-selection strategy for document-wide questions. `scripts/ingest_full_pdf.py` uses the RAG repository's PDF pipeline to ingest all pages into the `full_legal_109_pages` Qdrant collection.

```powershell
..\doc-to-lora\.venv\Scripts\python.exe scripts\internalize_full_pdf.py
..\vietnamese-rag-system\.venv\Scripts\python.exe scripts\ingest_full_pdf.py
```

## D2L training environment in WSL

Enable WSL and the virtual machine platform from an elevated PowerShell, then restart Windows:

```powershell
winget install --id Microsoft.WSL --exact --silent --accept-package-agreements --accept-source-agreements
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

After restarting, install Ubuntu and run the Linux dependency installer. See [install_d2l_training_wsl.sh](../scripts/install_d2l_training_wsl.sh). This installs the training packages only; `doc-to-lora/install.sh` also downloads datasets, which are much larger.

```powershell
wsl --install -d Ubuntu-24.04
wsl -d Ubuntu-24.04
```

In Ubuntu, run `cd /mnt/c/Users/namviet157/Documents/trustmargin-experiments && bash scripts/install_d2l_training_wsl.sh`. Check `nvidia-smi` inside WSL before starting a training job. The repository's published training scripts are sized for more than one GPU; this machine has one RTX 3060 with 12 GB VRAM, so dependency installation alone does not establish that a full paper-scale training run fits.

`doc-to-lora/train.py` already calls `wandb.init()` on its main process. Once the Linux training environment is installed, run `.venv-linux/bin/wandb login` interactively and set `WANDB_PROJECT=trustmargin-d2l-rag` in the Ubuntu shell before starting a training job. The TrustMargin evaluation has separate optional W&B logging described in the experiment README.
