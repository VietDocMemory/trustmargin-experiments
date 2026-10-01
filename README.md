# TrustMargin experiments: Doc-to-LoRA versus RAG

This is a research adaptation of [TrustMargin (arXiv:2606.08397)](https://arxiv.org/abs/2606.08397). The paper arbitrates **Direct versus RAG** answers. This project arbitrates **Doc-to-LoRA versus RAG** answers using the user-specified extension; it is not an exact reproduction of the paper.

## Architecture

```text
question ── RAGRetriever (Qdrant/BGE) ── passages ── Gemma base ── RAG answer
       └── document ── D2L hypernetwork ── LoRA ── same Gemma ── D2L answer
                                      both candidates + passages
                                                 │
                          six base-Gemma likelihoods → margin → decision
```

The RAG retriever runs in a separate Python environment because its `transformers<4.40` requirement conflicts with D2L's `transformers==4.51.3`. The retriever process exits before Gemma loads. Both answer paths and all likelihood scores use the **same checkpoint-loaded `google/gemma-2-2b-it` base model** and its tokenizer. The D2L context encoder is an additional model used only to generate LoRA weights. The RAG repository's Ollama/Qwen generator is never called. See [architecture audit](docs/ARCHITECTURE_AUDIT.md).

## Installation

Install the two source repositories in separate Python environments. The D2L repository's full `install.sh` targets Linux and includes training packages. For this experiment, a native Windows **inference-only** environment with CUDA has been smoke-tested; see [Windows setup](docs/WINDOWS_SETUP.md). The checkpoint is not bundled here.

```bash
python -m venv .venv-rag
.venv-rag/bin/python -m pip install -e ../vietnamese-rag-system
python -m venv .venv
.venv/bin/python -m pip install -e ../doc-to-lora
.venv/bin/python -m pip install -e '.[test]'
```

Set `RAG_PYTHON` to the first environment's Python executable. On Windows the path is `Scripts/python.exe`. Before generating, ingest the chosen document into Qdrant with the RAG repository's existing ingestion flow and record the collection/session ID. Supply the **same document** to D2L. Never use `ground_truth_context` as the D2L document: that would leak evaluation labels.

## Environment variables

Copy `.env.example` and export the values into your shell; this project does not silently load `.env` files. `RAG_REPO_PATH`, `D2L_REPO_PATH`, `D2L_CHECKPOINT`, and `RAG_PYTHON` are configurable. Relative paths resolve from this project root. The RAG repository's own `QDRANT_HOST` and `QDRANT_PORT` variables apply to the retriever subprocess.

```bash
export RAG_REPO_PATH=../vietnamese-rag-system
export D2L_REPO_PATH=../doc-to-lora
export D2L_CHECKPOINT=../doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin
export RAG_PYTHON=$PWD/.venv-rag/bin/python
```

Input JSONL needs `question`, `ground_truth_answer` or `gold_answers`, and optionally `id`, `split`, `question_type`, `collection_name`, `document_path`, or `document_text`. If the document is shared, use `--document`. `document_path` can be UTF-8 text or PDF. `--collection` is an existing Qdrant collection, not a file path. The included RAG evaluation dataset lacks stable IDs and document mappings; the runner derives IDs and requires an explicit document and collection.

## Smoke test

The first run should use three samples and a short document. On Windows, [the setup guide](docs/WINDOWS_SETUP.md) prepares a legal-document passage, ingests it into Qdrant, and creates three smoke rows. This setup avoids feeding the entire 109-page PDF into the D2L context encoder.

```bash
../vietnamese-rag-system/.venv/bin/python scripts/prepare_smoke_data.py
python scripts/smoke_test.py --input outputs/setup_smoke_input.jsonl --document outputs/setup_smoke_document.txt --collection setup_smoke_legal --limit 3
```

The script checks retrieval, both nonempty candidates, six finite likelihoods, finite margins, selection, and JSONL output. It prints one readable sample and per-sample CUDA peak memory. Inspect memory across samples; allocator reservation alone can rise without a leak.

## Generate candidates

```bash
python scripts/generate_candidates.py --input DATASET.jsonl --document DOCUMENT.pdf --collection COLLECTION
```

This writes `outputs/candidates.jsonl`. It caches generated LoRA tensors under `outputs/adapter_cache/` using a document hash and checkpoint identity. Existing candidate files are preserved unless `--overwrite` is supplied. Only a deliberate regeneration uses the retriever, adapter, or generator again.

## Score candidates

```bash
python scripts/score_candidates.py
```

This writes `outputs/scored_candidates.jsonl` with the six length-normalized answer-only teacher-forced log likelihoods. The base Gemma scores both candidates under question-only, question plus retrieved context, and context-only prompts. Prompt tokens are excluded. Empty answers are rejected.

## Run TrustMargin and oracle evaluation

```bash
python scripts/run_full_eval.py
python scripts/run_full_eval.py --lambda-bind 0.75 --tau -1 --overwrite
```

This stage reads cached scores and never loads a model. It reports D2L, RAG, oracle, and TrustMargin Exact Match, token F1, and ROUGE-L F1. For `question_type="unanswerable"`, it additionally reports abstention accuracy against the exact normalized phrase `Tài liệu không đề cập`. Oracle chooses the candidate with higher per-sample token F1, then EM; ties choose D2L. The report counts RAG better, D2L better, ties, both wrong, and oracle gain over the best single source. A near-zero gain triggers a complementarity warning.

The experimental margin is:

```text
M_prior = l_D(y_R) - l_D(y_D)
Delta(y) = l_R(y) - l_C(y)
M_bind = Delta(y_R) - Delta(y_D)
M = M_prior + lambda_bind * M_bind
select RAG iff M > tau; otherwise select D2L
```

Defaults are `lambda_bind=0.5` and `tau=-1.5`.

## Weights & Biases logging

Install the optional tracking dependency in the D2L Python environment. On Windows:

```powershell
py -m uv pip install --python ..\doc-to-lora\.venv\Scripts\python.exe -e '.[tracking]'
```

Authenticate interactively with `..\doc-to-lora\.venv\Scripts\wandb.exe login` after creating a new API key. Keep the key out of the repository, scripts, shell history, and experiment artifacts. The default project is `trustmargin-d2l-rag`; use `--wandb-project` and `--wandb-entity` to choose another destination.

To log a new evaluation automatically:

```powershell
..\doc-to-lora\.venv\Scripts\python.exe scripts\run_full_eval.py --wandb --wandb-project trustmargin-d2l-rag
```

Use `--wandb-artifact-file outputs/candidates.jsonl` when you also want the candidate-stage file attached to that run.

To upload the already generated smoke results without rerunning inference:

```powershell
..\doc-to-lora\.venv\Scripts\python.exe scripts\log_existing_run.py --results outputs\smoke_results.jsonl --report outputs\smoke_report.json --artifact-file outputs\smoke_candidates.jsonl --artifact-file outputs\smoke_scored.jsonl
```

Add `--mode offline` to `log_existing_run.py`, or `--wandb-mode offline` to `run_full_eval.py` or `smoke_test.py`, to store a run under ignored `outputs/wandb/` without authentication. After login, sync a saved run with `wandb sync` and its `offline-run-*` directory. Each run records the evaluation config, D2L/RAG/oracle/TrustMargin metrics, per-question decisions, and a versioned artifact with the selected JSONL and report files. Those artifacts include question and answer text; select a private W&B project if the documents are confidential. The full D2L checkpoint and generated adapter tensors are not uploaded.

## Validation threshold sweep

Prepare a **validation-only** scored file with `split="validation"` on every row. Then run:

```bash
python scripts/run_full_eval.py --input outputs/validation_scored.jsonl --output outputs/validation_results.jsonl --report outputs/validation_report.json --sweep
```

The sweep evaluates the configured six lambda values and six tau values and labels its result `validation-tuned`. It refuses mixed, missing, or test splits. Do not tune on the test set. Changing lambda or tau needs only this third stage.

## Output schema

One line of `outputs/results.jsonl` includes sample metadata, retrieved passages and scores, both answers, all six likelihoods, margins, decision, cache status, latencies, and peak CUDA memory. An abbreviated example:

```json
{"id":"q1","dataset":"validation","question":"What is X?","gold_answers":["X is ..."],"document_id":"doc1","retrieved_contexts":[{"content":"...","score":0.8,"page":1,"chunk_index":2}],"retrieval_scores":[0.8],"d2l_answer":"...","rag_answer":"...","scores":{"q_only":{"d2l":-2.1,"rag":-1.9},"q_context":{"d2l":-1.5,"rag":-1.1},"context_only":{"d2l":-1.8,"rag":-1.7}},"m_prior":0.2,"delta_d2l":0.3,"delta_rag":0.6,"m_bind":0.3,"trust_score":0.35,"lambda_bind":0.5,"tau":-1.5,"selected_source":"rag","final_answer":"...","latency":{"retrieval":0.1,"adapter_build":1.0,"rag_generation":0.5,"d2l_generation":0.6,"likelihood_scoring":0.8,"total":3.0},"memory":{"peak_allocated_mb":5000,"peak_reserved_mb":6000}}
```

## Known limitations

- The Windows inference environment and a short three-sample GPU smoke test have passed. All 109 pages of the legal PDF were independently internalized into 109 page adapters. The full document is about 90,000 tokens and cannot be supplied to D2L's single-context `internalize` API. The page adapters do not constitute one document adapter; the experiment's full-document candidate path remains unsupported without an explicit page-selection or aggregation method.
- Full-document internalization is limited by the checkpoint context encoder. Prepare a bounded, document-consistent input before evaluation; this runner does not silently truncate.
- PDF extraction order can be imperfect. Confirm the extracted document matches the Qdrant collection.
- ROUGE-L tokenization is primarily whitespace-based and may not fully reflect Vietnamese word boundaries. EM/F1 are lexical metrics, not a semantic judge.
- The adapter cache holds generated tensors and is only valid for the matching checkpoint and document content. The candidate cache is a stage artifact; changing prompts, model, document, or collection requires explicit regeneration.
