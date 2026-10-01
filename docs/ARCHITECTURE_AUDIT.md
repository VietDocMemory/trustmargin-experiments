# Architecture audit

## RAG repository

- `RAGRetriever.search(query, collection_name, top_k=5, score_threshold=0)` is async. The API passes `session_id` directly as `collection_name`. Qdrant stores a 1024 dimensional dense BGE-M3 vector and a sparse `text-sparse` vector per chunk. Retrieval expands segmented Vietnamese queries, fuses dense/sparse hits with RRF, reranks with `BAAI/bge-reranker-v2-m3`, then removes near duplicates. Output is a list of `{content, score, page, chunk_index}`; `score` is the reranker score. The embedding model is `BAAI/bge-m3`.
- Reuse `src.retrieval.search_engine.RAGRetriever` and, for ingestion if needed, `PDFIngestionPipeline` plus `VectorStoreManager`. Do not use the Ollama `RAGGenerator` (`qwen2.5:7b-instruct`), its Redis answer cache, or its LLM judge. The RAG evaluator has no reusable EM/F1 scorer.
- Existing evaluation data uses `question`, `ground_truth_answer`, `ground_truth_context`, and `question_type`; it has no stable ID or document ID. The runner must derive a stable ID and receive the correct collection/document mapping explicitly.

## D2L repository

- The public example loads a checkpoint with `torch.load`, then `ModulatedPretrainedModel.from_state_dict(..., train=False, use_sequence_packing=False)`. That constructs a PEFT wrapped causal LM and a separate context encoder. `model.internalize(document)` tokenizes context and generates in-memory LoRA weights; `model.generate(...)` combines and attaches them. `model.reset()` removes them. `model.generated_loras` permits a document-hash cache of generated weights. This is not a standard PEFT adapter directory.
- The example checkpoint is for `google/gemma-2-2b-it`; the loader also records the checkpoint's actual `base_model_name_or_path`. Verify it equals the configured model at runtime. Use `ctx_to_lora.model_loading.get_tokenizer` with its Gemma chat template for generation and scoring. The D2L evaluator has ROUGE-L F1 via `rouge_score`, but the full evaluation pipeline is too coupled to D2L training tasks to reuse.
- Use the checkpoint-loaded `model.base_model` for both candidate generation paths and all likelihood views, with `reset()` before base calls. This avoids loading another Gemma copy. The context encoder is a separate model and needs additional VRAM.

## Compatibility and limits

- RAG declares `transformers<4.40.0` and `peft<=0.11.0`; D2L pins `transformers==4.51.3` and requires a newer PEFT API. The environments cannot be combined reliably. Invoke the unchanged retriever through a small JSON subprocess launched with `RAG_PYTHON`, with the RAG repo as its working directory.
- D2L's full training installation depends on Linux-oriented packages such as DeepSpeed and FlashAttention. For this experiment, a native Windows inference environment with Torch 2.6 CUDA 12.4, the D2L checkpoint, and gated Gemma weights has since been installed. A three-sample GPU smoke test passed on the RTX 3060. The retrieval subprocess exits before Gemma loads, limiting overlap. A long full-document run is still unverified.
- The source RAG retriever has a potential diversity-filter edge case: its first selected passage bypasses the `top_k` break. The wrapper will cap returned results at `top_k` without modifying the source repo.
- The provided [TrustMargin paper](https://arxiv.org/abs/2606.08397) studies **Direct versus RAG**. This project tests an experimental **D2L versus RAG** extension with the user-specified formula and thresholds; it is not a reproduction of the paper's reported results. Both candidates are scored by the same unadapted Gemma base model and tokenizer.
