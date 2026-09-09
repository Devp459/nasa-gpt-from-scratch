# nasa-gpt-from-scratch

A GPT-style language model trained from scratch — tokenizer, transformer, and training loop all implemented directly on top of PyTorch primitives, with no pretrained weights or pretrained tokenizers anywhere in the pipeline — on a corpus of ~9,000 NASA technical reports pulled from the NASA Technical Reports Server (NTRS).

This is a learning/portfolio project built to understand, end to end, how a GPT actually works: not by fine-tuning an existing model, but by building the tokenizer, the attention mechanism, and the training loop by hand, then running the whole pipeline on real, messy, domain-specific text at a scale where the design choices actually matter.

## What "from scratch" means here

- **Tokenizer**: a byte-level BPE tokenizer implemented from the two core primitives (`get_stats`, `merge`), with GPT-2-style regex pre-splitting and incremental pair-count bookkeeping so training doesn't rescan the whole corpus on every merge step. No `tiktoken`, no HuggingFace `tokenizers`, no `sentencepiece`.
- **Model**: a GPT-style transformer assembled from `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` — no `nn.MultiheadAttention`, no `nn.Transformer`. Attention uses PyTorch's fused `F.scaled_dot_product_attention` kernel (same math as hand-rolled masked-softmax attention, just the fast kernel), plus weight tying and residual-branch init scaling.
- **Training**: AdamW with parameter-group weight decay, gradient accumulation, a cosine LR schedule with linear warmup, and gradient clipping — no pretrained checkpoint is ever loaded; every run starts from random init.
- **Data**: the model has never seen the corpus before this project — it's trained entirely on NASA NTRS text collected and filtered specifically for this project.

## Pipeline

```
fetch metadata → filter → fetch download links → download report text → train tokenizer → train model → generate
```

1. **Filter metadata** (`links/filter_ntrs_metadata.py`) — narrows NTRS's public metadata dump down to the documents worth training on, producing `candidates.ndjson`.
2. **Fetch download links** (`links/fetch_ntrs_download_links.py`) — hits the NTRS API per document ID to resolve the actual downloadable file (`fulltext` preferred, `pdf` fallback), producing `links.txt`. Resumable, parallelized with `multiprocessing`.
3. **Download report text** — two interchangeable downloaders, same resumable design (tracks completed URLs in a sibling `.done_urls` file so a run can be stopped and restarted):
   - `links/pdf_downloader.py` — concurrent (`ThreadPoolExecutor`), fast, more load on NTRS.
   - `links/download_ntrs_text.py` — sequential with a delay between requests, slower, gentler on NTRS.
4. **Train** (`training/train.py` for CPU, `training/train_gpu.py` for GPU) — trains (or loads) the BPE tokenizer, encodes the full corpus, splits 90/10 train/val, and trains the model with periodic validation, checkpointing, and sample generation.
5. **Generate** (`training/generate.py`) — loads a saved checkpoint and samples a continuation from a prompt, without retraining.

## Repo structure

```
links/
  filter_ntrs_metadata.py    # raw NTRS metadata -> candidates.ndjson
  fetch_ntrs_download_links.py  # candidates -> links.txt
  pdf_downloader.py           # concurrent downloader (fast)
  download_ntrs_text.py       # sequential downloader (slow, polite)
training/
  bpe_tokenizer.py            # from-scratch byte-level BPE tokenizer
  model.py                    # GPT model definition
  train.py                    # training loop, CPU-sized defaults
  train_gpu.py                # training loop, GPU-sized defaults (bigger model, mixed precision, torch.compile)
  generate.py                 # inference from a saved checkpoint
requirements.txt
Makefile
```

## Setup

```bash
make install
```

`torch` needs a CUDA-matched build for GPU training — a plain `pip install -r requirements.txt` installs CPU-only torch. Install the CUDA build separately first, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

(pick the CUDA tag that matches your driver, per [pytorch.org](https://pytorch.org/get-started/locally/))

## Usage

All scripts are configured with plain variables at the top of the file (`DATA_DIR`, `OUT_DIR`, etc.) — edit those, or override with CLI flags.

```bash
make links           # filter metadata, then fetch download links
make download-fast    # concurrent download
make download-slow    # sequential, rate-limited download
make train            # train the model (GPU-sized config)
make generate          # sample text from a trained checkpoint
```

## Model & training details

| | CPU config (`train.py`) | GPU config (`train_gpu.py`) |
|---|---|---|
| Parameters | ~13M | ~92M |
| Layers / heads / embd dim | 6 / 6 / 384 | 12 / 12 / 768 |
| Context length | 256 | 512 |
| Batch size / grad accum | 12 / 16 | 24 / 4 |
| Mixed precision | optional, off by default | on by default (bfloat16) |
| `torch.compile` | not used | used, with automatic fallback |

Both configs share the same tokenizer and model code — only scale and hardware-specific training-loop optimizations differ.

## Results

Trained on 8,926 NASA technical report text files (~394M characters, encoded to ~164M tokens with an 8,192-token BPE vocabulary; 147.9M train / 16.4M val tokens).

- **91.74M parameters** (vocab_size=8192, n_layer=12, n_head=12, n_embd=768, block_size=512)
- **Best validation loss: 2.6904** (iteration 11,750 of 15,000) — perplexity ≈ e^2.6904 ≈ 14.7
- ~89,000 tokens/sec training throughput on a single consumer GPU

At this loss level, the model reliably produces grammatical, domain-appropriate continuations (technical register, plausible NASA-report phrasing and structure) from a prompt, though it is not a general-purpose chatbot and was not trained or evaluated as one — see Limitations.

## Limitations / honest notes

- This is a **base language model**, not an instruction-following or chat model — it completes text, it doesn't answer questions or follow instructions unless the prompt is phrased as the start of a report-style passage.
- 92M parameters trained on ~164M tokens is small by modern standards (for comparison, GPT-2 small is 124M parameters trained on ~10B tokens) — the goal here was a correct, from-scratch implementation at a scale that trains in reasonable time on consumer hardware, not a state-of-the-art result.
- No safety filtering, no RLHF, no dataset deduplication beyond the metadata filtering step — outputs should be treated as a research/portfolio artifact, not a production system.

## License

Add a license of your choice (MIT is a common default for portfolio projects like this).