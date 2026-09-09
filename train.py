"""
What the file does, in order:
  1. Loads every .txt file in ./data_dir (extracted NTRS text).
  2. Trains a from-scratch BPE tokenizer (bpe_tokenizer.py) on that corpus,
     or loads one already trained (TOKENIZER_PATH), so re-runs don't
     retrain from scratch every time.
  3. Encodes the whole corpus with that tokenizer, splits 90/10 train/val.
  4. Trains the GPT from model.py with AdamW, gradient accumulation,
     cosine LR schedule with linear warmup, and gradient clipping,
     sized for a bigger model (~90M params by default), 
     with mixed precision (torch.autocast(bfloat16)) ON by
     default, torch.compile() applied when available (skipped
     automatically if it fails, e.g. on some Windows/Triton setups), and
     pinned-memory/non-blocking host->device transfer for the batch.
  5. Periodically evaluates on the held-out val split, checkpoints, and
     prints a short generated sample so you can watch quality improve.

"""

from __future__ import annotations

import argparse
import math
import os
import time
import unicodedata
from pathlib import Path

import torch

from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig


# ----------------------------------------------------------------------
# Edit these directly instead of passing --data_dir / --out_dir.
# TOKENIZER_PATH: if you already trained a tokenizer on a previous run
# point this at that tokenizer.json to skip retraining.
# ----------------------------------------------------------------------
DATA_DIR = "./data"                           # where to load .txt files from
OUT_DIR = ""                                  # where to save checkpoints and tokenizer.json
TOKENIZER_PATH = "./runs/gpt-v1"              # to reuse any previous tokenizer runs


def load_corpus(data_dir: str) -> str:
    paths = sorted(Path(data_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt files found in {data_dir}")
    parts = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[data] skipping {p.name}: {e}")
            continue
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
        parts.append(text)
    corpus = "\n\n".join(parts)
    print(f"[data] loaded {len(paths):,} files, {len(corpus):,} characters total")
    return corpus


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor,
                   block_size: int, batch_size: int, eval_iters: int, device: str) -> dict:
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, warmup_iters: int, lr_decay_iters: int,
           max_lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    ap.add_argument("--vocab_size", type=int, default=16384)

    # model — ~90M params at these defaults.
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--n_layer", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)

    '''
        training — batch size tuned to fit ~90M params + bf16 activations.
        If you hit CUDA OOM, lower --batch_size
        first (halve it), then --block_size, before touching model size.
    '''
    ap.add_argument("--batch_size", type=int, default=24, help="micro-batch size")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                     help="effective batch = batch_size * grad_accum_steps sequences")
    ap.add_argument("--max_iters", type=int, default=15000)
    ap.add_argument("--warmup_iters", type=int, default=300)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=20)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--compile", action="store_true", default=True,
                     help="use torch.compile for extra throughput (on by default, --no-compile to disable)")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train] cuda requested but not available, falling back to cpu - "
              "check your torch install (needs a CUDA build) and drivers")
        device = "cpu"
    print(f"[train] device={device}")
    if device == "cuda":
        print(f"[train] gpu: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    corpus = load_corpus(args.data_dir)

    tok_path = args.tokenizer_path or str(out_dir / "tokenizer.json")
    tokenizer = BPETokenizer()
    if os.path.exists(tok_path):
        print(f"[tokenizer] loading existing tokenizer from {tok_path}")
        tokenizer.load(tok_path)
    else:
        print(f"[tokenizer] training new BPE tokenizer, vocab_size={args.vocab_size}")
        train_sample = corpus if len(corpus) <= 20_000_000 else corpus[:20_000_000]
        tokenizer.train(train_sample, vocab_size=args.vocab_size, verbose=True)
        tokenizer.save(tok_path)
        print(f"[tokenizer] saved to {tok_path}")

    print("[data] encoding corpus with trained tokenizer...")
    ids = tokenizer.encode(corpus)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"[data] {len(data):,} tokens total, train={len(train_data):,}, val={len(val_data):,}")

    # ------------------------------------------------------------------
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)

    if args.compile and device == "cuda":
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed ({e}), continuing without it")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.max_lr,
        betas=(0.9, 0.95), device_type=device,
    )

    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    t0 = time.time()
    last_log_time = t0
    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(x, y)
            loss = loss / args.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()

        if it % args.log_interval == 0:
            now = time.time()
            dt = now - last_log_time
            last_log_time = now
            steps_since_last = args.log_interval if it > 0 else 1
            toks_since_last = args.batch_size * args.grad_accum_steps * args.block_size * steps_since_last
            tps = toks_since_last / max(dt, 1e-6)
            print(f"[train] iter {it:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{dt:.1f}s | ~{tps:,.0f} tok/s")

        if it > 0 and (it % args.eval_interval == 0 or it == args.max_iters - 1):
            losses = estimate_loss(model, train_data, val_data,
                                    args.block_size, args.batch_size, args.eval_iters, device)
            print(f"[eval] iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

            ckpt = {
                "model_state_dict": raw_model.state_dict(),
                "config": config,
                "iter": it,
                "val_loss": losses["val"],
            }
            torch.save(ckpt, out_dir / "last_checkpoint.pt")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(ckpt, out_dir / "best_checkpoint.pt")
                print(f"[eval] new best val loss {best_val_loss:.4f}, saved best_checkpoint.pt")

            raw_model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample_ids = raw_model.generate(context, max_new_tokens=150, temperature=0.8, top_k=50)[0].tolist()
            print("[sample]", tokenizer.decode(sample_ids).replace("\n", " ")[:300])
            raw_model.train()

    print(f"[train] done. total time {time.time() - t0:.1f}s, best val loss {best_val_loss:.4f}")

if __name__ == "__main__":
    main()