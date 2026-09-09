"""
Load a trained checkpoint and generate text from a prompt - no retraining
required. Loads the tokenizer and model config saved alongside the
checkpoint, encodes the prompt with the same from-scratch BPE tokenizer
used during training, and autoregressively samples a continuation with
temperature + top-k sampling.

Reports which training iteration and validation loss the checkpoint was
saved at, so you can sanity-check which run you're looking at.

"""

import argparse
import torch

from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig

CHECKPOINT_PATH = "./runs/gpt-v1/best_checkpoint.pt"               # path to the trained model checkpoint
TOKENIZER_PATH = "./runs/gpt-v1/tokenizer.json"                    # path to the trained tokenizer JSON file
PROMPT = "The results of the wind tunnel test show"                # prompt to generate text on
MAX_NEW_TOKENS = 200                                               # maximum number of new tokens to generate
TEMPERATURE = 0.8                                                  # temperature for sampling (higher = more random, lower = more greedy)
TOP_K = 50                                                         # top-k filtering (higher = more diverse, lower = more focused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    ap.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[generate] device={device}")

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[generate] loaded checkpoint from iter {ckpt['iter']}, val_loss {ckpt['val_loss']:.4f}")

    ids = tokenizer.encode(args.prompt)
    context = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(context, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k)[0].tolist()

    print("\n--- prompt ---")
    print(args.prompt)
    print("\n--- generated ---")
    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()