"""
GPT-style transformer, implemented from primitives rather than assembled
from high-level building blocks - no nn.MultiheadAttention, no
nn.Transformer, no pretrained weights loaded anywhere in this file. Every
module here is built from nn.Linear / nn.Embedding / nn.LayerNorm.

Two implementation choices worth calling out:

  1. Attention uses PyTorch's fused F.scaled_dot_product_attention instead
     of a hand-rolled "mask, softmax, matmul" version. Same math
     (softmax(QK^T / sqrt(d)) V, causally masked) - just the fast fused
     kernel instead of the explicit three-step version.
     
  2. Weight tying (input embedding == output projection) and residual-
     branch init scaling by (2 * n_layer) ** -0.5, both standard
     techniques from the original GPT-2 implementation, applied here to
     keep training stable as depth increases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192      # size of the from-scratch BPE vocab
    block_size: int = 256       # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True           # bias in Linear/LayerNorm


class CausalSelfAttention(nn.Module):
    # Multi-head causal self-attention combined into one module and computed with a fused kernel

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # combined Q,K,V projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # flagged for residual-scaled init
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # softmax(QK^T / sqrt(head_size)) V, causally masked, fused kernel.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
        Per-token feedforward same Linear->act->Linear
        shape, expanding to 4x n_embd in the middle.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")  # real GPT-2's activation (Phase 5)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
        One transformer block - pre-norm residual wiring: 
        x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),   # learned positional embedding
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: input embedding and output
        # projection are literally the same tensor.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        '''
            residual-branch init scaling: every c_proj layer that writes into
            the residual stream gets its init std scaled down by
            (2 * n_layer) ** -0.5, per the variance-accumulation
            experiment, residuals across n_layer blocks would otherwise
            compound variance and destabilize deep training.
        '''
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] {n_params/1e6:.2f}M parameters "
              f"(vocab={config.vocab_size}, n_layer={config.n_layer}, "
              f"n_head={config.n_head}, n_embd={config.n_embd}, block_size={config.block_size})")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressive sampling pattern:
        run the model, take the last time step's logits, softmax, sample,
        append, repeat. Cropping to block_size is required here since the
        positional embedding table only has block_size entries."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                              betas: tuple[float, float], device_type: str) -> torch.optim.Optimizer:
        """Optimizer setup: weight decay on every 2D+ tensor
        (weight matrices, embeddings), no decay on biases/LayerNorm params.
        Uses fused AdamW when running on CUDA, plain AdamW otherwise."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"[optim] decayed params: {len(decay_params)} tensors, {n_decay:,} elements")
        print(f"[optim] non-decayed params: {len(nodecay_params)} tensors, {n_nodecay:,} elements")

        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra = dict(fused=True) if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)
        print(f"[optim] using fused AdamW: {use_fused}")
        return optimizer
