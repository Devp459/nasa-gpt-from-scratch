"""
From-scratch byte-level BPE (Byte Pair Encoding) tokenizer.

Implements the core algorithm from scratch, the same way GPT-2's tokenizer
works internally:
  - operates on raw UTF-8 bytes (0-255), not characters
  - get_stats() / merge() as the two core primitives
  - GPT-2-style regex pre-splitting so merges never cross word/punctuation/
    whitespace boundaries
  - deterministic encode() that always applies the earliest-learned
    applicable merge (lowest merge id)

Performance note: training uses INCREMENTAL pair-count bookkeeping instead
of rescanning the whole corpus on every merge. A naive implementation
rescans every token on every single merge step, which is fine for a few KB
of toy text but becomes the dominant cost once the corpus is real (megabytes,
thousands of merges). This version instead keeps a running pair -> count map
and a pair -> set-of-sequence-indices index, and when a merge happens, only
updates the counts/positions actually touched by that merge instead of
recomputing everything from scratch - the same approach Karpathy's minbpe
uses.

No tiktoken, no HuggingFace tokenizers, no sentencepiece - every byte of
this file is the from-scratch algorithm.

"""

from __future__ import annotations

import json
import regex as re
from collections import Counter, defaultdict

'''
Same chunking regex GPT-2 uses to pre-split text before BPE merges are
allowed to happen (contractions, letter runs, digit runs, punctuation
runs, whitespace) - this is what stops BPE from merging across word or
whitespace boundaries. Reproduced here verbatim (this is a regex
pattern, not a pretrained model or a downloaded tokenizer artifact).
'''
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _get_chunk_ids(text: str) -> list[list[int]]:
    """Split text with the GPT-2 regex, return each chunk as a list of byte ids."""
    chunks = re.findall(GPT2_SPLIT_PATTERN, text)
    return [list(chunk.encode("utf-8")) for chunk in chunks]


class BPETokenizer:
    """A from-scratch, GPT-2-style byte-level BPE tokenizer."""

    def __init__(self):
        # merges: (id1, id2) -> new_id, in the order they were learned.
        # Order matters for encode() (earliest-learned merge wins).
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: id -> bytes, seeded with the 256 raw bytes, then every
        # learned merge's concatenated bytes.
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        assert vocab_size >= 256, "vocab_size must be >= 256 (the raw byte alphabet)"
        num_merges = vocab_size - 256

        chunks = _get_chunk_ids(text)
        if verbose:
            n_bytes = sum(len(c) for c in chunks)
            print(f"[bpe] training on {n_bytes:,} bytes across {len(chunks):,} chunks, "
                  f"target vocab_size={vocab_size} ({num_merges:,} merges)")

        # --- incremental bookkeeping setup -------------------------------
        # pair_counts: (a, b) -> total count across the whole corpus
        # pair_to_chunks: (a, b) -> set of chunk indices containing that pair
        # This lets us, after each merge, only re-examine the chunks that
        # actually contained the merged pair, instead of rescanning everything.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_chunks: dict[tuple[int, int], set[int]] = defaultdict(set)

        def register_chunk(i: int, ids: list[int]) -> None:
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += 1
                pair_to_chunks[(a, b)].add(i)

        for i, ids in enumerate(chunks):
            register_chunk(i, ids)

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = dict(self.vocab)

        for merge_i in range(num_merges):
            if not pair_counts:
                if verbose:
                    print(f"[bpe] stopping early at {256 + merge_i} tokens — "
                          f"no more pairs to merge")
                break

            # most frequent pair, tie-broken deterministically (lowest ids first)
            top_pair = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = 256 + merge_i
            merges[top_pair] = new_id
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]

            # Only touch chunks that actually contain this pair.
            affected = pair_to_chunks.pop(top_pair, set())
            pair_counts.pop(top_pair, None)

            for ci in affected:
                old_ids = chunks[ci]

                # Remove this chunk's old pair contributions before rewriting it.
                for a, b in zip(old_ids, old_ids[1:]):
                    pair_counts[(a, b)] -= 1
                    if pair_counts[(a, b)] <= 0:
                        del pair_counts[(a, b)]
                    pair_to_chunks[(a, b)].discard(ci)

                new_ids = _merge_ids(old_ids, top_pair, new_id)
                chunks[ci] = new_ids

                # Re-register only this chunk's new pairs.
                register_chunk(ci, new_ids)

            if verbose and (merge_i + 1) % 500 == 0:
                print(f"[bpe] merge {merge_i + 1}/{num_merges} -> vocab size {256 + merge_i + 1}")

        self.merges = merges
        self.vocab = vocab
        if verbose:
            n_bytes = sum(len(c) for c in chunks)
            n_tokens = sum(len(c) for c in chunks)
            print(f"[bpe] done. final vocab size={len(vocab)}, "
                  f"merges learned={len(merges)}, tokens after training={n_tokens:,}")

    # ------------------------------------------------------------------
    # Encode / decode — identical logic to the notebook, just class methods
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk_ids in _get_chunk_ids(text):
            ids.extend(self._encode_chunk(chunk_ids))
        return ids

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            '''
                always apply the EARLIEST-learned applicable merge (lowest
                merge id) - this is what makes encoding deterministic and
                consistent with how the vocabulary was built.
            '''
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "merges": [[list(pair), idx] for pair, idx in self.merges.items()],
            "vocab_size": len(self.vocab),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.merges = {tuple(pair): idx for pair, idx in payload["merges"]}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # rebuild vocab bytes in merge order (merges dict preserves insertion order in py3.7+)
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


if __name__ == "__main__":
    # quick smoke test
    sample = "The National Aeronautics and Space Administration (NASA) conducts wind tunnel research. " * 50
    tok = BPETokenizer()
    tok.train(sample, vocab_size=300, verbose=True)
    ids = tok.encode(sample[:200])
    back = tok.decode(ids)
    assert back == sample[:200], "round-trip failed"
    print("round-trip OK, sample ids:", ids[:20])
