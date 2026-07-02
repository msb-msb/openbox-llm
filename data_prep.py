"""
data_prep.py — stream a FineWeb-Edu sample, tokenize with GPT-2 BPE, cache to disk.

Rung 2a needs a REAL held-out val split (generalization, not overfit). We:
  1. stream HuggingFaceFW/fineweb-edu (sample-10BT) over HTTP (no full download),
  2. tokenize with tiktoken's GPT-2 BPE (no training, fully reproducible),
  3. concatenate docs with the <|endoftext|> separator,
  4. write a fixed, DISJOINT train/val split to data/{train,val}.bin as uint16.

Tokenized once and cached — reruns of training never re-tokenize. The two
training arms (nsa / full) both read these exact bytes, so the data is identical.

Note on scale: the workstation's link to HF measured ~0.5 MB/s, so the default
budget (30M train + 3M val tokens) is chosen to (a) stay a one-time ~15 min
download and (b) exceed the ~16-20M tokens a default training run consumes, so
val loss actually measures generalization (train is <1 epoch, val fully held out).
Raise --train_tokens if you have bandwidth to spare.
"""

import argparse
import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
DTYPE = np.uint16  # GPT-2 vocab is 50257 < 65536, so uint16 is exact + compact


def fill(stream_iter, enc, budget, doc_batch=256):
    """Pull docs from the stream until we've collected >= `budget` tokens.

    Returns (uint16 array of tokens, n_docs_consumed). Docs are tokenized in
    batches (tiktoken releases the GIL, so batched encode is much faster).
    """
    eot = enc.eot_token
    out = []
    n_tok = 0
    n_doc = 0
    texts = []
    for ex in stream_iter:
        texts.append(ex["text"])
        if len(texts) >= doc_batch:
            for ids in enc.encode_ordinary_batch(texts):
                ids.append(eot)
                out.append(np.array(ids, dtype=DTYPE))
                n_tok += len(ids)
            n_doc += len(texts)
            texts = []
            print(f"    ... {n_tok/1e6:6.2f}M / {budget/1e6:.1f}M tokens "
                  f"({n_doc} docs)", end="\r", flush=True)
            if n_tok >= budget:
                break
    if texts and n_tok < budget:  # flush tail
        for ids in enc.encode_ordinary_batch(texts):
            ids.append(eot)
            out.append(np.array(ids, dtype=DTYPE))
            n_tok += len(ids)
        n_doc += len(texts)
    print()
    return np.concatenate(out), n_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--train_tokens", type=int, default=30_000_000)
    ap.add_argument("--val_tokens", type=int, default=3_000_000)
    ap.add_argument("--force", action="store_true", help="re-tokenize even if cache exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_bin = os.path.join(args.out_dir, "train.bin")
    val_bin = os.path.join(args.out_dir, "val.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    if not args.force and os.path.exists(train_bin) and os.path.exists(val_bin) \
            and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        print(f"cache present, skipping. train={meta['train_tokens']/1e6:.2f}M "
              f"val={meta['val_tokens']/1e6:.2f}M tokens (use --force to redo)")
        return

    enc = tiktoken.get_encoding("gpt2")
    print(f"streaming {DATASET} ({CONFIG}) — GPT-2 BPE, vocab {enc.n_vocab}")
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    it = iter(ds)

    # val comes from the HEAD of the stream, train from the tail — disjoint docs.
    print(f"  collecting VAL ({args.val_tokens/1e6:.1f}M tokens)...")
    val, val_docs = fill(it, enc, args.val_tokens)
    print(f"  collecting TRAIN ({args.train_tokens/1e6:.1f}M tokens)...")
    train, train_docs = fill(it, enc, args.train_tokens)

    train.tofile(train_bin)
    val.tofile(val_bin)
    meta = {
        "dataset": DATASET, "config": CONFIG, "encoding": "gpt2",
        "vocab_size": enc.n_vocab, "dtype": "uint16",
        "train_tokens": int(train.size), "val_tokens": int(val.size),
        "train_docs": train_docs, "val_docs": val_docs,
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"\nwrote {train_bin} ({train.size/1e6:.2f}M tok) and "
          f"{val_bin} ({val.size/1e6:.2f}M tok)")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
    # The HF streaming client spins a background HTTP thread whose teardown can
    # race Python finalization and print a scary (harmless) GIL error. Data is
    # already safely on disk by here, so exit hard and skip that finalization.
    import sys
    sys.stdout.flush()
    os._exit(0)
