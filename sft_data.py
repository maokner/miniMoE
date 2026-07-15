"""
Prepare an instruction-tuning dataset for miniMoE SFT.

Downloads a chat dataset from the Hugging Face Hub, renders each conversation
with the canonical chat template (chat_template.py), tokenizes with GPT-2 BPE,
and packs the result into fixed-length shards - the same shard-and-mmap layout
that fineweb.py uses for pretraining, so sft_train.py can stream it cheaply.

Two parallel arrays are written per shard:
  sft_<split>_<i>_tokens.npy : uint16 token ids
  sft_<split>_<i>_mask.npy   : uint8, 1 where the token is a supervised target
                               (assistant content + its terminal EOS), else 0

Packing note: conversations are concatenated into one stream and chunked into
BLOCK_SIZE blocks, exactly like the pretraining pipeline packs documents. A block
can therefore span two conversations; because every conversation begins with EOS
and uses absolute positions the same way pretraining did, this matches what the
base model already saw. The prompt mask ensures loss only ever lands on
assistant tokens regardless of block boundaries.

Defaults target HuggingFaceTB/smol-smoltalk, the small-model variant of SmolTalk
(what SmolLM2-360M was tuned on), which is a good fit for a model this size.
Override with env vars, e.g.:

    DATASET=HuggingFaceTB/smoltalk DATASET_CONFIG=all python sft_data.py
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from chat_template import render_conversation

DATASET = os.environ.get("DATASET", "HuggingFaceTB/smol-smoltalk")
DATASET_CONFIG = os.environ.get("DATASET_CONFIG", "")  # "" = default config
DATA_DIR = os.environ.get("SFT_DATA_DIR", "sft_data")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 1024))
SHARD_TOKENS = int(os.environ.get("SHARD_TOKENS", 100_000_000))
MESSAGES_KEY = os.environ.get("MESSAGES_KEY", "messages")

# HF split name -> our split name. smol-smoltalk ships train/test; we use test
# as validation for the SFT run.
SPLIT_MAP = {"train": "train", "test": "val"}


def iter_examples(hf_split):
    kwargs = {"split": hf_split}
    if DATASET_CONFIG:
        ds = load_dataset(DATASET, DATASET_CONFIG, **kwargs)
    else:
        ds = load_dataset(DATASET, **kwargs)
    return ds


def write_shard(split, index, tokens, mask):
    os.makedirs(DATA_DIR, exist_ok=True)
    base = os.path.join(DATA_DIR, f"sft_{split}_{index:04d}")
    np.save(f"{base}_tokens.npy", np.asarray(tokens, dtype=np.uint16))
    np.save(f"{base}_mask.npy", np.asarray(mask, dtype=np.uint8))
    return base


def build_split(hf_split, split, enc):
    ds = iter_examples(hf_split)
    if MESSAGES_KEY not in ds.column_names:
        raise KeyError(
            f"Dataset {DATASET} split {hf_split} has no '{MESSAGES_KEY}' column; "
            f"columns are {ds.column_names}. Set MESSAGES_KEY to the right one."
        )

    shard_index = 0
    tok_buf = np.empty(SHARD_TOKENS, dtype=np.uint16)
    mask_buf = np.empty(SHARD_TOKENS, dtype=np.uint8)
    fill = 0
    total_tokens = 0
    total_target = 0
    skipped = 0

    def flush(count):
        nonlocal shard_index
        path = write_shard(split, shard_index, tok_buf[:count], mask_buf[:count])
        shard_index += 1
        return path

    for example in tqdm(ds, desc=f"{split}"):
        # Guard per-conversation: one malformed row must never abort the split.
        try:
            tokens, is_target = render_conversation(example[MESSAGES_KEY], enc)
        except Exception:
            skipped += 1
            continue
        if tokens is None:
            skipped += 1
            continue

        total_tokens += len(tokens)
        total_target += sum(is_target)

        pos = 0
        n = len(tokens)
        while pos < n:
            space = SHARD_TOKENS - fill
            take = min(space, n - pos)
            tok_buf[fill:fill + take] = tokens[pos:pos + take]
            mask_buf[fill:fill + take] = is_target[pos:pos + take]
            fill += take
            pos += take
            if fill == SHARD_TOKENS:
                flush(SHARD_TOKENS)
                fill = 0

    if fill > 0:
        flush(fill)

    print(
        f"[{split}] {total_tokens:,} tokens across {shard_index} shard(s), "
        f"{total_target:,} supervised ({100 * total_target / max(total_tokens, 1):.1f}%), "
        f"{skipped:,} conversations skipped"
    )


def main():
    enc = tiktoken.get_encoding("gpt2")
    print(f"dataset: {DATASET}" + (f" ({DATASET_CONFIG})" if DATASET_CONFIG else ""))
    print(f"output dir: {DATA_DIR}, block size: {BLOCK_SIZE}")
    for hf_split, split in SPLIT_MAP.items():
        try:
            build_split(hf_split, split, enc)
        except ValueError as err:
            # A missing split (e.g. no test split) is non-fatal for prep.
            print(f"[{split}] skipped: {err}")


if __name__ == "__main__":
    main()
