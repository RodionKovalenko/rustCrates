"""Pretraining data: German Wikipedia (wikimedia/wikipedia, 20231101.de on Hugging Face) as token shards.

Downloads the Parquet parts, tokenizes every article with the project tokenizer and writes the token ids as
little-endian uint16 shard files (the vocabulary has < 65536 tokens):

    <out>/train_00000.bin, train_00001.bin, ...   each about --shard-tokens tokens
    <out>/val_00000.bin                           held-out articles (--val-articles)
    <out>/meta.json                               token counts, source parts, tokenizer, eos id

Every article is "title\\n\\ntext" followed by the <eos> token, so the model learns where a document ends.
Nothing is committed to git: on Kaggle the shards land in the notebook output and later runs reuse them as input.

    python kaggle/prepare_pretrain.py --out /kaggle/working/pretrain --parts 0-7
    python kaggle/prepare_pretrain.py --out data/pretrain --parquet local_file.parquet   # local test
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

DATASET_URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.de/train-{:05d}-of-00020.parquet"
NUM_PARTS = 20
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "neural_networks", "tokenizers", "gtp_neox_tokenizer.json")
EOS = "<eos>"


def parse_parts(spec):
    """'0-3,7' -> [0, 1, 2, 3, 7]"""
    parts = []
    for item in spec.split(","):
        if "-" in item:
            a, b = item.split("-")
            parts.extend(range(int(a), int(b) + 1))
        elif item:
            parts.append(int(item))
    if any(p < 0 or p >= NUM_PARTS for p in parts):
        sys.exit(f"parts must be in 0..{NUM_PARTS - 1}")
    return parts


def download(part, directory):
    path = os.path.join(directory, f"de_{part:05d}.parquet")
    if not os.path.exists(path):
        url = DATASET_URL.format(part)
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


class ShardWriter:
    def __init__(self, out, prefix, shard_tokens):
        self.out, self.prefix, self.shard_tokens = out, prefix, shard_tokens
        self.buffer, self.buffered, self.index, self.total, self.files = [], 0, 0, 0, []

    def add(self, ids):
        self.buffer.append(np.asarray(ids, dtype=np.uint16))
        self.buffered += len(ids)
        if self.buffered >= self.shard_tokens:
            self.flush()

    def flush(self):
        if not self.buffered:
            return
        name = f"{self.prefix}_{self.index:05d}.bin"
        np.concatenate(self.buffer).astype("<u2").tofile(os.path.join(self.out, name))
        self.files.append({"file": name, "tokens": self.buffered})
        self.total += self.buffered
        print(f"  wrote {name}: {self.buffered / 1e6:.1f} M tokens", flush=True)
        self.buffer, self.buffered, self.index = [], 0, self.index + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output folder for the shards")
    parser.add_argument("--parts", default="0", help="Wikipedia parquet parts to use, e.g. '0-7' (20 parts, ~150 M tokens each)")
    parser.add_argument("--parquet", nargs="*", help="local parquet files instead of downloading --parts")
    parser.add_argument("--download-dir", default=None, help="where downloaded parquet files are kept (default: <out>/download)")
    parser.add_argument("--shard-tokens", type=int, default=50_000_000, help="tokens per shard file (2 bytes each)")
    parser.add_argument("--val-articles", type=int, default=2000, help="articles held out for validation")
    parser.add_argument("--batch", type=int, default=2000, help="articles tokenized per call")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eos = tokenizer.token_to_id(EOS)
    assert eos is not None and tokenizer.get_vocab_size() <= 65536

    if args.parquet:
        files = args.parquet
        sources = [os.path.basename(f) for f in files]
    else:
        download_dir = args.download_dir or os.path.join(args.out, "download")
        os.makedirs(download_dir, exist_ok=True)
        parts = parse_parts(args.parts)
        files = [download(p, download_dir) for p in parts]
        sources = [DATASET_URL.format(p) for p in parts]

    train = ShardWriter(args.out, "train", args.shard_tokens)
    val = ShardWriter(args.out, "val", args.shard_tokens)
    rng = random.Random(42)
    articles, started = 0, time.time()

    for path in files:
        table = pq.read_table(path, columns=["title", "text"])
        titles, texts = table.column("title").to_pylist(), table.column("text").to_pylist()
        # The validation articles come from the first file, spread over it at random (fixed seed).
        val_rows = set(rng.sample(range(len(texts)), min(args.val_articles, len(texts)))) if path == files[0] else set()
        for start in range(0, len(texts), args.batch):
            rows = range(start, min(start + args.batch, len(texts)))
            documents = [f"{titles[i]}\n\n{texts[i]}" for i in rows]
            for i, encoding in zip(rows, tokenizer.encode_batch(documents)):
                ids = encoding.ids + [eos]
                (val if i in val_rows else train).add(ids)
            articles += len(rows)
        print(f"{os.path.basename(path)}: {articles} articles, {(train.total + train.buffered) / 1e6:.0f} M train tokens, {time.time() - started:.0f} s", flush=True)

    train.flush()
    val.flush()
    meta = {
        "dataset": "wikimedia/wikipedia 20231101.de (CC BY-SA 4.0 / GFDL)",
        "sources": sources,
        "tokenizer": "src/neural_networks/tokenizers/gtp_neox_tokenizer.json",
        "dtype": "uint16 little-endian",
        "eos_id": eos,
        "articles": articles,
        "train_tokens": train.total,
        "val_tokens": val.total,
        "train_files": train.files,
        "val_files": val.files,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {train.total / 1e6:.0f} M train tokens in {len(train.files)} files, {val.total / 1e6:.1f} M validation tokens")


if __name__ == "__main__":
    main()
