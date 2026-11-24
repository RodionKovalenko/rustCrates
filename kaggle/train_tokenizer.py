"""German byte-level BPE tokenizer for the pretraining (German Wikipedia).

Trains on one Wikipedia parquet part (default part 1: the validation articles of the pretraining come from part 0)
and writes a tokenizers JSON with the same special tokens the project uses (<eos> ends every article):

    python kaggle/train_tokenizer.py --out src/neural_networks/tokenizers/german_bpe_32k.json --vocab 32000

The GPT-NeoX tokenizer of the first pretraining runs is built for English: German words split into more pieces and
its 50 k entries take 38 M of the model's 63 M parameters (embedding and head). 32 k German entries give fewer tokens
per text and leave the parameters to the transformer layers.
"""

import argparse
import os
import sys
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_pretrain import download  # noqa: E402

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<eos>", "<bos>", "<sep>", "<context>", "</context>"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="tokenizer JSON to write")
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--part", type=int, default=1, help="Wikipedia parquet part to train on")
    parser.add_argument("--parquet", help="local parquet file instead of downloading --part")
    parser.add_argument("--download-dir", default="/tmp/wiki_download")
    parser.add_argument("--max-articles", type=int, default=0, help="limit (0 = all articles of the part)")
    args = parser.parse_args()

    if args.parquet:
        path = args.parquet
    else:
        os.makedirs(args.download_dir, exist_ok=True)
        path = download(args.part, args.download_dir)
    table = pq.read_table(path, columns=["title", "text"])
    titles, texts = table.column("title").to_pylist(), table.column("text").to_pylist()
    if args.max_articles:
        titles, texts = titles[: args.max_articles], texts[: args.max_articles]
    print(f"{len(texts)} articles from {os.path.basename(path)}", flush=True)

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    started = time.time()
    tokenizer.train_from_iterator((f"{t}\n\n{x}" for t, x in zip(titles, texts)), trainer=trainer, length=len(texts))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tokenizer.save(args.out)
    print(f"vocabulary {tokenizer.get_vocab_size()}, eos id {tokenizer.token_to_id('<eos>')}, {time.time() - started:.0f} s -> {args.out}")

    # Tokens per word against the GPT-NeoX tokenizer on a sample.
    neox = Tokenizer.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "neural_networks", "tokenizers", "gtp_neox_tokenizer.json"))
    sample = [f"{t}\n\n{x}" for t, x in zip(titles[-500:], texts[-500:])]
    words = sum(len(s.split()) for s in sample)
    for name, tok in [("german", tokenizer), ("gpt-neox", neox)]:
        n = sum(len(e.ids) for e in tok.encode_batch(sample))
        print(f"{name}: {n / words:.2f} tokens per word")


if __name__ == "__main__":
    main()
