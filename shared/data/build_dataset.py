"""
Encode raw .txt splits with Sennrich BPE (subword-nmt) into concatenated .npy streams.

Expects codes.txt already produced by:
    subword-nmt learn-bpe -s 29997 < {dataset}_train.txt > codes.txt

If vocab.json is missing, it is built from the train split (specials first,
then most frequent BPE pieces up to --vocab-size).

Usage (from repo root):
    python -m shared.data.build_dataset
    python -m shared.data.build_dataset --raw-dir shared/data/raw/imdb --dataset-name imdb
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import numpy.lib.format as npf

from shared.data.tokenizer import CODES_NAME, VOCAB_NAME, Tokenizer

DEFAULT_RAW_DIR = "shared/data/raw/imdb"
DEFAULT_DATASET_NAME = "imdb"
DEFAULT_VOCAB_SIZE = 30000
ENCODE_BATCH_LINES = 256
NPY_WRITE_CHUNK_TOKENS = 4_000_000


def save_token_ids_npy(
    bin_path: Path,
    output_path: Path,
    count: int,
    chunk_tokens: int = NPY_WRITE_CHUNK_TOKENS,
) -> None:
    """Convert a raw int32 binary file to .npy without loading all tokens at once."""
    header = npf.header_data_from_array_1_0(np.empty(count, dtype=np.int32))
    with open(output_path, "wb") as out_f:
        npf.write_array_header_1_0(out_f, header)
        mm = np.memmap(bin_path, dtype=np.int32, mode="r", shape=(count,))
        for start in range(0, count, chunk_tokens):
            end = min(start + chunk_tokens, count)
            out_f.write(mm[start:end].tobytes())


def encode_file_to_npy(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    eos_id: int,
    batch_lines: int = ENCODE_BATCH_LINES,
) -> tuple[int, int, int]:
    """
    Encode a text file line-by-line and write token ids incrementally to .npy.

    Returns (token_count, min_id, max_id).
    """
    tmp_bin = output_path.with_suffix(".tmp.bin")
    token_count = 0
    token_min = np.iinfo(np.int32).max
    token_max = np.iinfo(np.int32).min
    line_count = 0
    batch: list[str] = []

    def _flush(lines: list[str], out_bin) -> None:
        nonlocal token_count, token_min, token_max
        for ids in tokenizer.encode_batch(lines):
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int32)
            arr.tofile(out_bin)
            token_count += arr.size
            token_min = min(token_min, int(arr.min()))
            token_max = max(token_max, int(arr.max()))

    try:
        with open(tmp_bin, "wb") as out_bin, open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                batch.append(line)
                line_count += 1

                if len(batch) < batch_lines:
                    continue

                _flush(batch, out_bin)
                batch.clear()

                if line_count % 50_000 == 0:
                    print(f"  encoded {line_count:,} lines -> {token_count:,} tokens")

            if batch:
                _flush(batch, out_bin)

            eos_arr = np.array([eos_id], dtype=np.int32)
            eos_arr.tofile(out_bin)
            token_count += 1
            token_min = min(token_min, eos_id)
            token_max = max(token_max, eos_id)

        save_token_ids_npy(tmp_bin, output_path, token_count)
    finally:
        tmp_bin.unlink(missing_ok=True)

    return token_count, token_min, token_max


def load_or_build_tokenizer(
    tok_dir: Path,
    train_txt: Path,
    vocab_size: int,
) -> Tokenizer:
    codes_path = tok_dir / CODES_NAME
    vocab_path = tok_dir / VOCAB_NAME
    if not codes_path.exists():
        print(f"codes.txt not found: {codes_path}", file=sys.stderr)
        print(
            "Run: subword-nmt learn-bpe -s 29997 < "
            f"{train_txt} > {codes_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if vocab_path.exists():
        tokenizer = Tokenizer.from_file(tok_dir)
        print(f"Loaded vocab ({tokenizer.vocab_size} types) from {vocab_path}")
        return tokenizer

    tokenizer = Tokenizer.from_codes(codes_path)
    print("Building vocab.json from train...")
    started = time.time()
    with train_txt.open("r", encoding="utf-8") as handle:
        tokenizer.build_vocab(handle, vocab_size=vocab_size)
    tokenizer.save_vocab(vocab_path)
    print(
        f"Saved {tokenizer.vocab_size} types in "
        f"{(time.time() - started) / 60:.1f} min -> {vocab_path}"
    )
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode splits with Sennrich BPE (codes.txt + vocab.json) to .npy."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocab cap when building vocab.json (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help=f"Directory with raw .txt splits (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset prefix for input/output files (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=ENCODE_BATCH_LINES,
        help="Lines per encode_batch chunk during tokenization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(f"shared/data/processed/{args.dataset_name}/{args.vocab_size}_sennrich")
    tok_dir = Path(f"shared/data/tokenizers/{args.dataset_name}/{args.vocab_size}_sennrich")
    train_txt = raw_dir / f"{args.dataset_name}_train.txt"

    if not train_txt.exists():
        print(f"Training file not found: {train_txt}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    print(f"vocab_size_cap={args.vocab_size}")
    print(f"train={train_txt}")
    print(f"codes={tok_dir / CODES_NAME}")

    tokenizer = load_or_build_tokenizer(tok_dir, train_txt, args.vocab_size)
    eos_id = tokenizer.eos_id
    print(
        f"pad={tokenizer.pad_id} unk={tokenizer.unk_id} "
        f"eos={eos_id} vocab_size={tokenizer.vocab_size}"
    )
    gc.collect()

    for split in ["train", "validation", "test"]:
        input_path = raw_dir / f"{args.dataset_name}_{split}.txt"
        output_path = out_dir / f"{args.dataset_name}-{split}.npy"

        if not input_path.exists():
            print(f"Skipping missing split: {input_path}")
            continue

        print(f"Tokenizing {split}...")
        started = time.time()
        token_count, token_min, token_max = encode_file_to_npy(
            tokenizer,
            input_path,
            output_path,
            eos_id=eos_id,
            batch_lines=args.batch_lines,
        )
        elapsed = time.time() - started
        print(
            split,
            (token_count,),
            token_min,
            token_max,
            output_path,
            f"({elapsed / 60:.1f} min)",
        )
        gc.collect()


if __name__ == "__main__":
    main()
