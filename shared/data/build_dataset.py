"""
Train a ByteLevel BPE tokenizer with HuggingFace `tokenizers` and export .npy datasets.

Uses the GPT-2 pretokenizer regex (same PAT as pre_tokenizing.py / Tokenizer.py),
256 byte tokens + special tokens, and line-by-line encoding with <|endoftext|> per split.

Usage (from repo root):
    python -m shared.data.build_dataset --vocab-size 32768
    python -m shared.data.build_dataset --vocab-size 110592 --raw-dir shared/data/raw/wikitext103 --dataset-name wikitext103
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import numpy as np
import numpy.lib.format as npf
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

# Same regex as shared/tools/utils/pre_tokenizing.py and Tokenizer.py
GPT2_PRETOKENIZER_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
DEFAULT_RAW_DIR = "shared/data/raw/wikitext103"
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_DATASET_NAME = "wikitext103"
DEFAULT_VOCAB_SIZE = 30000
ENCODE_BATCH_LINES = 256
NPY_WRITE_CHUNK_TOKENS = 4_000_000

def build_bytelevel_bpe_tokenizer(
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Create a ByteLevel BPE tokenizer with the project pretokenizer regex."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(GPT2_PRETOKENIZER_REGEX),
                behavior="isolated",
                invert=False,
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False,
                use_regex=False,
            ),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=0,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def _normalize_raw_line(raw_line: bytes) -> str:
    return raw_line.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8", errors="ignore")


def iter_training_text(path: str, special_tokens: list[str]):
    """
    Yield training text with the same normalization and special-token removal
    used in parallel_pre_tokenize during custom BPE training.

    Reads line-by-line and splits on special tokens incrementally so the full
    file is never loaded into memory.
    """
    if not special_tokens:
        with open(path, "rb") as handle:
            for raw_line in handle:
                line = _normalize_raw_line(raw_line)
                if line:
                    yield line
        return

    pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
    current_parts: list[str] = []

    with open(path, "rb") as handle:
        for raw_line in handle:
            line = _normalize_raw_line(raw_line)
            pos = 0
            for match in pattern.finditer(line):
                if match.start() > pos:
                    current_parts.append(line[pos : match.start()])
                part = "".join(current_parts)
                if part:
                    yield part
                current_parts.clear()
                pos = match.end()
            if pos < len(line):
                current_parts.append(line[pos:])

    trailing = "".join(current_parts)
    if trailing:
        yield trailing


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

    try:
        with open(tmp_bin, "wb") as out_bin, open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                batch.append(line)
                line_count += 1

                if len(batch) < batch_lines:
                    continue

                for encoding in tokenizer.encode_batch(batch):
                    arr = np.asarray(encoding.ids, dtype=np.int32)
                    arr.tofile(out_bin)
                    token_count += arr.size
                    if arr.size:
                        token_min = min(token_min, int(arr.min()))
                        token_max = max(token_max, int(arr.max()))
                batch.clear()

                if line_count % 50_000 == 0:
                    print(f"  encoded {line_count:,} lines -> {token_count:,} tokens")

            if batch:
                for encoding in tokenizer.encode_batch(batch):
                    arr = np.asarray(encoding.ids, dtype=np.int32)
                    arr.tofile(out_bin)
                    token_count += arr.size
                    if arr.size:
                        token_min = min(token_min, int(arr.min()))
                        token_max = max(token_max, int(arr.max()))

            eos_arr = np.array([eos_id], dtype=np.int32)
            eos_arr.tofile(out_bin)
            token_count += 1
            token_min = min(token_min, eos_id)
            token_max = max(token_max, eos_id)

        save_token_ids_npy(tmp_bin, output_path, token_count)
    finally:
        tmp_bin.unlink(missing_ok=True)

    return token_count, token_min, token_max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ByteLevel BPE and export tokenized .npy datasets."
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Final vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
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
        help=f"Dataset prefix for output files (default: {DEFAULT_DATASET_NAME})",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=DEFAULT_SPECIAL_TOKENS,
        help="Special tokens added after the 256 byte tokens",
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
    out_dir = Path(f"shared/data/processed/{args.dataset_name}/{args.vocab_size}_hf")
    tok_dir = Path(f"shared/data/tokenizers/{args.dataset_name}/{args.vocab_size}_hf")
    train_txt = raw_dir / f"{args.dataset_name}_train.txt"

    if not train_txt.exists():
        print(f"Training file not found: {train_txt}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    print(f"vocab_size={args.vocab_size}")
    print(f"train={train_txt}")
    print(f"regex={GPT2_PRETOKENIZER_REGEX}")

    tokenizer, trainer = build_bytelevel_bpe_tokenizer(args.vocab_size, args.special_tokens)

    print("Training ByteLevel BPE...")
    started = time.time()
    training_text = iter_training_text(str(train_txt), args.special_tokens)
    tokenizer.train_from_iterator(training_text, trainer=trainer)
    print(f"Training done in {(time.time() - started) / 60:.1f} min")

    tokenizer_path = tok_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {tokenizer_path}")

    eos_id = tokenizer.token_to_id(args.special_tokens[0])
    if eos_id is None:
        raise ValueError(f"Special token {args.special_tokens[0]!r} was not added to the vocab")

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
