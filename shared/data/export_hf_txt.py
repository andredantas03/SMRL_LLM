"""
Export a HuggingFace dataset saved with save_to_disk() to a single .txt
with <|endoftext|> between documents.

Usage (from repo root):
    python -m shared.data.export_hf_txt --input shared/data/raw/imdb
    python -m shared.data.export_hf_txt --input shared/data/raw/ag_news --split train
    python -m shared.data.export_hf_txt --input shared/data/raw/imdb --split train --output shared/data/raw/imdb/imdb_train.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

DEFAULT_EOS = "<|endoftext|>"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_SPLIT = "train"


def resolve_split(dataset: Dataset | DatasetDict, split: str | None) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if split is None:
        if "train" in dataset:
            split = "train"
        else:
            split = next(iter(dataset.keys()))
        print(f"No --split given; using '{split}'", file=sys.stderr)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise SystemExit(f"Split {split!r} not found. Available: {available}")
    return dataset[split]


def export_to_txt(
    dataset: Dataset,
    output_path: Path,
    text_column: str,
    eos: str,
) -> tuple[int, int]:
    if text_column not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise SystemExit(f"Column {text_column!r} not found. Available: {cols}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = example[text_column]
            if text is None:
                skipped += 1
                continue
            text = str(text).strip()
            if not text:
                skipped += 1
                continue
            handle.write(text)
            handle.write("\n")
            handle.write(eos)
            handle.write("\n")
            written += 1

    return written, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a save_to_disk() dataset and write one .txt with EOS between documents."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path passed to datasets.load_from_disk().",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Split to export when the path is a DatasetDict (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help=f"Text field name (default: {DEFAULT_TEXT_COLUMN}).",
    )
    parser.add_argument(
        "--eos",
        default=DEFAULT_EOS,
        help=f"Document separator (default: {DEFAULT_EOS}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .txt path. Default: <input>/<input.name>_<split>.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    dataset = load_from_disk(str(input_path))
    split_ds = resolve_split(dataset, args.split)

    output_path = (
        Path(args.output)
        if args.output
        else input_path / f"{input_path.name}_{args.split}.txt"
    )

    print(f"input={input_path}")
    print(f"split={args.split} n={len(split_ds)}")
    print(f"text_column={args.text_column}")
    print(f"output={output_path}")

    written, skipped = export_to_txt(split_ds, output_path, args.text_column, args.eos)
    print(f"wrote {written} documents ({skipped} empty skipped) -> {output_path}")


if __name__ == "__main__":
    main()
