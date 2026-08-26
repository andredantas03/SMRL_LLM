"""
Split a concatenated LM .npy (build_dataset.py) back into documents and
attach labels from a save_to_disk() HuggingFace dataset.

Two ways the .npy can look:

1. Docs separated by <|endoftext|> (id eos_id), optional extra EOS at EOF.
   Split on that id (original pipeline).

2. One document per line, no EOS between them, extra EOS only at EOF
   (current export_hf_txt with empty --eos). Re-encode each HF text and
   check that the concatenation matches the .npy prefix.

Usage (from repo root):
    python -m shared.data.segment_npy_labels \\
        --npy shared/data/processed/ag_news/30000_sennrich/ag_news-train.npy \\
        --hf-dir shared/data/raw/ag_news --split train

    python -m shared.data.segment_npy_labels \\
        --npy shared/data/processed/ag_news/30000_sennrich/ag_news-test.npy \\
        --hf-dir shared/data/raw/ag_news --split test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from shared.data.export_hf_txt import resolve_split
from shared.data.tokenizer import DEFAULT_EOS, Tokenizer

DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"


def collect_texts_and_labels(
    dataset: Dataset,
    text_column: str,
    label_column: str,
) -> tuple[list[str], np.ndarray, int]:
    if text_column not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise SystemExit(f"Column {text_column!r} not found. Available: {cols}")
    if label_column not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise SystemExit(f"Column {label_column!r} not found. Available: {cols}")

    texts: list[str] = []
    labels: list[int] = []
    skipped = 0
    for example in dataset:
        text = example[text_column]
        if text is None or not str(text).strip():
            skipped += 1
            continue
        text = re.sub(r"<br\s*/?>", " ", str(text), flags=re.I)
        text = " ".join(text.split())
        if not text:
            skipped += 1
            continue
        texts.append(text)
        labels.append(int(example[label_column]))
    return texts, np.asarray(labels, dtype=np.int64), skipped


def split_on_eos(token_ids: np.ndarray, eos_id: int) -> list[np.ndarray]:
    """Cut the concatenated stream on eos_id. Drops empty segments."""
    eos_pos = np.flatnonzero(token_ids == eos_id)
    if eos_pos.size == 0:
        return []

    docs: list[np.ndarray] = []
    prev = 0
    for pos in eos_pos.tolist():
        seg = token_ids[prev:pos]
        if seg.size:
            docs.append(np.asarray(seg, dtype=np.int32))
        prev = pos + 1
    return docs


def docs_from_texts(tokenizer: Tokenizer, texts: list[str]) -> list[np.ndarray]:
    docs: list[np.ndarray] = []
    for text in texts:
        ids = tokenizer.encode(text)
        if not ids:
            raise SystemExit("A document encoded to zero tokens; check the tokenizer.")
        docs.append(np.asarray(ids, dtype=np.int32))
    return docs


def pack_docs(docs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(docs) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, doc in enumerate(docs):
        offsets[i + 1] = offsets[i] + doc.size
    tokens = np.empty(int(offsets[-1]), dtype=np.int32)
    for i, doc in enumerate(docs):
        tokens[offsets[i] : offsets[i + 1]] = doc
    return tokens, offsets


def infer_tokenizer_dir(npy_path: Path) -> Path:
    # .../processed/{name}/{vocab}_sennrich/{name}-{split}.npy
    # -> .../tokenizers/{name}/{vocab}_sennrich/
    vocab_dir = npy_path.parent.name
    dataset_name = npy_path.parent.parent.name
    return Path("shared/data/tokenizers") / dataset_name / vocab_dir


def stem_prefix(npy_path: Path) -> str:
    return npy_path.stem  # e.g. ag_news-train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a concatenated .npy and attach HuggingFace labels."
    )
    parser.add_argument("--npy", required=True, help="Concatenated token id .npy from build_dataset.py.")
    parser.add_argument("--hf-dir", required=True, help="Path for datasets.load_from_disk().")
    parser.add_argument("--split", default="train", help="DatasetDict split (default: train).")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory (codes.txt + vocab.json). Default: inferred from --npy.",
    )
    parser.add_argument("--eos", default=DEFAULT_EOS, help=f"Doc separator token (default: {DEFAULT_EOS}).")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for *-docs.npy, *-offsets.npy, *-labels.npy (default: same as --npy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npy_path = Path(args.npy)
    if not npy_path.exists():
        print(f"npy not found: {npy_path}", file=sys.stderr)
        sys.exit(1)

    tok_dir = Path(args.tokenizer) if args.tokenizer else infer_tokenizer_dir(npy_path)
    tokenizer = Tokenizer.from_file(tok_dir, eos=args.eos)
    eos_id = tokenizer.eos_id

    token_ids = np.asarray(np.load(npy_path, mmap_mode="r"))
    hf_path = Path(args.hf_dir)
    dataset = load_from_disk(str(hf_path))
    split_ds = resolve_split(dataset, args.split)
    texts, labels, skipped = collect_texts_and_labels(
        split_ds, args.text_column, args.label_column
    )

    docs_eos = split_on_eos(token_ids, eos_id=eos_id)
    if len(docs_eos) == len(labels):
        docs = docs_eos
        mode = "eos"
    else:
        docs = docs_from_texts(tokenizer, texts)
        packed, _ = pack_docs(docs)
        if packed.size > token_ids.size or not np.array_equal(
            token_ids[: packed.size], packed
        ):
            raise SystemExit(
                f"Document/label mismatch: {len(docs_eos)} EOS segments vs "
                f"{len(labels)} labels (skipped empty texts={skipped}). "
                "Re-encoded texts also do not match the .npy prefix. "
                "Rebuild the .npy (build_dataset) with the same texts as --hf-dir."
            )
        mode = "re-encode"
        print(
            f"EOS split gave {len(docs_eos)} docs; using re-encode of {len(docs)} HF texts "
            f"(npy prefix matches)."
        )

    if len(docs) != len(labels):
        raise SystemExit(
            f"Document/label mismatch: {len(docs)} docs vs {len(labels)} labels "
            f"(skipped empty texts={skipped})."
        )

    out_dir = Path(args.output_dir) if args.output_dir else npy_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = stem_prefix(npy_path)

    tokens, offsets = pack_docs(docs)
    tokens_path = out_dir / f"{prefix}-docs.npy"
    offsets_path = out_dir / f"{prefix}-offsets.npy"
    labels_path = out_dir / f"{prefix}-labels.npy"
    meta_path = out_dir / f"{prefix}-meta.json"

    np.save(tokens_path, tokens)
    np.save(offsets_path, offsets)
    np.save(labels_path, labels)
    meta_path.write_text(
        json.dumps(
            {
                "n_docs": int(len(docs)),
                "n_tokens": int(tokens.size),
                "mode": mode,
                "eos": args.eos,
                "eos_id": int(eos_id),
                "newline_id": tokenizer.newline_id,
                "npy": str(npy_path),
                "hf_dir": str(hf_path),
                "split": args.split,
                "label_column": args.label_column,
                "skipped_empty_texts": int(skipped),
                "tokenizer": str(tok_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lengths = np.diff(offsets)
    print(f"docs={len(docs)} tokens={tokens.size} labels={labels.tolist()[:8]}...")
    print(f"label_counts={ {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))} }")
    print(f"doc_len min/mean/max={int(lengths.min())}/{float(lengths.mean()):.1f}/{int(lengths.max())}")
    print(f"wrote {tokens_path}")
    print(f"wrote {offsets_path}")
    print(f"wrote {labels_path}")
    print(f"wrote {meta_path}")
    print("load: tokens[offsets[i]:offsets[i+1]], labels[i]")


if __name__ == "__main__":
    main()
