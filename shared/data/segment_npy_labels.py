"""
Split a concatenated LM .npy (build_dataset.py) back into documents and
attach the original labels from a save_to_disk() HuggingFace dataset.

The token stream from encode_file_to_npy looks like:

    doc0_tokens  <|endoftext|>  \\n  doc1_tokens  <|endoftext|>  \\n  ...  <|endoftext|>

Labels are taken in the same order as export_hf_txt.py (non-empty text only).

Usage (from repo root):
    python -m shared.data.segment_npy_labels \\
        --npy shared/data/processed/imdb/30000_hf/imdb-train.npy \\
        --hf-dir shared/data/raw/imdb --split train

    python -m shared.data.segment_npy_labels \\
        --npy shared/data/processed/ag_news/30000_hf/ag_news-train.npy \\
        --hf-dir shared/data/raw/ag_news --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer

from shared.data.export_hf_txt import resolve_split

DEFAULT_EOS = "<|endoftext|>"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"


def collect_labels(
    dataset: Dataset,
    text_column: str,
    label_column: str,
) -> tuple[np.ndarray, int]:
    if text_column not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise SystemExit(f"Column {text_column!r} not found. Available: {cols}")
    if label_column not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise SystemExit(f"Column {label_column!r} not found. Available: {cols}")

    labels: list[int] = []
    skipped = 0
    for example in dataset:
        text = example[text_column]
        if text is None or not str(text).strip():
            skipped += 1
            continue
        labels.append(int(example[label_column]))
    return np.asarray(labels, dtype=np.int64), skipped


def newline_id(tokenizer: Tokenizer) -> int | None:
    ids = tokenizer.encode("\n").ids
    if len(ids) == 1:
        return int(ids[0])
    return None


def split_on_eos(token_ids: np.ndarray, eos_id: int, nl_id: int | None) -> list[np.ndarray]:
    """
    Cut the concatenated stream on eos_id. Drops the extra trailing EOS that
    encode_file_to_npy appends, and strips the newline tokens from the
    '<|endoftext|>\\n' separator lines.
    """
    eos_pos = np.flatnonzero(token_ids == eos_id)
    if eos_pos.size == 0:
        raise SystemExit(f"No EOS token id={eos_id} found in the .npy stream.")

    docs: list[np.ndarray] = []
    prev = 0
    for pos in eos_pos.tolist():
        seg = token_ids[prev:pos]
        if nl_id is not None:
            if seg.size and int(seg[0]) == nl_id:
                seg = seg[1:]
            if seg.size and int(seg[-1]) == nl_id:
                seg = seg[:-1]
        if seg.size:
            docs.append(np.asarray(seg, dtype=np.int32))
        prev = pos + 1
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
        help="tokenizer.json used to encode the .npy (default: inferred from --npy path).",
    )
    parser.add_argument("--eos", default=DEFAULT_EOS, help=f"Special token that separates docs (default: {DEFAULT_EOS}).")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for *-docs.npy, *-offsets.npy, *-labels.npy (default: same as --npy).",
    )
    return parser.parse_args()


def infer_tokenizer_path(npy_path: Path) -> Path:
    # .../processed/{name}/{vocab}_hf/{name}-{split}.npy
    # -> .../tokenizers/{name}/{vocab}_hf/tokenizer.json
    vocab_dir = npy_path.parent.name
    dataset_name = npy_path.parent.parent.name
    return Path("shared/data/tokenizers") / dataset_name / vocab_dir / "tokenizer.json"


def stem_prefix(npy_path: Path) -> str:
    return npy_path.stem  # e.g. imdb-train


def main() -> None:
    args = parse_args()
    npy_path = Path(args.npy)
    if not npy_path.exists():
        print(f"npy not found: {npy_path}", file=sys.stderr)
        sys.exit(1)

    tok_path = Path(args.tokenizer) if args.tokenizer else infer_tokenizer_path(npy_path)
    if not tok_path.exists():
        print(f"tokenizer not found: {tok_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tok_path))
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit(f"Special token {args.eos!r} not in {tok_path}")
    nl_id = newline_id(tokenizer)

    token_ids = np.load(npy_path, mmap_mode="r")
    docs = split_on_eos(token_ids, eos_id=int(eos_id), nl_id=nl_id)

    hf_path = Path(args.hf_dir)
    dataset = load_from_disk(str(hf_path))
    split_ds = resolve_split(dataset, args.split)
    labels, skipped = collect_labels(split_ds, args.text_column, args.label_column)

    if len(docs) != len(labels):
        raise SystemExit(
            f"Document/label mismatch: {len(docs)} segments vs {len(labels)} labels "
            f"(skipped empty texts={skipped}). Check EOS splitting and --split."
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
                "eos": args.eos,
                "eos_id": int(eos_id),
                "newline_id": nl_id,
                "npy": str(npy_path),
                "hf_dir": str(hf_path),
                "split": args.split,
                "label_column": args.label_column,
                "skipped_empty_texts": int(skipped),
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
    print(
        "load: tokens[offsets[i]:offsets[i+1]], labels[i]"
    )


if __name__ == "__main__":
    main()
