"""
Sennrich BPE (subword-nmt) + integer vocabulary in one object.

Machine A — codes.txt: apply-bpe (characters + </w> merges, @@ on encode).
Machine B — vocab.json: piece → id. Specials are fixed:
    [PAD]=0, [UNK]=1, <|endoftext|>=2

Layout (created by build_dataset / learn-bpe):
    shared/data/tokenizers/{dataset}/{vocab_size}_sennrich/
        codes.txt
        vocab.json

Usage:
    from shared.data.tokenizer import Tokenizer
    tok = Tokenizer.from_file("shared/data/tokenizers/ag_news/30000_sennrich")
    ids = tok.encode("The cat sat")
    text = tok.decode(ids)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from subword_nmt.apply_bpe import BPE

DEFAULT_EOS = "<|endoftext|>"
DEFAULT_PAD = "[PAD]"
DEFAULT_UNK = "[UNK]"
DEFAULT_SPECIAL_TOKENS = (DEFAULT_PAD, DEFAULT_UNK, DEFAULT_EOS)
DEFAULT_TOKENIZERS_ROOT = Path("shared/data/tokenizers")
DEFAULT_VOCAB_SIZE = 30000
CODES_NAME = "codes.txt"
VOCAB_NAME = "vocab.json"


class Tokenizer:
    def __init__(
        self,
        bpe: BPE,
        stoi: dict[str, int],
        eos: str = DEFAULT_EOS,
        pad: str = DEFAULT_PAD,
        unk: str = DEFAULT_UNK,
        path: Path | None = None,
    ):
        self.bpe = bpe
        self.stoi = dict(stoi)
        self.itos = _itos_from_stoi(self.stoi)
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.path = path

        for name in (pad, unk, eos):
            if name not in self.stoi:
                raise ValueError(f"Special token {name!r} is missing from the vocabulary")

        self.pad_id = int(self.stoi[pad])
        self.unk_id = int(self.stoi[unk])
        self.eos_id = int(self.stoi[eos])

    @classmethod
    def from_codes(
        cls,
        codes_path: Path | str,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
        eos: str = DEFAULT_EOS,
        pad: str = DEFAULT_PAD,
        unk: str = DEFAULT_UNK,
    ) -> "Tokenizer":
        """Load machine A only. Vocabulary is just the three specials (for build_vocab)."""
        codes_path = Path(codes_path)
        bpe = _load_bpe(codes_path, glossaries=[eos])
        stoi = {tok: i for i, tok in enumerate(special_tokens)}
        return cls(bpe, stoi, eos=eos, pad=pad, unk=unk, path=codes_path.parent)

    @classmethod
    def from_file(cls, path: Path | str, eos: str = DEFAULT_EOS) -> "Tokenizer":
        """Load a tokenizer directory, or a codes.txt / vocab.json inside it."""
        path = Path(path)
        tok_dir = _resolve_tokenizer_dir(path)
        codes_path = tok_dir / CODES_NAME
        vocab_path = tok_dir / VOCAB_NAME
        if not codes_path.exists():
            raise FileNotFoundError(codes_path)

        bpe = _load_bpe(codes_path, glossaries=[eos])
        if vocab_path.exists():
            stoi = _load_vocab(vocab_path)
        else:
            stoi = {tok: i for i, tok in enumerate(DEFAULT_SPECIAL_TOKENS)}
        return cls(bpe, stoi, eos=eos, path=tok_dir)

    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        suffix: str = "sennrich",
        root: Path | str = DEFAULT_TOKENIZERS_ROOT,
    ) -> "Tokenizer":
        tok_dir = Path(root) / dataset_name / f"{vocab_size}_{suffix}"
        return cls.from_file(tok_dir)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def newline_id(self) -> int | None:
        """Sennrich encode strips newlines; segmentation should use EOS only."""
        return None

    def tokenize(self, text: str) -> list[str]:
        """Machine A: text → BPE pieces (no ids)."""
        text = text.rstrip("\r\n")
        if not text:
            return []
        if text == self.eos:
            return [self.eos]
        segmented = self.bpe.process_line(text).strip()
        if not segmented:
            return []
        return segmented.split()

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """A then B: text → ids."""
        ids = [self.stoi.get(piece, self.unk_id) for piece in self.tokenize(text)]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = False) -> list[list[int]]:
        return [self.encode(text, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        skip = {self.pad_id, self.unk_id, self.eos_id} if skip_special_tokens else {self.pad_id}
        pieces: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in skip:
                continue
            if token_id < 0 or token_id >= len(self.itos) or self.itos[token_id] is None:
                continue
            pieces.append(self.itos[token_id])
        text = " ".join(pieces)
        return text.replace(f"{self.bpe.separator} ", "").replace(self.bpe.separator, "")

    def token_to_id(self, token: str) -> int | None:
        tid = self.stoi.get(token)
        return None if tid is None else int(tid)

    def id_to_token(self, token_id: int) -> str | None:
        token_id = int(token_id)
        if token_id < 0 or token_id >= len(self.itos):
            return None
        return self.itos[token_id]

    def build_vocab(
        self,
        lines: Iterable[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        """
        Machine B from a train corpus: count BPE pieces, keep specials first,
        then the most frequent pieces until vocab_size.
        """
        if vocab_size < len(special_tokens):
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than {len(special_tokens)} special tokens"
            )

        counts: Counter[str] = Counter()
        special_set = set(special_tokens)
        for line in lines:
            for piece in self.tokenize(line):
                if piece in special_set:
                    continue
                counts[piece] += 1

        stoi = {tok: i for i, tok in enumerate(special_tokens)}
        remaining = vocab_size - len(special_tokens)
        for piece, _ in counts.most_common(remaining):
            if piece not in stoi:
                stoi[piece] = len(stoi)

        self.stoi = stoi
        self.itos = _itos_from_stoi(stoi)
        self.pad_id = int(stoi[self.pad])
        self.unk_id = int(stoi[self.unk])
        self.eos_id = int(stoi[self.eos])

    def save_vocab(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path is not None else (self.path or Path(".")) / VOCAB_NAME
        if path.is_dir():
            path = path / VOCAB_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.stoi, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _resolve_tokenizer_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.name in {CODES_NAME, VOCAB_NAME}:
        return path.parent
    if path.exists() and path.suffix == ".txt":
        return path.parent
    raise FileNotFoundError(path)


def _load_bpe(codes_path: Path, glossaries: list[str]) -> BPE:
    with codes_path.open("r", encoding="utf-8") as handle:
        bpe = BPE(handle, glossaries=glossaries)
    # apply_bpe builds '^({})$' without escaping; <|endoftext|> contains '|'.
    escaped = "|".join(re.escape(g) for g in glossaries)
    bpe.glossaries_regex = re.compile(f"^({escaped})$") if glossaries else None
    return bpe


def _load_vocab(vocab_path: Path) -> dict[str, int]:
    raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in raw.items()}


def _itos_from_stoi(stoi: dict[str, int]) -> list[str | None]:
    if not stoi:
        return []
    size = max(stoi.values()) + 1
    itos: list[str | None] = [None] * size
    for token, idx in stoi.items():
        itos[int(idx)] = token
    return itos
