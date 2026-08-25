"""
Wrapper around the project's HuggingFace ByteLevel BPE (`tokenizer.json`).
Layout on disk (created by build_dataset.py):
    shared/data/tokenizers/{dataset_name}/{vocab_size}_hf/tokenizer.json
Usage:
    from shared.data.tokenizer import Tokenizer
    tok = Tokenizer.from_dataset("imdb", vocab_size=30000)
    ids = tok.encode("a short review")
    text = tok.decode(ids)
"""

from __future__ import annotations
from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer

DEFAULT_EOS = "<|endoftext|>"
DEFAULT_TOKENIZERS_ROOT = Path("shared/data/tokenizers")
DEFAULT_VOCAB_SIZE = 30000

class Tokenizer:
    def __init__(
        self,
        hf_tokenizer: HFTokenizer,
        eos: str = DEFAULT_EOS,
        path: Path | None = None,
    ):
        self._tok = hf_tokenizer
        self.eos = eos
        self.path = path
        eos_id = hf_tokenizer.token_to_id(eos)
        if eos_id is None:
            raise ValueError(f"Special token {eos!r} not in tokenizer")
        self.eos_id = int(eos_id)
        # No dedicated pad token in these BPEs; ClassificationDataset uses EOS.
        self.pad_id = self.eos_id

    @classmethod
    def from_file(cls, path: Path | str, eos: str = DEFAULT_EOS) -> "Tokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(HFTokenizer.from_file(str(path)), eos=eos, path=path)
    
    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())
   
    @property
    def newline_id(self) -> int | None:
        ids = self._tok.encode("\n").ids
        if len(ids) == 1:
            return int(ids[0])
        return None
    
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(self._tok.encode(text).ids)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if int(i) != self.eos_id]
        return self._tok.decode(list(map(int, ids)))
    
    def token_to_id(self, token: str) -> int | None:
        tid = self._tok.token_to_id(token)
        return None if tid is None else int(tid)
    
    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(int(token_id))