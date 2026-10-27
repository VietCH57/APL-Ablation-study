"""Phone vocabulary.

configs/vocab.json maps the 68 output phones (39 standard phones, 28 accented phones marked
with '*', and 'err') to ids 0..67, and the single-space token " " to 68. The space token is used
as the CTC blank: it never appears inside a phone sequence because sequences are split on
whitespace.

The canonical phone sequence fed to the linguistic encoder uses the same ids plus one extra
padding id (= number of vocab entries).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


class Vocab:
    def __init__(self, token_to_id: Dict[str, int], blank_token: str = " "):
        ids = sorted(token_to_id.values())
        if ids != list(range(len(ids))):
            raise ValueError("vocab ids must be contiguous and start at 0")
        if blank_token not in token_to_id:
            raise ValueError(f"blank token {blank_token!r} not in vocab")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.blank_token = blank_token
        self.blank_id = token_to_id[blank_token]
        # padding id for the canonical sequence (embedding table has one extra row)
        self.pad_id = len(token_to_id)

    @classmethod
    def from_file(cls, path: str | Path, blank_token: str = " ") -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), blank_token=blank_token)

    @property
    def num_classes(self) -> int:
        """Size of the CTC output layer (phones + blank)."""
        return len(self.token_to_id)

    @property
    def num_embeddings(self) -> int:
        """Size of the canonical-phone embedding table (phones + blank + pad)."""
        return len(self.token_to_id) + 1

    @property
    def phones(self) -> List[str]:
        return [t for t, _ in sorted(self.token_to_id.items(), key=lambda kv: kv[1]) if t != self.blank_token]

    def encode(self, text: str | Sequence[str]) -> List[int]:
        toks = text.split() if isinstance(text, str) else list(text)
        missing = [t for t in toks if t not in self.token_to_id]
        if missing:
            raise KeyError(f"phones not in vocab: {sorted(set(missing))}")
        return [self.token_to_id[t] for t in toks]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token[int(i)] for i in ids if int(i) != self.blank_id and int(i) != self.pad_id]
